import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import math
import pdb
import sys

from .deeplearning import masked_log
from .helpers import minfb, TermColours, ManualLRScheduler, CF, extend

def create_closure(xi, theta, tPs, tQs, tMuHat, netflow,
                   ng, dev, tau, scale, masks, treat_idcs, years,
                   optim, cf, train0, beta, key):
    additional = [None, None, None, None, None, xi, None, 0.0]
    def closure(candidate_theta):
        nonlocal xi, key
        diagnostics = {}
        resid, ssh, sss, l, conds, margs, _, candidate_xi, key = match_moments(xi,
                                              candidate_theta,
                                              tPs, tQs, tMuHat,
                                              netflow, ng, dev,
                                              tau, scale, masks, treat_idcs,
                                              years, skiptrain=False,
                                              cf = cf, train0 = train0,
                                              beyond = 0, beta = beta, key = key,
                                              diagnostics = diagnostics)
        xi = candidate_xi
        additional[:6] = [ssh, sss, l, conds, margs, xi]
        mass_max = diagnostics["max_mass_deviation"]
        additional[7] = max(additional[7], mass_max if np.isfinite(mass_max) else float("inf"))
        # 2026-10-02: Snapshot each evaluated pair before the outer optimizer changes theta or xi.
        value = float(resid)
        best = additional[6]
        if np.isfinite(value) and value < (best[0] if best else float("inf")):
            additional[6] = (value, candidate_theta, xi,
                                     (ssh, sss, l, conds, margs))
        return resid
    return closure, additional

def check_grad_norm(grads):
    total_norm = 0
    for p in jax.tree_util.tree_leaves(grads):
        total_norm += jnp.linalg.norm(p) ** 2
    total_norm = total_norm ** 0.5

    return total_norm

def tJ(dim, dev):
    return jnp.eye(dim)[:, :-1]

def tiota(dim, dev):
    return jnp.ones(dim).reshape(-1, 1)


def choices_vectorised(mus, p, q, dev):
    nty0 = mus.shape[1]
    tMuM = jnp.matmul(tJ(nty0, dev).T, mus)
    tMuF = jnp.matmul(mus, tJ(nty0, dev))
    return jnp.concatenate(
        [jnp.matmul(tMuM.reshape(-1, p.shape[1]), p.T),
         jnp.matmul(tMuF.reshape(-1, q.shape[1]), q.T)],
        axis=1)

def incumbent_population(mus, d, p0, q0, p1, q1):
    tMuM = mus[:,:-1,:]
    tMuF = mus[:,:,:-1]
    alpha = d.reshape(-1, 1, 1, 1)
    p = (1.0 - alpha) * p0[None] + alpha * p1[None]
    q = (1.0 - alpha) * q0[None] + alpha * q1[None]
    M = jnp.einsum('nij,nijk->nk', tMuM, p)
    F = jnp.einsum('nij,nijk->nk', tMuF, q)
    return jnp.concatenate([M, F], axis=1)

def create_netflow(ee, tMuHat, tPs, tQs, treat_idcs, train0, fixed=False):
    # 2026-11-03: Apply the profile's fixed or proportional choice to both entry and exit.
    entries, exits = jnp.tile(ee[:, 0], 2), jnp.tile(ee[:, 1], 2)
    if fixed:
        return lambda s: (entries - exits).astype(s.dtype)
    start = 0 if train0 else treat_idcs[0]
    regimes = jnp.zeros(len(tMuHat), dtype=tMuHat.dtype)
    regimes = regimes.at[jnp.asarray(treat_idcs)].set(1)
    reference = incumbent_population(tMuHat[start:], regimes[start:],
                                     tPs[2], tQs[2], tPs[1], tQs[1]).mean(0)
    entry_rates, exit_rates = entries / reference, exits / reference
    return lambda s: (entry_rates - exit_rates).astype(s.dtype) * s

def choices(mus, t, d, p0, q0, p1, q1, netflow, dt, dev):
    Sincumb = incumbent_population(mus, d, p0, q0, p1, q1)
    S0 = Sincumb + netflow(Sincumb)
    # 2026-10-02: Normalize each sampled economy separately after net flows.
    S = S0 / S0.sum(axis=1, keepdims=True)
    return jnp.concatenate([S, t + dt, d], axis = 1)

def check_mass(mus, s):
    ntypes = s.shape[1] // 2
    f_resid = jnp.square(mus.sum(1)[:, :-1] - s[:,ntypes:(2*ntypes)]).mean()
    m_resid = jnp.square(mus.sum(2)[:, :-1] - s[:,0:(ntypes)]).mean()
    total = jnp.sqrt(0.5 * f_resid + 0.5 * m_resid)
    offdiag = (mus.mean(0)[0,1] + mus.mean(0)[1,0])
    print(f"{TermColours.YELLOW} {total:.2f} {TermColours.RESET}", end='')
    #print(f"{TermColours.CYAN} {offdiag:.2f} {TermColours.RESET}", end='')

def matching_entropy(mus, male_margins, female_margins, scales, maskc):
    # 2026-10-02: Share entropy with reporting while keeping Bellman margins explicit.
    sigma_m_type, sigma_f_type = scales
    mum_cond = mus[:, :-1, :] / male_margins[:, :, None]
    muf_cond = mus[:, :, :-1] / female_margins[:, None, :]
    log_m = masked_log(mum_cond, maskc[:-1, :])
    log_f = masked_log(muf_cond, maskc[:, :-1])
    H_m_by_type = (mus[:, :-1, :] * log_m).sum(axis=2)
    H_f_by_type = (mus[:, :, :-1] * log_f).sum(axis=1)
    entropy_c_m = (H_m_by_type * sigma_m_type).sum(axis=1)
    entropy_c_f = (H_f_by_type * sigma_f_type).sum(axis=1)
    return -(entropy_c_m + entropy_c_f)


# Define the residuals function (unconstrained + sinkhorn)
def residuals(ng0, xi, transitions, netflow,
              beta, theta, tau, scale, masks, ts, dev, key,
              return_mass=False, return_max_mass=False):

    maskc, mask0 = masks # maskc now has all information
    _, _, tP0, tQ0, tP1, tQ1 = transitions
    ndim = tP0.shape[0]

    # 2026-11-13: Explicit random keys; mask rejected draws instead of dropping rows.
    keym, keyf, keyp, keyt, keyd = jax.random.split(key, 5)
    concentrations = jnp.asarray([2.0] * ndim, dtype=ts.dtype)
    pm = 0.5 + (jax.random.uniform(keyp, (ng0, 1), dtype=ts.dtype) - 0.5) / 5
    s0 = jnp.concatenate((jax.random.dirichlet(keym, concentrations, (ng0, )) * pm,
                    jax.random.dirichlet(keyf, concentrations, (ng0, )) * (1-pm)), axis = 1)
    valid = jnp.all(s0 > 0.001, axis=1) # 0.04 works
    s = jnp.where(valid[:, None], s0, s0[jnp.argmax(valid)])
    ng = s.shape[0]
    rts = ts[jax.random.randint(keyt, (ng, 1), 0, ts.size)]
    rds = jax.random.randint(keyd, (ng, 1), 0, 2).astype(ts.dtype)
    st = jnp.concatenate((s, rts, rds), axis = 1)

    mus, vcur = xi(st)
    dt = ts[1] - ts[0]
    # 2026-11-13: JAX differentiates a function of mus, rather than an existing graph.
    def objective(mus):
        stnext = choices(mus, rts, rds, tP0, tQ0, tP1, tQ1, netflow, dt, dev)
        _, vnext = xi(stnext)

        vmapper = jax.vmap(lambda t, d: tau(theta, t, d, dev), in_axes=(0, 0))
        phi0 = vmapper(rts, rds)
        unregularised = (mus * phi0).sum(axis=(1,2))

        # Choo/Siow (2006): 2ec-em-ef, Galichon has 2ec+em+ef (sometimes other)
        # 2 e_c - e_m - e_f is the one consistent with dE/dmu = Phi
        # -entropy is the largest for uniform distribution (all mus equal)
        # we maximise, thus we punish mus close to 0 or 1 (due to adding up)
        # 1) fun = unregularised - (2 * entropyc + entropym + entropyf)
        # 2) fun = unregularised - (2 * entropyc - entropym - entropyf)

        # entropyc = (masked_log(mus[:,:-1,:-1], maskc[:-1,:-1]) * mus[:,:-1,:-1]).sum(dim = (1, 2))
        # entropyf = (masked_log(mus[:,-1,:], mask0) * mus[:,-1,:]).sum(dim=1)
        # entropym = (masked_log(mus[:,:,-1], mask0) * mus[:,:,-1]).sum(dim=1)
        # entropy = -(2 * entropyc - entropym - entropyf)

        entropy = matching_entropy(mus, s[:, :ndim], s[:, ndim:(2*ndim)],
                                   scale(theta, dev), maskc)

        fun = unregularised + entropy

        sumL = jnp.sum(fun + beta * vnext)
        return sumL, (fun, vnext)
    grads, (fun, vnext) = jax.grad(objective, has_aux=True)(mus)

    margs = jnp.concatenate([mus[:,:-1,:].sum(2), mus[:,:,:-1].sum(1)], axis=1)

    lambda1, lambda2, lambda3 = 1.0, 1.0, 0.0 # last part from sinkhorn
    # 2026-10-02: Eliminate current margin multipliers; retain continuation derivatives.
    g = grads
    foc = g[:, :-1, :-1] - g[:, :-1, -1:] - g[:, -1:, :-1]
    r1 = lambda1 * jnp.square(foc * maskc[:-1, :-1])
    r1 = jnp.pad(r1, ((0, 0), (0, 1), (0, 1))).reshape(ng, -1)
    r2 = lambda2 * jnp.square((vcur - fun - beta * vnext).reshape(ng, 1))
    r3 = lambda3 * jnp.square(s - margs).reshape(ng, -1)

    resid_v = jnp.concatenate((r1, r2, r3), axis=1)
    mean_resid_v = jnp.where(valid[:, None], resid_v, 0).sum() / (valid.sum() * resid_v.shape[1])

    if return_max_mass:
        mass_max = jnp.max(jnp.where(valid[:, None], jnp.abs(s - margs), 0))
        return mean_resid_v, mass_max
    if return_mass:
        # Report only accepted draws, as in the legacy filtered-sample check.
        mass_rms = jnp.sqrt(jnp.where(valid[:, None], jnp.square(s - margs), 0).sum()
                            / (valid.sum() * s.shape[1]))
        return mean_resid_v, mass_rms
    return mean_resid_v


@eqx.filter_jit
def minimise_inner(xi, theta, beta, transitions, netflow,
                   ng, ts, tau, scale, masks, dev, key, return_diagnostics=False):

    epochs = 600
    optimiser = optax.adam(0.001)
    params = xi.params
    opt_state = optimiser.init(params)
    loss_grad = jax.value_and_grad(lambda params, key: residuals(
        ng, xi.with_params(params), transitions, netflow,
        beta, theta, tau, scale, masks, ts, dev, key, return_max_mass=True), has_aux=True)

    def step(carry, epoch):
        params, opt_state, key = carry
        key, draw_key = jax.random.split(key)
        (loss, mass_max), grads = loss_grad(params, draw_key)
        updates, opt_state = optimiser.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, key), (loss, mass_max)

    (params, opt_state, key), (losses, mass_errors) = jax.lax.scan(
        step, (params, opt_state, key), jnp.arange(epochs))
    result = (losses[-1], xi.with_params(params), key)
    if return_diagnostics:
        return (*result, mass_errors.max())
    return result

def overallPQ(tPs, tQs, n0, n1, n2):
    tP = (n0 * tPs[0] + n1 * tPs[1] + n2 * tPs[2] ) / (n0 + n1 + n2)
    tQ = (n0 * tQs[0] + n1 * tQs[1] + n2 * tQs[2] ) / (n0 + n1 + n2)
    return tP, tQ, tPs[2], tQs[2], tPs[1], tQs[1]

def match_moments(xi, theta, tPs, tQs, tMuHat, netflow,
                  ng, dev, tau, scale, masks, treat_idcs, years, beta,
                  skiptrain = False, cf = CF.None_, train0 = True, beyond = 0,
                  bellman_welfare = False, key=None, diagnostics=None):

    beta = jnp.asarray(beta, dtype=theta.dtype)
    ts0 = jnp.asarray(years, dtype=theta.dtype)
    ts = (ts0 - jnp.min(ts0)) / (jnp.max(ts0) - jnp.min(ts0))
    if beyond > 0:
        ts = jnp.concatenate((ts, ts[-1] + (ts[1] - ts[0]) * jnp.arange(1, beyond + 1)))


    nT, nty0, nty0 = tMuHat.shape
    maskc, mask0 = masks

    # regimes
    pre = range(0, treat_idcs[0])
    post = range(treat_idcs[-1] + 1, nT)
    future = range(nT, nT + beyond) if beyond > 0 else []

    transitions = overallPQ(tPs, tQs, int(train0)*len(pre),
                            len(treat_idcs), len(post))
    tP, tQ, tP0, tQ0, tP1, tQ1 = transitions

    if not skiptrain:
        xi = xi.train()
        loss, xi, key, mass_max = minimise_inner(xi, theta, beta, transitions, netflow,
                                      ng, ts, tau, scale, masks, dev, key,
                                      return_diagnostics=True)
        if diagnostics is not None:
            diagnostics["max_mass_deviation"] = float(mass_max)
    else:
        loss = jnp.asarray(0.0, dtype=theta.dtype)


    def regime_at(i):
        # 2026-10-02: Use one scenario regime for welfare and the matching rollout.
        flag = {CF.None_: int(i in treat_idcs),
                CF.HighCost: 0, CF.LowCost: 1}[cf]
        return jnp.asarray([[flag]], dtype=theta.dtype)

    def transition_mu(xi, p0, q0, p1, q1):
        def step(mus, t, current_regime, next_regime):
            sst = choices(mus, t, current_regime, p0, q0, p1, q1,
                          netflow, ts[1] - ts[0], dev)
            # 2026-10-02: Evaluate next year's matching under next year's regime.
            sst = jnp.concatenate([sst[:, :-1], next_regime], axis=1)
            mus, vs = xi(sst)
            return mus, vs
        return step

    tM = tMuHat[:,:-1,:].sum(2)
    tF = tMuHat[:,:,:-1].sum(1)
    ss_hat = jnp.concatenate((tM, tF), axis=1)

    # initial state
    idx0 = 0 if train0 else treat_idcs[0]
    mu_cur = tMuHat[idx0:idx0+1,:,:]

    # we recursively define the whole path
    xi = xi.eval()
    transition = transition_mu

    # same transition for all regimes
    _, v = xi(jnp.concatenate([ss_hat[idx0:idx0+1], ts[idx0].reshape(1,1),
                              regime_at(idx0)], axis=1))
    walker = transition(xi, tP0, tQ0, tP1, tQ1)
    regimes = ([(pre, walker)] if train0 else []) + [(treat_idcs, walker), (post, walker)] + ([(future, walker)] if beyond > 0 else [])

    mu_star = jnp.zeros((nT+beyond, nty0, nty0), dtype=tMuHat.dtype)
    v_star = jnp.zeros((nT+beyond, 1), dtype=tMuHat.dtype)

    for (idcs, walker) in regimes:
        for i in idcs:
            mu_star = mu_star.at[i, :, :].set(mu_cur[0])
            current_regime = regime_at(i)
            next_regime = regime_at(i + 1)
            if bellman_welfare:
                # 2026-10-22: Report current-state V before the walker advances to next year's value.
                v_star = v_star.at[i, :].set(v)
            else:
                unreg = (mu_cur * tau(theta, ts[i].reshape(1,1), current_regime, dev)).sum()
                vdir = unreg + matching_entropy(
                    mu_cur, mu_cur[:, :-1, :].sum(2), mu_cur[:, :, :-1].sum(1),
                    scale(theta, dev), maskc).sum()
                v_star = v_star.at[i, :].set(vdir) # vs v (approximate value fct)
            mu_cur, v = walker(mu_cur, ts[i].reshape(1,1), current_regime, next_regime)

    star_idcs = slice(idx0, len(mu_star) - beyond)
    if beyond == 0:
        matched = conditional_kl_loss(tMuHat[idx0:,:,:], mu_star[star_idcs,:,:], masks)
    else:
        # 2026-10-02: Duplicate model outputs into hat for extended-plot alignment, not data-fit validation.
        matched = conditional_kl_loss(mu_star[idx0:,:,:], mu_star[idx0:,:,:], masks)
    kl_div, cond_m_hat, cond_m_star, cond_f_hat, cond_f_star, m_hat, m_star, f_hat, f_star = matched
    # 2026-10-02: Report the KL, inner residual and total from the same evaluation.
    # Keep the FP32 inner residual, but combine outer scores in host FP64.
    # Mixing a NumPy scalar with a JAX scalar here would demote the sum to FP32.
    total_loss = np.float64(kl_div) + 2.0 * np.float64(loss)
    return (total_loss, tMuHat, mu_star, loss,
            (cond_m_hat, cond_m_star, cond_f_hat, cond_f_star),
            (m_hat, m_star, f_hat, f_star),
            v_star, xi, key)

# earlier frobenius norm matching has a margin-mismatch component, which is controlled by the markov kernel
# and entry/exit. as a result, being a square criterion it balances this over different cells.
# now matching is conditional on margins which is invariant wrt the markov kernel errors.
# KL criterion minimised between phat and pstar.
# theoretically much cleaner and robust to accumulation of mass over/underestimation due to conditionality
def conditional_kl_loss(mu_data, mu_model, masks):
    """Compute conditional fit terms in JAX and accumulate the outer score in FP64.

    This host-side score is used by Nelder-Mead/reporting, outside inner AD/JIT.
    Probabilities, logarithms, and returned conditional arrays keep their dtype.
    """
    T, ndim1, _ = mu_data.shape
    ndim = ndim1 - 1

    def by_gender(mu_d, mu_m, mc):
        data_rows  = mu_d[:, :ndim, :]
        model_rows = mu_m[:, :ndim, :]

        S_hat = data_rows.sum(axis=-1, keepdims=True)
        S_mod = model_rows.sum(axis=-1, keepdims=True)

        p_hat = data_rows / (S_hat + 1e-12)
        p_mod = model_rows / (S_mod + 1e-12)

        log_phat = masked_log(p_hat, mc[:-1,:])
        log_pmod = masked_log(p_mod, mc[:-1,:])
        # Transfer only the small historical-fit array, not the training graph.
        # Weight cells before reducing so no row sum is first rounded to FP32.
        weighted_terms = S_hat * (p_hat * (log_phat - log_pmod))
        score = np.sum(np.asarray(weighted_terms), dtype=np.float64)

        return p_hat, p_mod, S_hat, S_mod, score

    maskc, _ = masks
    cond_m_hat, cond_m_star, m_hat, m_star, ckl_m = by_gender(mu_data, mu_model, maskc)
    cond_f_hat, cond_f_star, f_hat, f_star, ckl_f = by_gender(mu_data.swapaxes(1,2), mu_model.swapaxes(1,2), maskc.T)

    return ckl_m + ckl_f, cond_m_hat, cond_m_star, cond_f_hat, cond_f_star, m_hat, m_star, f_hat, f_star

