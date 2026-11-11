import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.autograd import grad as autograd_grad
from torch.func import functional_call
import math
import pdb
import sys

from .deeplearning import masked_log
from .helpers import minfb, TermColours, ManualLRScheduler, CF, extend

def create_closure(xi, theta, tPs, tQs, tMuHat, netflow,
                   ng, dev, tau, scale, masks, treat_idcs, years,
                   optim, cf, train0, beta, calcgrad = True):
    additional_outputs = [None, None, None, None, None]
    if calcgrad:
        # 2026-10-02: Restore structural gradients when loading detached checkpoints.
        theta.requires_grad_(True)
    def closure():
        optim.zero_grad()
        # 2026-11-10: A failed Nelder-Mead trial must not change the next NN warm start.
        xi_before = deepcopy(xi.state_dict()) if not calcgrad else None
        xi_params = {} if calcgrad else None
        resid, ssh, sss, l, conds, margs, _, _ = match_moments(xi,
                                              theta,
                                              tPs, tQs, tMuHat,
                                              netflow, ng, dev,
                                              tau, scale, masks, treat_idcs,
                                              years, skiptrain=False,
                                              cf = cf, train0 = train0,
                                              beyond = 0, xi_params = xi_params,
                                              beta = beta)
        if not calcgrad and not torch.isfinite(resid):
            xi.load_state_dict(xi_before)
            print("Nonfinite trial loss; restoring previous NN weights.")
            return resid.new_tensor(float("inf"))
        if calcgrad:
            resid.backward(inputs=(theta,))
            with torch.no_grad():
                for name, param in xi.named_parameters():
                    param.copy_(xi_params[name])
        additional_outputs[0] = ssh.detach().cpu()
        additional_outputs[1] = sss.detach().cpu()
        additional_outputs[2] = l.detach().cpu()
        additional_outputs[3] = [c.detach().cpu() for c in conds]
        additional_outputs[4] = [mgs.detach().cpu() for mgs in margs]
        return resid
    return closure, additional_outputs

def check_grad_norm(optimizer):
    total_norm = 0
    for param_group in optimizer.param_groups:
        for p in param_group['params']:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    return total_norm

def tJ(dim, dev):
    return torch.eye(dim).to(device=dev)[:, :-1]

def tiota(dim, dev):
    return torch.ones(dim).to(device=dev).view(-1, 1)


def choices_vectorised(mus, p, q, dev):
    nty0 = mus.shape[1]
    tMuM = torch.matmul(tJ(nty0, dev).T, mus)
    tMuF = torch.matmul(mus, tJ(nty0, dev))
    return torch.cat(
        [torch.matmul(tMuM.view(-1, p.shape[1]), p.T),
         torch.matmul(tMuF.view(-1, q.shape[1]), q.T)],
        dim=1)

def regime_transitions(d, p0, q0, p1, q1):
    alpha = d.view(-1, 1, 1, 1)
    p = (1.0 - alpha) * p0.unsqueeze(0) + alpha * p1.unsqueeze(0)
    q = (1.0 - alpha) * q0.unsqueeze(0) + alpha * q1.unsqueeze(0)
    return p, q

def incumbent_population(mus, d, p0, q0, p1, q1):
    tMuM = mus[:,:-1,:]
    tMuF = mus[:,:,:-1]
    p, q = regime_transitions(d, p0, q0, p1, q1)
    M = torch.einsum('nij,nijk->nk', tMuM, p)
    F = torch.einsum('nij,nijk->nk', tMuF, q)
    return torch.cat([M, F], dim=1)

def create_netflow(ee, tMuHat, tPs, tQs, treat_idcs, train0, fixed=False):
    # 2026-11-03: Apply the profile's fixed or proportional choice to both entry and exit.
    entries, exits = ee[:, 0].repeat(2), ee[:, 1].repeat(2)
    if fixed:
        return lambda s: (entries - exits).to(s)
    start = 0 if train0 else treat_idcs[0]
    regimes = tMuHat.new_zeros(len(tMuHat))
    regimes[treat_idcs] = 1
    reference = incumbent_population(tMuHat[start:], regimes[start:],
                                     tPs[2], tQs[2], tPs[1], tQs[1]).mean(0)
    entry_rates, exit_rates = entries / reference, exits / reference
    if not torch.all(torch.isfinite(exit_rates) & (exit_rates >= 0) & (exit_rates <= 1)):
        raise ValueError("Calibrated annual exit rates must lie between zero and one.")
    if not torch.all(torch.isfinite(entry_rates) & (entry_rates >= 0)):
        raise ValueError("Calibrated annual entry rates must be finite and nonnegative.")
    return lambda s: (entry_rates.to(s) - exit_rates.to(s)) * s

def choices(mus, t, d, p0, q0, p1, q1, netflow, dt, dev):
    Sincumb = incumbent_population(mus, d, p0, q0, p1, q1)
    S0 = Sincumb + netflow(Sincumb)
    # 2026-10-02: Normalize each sampled economy separately after net flows.
    S = S0 / S0.sum(dim=1, keepdim=True)
    return torch.cat([S, t + dt, d], dim = 1)

def check_mass(mus, s):
    ntypes = s.shape[1] // 2
    f_resid = torch.square(mus.sum(1)[:, :-1] - s[:,ntypes:(2*ntypes)]).mean()
    m_resid = torch.square(mus.sum(2)[:, :-1] - s[:,0:(ntypes)]).mean()
    total = torch.sqrt(0.5 * f_resid + 0.5 * m_resid).detach()
    offdiag = (mus.mean(0)[0,1] + mus.mean(0)[1,0]).detach()
    print(f"{TermColours.YELLOW} {total:.2f} {TermColours.RESET}", end='')
    #print(f"{TermColours.CYAN} {offdiag:.2f} {TermColours.RESET}", end='')

def matching_entropy(mus, male_margins, female_margins, scales, maskc):
    # 2026-10-02: Share entropy with reporting while keeping Bellman margins explicit.
    sigma_m_type, sigma_f_type = scales
    mum_cond = mus[:, :-1, :] / male_margins.unsqueeze(2)
    muf_cond = mus[:, :, :-1] / female_margins.unsqueeze(1)
    log_m = masked_log(mum_cond, maskc[:-1, :])
    log_f = masked_log(muf_cond, maskc[:, :-1])
    H_m_by_type = (mus[:, :-1, :] * log_m).sum(dim=2)
    H_f_by_type = (mus[:, :, :-1] * log_f).sum(dim=1)
    entropy_c_m = (H_m_by_type * sigma_m_type).sum(dim=1)
    entropy_c_f = (H_f_by_type * sigma_f_type).sum(dim=1)
    return -(entropy_c_m + entropy_c_f)


def dual_residuals(mus, s, d, phi, scales, values, next_values,
                   transitions, netflow, beta, maskc):
    """Return the dynamic dual equations and value gap separately, before squaring."""
    # 2026-11-03: Include single options and the discounted fixed-flow contribution in the dual.
    vcur, wm, wf, J = values
    wmnext, wfnext, Jnext = next_values
    sigma, tau = scales
    _, _, p0, q0, p1, q1 = transitions
    p, q = regime_transitions(d, p0, q0, p1, q1)
    flow = netflow(s)
    c = 1.0 + flow.sum(dim=-1, keepdim=True)

    pm = mus[:, :-1, :] / mus[:, :-1, :].sum(2, keepdim=True)
    pf = mus[:, :, :-1] / mus[:, :, :-1].sum(1, keepdim=True)
    am = wm.unsqueeze(2) + sigma.view(1, -1, 1) * masked_log(pm, maskc[:-1, :])
    af = wf.unsqueeze(1) + tau.view(1, 1, -1) * masked_log(pf, maskc[:, :-1])
    am = am - beta * torch.einsum('nijk,nk->nij', p, wmnext) / c.unsqueeze(-1)
    af = af - beta * torch.einsum('nijk,nk->nij', q, wfnext) / c.unsqueeze(-1)
    budget = (torch.nn.functional.pad(am, (0, 0, 0, 1))
              + torch.nn.functional.pad(af, (0, 1, 0, 0)) - phi)

    w, wnext = torch.cat([wm, wf], dim=1), torch.cat([wmnext, wfnext], dim=1)
    source = J - beta * ((flow / c * wnext).sum(1) + Jnext)
    # The gap is a consistency residual until the dynamic equations and tail conditions hold.
    gap = (s * w).sum(1) + J - vcur
    return torch.cat([budget[:, maskc], source.unsqueeze(1)], dim=1), gap


# Define the residuals function (unconstrained + sinkhorn)
def residuals(ng0, xi, transitions, netflow,
              beta, theta, tau, scale, masks, ts, dev, use_dual=True):

    maskc, mask0 = masks # maskc now has all information
    _, _, tP0, tQ0, tP1, tQ1 = transitions
    ndim = tP0.shape[0]

    concentrations = torch.tensor([2.0] * ndim).to(device = dev)
    dirichlet = torch.distributions.Dirichlet(concentrations)
    pm = 0.5 + (torch.rand((ng0, 1), device = dev) - 0.5) / 5
    s0 = torch.cat((dirichlet.sample((ng0, )) * pm,
                    dirichlet.sample((ng0, )) * (1-pm)), dim = 1)
    s = s0[torch.all(s0 > 0.001, dim=1)] # 0.04 works
    ng = s.shape[0]
    rts = ts[torch.randint(0, ts.numel(), (ng, 1))]
    rds = torch.randint(0, 2, (ng, 1), device = dev)
    st = torch.cat((s, rts, rds), dim = 1)

    mus, vcur, wm, wf, J = xi(st)
    dt = ts[1] - ts[0]
    stnext = choices(mus, rts, rds, tP0, tQ0, tP1, tQ1, netflow, dt, dev)
    check_mass(mus, s)
    _, vnext, wmnext, wfnext, Jnext = xi(stnext)

    vmapper = torch.vmap(lambda t, d: tau(theta, t, d, dev), in_dims=(0, 0))
    phi0 = vmapper(rts, rds)
    unregularised = (mus * phi0).sum(dim=(1,2))

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

    scales = scale(theta, dev)
    entropy = matching_entropy(mus, s[:, :ndim], s[:, ndim:(2*ndim)], scales, maskc)

    fun = unregularised + entropy

    sumL = torch.sum(fun + beta * vnext)
    grads = autograd_grad(outputs=sumL, inputs=mus, create_graph=True)

    # 2026-10-02: Eliminate current margin multipliers; retain continuation derivatives.
    g = grads[0]
    foc = g[:, :-1, :-1] - g[:, :-1, -1:] - g[:, -1:, :-1]
    # 2026-11-06: Preserve the original padded primal denominator for all four explicit terms.
    normalisation = (ndim + 1)**2 + 1 + 2*ndim
    loss_foc = (foc * maskc[:-1, :-1]).square().sum((1, 2)).mean() / normalisation
    loss_bellman = (vcur - fun - beta * vnext).square().mean() / normalisation
    loss_dual = loss_gap = vcur.new_zeros(())
    if use_dual:
        # 2026-11-06: Restrict only the added dual equations; keep the original primal draws unchanged.
        valid = (stnext[:, :2*ndim] > 0).all(dim=1)
        if not valid.any():
            raise ValueError("No sampled states have positive next populations under additive flows.")
        dual, gap = dual_residuals(
            mus[valid], s[valid], rds[valid], phi0[valid], scales,
            tuple(x[valid] for x in (vcur, wm, wf, J)),
            tuple(x[valid] for x in (wmnext, wfnext, Jnext)),
            transitions, netflow, beta, maskc)
        loss_dual = dual.square().sum(1).mean() / normalisation
        loss_gap = gap.square().mean() / normalisation
    loss = loss_foc + loss_bellman + loss_dual + loss_gap

    torch.cuda.empty_cache()

    return loss


def prepare_differentiable_adam(xi, loss, optimiser, xi_params):
    # 2026-10-02: Differentiate the final Adam update, holding earlier weights and moments fixed.
    grads = autograd_grad(loss, tuple(xi.parameters()), create_graph=True)
    xi_params.update({name: param.clone() for name, param in xi.named_parameters()})
    state = optimiser.state_dict()
    state['param_groups'][0].update(differentiable=True, foreach=False)
    optimiser = optim.Adam(xi_params.values(), differentiable=True, foreach=False)
    optimiser.load_state_dict(state)
    for param, grad in zip(xi_params.values(), grads):
        param.grad = grad
        # 2026-10-02: Keep sqrt derivatives finite for weights with zero Adam variance.
        optimiser.state[param]['exp_avg_sq'].clamp_min_(2 * torch.finfo(param.dtype).tiny)
    return optimiser


def minimise_inner(xi, theta, beta, transitions, netflow,
                   ng, ts, tau, scale, masks, dev, xi_params = None):

    epochs = 600 # 2026-11-06: Allow more inner updates for the joint primal/dual solve.
    optimiser = optim.Adam(xi.parameters())  #, lr = .1) # , weight_decay = 0.01)

    for epoch in range(0, epochs):
        optimiser.zero_grad()
        # 2026-11-11: Park the dual and use the original primal objective.
        loss = residuals(ng, xi, transitions, netflow,
                         beta, theta, tau, scale, masks, ts, dev, use_dual=False)
        if xi_params is not None and epoch == epochs - 1:
            optimiser = prepare_differentiable_adam(xi, loss, optimiser, xi_params)
        else:
            loss.backward(inputs=tuple(xi.parameters()))
        optimiser.step()
        grad_norm = check_grad_norm(optimiser)
        if epoch % (epochs // 100) == 0:
            print(f"{int((epoch/epochs) * 100)}%: {loss.item():.4f} [{grad_norm:.4f}] ",
                  end='\t', flush = "True")
    return loss

def overallPQ(tPs, tQs, n0, n1, n2):
    tP = (n0 * tPs[0] + n1 * tPs[1] + n2 * tPs[2] ) / (n0 + n1 + n2)
    tQ = (n0 * tQs[0] + n1 * tQs[1] + n2 * tQs[2] ) / (n0 + n1 + n2)
    return tP, tQ, tPs[2], tQs[2], tPs[1], tQs[1]

def match_moments(xi, theta, tPs, tQs, tMuHat, netflow,
                  ng, dev, tau, scale, masks, treat_idcs, years, beta,
                  skiptrain = False, cf = CF.None_, train0 = True, beyond = 0,
                  xi_params = None, bellman_welfare = False):

    beta = torch.as_tensor(beta, device=dev)
    ts0 = torch.tensor(years, device=dev)
    ts = (ts0 - torch.min(ts0)) / (torch.max(ts0) - torch.min(ts0))
    if beyond > 0:
        ts = torch.cat((ts, ts[-1] + (ts[1] - ts[0]) * torch.arange(1, beyond + 1, device=dev)))


    print("theta: ", theta.detach().cpu().numpy())

    nT, nty0, nty0 = tMuHat.size()
    maskc, mask0 = masks

    # regimes
    pre = range(0, treat_idcs[0])
    post = range(treat_idcs[-1] + 1, nT)
    future = range(nT, nT + beyond) if beyond > 0 else []

    transitions = overallPQ(tPs, tQs, int(train0)*len(pre),
                            len(treat_idcs), len(post))
    tP, tQ, tP0, tQ0, tP1, tQ1 = transitions

    if not skiptrain:
        xi.train()
        loss = minimise_inner(xi, theta, beta, transitions, netflow,
                              ng, ts, tau, scale, masks, dev, xi_params)
    else:
        loss = torch.tensor(0.0, device=dev)


    def regime_at(i):
        # 2026-10-02: Use one scenario regime for welfare and the matching rollout.
        flag = {CF.None_: int(i in treat_idcs),
                CF.HighCost: 0, CF.LowCost: 1}[cf]
        return torch.tensor([[flag]], device=dev)

    def transition_mu(xi, p0, q0, p1, q1):
        def step(mus, t, current_regime, next_regime):
            sst = choices(mus, t, current_regime, p0, q0, p1, q1,
                          netflow, ts[1] - ts[0], dev)
            # 2026-10-02: Evaluate next year's matching under next year's regime.
            sst = torch.cat([sst[:, :-1], next_regime], dim=1)
            return xi(sst)
        def step_household(mus): # not in use
            raise NotImplementedError()
        def step_matching(mus): # not in use
            raise NotImplementedError()
        return step

    tM = tMuHat[:,:-1,:].sum(2)
    tF = tMuHat[:,:,:-1].sum(1)
    ss_hat = torch.cat((tM, tF), dim=1)

    # initial state
    idx0 = 0 if train0 else treat_idcs[0]
    mu_cur = tMuHat[idx0,:,:].unsqueeze(0)

    # we recursively define the whole path
    xi.eval()
    transition = transition_mu

    # same transition for all regimes
    xi_eval = xi if xi_params is None else lambda s: functional_call(xi, xi_params, (s,))
    _, v, wm, wf, J = xi_eval(torch.cat([ss_hat[idx0:idx0+1], ts[idx0].view(1,1),
                                        regime_at(idx0)], dim=1))
    walker = transition(xi_eval, tP0, tQ0, tP1, tQ1)
    regimes = ([(pre, walker)] if train0 else []) + [(treat_idcs, walker), (post, walker)] + ([(future, walker)] if beyond > 0 else [])

    mu_star = torch.zeros(nT+beyond, nty0, nty0).to(dev)
    v_star = torch.zeros(nT+beyond, 1).to(dev)
    wm_star = mu_star.new_zeros(nT+beyond, nty0-1)
    wf_star = torch.zeros_like(wm_star)
    J_star = torch.zeros_like(v_star)

    for (idcs, walker) in regimes:
        for i in idcs:
            mu_star[i, :, :] = mu_cur
            # 2026-11-03: Record current-state individual values before the shared walker advances.
            wm_star[i, :], wf_star[i, :], J_star[i, :] = wm, wf, J
            current_regime = regime_at(i)
            next_regime = regime_at(i + 1)
            if bellman_welfare:
                # 2026-10-22: Report current-state V before the walker advances to next year's value.
                v_star[i, :] = v
            else:
                unreg = (mu_cur * tau(theta, ts[i].view(1,1), current_regime, dev)).sum()
                v_star[i, :] = unreg + matching_entropy(
                    mu_cur, mu_cur[:, :-1, :].sum(2), mu_cur[:, :, :-1].sum(1),
                    scale(theta, dev), maskc).sum()
            mu_cur, v, wm, wf, J = walker(mu_cur, ts[i].view(1,1), current_regime, next_regime)

    star_idcs = range(idx0, len(mu_star) - beyond)
    if beyond == 0:
        matched = conditional_kl_loss(tMuHat[idx0:,:,:], mu_star[star_idcs,:,:], masks)
    else:
        # 2026-10-02: Duplicate model outputs into hat for extended-plot alignment, not data-fit validation.
        matched = conditional_kl_loss(mu_star[idx0:,:,:], mu_star[idx0:,:,:], masks)
    kl_div, cond_m_hat, cond_m_star, cond_f_hat, cond_f_star, m_hat, m_star, f_hat, f_star = matched
    # 2026-10-02: Report the KL, inner residual and total from the same evaluation.
    total_loss = kl_div + 2.0 * (loss.detach() if xi_params is None else loss)
    print("D_KL: ", kl_div.item(), " inner loss: ", loss.item(), " total loss: ", total_loss.item())

    torch.cuda.empty_cache()

    return (total_loss, tMuHat, mu_star, loss,
            (cond_m_hat, cond_m_star, cond_f_hat, cond_f_star),
            (m_hat, m_star, f_hat, f_star),
            v_star, (wm_star, wf_star, J_star))

# earlier frobenius norm matching has a margin-mismatch component, which is controlled by the markov kernel
# and entry/exit. as a result, being a square criterion it balances this over different cells.
# now matching is conditional on margins which is invariant wrt the markov kernel errors.
# KL criterion minimised between phat and pstar.
# theoretically much cleaner and robust to accumulation of mass over/underestimation due to conditionality
def conditional_kl_loss(mu_data, mu_model, masks):
    T, ndim1, _ = mu_data.shape
    ndim = ndim1 - 1

    def by_gender(mu_d, mu_m, mc):
        data_rows  = mu_d[:, :ndim, :]
        model_rows = mu_m[:, :ndim, :]

        S_hat = data_rows.sum(dim=-1, keepdim=True)
        S_mod = model_rows.sum(dim=-1, keepdim=True)

        p_hat = data_rows / (S_hat + 1e-12)
        p_mod = model_rows / (S_mod + 1e-12)

        log_phat = masked_log(p_hat, mc[:-1,:])
        log_pmod = masked_log(p_mod, mc[:-1,:])
        row_kl = (p_hat * (log_phat - log_pmod)).sum(dim=-1)

        return p_hat, p_mod, S_hat, S_mod, (S_hat.squeeze(-1) * row_kl).sum()

    maskc, _ = masks
    cond_m_hat, cond_m_star, m_hat, m_star, ckl_m = by_gender(mu_data, mu_model, maskc)
    cond_f_hat, cond_f_star, f_hat, f_star, ckl_f = by_gender(mu_data.transpose(1,2), mu_model.transpose(1,2), maskc.T)

    return ckl_m + ckl_f, cond_m_hat, cond_m_star, cond_f_hat, cond_f_star, m_hat, m_star, f_hat, f_star

