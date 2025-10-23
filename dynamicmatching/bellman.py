import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad as autograd_grad
import math
import pdb
import sys

from .deeplearning import masked_log
from .helpers import tauMflex, tauM, minfb, TermColours, ManualLRScheduler, CF, extend

def create_closure(xi, theta, tPs, tQs,
                   tMuHat, ng, dev, tau, masks, treat_idcs, years,
                   optim, cf, train0, calcgrad = True):
    additional_outputs = [None, None, None]
    def closure():
        optim.zero_grad()
        resid, ssh, sss, l = match_moments(xi,
                                           theta,
                                           tPs, tQs,
                                           tMuHat, ng, dev,
                                           tau, masks, treat_idcs,
                                           years, skiptrain=False,
                                           cf = cf, train0 = train0)
        if calcgrad:
            resid.backward()
        additional_outputs[0] = ssh.detach().cpu()
        additional_outputs[1] = sss.detach().cpu()
        additional_outputs[2] = l.detach().cpu()
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

def choices(mus, t, d, p, q, dev):
    tMuM = mus[:,:-1,:]
    tMuF = mus[:,:,:-1]
    M = torch.einsum('nij,ijk->nk', tMuM, p)
    F = torch.einsum('nij,ijk->nk', tMuF, q)
    return torch.cat([M, F, t + 1, d], dim = 1)

def check_mass(mus, s):
    ntypes = s.shape[1] // 2
    f_resid = torch.square(mus.sum(1)[:, :-1] - s[:,ntypes:(2*ntypes)]).mean()
    m_resid = torch.square(mus.sum(2)[:, :-1] - s[:,0:(ntypes)]).mean()
    total = torch.sqrt(0.5 * f_resid + 0.5 * m_resid).detach()
    offdiag = (mus.mean(0)[0,1] + mus.mean(0)[1,0]).detach()
    print(f"{TermColours.YELLOW} {total:.2f} {TermColours.RESET}", end='')
    print(f"{TermColours.CYAN} {offdiag:.2f} {TermColours.RESET}", end='')

# Define the residuals function (unconstrained + sinkhorn)
def residuals(ng0, xi, tP, tQ, beta, theta, tau, masks, ts, dev):

    maskc, mask0 = masks # maskc now has all information
    ndim = tP.shape[0]

    concentrations = torch.tensor([2.0] * ndim).to(device = dev)
    dirichlet = torch.distributions.Dirichlet(concentrations)
    pm = 0.5 + (torch.rand((ng0, 1), device = dev) - 0.5) / 5
    s0 = torch.cat((dirichlet.sample((ng0, )) * pm,
                    dirichlet.sample((ng0, )) * (1-pm)), dim = 1)
    s = s0[torch.all(s0 > 0.04, dim=1)]
    ng = s.shape[0]
    rts = ts[torch.randint(0, ts.numel(), (ng, 1))]
    rds = torch.randint(0, 2, (ng, 1), device = dev)
    st = torch.cat((s, rts, rds), dim = 1)

    mus, vcur = xi(st)
    check_mass(mus, s)
    # TODO: transition matrix by d or weighted (#of per) comb of them?
    stnext = choices(mus, rts, rds, tP, tQ, dev)
    _, vnext = xi(stnext)

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

    mum_cond = mus[:,:-1,:] / s[:,0:ndim].reshape((ng, ndim, 1))
    muf_cond = mus[:,:,:-1] / s[:,ndim:(2*ndim)].reshape((ng, 1, ndim))
    entropy_c_m = mus[:,:-1,:] * masked_log(mum_cond, maskc[:-1,:])
    entropy_c_f = mus[:,:,:-1] * masked_log(muf_cond, maskc[:,:-1])
    entropy = -entropy_c_m.sum((1,2)) - entropy_c_f.sum((1,2))

    fun = unregularised + entropy

    sumL = torch.sum(fun + beta * vnext)
    grads = autograd_grad(outputs=sumL, inputs=mus, create_graph=True)
    m2 = torch.cat([torch.cat([maskc[:-1, :-1], mask0[:-1].unsqueeze(1)], dim=1),
                    mask0.unsqueeze(0)], dim=0)

    margs = torch.cat([mus[:,:-1,:].sum(2), mus[:,:,:-1].sum(1)], dim=1)

    lambda1, lambda2, lambda3 = 1.0, 1.0, 0.0 # last part from sinkhorn
    r1 = lambda1 * torch.square((grads[0] * m2).view(ng, -1))
    r2 = lambda2 * torch.square((vcur - fun - beta * vnext).view(ng, 1))
    r3 = lambda3 * torch.square(s - margs).view(ng, -1)

    resid_v = torch.cat((r1, r2, r3), dim=1)
    mean_resid_v = torch.mean(resid_v)

    torch.cuda.empty_cache()

    return mean_resid_v


def minimise_inner(xi, theta, beta, tP, tQ, ng, ts, tau, masks, dev):

    epochs = 1000
    optimiser = optim.SGD(xi.parameters(), lr = .1) # , weight_decay = 0.01)

    for epoch in range(0, epochs):
        optimiser.zero_grad()
        loss = residuals(ng, xi, tP, tQ, beta, theta, tau, masks, ts, dev)
        loss.backward(retain_graph=True)
        optimiser.step()
        if epoch < epochs - 1: # detach gradients in trajectory until (only keep for last)
            for param in xi.parameters():
                param.data = param.data.detach()
        grad_norm = check_grad_norm(optimiser)
        if epoch % (epochs // 100) == 0:
            print(f"{int((epoch/epochs) * 100)}%: {loss.item():.4f} [{grad_norm:.4f}] ",
                  end='\t', flush = "True")
    return loss

def match_moments(xi, theta, tPs, tQs,
                  tMuHat, ng, dev, tau, masks, treat_idcs, years,
                  skiptrain = False, cf = CF.None_, train0 = True):

    beta = torch.tensor(0.9, device=dev)
    ts0 = torch.tensor(years, device=dev)
    ts = (ts0 - torch.min(ts0)) / (torch.max(ts0) - torch.min(ts0))
    # print("theta: ", theta.detach().cpu().numpy())

    nT, nty0, nty0 = tMuHat.size()

    # regimes
    pre = range(0, treat_idcs[0])
    post = range(treat_idcs[-1] + 1, nT)
    control_idcs = list(pre) + list(post)

    # these won't depend on phi (leaves in the autograd graph)
    # dldTheta is gradient with respect to inner loss function
    # emulate dl/dphi = dl/dxi * dxi/dphi + dl/dphi
    # dl/dxi = 0 by the envelope theorem at xi = xi_opt
    # update: now gradient graph is kept and phi is set to requires_grad

    n0, n1, n2 = int(train0) * len(pre), len(treat_idcs), len(post)
    tP = (n0 * tPs[0] + n1 * tPs[1] + n2 * tPs[2] ) / (n0 + n1 + n2)
    tQ = (n0 * tQs[0] + n1 * tQs[1] + n2 * tQs[2] ) / (n0 + n1 + n2)

    if not skiptrain:
        xi.train()
        loss = minimise_inner(xi, theta, beta, tP, tQ,
                              ng, ts, tau, masks, dev)
    else:
        loss = None


    def transition_mu(xi, p, q, cf):
        def step(mus, t, d):
            sst = choices(mus, t, d, p, q, dev)
            mus, _ = xi(sst)
            return mus
        def step_household(mus): # not in use
            raise NotImplementedError()
        def step_matching(mus): # not in use
            raise NotImplementedError()
        return {CF.None_: step,
                CF.HouseholdOnly: step_household,
                CF.MatchingOnly: step_matching}[cf]



    tM = tMuHat[:,:-1,:].sum(2)
    tF = tMuHat[:,:,:-1].sum(1)
    ss_hat = torch.cat((tM, tF), dim=1)

    # initial state
    idx0 = 0 if train0 else treat_idcs[0]
    ss_cur = ss_hat[idx0, :].view(1, -1)
    mu_cur = tMuHat[idx0,:,:].unsqueeze(0)

    # we recursively define the whole path
    xi.eval()
    transition = transition_mu
    walker_pre = transition(xi, tPs[0], tQs[0], cf)
    walker_treat = transition(xi, tPs[1], tQs[1], cf)
    walker_post = transition(xi, tPs[2], tQs[2], cf)
    regimes = ([(pre, walker_pre)] if train0 else []) + [(treat_idcs, walker_treat), (post, walker_post)]

    ss_star = torch.zeros(nT, ss_cur.shape[1]).to(dev)
    mu_star = torch.zeros(nT, nty0, nty0).to(dev)

    for (idcs, walker) in regimes:
        for i in idcs:
           mu_star[i, :, :] = mu_cur
           d = torch.tensor(int(i in treat_idcs), device = dev)
           mu_cur = walker(mu_cur, ts[i].view(1,1), d.view(1,1)) # ts[i] to be safe
           #ss_star[i, ] = ss_cur
           #ss_cur = walker(ss_cur)

    resid = torch.square(tMuHat[idx0:,:,:] - mu_star[idx0,:,:]).sum()

    torch.cuda.empty_cache()

    return (resid, tMuHat, mu_star, loss)
