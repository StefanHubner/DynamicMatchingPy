import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad as autograd_grad
import math
import pdb

from .deeplearning import masked_log
from .helpers import tauMflex, tauM, minfb, TermColours, ManualLRScheduler, CF

def create_closure(xi0, xi1, xi2, theta0, theta1, theta2, tPs, tQs,
                   tMuHat, ng, dev, tau, masks, treat_idcs, optim, cf):
    # Use a list as a mutable container
    additional_outputs = [None, None, None, None, None]
    def closure():
        optim.zero_grad()
        resid, ssh, sss, l0, l1, l2 = match_moments(xi0, xi1, xi2,
                                                theta0, theta1, theta2,
                                                tPs, tQs,
                                                tMuHat, ng, dev,
                                                tau, masks, treat_idcs,
                                                skiptrain=False, cf = cf)
        resid.backward()
        additional_outputs[0] = ssh.detach().cpu()
        additional_outputs[1] = sss.detach().cpu()
        additional_outputs[2] = l0.detach().cpu()
        additional_outputs[3] = l1.detach().cpu()
        additional_outputs[4] = l2.detach().cpu()
        print(resid)
        return resid
    return closure, additional_outputs


def tJ(dim, dev):
    return torch.eye(dim).to(device=dev)[:, :-1]

def tiota(dim, dev):
    return torch.ones(dim).to(device=dev).view(-1, 1)

def extend(phi):
    return torch.cat((torch.cat((phi, torch.zeros(phi.shape[0], 1,
                                                  device = phi.device)
                                 ), dim=1),
                      torch.zeros(1, phi.shape[1] + 1,
                                  device = phi.device)), dim=0)


def choices_vectorised(mus, p, q, dev):
    nty0 = mus.shape[1]
    tMuM = torch.matmul(tJ(nty0, dev).T, mus)
    tMuF = torch.matmul(mus, tJ(nty0, dev))
    return torch.cat(
        [torch.matmul(tMuM.view(-1, p.shape[1]), p.T),
         torch.matmul(tMuF.view(-1, q.shape[1]), q.T)],
        dim=1)

def choices(mus, p, q, dev):
    tMuM = mus[:,:-1,:]
    tMuF = mus[:,:,:-1]
    M = torch.einsum('nij,ijk->nk', tMuM, p)
    F = torch.einsum('nij,ijk->nk', tMuF, q)
    return torch.cat([M, F], dim = 1)


# Define the residuals function (unconstrained + sinkhorn)
def residuals(ng0, xi, tP, tQ, beta, phi, masks, dev):

    maskc, mask0 = masks
    phi0 = extend(phi)
    ndim = tP.shape[0]
    concentrations = torch.tensor([1.0] * (tP.shape[0] * 2)).to(device = dev)
    dirichlet = torch.distributions.Dirichlet(concentrations)
    s0 = dirichlet.sample((ng0, ))
    s = s0[torch.all(s0 > 0.01, dim=1)] # 0.02 worked
    ng = s.shape[0]

    mus, vcur = xi(s)
    snext = choices(mus, tP, tQ, dev)
    _, vnext = xi(snext)

    unregularised = (mus * phi0).sum(dim=(1,2))
    entropyc = (masked_log(mus, maskc) * mus).sum(dim = (1, 2))
    entropyf = (masked_log(mus[:,-1,:], mask0) * mus[:,-1,:]).sum(dim=1)
    entropym = (masked_log(mus[:,:,-1], mask0) * mus[:,:,-1]).sum(dim=1)

    # Choo/Siow (2006): 2ec-em-ef, Galichon has 2ec+em+ef (sometimes other)
    # 2 e_c + e_m + e_f according to my derivations
    # -entropy is the largest for uniform distribution (all mus equal)
    # we maximise, thus we punish mus close to 0 or 1 (due to adding up)
    fun = unregularised - (2 * entropyc + entropym + entropyf)

    sumL = torch.sum(fun + beta * vnext)
    grads = autograd_grad(outputs=sumL, inputs=mus, create_graph=True)

    lambda1, lambda2 = 1.0, 1.0
    r1 = lambda1 * torch.square(grads[0].view(ng, -1))
    r2 = lambda2 * torch.square((vcur - fun - beta * vnext).view(ng, 1))

    resid_v = torch.cat((r1, r2), dim=1)
    mean_resid_v = torch.mean(resid_v)

    torch.cuda.empty_cache()

    return mean_resid_v


def minimise_inner(xi, theta, beta, tP, tQ, ng, tau, masks, dev):

    phi = tau(theta, dev)

    epochs = 10000
    optimiser = optim.Adam(xi.parameters(), lr = .0001) #, weight_decay = 0.01)

    def calculate_loss():
        optimiser.zero_grad()
        for attempts in range(1, 50): # safeguard for "bad draw" of s
            val = residuals(ng, xi, tP, tQ, beta, phi, masks, dev)
            if not val.isnan():
                break
            else:
                print(f"{TermColours.RED}.{TermColours.RESET}", end='')
        val.backward(retain_graph=True)

        #for name, param in xi.named_parameters():
        #    if torch.isnan(param.grad).any():
        #        pdb.set_trace()
        # 2/4/25 tried without clipping
        #torch.nn.utils.clip_grad_norm_(xi.parameters(), max_norm=1.0)
        #torch.nn.utils.clip_grad_value_(xi.parameters(), 1.0)
        return val

    for epoch in range(0, epochs):
        loss = optimiser.step(calculate_loss)
        if epoch % (epochs // 100) == 0:
            print(f"{int((epoch/epochs) * 100)}%: {loss.item():.4f}",
                  end='\t', flush = "True")

    return loss

def match_moments(xi0, xi1, xi2, theta0, theta1, theta2, tPs, tQs,
                  tMuHat, ng, dev, tau, masks, treat_idcs,
                  skiptrain = False, cf = CF.None_):

    beta = torch.tensor(0.9, device=dev)

    # these won't depend on phi (leaves in the autograd graph)
    # dldTheta is gradient with respect to inner loss function
    # emulate dl/dphi = dl/dxi * dxi/dphi + dl/dphi
    # dl/dxi = 0 by the envelope theorem at xi = xi_opt
    # update: now gradient graph is kept and phi is set to requires_grad

    if not skiptrain:
        xi0.train()
        loss0 = minimise_inner(xi0, theta0, beta, tPs[0], tQs[0],
                               ng, tau, masks, dev)
        xi1.train()
        loss1 = minimise_inner(xi1, theta1, beta, tPs[1], tQs[1],
                               ng, tau, masks, dev)
        xi2.train()
        loss2 = minimise_inner(xi2, theta2, beta, tPs[2], tQs[2],
                               ng, tau, masks, dev)
    else:
        loss0, loss1, loss2 = None, None, None

    nT, nty0, nty0 = tMuHat.size()

    def transition_s(xi, p, q): # deprecated
        def step(ss):
            mus, _ = xi(ss)
            return choices(mus, p, q, dev)
        return step
    def transition_mu(xi, p, q, cf):
        def step(mus):
            ss = choices(mus, p, q, dev)
            mus, _ = xi(ss)
            return mus
        def step_household(mus): # not in use
            ss = choices(mus, p, q, dev)
            tM = torch.matmul(torch.matmul(tJ(nty0, dev).T, mus), tiota(nty0, dev))
            tF = torch.matmul(tiota(nty0, dev).T, torch.matmul(mus, tJ(nty0, dev)))
            mus, _ = xi(torch.cat([tM.squeeze(), tF.squeeze()], dim=0).view(1, -1))
            return mus
        def step_matching(mus): # not in use
            ss = choices(mus, p, q, dev)
            mus, _ = xi(ss)
            return mus
        return {CF.None_: step,
                CF.HouseholdOnly: step_household,
                CF.MatchingOnly: step_matching}[cf]



    nmembs = torch.full(tMuHat.shape, 2, device = dev)
    nmembs[:, :, -1] = nmembs[:, -1, :] = 1
    tF = (tMuHat * nmembs).sum(1)
    tM = (tMuHat * nmembs).sum(2)
    ss_hat = torch.cat((tM[:,:].view(-1, nty0-1),
                       tF[:,:].view(-1, nty0-1)), dim=1)

    # initial state
    ss_cur = ss_hat[0, :].view(1, -1)
    mu_cur = tMuHat[0,:,:].unsqueeze(0)
    pre = range(0, treat_idcs[0])
    post = range(treat_idcs[-1] + 1, nT)
    control_idcs = list(pre) + list(post)

    # we recursively define the whole path
    xi0.eval()
    xi1.eval()
    xi2.eval()
    transition = transition_mu
    walker_pre = transition(xi0, tPs[0], tQs[0], cf)
    walker_treat = transition(xi1, tPs[1], tQs[1], cf)
    walker_post = transition(xi2, tPs[2], tQs[2], cf)
    regimes = [(pre, walker_pre),
               (treat_idcs, walker_treat),
               (post, walker_post)]

    ss_star = torch.zeros(nT, ss_cur.shape[1]).to(dev)
    mu_star = torch.zeros(nT, nty0, nty0).to(dev)

    for (idcs, walker) in regimes:
        for i in idcs:
           mu_star[i, :, :] = mu_cur
           mu_cur = walker(mu_cur)
           #ss_star[i, ] = ss_cur
           #ss_cur = walker(ss_cur)

    resid = torch.square(tMuHat - mu_star).sum()

    torch.cuda.empty_cache()

    return (resid, tMuHat, mu_star, loss0, loss1, loss2)


