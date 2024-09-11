import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad as autograd_grad
from torch.optim.lr_scheduler import MultiStepLR
import math

from .deeplearning import Perceptron
from .helpers import tauMflex, tauM, minfb, TermColours, ManualLRScheduler

# Define the dr function
def dr(s, xi):
    mu, _, _ = xi(s)
    #ng, cols = x.size()
    ng, nout = mu.size()
    #ndim = int(((cols - 1)/2) ** .5)
    ndim = int(math.sqrt(nout))
    return mu.view(ng, ndim, ndim)

# Define the residuals function (unconstrained)
def residuals(ng0, xi, tP, tQ, beta, phi, dev):

    ndim = tP.shape[0]
    concentrations = torch.tensor([2.5] * (tP.shape[0] * 2)).to(device = dev)
    dirichlet = torch.distributions.Dirichlet(concentrations)
    s0 = dirichlet.sample((ng0, ))[:,:-1]
    s = s0[torch.all(s0 > 0.01, dim=1)] # saw-shape?
    ng = s.shape[0]
    #s.requires_grad = True

    mu, _, V = xi(s) # mu's (singles are implied), continuation value

    # Transition of endogenous state
    muc = mu.view(s.shape[0], ndim, ndim)
    M = s[:, :ndim]
    last = (1 - s[:, ndim:].sum(dim=1) - M.sum(dim = 1)).view(-1, 1)
    F = torch.cat((s[:,ndim:], last), dim = 1)

    mum0 = torch.max(M - torch.sum(muc, dim=2), torch.tensor(10e-6))
    mu0f = torch.max(F - torch.sum(muc, dim=1), torch.tensor(10e-6))
    mum = torch.cat((muc, mum0.view(ng, ndim, 1)), dim=2)
    muf = torch.cat((muc, mu0f.view(ng, 1, ndim)), dim=1)

    mnext = torch.matmul(mum.view(ng, ndim * (ndim + 1)), tP.T)
    fnext = torch.matmul(muf.view(ng, ndim * (ndim + 1)), tQ.T)
    snext = torch.cat((mnext, fnext), dim = 1)[:,:-1]

    # Compute next period's value and expected marginal social surplus growth
    _, _, vnext = xi(snext)

    unregularised = torch.multiply(mu, phi.view(-1)).sum(dim = 1)
    entropyc = torch.multiply(mu, torch.log(mu)).sum(dim = 1)
    entropym = torch.multiply(mum0, torch.log(mum0)).sum(dim = 1)
    entropyf = torch.multiply(mu0f, torch.log(mu0f)).sum(dim = 1)
    # Choo/Siow (2006): 2ec-em-ef, Galichon has 2ec+em+ef (sometimes other)
    fun = unregularised - (2 * entropyc + entropym + entropyf)

    # instead of constraining only mu, constrain mu, mum0, mu0f
    sumL = torch.sum(fun + beta * vnext)
    grads = autograd_grad(outputs=sumL, inputs=mu, create_graph=True)
    gradsm0 = autograd_grad(outputs=sumL, inputs=mum0, create_graph=True)
    grads0f = autograd_grad(outputs=sumL, inputs=mu0f, create_graph=True)
    dLdMu = torch.cat((grads[0], gradsm0[0], grads0f[0]), dim=1)
    allMu = torch.cat((mu, mum0, mu0f), dim=1)

    lambda1, lambda2 = 1.0, 1.0
    r1 = lambda1 * torch.square(dLdMu) # grads[0] vs dLdMu
    r2 = lambda2 * torch.square((V - fun - beta * vnext).view(ng, 1))

    resid_v = torch.cat((r1, r2), dim=1)
    mean_resid_v = torch.mean(resid_v)

    torch.cuda.empty_cache()

    return mean_resid_v


def minimise_inner(xi, theta, beta, tP, tQ, ng, tau, dev):

    phi = tau(theta, dev)

    epochs = 20000
    milestones = [epochs // 10, epochs // 4, epochs // 2]
    milestones = []
    optimiser = optim.SGD(xi.parameters(),
                          lr=0.01)
    scheduler = MultiStepLR(optimiser, milestones = milestones, gamma=0.1)
    # scheduler = ManualLRScheduler(optimiser, factor=0.5, min_lr=1e-8)

    def calculate_loss():
        optimiser.zero_grad()
        for attempts in range(1, 50): # safeguard for "bad draw" of s
            val = residuals(ng, xi, tP, tQ, beta, phi, dev)
            if not val.isnan():
                break
            else:
                print(f"{TermColours.RED}.{TermColours.RESET}", end='')
        val.backward(retain_graph=True)
        #torch.nn.utils.clip_grad_norm_(xi.parameters(), max_norm=1.0)
        #torch.nn.utils.clip_grad_value_(xi.parameters(), 1.0)
        return val

    for epoch in range(1, epochs + 1):
        loss = optimiser.step(calculate_loss)
        if epoch % (epochs // 100) == 0:
            print(f"{int((epoch/epochs) * 100)}%: {loss.item():.4f}",
                  end='\t', flush = "True")

    return loss

def match_moments(xi0, xi1, theta0, theta1, tPs, tQs,
                  tMuHat, ng, dev, tau, treat_idcs, skiptrain = False):

    beta = torch.tensor(0.95, device=dev)

    # these won't depend on phi (leaves in the autograd graph)
    # dldTheta is gradient with respect to inner loss function
    # emulate dl/dphi = dl/dxi * dxi/dphi + dl/dphi
    # dl/dxi = 0 by the envelope theorem at xi = xi_opt
    # update: now gradient graph is kept and phi is set to requires_grad

    if not skiptrain:
        loss0 = minimise_inner(xi0, theta0, beta, tPs[1], tQs[1],
                               ng, tau, dev)
        loss1 = minimise_inner(xi1, theta1, beta, tPs[1], tQs[1],
                               ng, tau, dev)
    else:
        loss0, loss1 = None, None

    tJ = lambda dim: torch.eye(dim).to(device=dev)[:, :-1]
    tiota = lambda dim: torch.ones(dim).to(device=dev).view(-1, 1)
    nT, nty0, nty0 = tMuHat.size()

    def match(xi, ss):
        mus0 = dr(ss, xi).view(nty0-1, nty0-1)
        mum0s = ss[:,:(nty0-1)]
        muf0s = torch.cat((ss[:,-(nty0-2):],
                            1 - ss.sum().view(1, 1)), dim = 1)
        zero = torch.tensor(0.0, device = dev).view(1, 1)
        mus = torch.cat((torch.cat((mus0, mum0s), dim = 0),
                         torch.cat((muf0s.view(-1, 1), zero), dim=0)),
                        dim=1)
        return mus

    def choices(mus, p, q):
        tMuM = torch.matmul(tJ(nty0).T, mus)
        tMuF = torch.matmul(mus, tJ(nty0))
        return torch.cat(
            [torch.matmul(tMuM.view(-1, p.shape[1]), p.T),
             torch.matmul(tMuF.view(-1, q.shape[1]), q.T)],
            dim=1)[:,:-1]

    def transition(xi, p, q):
        def step(ss):
            return choices(match(xi, ss), p, q)
        return step

    # calculate sshat from data (marginals of muhat)
    tMuM = torch.bmm(tJ(nty0).T.repeat(nT, 1, 1), tMuHat)
    tMuF = torch.bmm(tMuHat, tJ(nty0).repeat(nT, 1, 1))
    tM = torch.bmm(tMuM, tiota(nty0).repeat(nT, 1, 1)).view(-1, nty0 - 1)
    tF = torch.bmm(tiota(nty0).T.repeat(nT, 1, 1), tMuF).view(-1, nty0 - 1)
    ss_hat = torch.cat((tM[:,:].view(-1, nty0-1),
                       tF[:,:-1].view(-1, nty0-2)), dim=1)

    # initial state
    ss_cur = ss_hat[0, :].view(1, -1)
    pre = range(0, treat_idcs[0])
    post = range(treat_idcs[-1] + 1, nT)
    control_idcs = list(pre) + list(post)

    # we recursively define the whole path
    walker_pre = transition(xi0, tPs[0], tQs[0])
    walker_treat = transition(xi1, tPs[1], tQs[1])
    walker_post = transition(xi0, tPs[2], tQs[2])
    regimes = [(pre, walker_pre),
               (treat_idcs, walker_treat),
               (post, walker_post)]

    ss_star = torch.zeros(nT, ss_cur.shape[1]).to(dev)

    for (idcs, walker) in regimes:
        for i in idcs:
           ss_star[i, ] = ss_cur
           ss_cur = walker(ss_cur)

    resid = torch.square(ss_hat - ss_star).sum()

    torch.cuda.empty_cache()

    return (resid, ss_hat, ss_star, loss0, loss1)


