import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad as autograd_grad
from torch.optim.lr_scheduler import MultiStepLR

from .helpers import tauMflex, tauM, minfb, TermColours

# Define the dr function
def dr(s, xi):
    x = xi(s)
    ng, cols = x.size()
    ndim = int(((cols - 1)/2) ** .5)
    #mu = torch.sigmoid(x[:, :ndim**2])
    mu = x[:, :ndim**2]
    return mu.view(ng, ndim, ndim)

# Define the residuals function
def residuals(ng0, xi, tP, tQ, beta, phi, dev):

    ndim = tP.shape[0]
    concentrations = torch.tensor([5.5] * (tP.shape[0] * 2)).to(device = dev)
    dirichlet = torch.distributions.Dirichlet(concentrations)
    s0 = dirichlet.sample((ng0, ))[:,:-1]
    s = s0[torch.all(s0 > 0.05, dim=1)]
    ng = s.shape[0]
    #s.requires_grad = True

    x = xi(s)
    mu = x[:, :ndim**2]  # mu's (singles are implied)
    nu = x[:, ndim**2:2*ndim**2]  # nu's (LM) 
    V = x[:, 2*ndim**2]  # cont. value

    # Transition of endogenous state
    muc = mu.view(s.shape[0], ndim, ndim)
    M = s[:, :ndim]
    last = (1 - s[:, ndim:].sum(dim=1) - M.sum(dim = 1)).view(-1, 1)
    F = torch.cat((s[:,ndim:], last), dim = 1)

    mum0 = M - torch.sum(muc, dim=2)
    mu0f = F - torch.sum(muc, dim=1)
    mum = torch.cat((muc, mum0.view(ng, ndim, 1)), dim=2)
    muf = torch.cat((muc, mu0f.view(ng, 1, ndim)), dim=1)

    mnext = torch.matmul(mum.view(ng, ndim * (ndim + 1)), tP.T)
    fnext = torch.matmul(muf.view(ng, ndim * (ndim + 1)), tQ.T)
    snext = torch.cat((mnext, fnext), dim = 1)[:,:-1]

    # Compute next period's value and expected marginal social surplus growth
    xnext = xi(snext)
    vnext = xnext[:, -1]

    unregularised = torch.multiply(mu, phi.view(-1)).sum(dim = 1)
    entropyc = -torch.multiply(mu, torch.log(mu)).sum(dim = 1)
    entropym = -torch.multiply(mum0, torch.log(mum0)).sum(dim = 1)
    entropyf = -torch.multiply(mu0f, torch.log(mu0f)).sum(dim = 1)
    fun = unregularised + (2 * entropyc + entropym + entropyf)

    # instead of constraining only mu, constrain mu, mum0, mu0f
    sumL = torch.sum(fun + beta * vnext)
    grads = autograd_grad(outputs=sumL, inputs=mu, create_graph=True)
    gradsm0 = autograd_grad(outputs=sumL, inputs=mum0, create_graph=True)
    grads0f = autograd_grad(outputs=sumL, inputs=mu0f, create_graph=True)
    dLdMu = torch.cat((grads[0], gradsm0[0], grads0f[0]), dim=1)
    allMu = torch.cat((mu, mum0, mu0f), dim=1)
    compsla = minfb(allMu, dLdMu)

    lambda1, lambda2, lambda3 = 1.0, 1.0, 1.0
    r1 = lambda1 * torch.square(nu - grads[0]) # grads[0] vs dLdMu
    r2 = lambda2 * torch.square(compsla)
    r3 = lambda3 * torch.square((V - fun - beta * vnext).view(ng, 1))

    resid_v = torch.cat((r1, r2, r3), dim=1)
    mean_resid_v = torch.mean(resid_v)

    return mean_resid_v

class Perceptron(nn.Module):
    def __init__(self, hidden_sizes, ntypes):
        super(Perceptron, self).__init__()
        layers = []
        input_size = 2 * ntypes - 1 # adding up
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        nout = 2 * (ntypes ** 2) + 1  # mu, nu, V (dim mu = dim nu)
        layers.append(nn.Linear(input_size, nout))
        self.layers = nn.Sequential(*layers)
        self.nout = nout
        self.ntypes = ntypes
        self._initialize_weights()
    # this makes sure that startup mus are feasiable
    # in case of nan's adjust this
    def _initialize_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                #nn.init.uniform_(m.weight, a=0.0000001, b=0.0000002)
                nn.init.uniform_(m.weight, a=0.00, b=0.05)
                if m.bias is not None:
                    #nn.init.uniform_(m.bias, a=0.00001, b=0.00003)
                    nn.init.uniform_(m.bias, a=0.00, b=0.03)
    def forward(self, x):
        return self.layers(x)

def minimise_inner(theta, beta, tP, tQ, ng, tau, dev):

    phi = tau(theta, dev)

    network = Perceptron([64, 64], tP.shape[0])
    xi = network.to(dev)

    epochs = 5000
    milestones = [epochs // 4, epochs // 2, 3 * epochs // 4]
    optimiser = optim.SGD(xi.parameters(), lr=0.000001)
    scheduler = MultiStepLR(optimiser, milestones = milestones, gamma=0.5)

    def calculate_loss():
        optimiser.zero_grad()
        for attempts in range(1, 50): # safeguard for "bad draw" of s
            val = residuals(ng, xi, tP, tQ, beta, phi, dev)
            if not val.isnan():
                break
            else:
                print(f"{TermColours.RED}.{TermColours.RESET}", end='')
        val.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(xi.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_value_(xi.parameters(), 1.0)
        return (val, attempts)

    for epoch in range(1, epochs + 1):
        loss, _ = optimiser.step(calculate_loss)
        scheduler.step()
        if epoch % (epochs // 100) == 0:
            print(f"{int((epoch/epochs) * 100)}%: {loss.item():.4f}",
                  end='\t', flush = "True")

    return xi, loss

def match_moments(theta0, theta1, tP, tQ, tMuHat, ng, dev, tau, treat_idcs):

    beta = torch.tensor(0.9, device=dev)

    # these won't depend on phi (leaves in the autograd graph)
    # dldTheta is gradient with respect to inner loss function
    # emulate dl/dphi = dl/dxi * dxi/dphi + dl/dphi
    # dl/dxi = 0 by the envelope theorem at xi = xi_opt
    # update: now gradient graph is kept and phi is set to requires_grad
    xi0, loss0 = minimise_inner(theta0, beta, tP, tQ, ng, tau, dev)
    xi1, loss1 = minimise_inner(theta1, beta, tP, tQ, ng, tau, dev)

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
            [torch.matmul(tMuM.view(-1, tP.shape[1]), p.T),
             torch.matmul(tMuF.view(-1, tQ.shape[1]), q.T)],
            dim=1)[:,:-1]

    def transition(xi, tP, tQ):
        def step(ss):
            return choices(match(xi, ss), tP, tQ)
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
    walker_c = transition(xi0, tP, tQ)
    walker_t = transition(xi1, tP, tQ)
    regimes = [(pre, walker_c), (treat_idcs, walker_t), (post, walker_c)]

    ss_star = torch.zeros(nT, ss_cur.shape[1]).to(dev)

    for (idcs, walker) in regimes:
        for i in idcs:
           ss_star[i, ] = ss_cur
           ss_cur = walker(ss_cur)

    resid = torch.square(ss_hat - ss_star).sum()

    return (resid, ss_hat, ss_star)


