def residuals_constr(ng0, xi, tP, tQ, beta, phi, dev):

    ndim = tP.shape[0]
    concentrations = torch.tensor([5.5] * (tP.shape[0] * 2)).to(device = dev)
    dirichlet = torch.distributions.Dirichlet(concentrations)
    s0 = dirichlet.sample((ng0, ))[:,:-1]
    s = s0[torch.all(s0 > 0.05, dim=1)]
    ng = s.shape[0]

    mu, nu, V = xi(s)

    # Transition of endogenous state
    muc = mu.view(s.shape[0], ndim, ndim)
    M = s[:, :ndim]
    last = (1 - s[:, ndim:].sum(dim=1) - M.sum(dim = 1)).view(-1, 1)
    F = torch.cat((s[:,ndim:], last), dim = 1)

    mum0 = torch.max(M - torch.sum(muc, dim=2), torch.tensor(10e-3))
    mu0f = torch.max(F - torch.sum(muc, dim=1), torch.tensor(10e-3))
    mum = torch.cat((muc, mum0.view(ng, ndim, 1)), dim=2)
    muf = torch.cat((muc, mu0f.view(ng, 1, ndim)), dim=1)

    mnext = torch.matmul(mum.view(ng, ndim * (ndim + 1)), tP.T)
    fnext = torch.matmul(muf.view(ng, ndim * (ndim + 1)), tQ.T)
    snext = torch.cat((mnext, fnext), dim = 1)[:,:-1]

    # Compute next period's value and expected marginal social surplus growth
    xnext = xi(snext)
    vnext = xnext[:, -1]

    unregularised = torch.multiply(mu, phi.view(-1)).sum(dim = 1)
    entropyc = torch.multiply(mu, torch.log(mu)).sum(dim = 1)
    entropym = torch.multiply(mum0, torch.log(mum0)).sum(dim = 1)
    entropyf = torch.multiply(mu0f, torch.log(mu0f)).sum(dim = 1)
    fun = unregularised - (2 * entropyc + entropym + entropyf)

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

## grid search

step = 1
grids = [torch.linspace(start=value - 2 * step, end=value + 2 * step, steps=5, device = dev) for value in theta]
grid_5d = torch.meshgrid(*grids, indexing='ij')
grid_points = torch.stack([g.flatten() for g in grid_5d], dim=1)
indices = torch.randperm(grid_points.size(0))
random_grid = grid_points[indices]

results = []
for point in random_grid:
    loss = match_moments(point, tP, tQ, tMuHat, ng, dev, tauM, treat_idcs)
    results.append((point, loss))
    print(f"{TermColours.RED}{loss}: {point.tolist()}{TermColours.RESET}")
    cur_min = min(results, key=lambda x: x[1])
    print(f"{TermColours.GREEN}{cur_min[1]}: {cur_min[0].tolist()}{TermColours.RESET}")

# Sort results by the loss value
results.sort(key=lambda x: x[1])



