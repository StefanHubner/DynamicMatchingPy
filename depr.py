import pdb

def log_sum_exp_stable(H, a, epsilon, dim):
    H_shifted = H - torch.min(H, dim=dim, keepdim=True)[0]
    s = torch.sum(a * torch.exp(-H_shifted / epsilon), dim = dim)
    return -epsilon * torch.log(s) + torch.min(H, dim=dim)[0]

def mina(H, a, epsilon):
    return log_sum_exp_stable(H, a, epsilon, dim=0)

def minb(H, b, epsilon):
    return log_sum_exp_stable(H, b, epsilon, dim=1)

def compute_transport_plan(f, g, a, b, epsilon):
    K = (f.unsqueeze(1) + g.unsqueeze(0)) / epsilon
    T = a.unsqueeze(1) + K + b.unsqueeze(0)
    return torch.exp(T)


def sinkhorn_log(A, row_margins, col_margins, iter, masks):
   eps = 1e-10
   cmask, smask = masks
   H = masked_log(A, cmask)
   a = masked_log(row_margins, smask)
   b = masked_log(col_margins, smask)
   f = torch.zeros(H.shape[0], device = H.device)
   g = torch.zeros(H.shape[1], device = H.device)
   for _ in range(iter):
       f = mina(H - g, a, eps)
       g = minb(H - f, b, eps)
   return torch.exp(H + f.unsqueeze(1) + g.unsqueeze(0))


class Perceptron(nn.Module):
    def __init__(self, hidden_sizes, ntypes, nout, llb = 0.0):
        super(Perceptron, self).__init__()
        layers = []
        input_size = 2 * ntypes - 1 # adding up
        self.nout = nout + 1
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, self.nout))
        self.layers = nn.Sequential(*layers)
        self.ntypes = ntypes
        self.lastlayerbias = llb
        self._initialize_weights()
    # this makes sure that startup mus are feasiable
    # in case of nan's adjust this
    def _initialize_weights(self):
        gain = nn.init.calculate_gain('relu')
        for idx, m in enumerate(self.layers):
            if isinstance(m, nn.Linear):
                bound = gain * math.sqrt(6 / m.weight.size(1))
                nn.init.uniform_(m.weight, a=-bound, b=+bound)
                if m.bias is not None:
                    #bias for last layer to be inside the feasible region
                    b = self.lastlayerbias * int(idx == len(self.layers) - 1)
                    nn.init.constant_(m.bias, b)
    def forward(self, x):
        y = self.layers(x)
        mupar = torch.sigmoid(y[:,:-1])
        V = torch.exp(y[:,-1]) # torch.exp
        return (mupar, (), V)


def match(xi, ss, dr, dev):
    muc, _, _ = dr(ss, xi)
    #mum0s = ss[:,:(nty0-1)]
    #muf0s = torch.cat((ss[:,-(nty0-2):],
    #                    1 - ss.sum().view(1, 1)), dim = 1)
    nty0 = muc.shape[1] + 1
    mum0, mu0f = calculate_singles(muc, ss)
    zero = torch.tensor(0.0, device = dev).view(1, 1, 1)
    mus = torch.cat((torch.cat((muc, mum0.view(1, 1, -1)), dim = 1),
                     torch.cat((mu0f.view(1, -1, 1), zero), dim=1)),
                    dim=2)
    return mus.view(nty0, nty0)

# Define the dr function
def build_dr(tau, dev):
    def dr(s, xi):
        mupar, nu, v = xi(s)
        mu = torch.vmap(lambda p: tau(p, dev), in_dims=0)(mupar)
        ng, ndim, ndim = mu.size()
        return mu.view(ng, ndim, ndim), nu, v
    return dr

def smooth_max(x, min_value = 10e-09, slope=100):
    x_stack = torch.stack([x, torch.full_like(x, min_value)])
    return torch.logsumexp(x_stack * slope, dim=0) / slope

def calculate_singles(muc, s, maxf = smooth_max):
    ndim = (s.shape[1] + 1) // 2
    M = s[:, :ndim]
    last = (1 - s[:, ndim:].sum(dim=1) - M.sum(dim = 1)).view(-1, 1)
    F = torch.cat((s[:,ndim:], last), dim = 1)
    mum0 = maxf(M - torch.sum(muc, dim=2))
    mu0f = maxf(F - torch.sum(muc, dim=1))
    return mum0, mu0f


# Define the residuals function (unconstrained)
def residuals(ng0, xi, tP, tQ, beta, phi, dr, dev):

    ndim = tP.shape[0]
    concentrations = torch.tensor([1.5] * (tP.shape[0] * 2)).to(device = dev)
    dirichlet = torch.distributions.Dirichlet(concentrations)
    s0 = dirichlet.sample((ng0, ))[:,:-1]
    s = s0[torch.all(s0 > 0.01, dim=1)]
    ng = s.shape[0]
    #s.requires_grad = True
    maxf = smooth_max # lambda t: torch.max(t, torch.tensor(10e-6))

    muc, _, V = dr(s, xi) # mu's (singles are implied), continuation value
    #print(muc[0,:,:])

    # Transition of endogenous state
    mum0, mu0f = calculate_singles(muc, s, maxf)
    mum = torch.cat((muc, mum0.view(ng, ndim, 1)), dim=2)
    muf = torch.cat((muc, mu0f.view(ng, 1, ndim)), dim=1)

    mnext = torch.matmul(mum.view(ng, ndim * (ndim + 1)), tP.T)
    fnext = torch.matmul(muf.view(ng, ndim * (ndim + 1)), tQ.T)
    snext = torch.cat((mnext, fnext), dim = 1)[:,:-1]

    # Compute next period's value and expected marginal social surplus growth
    _, _, vnext = dr(snext, xi)

    # the maxf will mess with the regularisation a bit
    # also it regularises the sparse cases which will lead to a loss offset
    musafe = maxf(muc)
    unregularised = torch.multiply(musafe, phi).sum(dim = (1, 2))
    entropyc = torch.multiply(musafe, torch.log(musafe)).sum(dim = (1, 2))
    entropym = torch.multiply(mum0, torch.log(mum0)).sum(dim = 1)
    entropyf = torch.multiply(mu0f, torch.log(mu0f)).sum(dim = 1)
    # Choo/Siow (2006): 2ec-em-ef, Galichon has 2ec+em+ef (sometimes other)
    # 2 e_c + e_m + e_f according to my derivations
    # -entropy is the largest for uniform distribution (all mus equal)
    # we maximise, thus we punish mus close to 0 or 1 (due to adding up)
    fun = unregularised - (2 * entropyc + entropym + entropyf)

    # instead of constraining only mu, constrain mu, mum0, mu0f
    sumL = torch.sum(fun + beta * vnext)
    grads = autograd_grad(outputs=sumL, inputs=musafe, create_graph=True)
    gradsm0 = autograd_grad(outputs=sumL, inputs=mum0, create_graph=True)
    grads0f = autograd_grad(outputs=sumL, inputs=mu0f, create_graph=True)
    dLdMu = torch.cat((grads[0].view(ng, -1), gradsm0[0], grads0f[0]), dim=1)
    # allMu = torch.cat((mu, mum0, mu0f), dim=1)

    lambda1, lambda2 = 1.0, 1.0
    # should singles be regularised?
    r1 = lambda1 * torch.square(dLdMu) # grads[0].view(ng, -1) vs dLdMu
    r2 = lambda2 * torch.square((V - fun - beta * vnext).view(ng, 1))
    # maybe normalise deviations (relative deviations between V and Vnext)?

    resid_v = torch.cat((r1, r2), dim=1)
    mean_resid_v = torch.mean(resid_v)

    torch.cuda.empty_cache()

    return mean_resid_v

def v(b):
    def v0(T):
        if T == 0:
            return 0
        else:
            return 1 + b * v0(T - 1)
    return v0

[list(map(f, range(1, 10))) for f in map(v, [0.9, 0.8, 0.66, 0.5, 0.25, 0])]

# Assuming you have your matrices as pandas dataframes
def data(label):
    p = json.load(open("/tmp/tP" + label + ".json", "r"))
    q = json.load(open("/tmp/tQ" + label + ".json", "r"))
    muhat = json.load(open("/tmp/tMuHat" + label + ".json", "r"))
    # Convert dataframes to lists of lists
    return Dataset.from_dict({'p': [p],
                              'q': [q],
                              'couplings': [muhat]})

# Combine datasets into a DatasetDict
# they are really all just functions of muhat with the J_m, J_f projections
dataset_dict = DatasetDict({
    'K': data('Kids'),
    'M': data('Marriage'),
    'KM': data('KM')
})

hf_rwtoken = "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"
dataset_dict.push_to_hub("StefanHubner/DivorceData", token = hf_rwtoken)


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



