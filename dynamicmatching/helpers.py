import torch

class TermColours:
    RED = "\033[31m"
    RESET = "\033[0m"
    GREEN = "\033[32m"

# vector basis function
def vbasis(n, i, dev):
    b = torch.zeros(n, device = dev)
    b[i] = 1
    return b

# matrix basis function
def mbasis(n, i, j, dev):
    b = torch.zeros(n, n, device = dev)
    b[i,j] = 1
    return b

# tau for married only (3x3)
def tauM(par, treat, dev):
    theta = par[:-1]
    kappa = par[-1]
    n = theta.shape[0]
    rs = theta + vbasis(n, n - 1, dev) * kappa * treat
    b = torch.stack((
        mbasis(3, 0, 0, dev),
        mbasis(3, 1, 1, dev),
        mbasis(3, 0, 1, dev) + mbasis(3, 1, 0, dev),
        mbasis(3, 2, 2, dev))) # by convention the last one is (c, c)
    return torch.multiply(rs.view(-1, 1, 1), b).sum(dim=0)

def tauMflex(par, dev):
    b = torch.stack((
        mbasis(3, 0, 0, dev),
        mbasis(3, 1, 1, dev),
        mbasis(3, 0, 1, dev) + mbasis(3, 1, 0, dev),
        mbasis(3, 2, 2, dev))) # by convention the last one is (c, c)
    return torch.multiply(par.view(-1, 1, 1), b).sum(dim=0)

# Function to minimize fb
def minfb(a, b):
    return a + b - torch.sqrt(a**2 + b**2 + 1e-8)


class ManualLRScheduler:
    def __init__(self, optimizer, factor=0.1, min_lr=1e-8):
        self.optimizer = optimizer
        self.factor = factor
        self.min_lr = min_lr

    def step(self):
        for param_group in self.optimizer.param_groups:
            new_lr = max(param_group['lr'] * self.factor, self.min_lr)
            param_group['lr'] = new_lr

