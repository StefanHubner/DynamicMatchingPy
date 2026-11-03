import torch
import numpy as np
from enum import Enum

class CF(Enum):
    None_ = 0          # No Coutnerfactural
    HighCost = 1
    LowCost = 2

class TermColours:
    BLACK   = "\033[30m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"
    RESET   = "\033[0m"
    BRIGHT_BLACK   = "\033[90m"
    BRIGHT_RED     = "\033[91m"
    BRIGHT_GREEN   = "\033[92m"
    BRIGHT_YELLOW  = "\033[93m"
    BRIGHT_BLUE    = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN    = "\033[96m"
    BRIGHT_WHITE   = "\033[97m"

def extend(phi):
    return torch.cat((torch.cat((phi, torch.zeros(phi.shape[0], 1,
                                                  device = phi.device)
                                 ), dim=1),
                      torch.zeros(1, phi.shape[1] + 1,
                                  device = phi.device)), dim=0)

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

def tauMcal(par, t, d, dev):
    b = torch.stack((
        mbasis(2, 0, 0, dev),
        mbasis(2, 1, 1, dev),
        d * mbasis(2, 1, 1, dev)))
    psi = -torch.log(torch.tensor([1.1], device = dev)) # theoretical 1.1 hazard ratio at cutoff
    p = torch.cat([ par[[0, 1]], psi ]).view(-1, 1, 1)
    const = torch.multiply(b, p).sum(dim = 0)
    return extend(const)

def scaleMcal(par, dev):
    one = torch.ones(1, device=dev)
    a = torch.cat([one, torch.exp(par[2:])])
    return a, a

def tauMScal(par, t, d, dev):
    b_const = torch.stack((
        mbasis(3, 0, 0, dev),
        mbasis(3, 1, 1, dev),
        mbasis(3, 2, 2, dev),
        mbasis(3, 2, 1, dev),
        mbasis(3, 1, 2, dev),
        d * mbasis(3, 1, 1, dev),
        d * mbasis(3, 2, 2, dev),
        d * mbasis(3, 2, 1, dev),
        d * mbasis(3, 1, 2, dev)))
    rf = [0.046/0.035, 0.18/0.155, 1.0, 1.0] # odds ratio
    psi = -torch.log(torch.tensor(rf, device = dev)) 
    p_const = torch.cat([ par[[0, 1, 2, 3, 4]], psi ]).view(-1, 1, 1)
    const = torch.multiply(b_const, p_const).sum(dim = 0)
    return extend(const)

def scaleMScal(par, dev):
    one = torch.ones(1, device=dev)
    a = torch.cat([one, torch.exp(par[5:])])
    return a, a

def tauMScaltrend(par, t, d, dev):
    b_const = torch.stack((
        mbasis(3, 0, 0, dev),
        mbasis(3, 1, 1, dev),
        mbasis(3, 2, 2, dev),
        mbasis(3, 2, 1, dev),
        mbasis(3, 1, 2, dev),
        d * mbasis(3, 1, 1, dev),
        d * mbasis(3, 2, 2, dev),
        d * mbasis(3, 2, 1, dev),
        d * mbasis(3, 1, 2, dev)))
    b_slope = torch.stack((
        mbasis(3, 0, 0, dev),
        mbasis(3, 1, 1, dev),
        mbasis(3, 2, 2, dev),
        mbasis(3, 2, 1, dev),
        mbasis(3, 1, 2, dev)))
    rf = [0.046/0.035, 0.18/0.155, 1.0, 1.0] # odds ratios
    psi = -torch.log(torch.tensor(rf, device = dev))
    p_const = torch.cat([ par[[0, 1, 2, 3, 4]], psi ]).view(-1, 1, 1)
    p_slope = par[[5, 6, 7, 8, 9]].view(-1, 1, 1)
    const = torch.multiply(b_const, p_const).sum(dim = 0)
    trend = t * torch.multiply(p_slope, b_slope).sum(dim = 0)
    return extend(const + trend)

def scaleMScaltrend(par, dev):
    one = torch.ones(1, device=dev)
    a = torch.cat([one, torch.exp(par[10:])])
    return a, a

maskcM  = [[True,   False, True],
           [False,  True , True],
           [True,   True, False]]
masks0M = [True, True, False]
masksM  = (maskcM, masks0M)


maskcMS = [[True,  False, False, True],
           [False, True,  True, True],
           [False, True,  True,  True],
           [True,  True,  True,  False]]
masks0MS = [True, True, True, False]
masksMS = (maskcMS, masks0MS)

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


# Function to minimize fb
def minfb(a, b):
    return a + b - torch.sqrt(a**2 + b**2 + 1e-8)

