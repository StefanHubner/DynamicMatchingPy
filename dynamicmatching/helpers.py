import torch
import numpy as np
from enum import Enum

class CF(Enum):
    None_ = 0          # No Coutnerfactural
    MatchingOnly = 1   # Only matching channel
    HouseholdOnly = 2  # Only household channel

class TermColours:
    RED = "\033[31m"
    RESET = "\033[0m"
    GREEN = "\033[32m"

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

# tau for married new prototype (2x2) 
def tauMproto(par, dev):
    b = torch.stack((
        mbasis(2, 0, 0, dev),
        mbasis(2, 1, 1, dev)))
    return torch.multiply(par.view(-1, 1, 1), b).sum(dim=0)

# takes 4 parameters
def tauMflex(par, dev):
    b = torch.stack((
        mbasis(3, 0, 0, dev),
        mbasis(3, 1, 1, dev),
        mbasis(3, 0, 1, dev) + mbasis(3, 1, 0, dev),
        mbasis(3, 2, 2, dev))) # by convention the last one is (c, c)
    return torch.multiply(par.view(-1, 1, 1), b).sum(dim=0)

# takes 8 parameters
def tauKMsimple(par, dev):
    b = torch.stack((
        mbasis(6, 0, 0, dev), # znzn
        mbasis(6, 3, 3, dev), # knkn
        mbasis(6, 1, 1, dev), # zeze
        mbasis(6, 4, 4, dev), # keke
        mbasis(6, 0, 1, dev) + mbasis(6, 1, 0, dev), # znze,zezn
        mbasis(6, 3, 4, dev) + mbasis(6, 4, 3, dev), # knke,kekn
        mbasis(6, 2, 2, dev), # zczc
        mbasis(6, 5, 5, dev))) # kckc
    return torch.multiply(par.view(-1, 1, 1), b).sum(dim=0)



maskcM = [[True,  True,  False, False],
          [True,  True,  False, False],
          [False, False, True,  False],
          [False, False, False, False]]
maskcKM = np.kron(np.eye(2), np.matrix(maskcM)[:-1,:-1])
maskcKM = np.vstack((maskcKM, np.zeros((1, 6))))
maskcKM = np.hstack((maskcKM, np.zeros((7, 1))))
maskcKM = (maskcKM==1).tolist()

mask0M = [True, True, False, False]
mask0KM = [True, True, False, True, True, False, False]

masksKM = (maskcKM, mask0KM)
masksM = (maskcM, mask0M)

maskcMp = [[True,   False, False],
           [False,  True , False],
           [False,  False, False]]
masks0Mp = [True, True, False]
masksMproto = (maskcMp, masks0Mp)



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

