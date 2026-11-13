import jax.numpy as jnp
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
    return jnp.concatenate((jnp.concatenate((phi, jnp.zeros((phi.shape[0], 1),
                                                           dtype=phi.dtype)
                                           ), axis=1),
                            jnp.zeros((1, phi.shape[1] + 1),
                                      dtype=phi.dtype)), axis=0)

# vector basis function
def vbasis(n, i, dev):
    b = jnp.zeros(n, dtype=jnp.float32)
    b = b.at[i].set(1)
    return b

# matrix basis function
def mbasis(n, i, j, dev):
    b = jnp.zeros((n, n), dtype=jnp.float32)
    b = b.at[i,j].set(1)
    return b

def surplus(n, cells):
    def tau(levels, psi, d):
        b = jnp.stack([mbasis(n, i, j, None) for i, j in cells])
        p = levels + jnp.concatenate((jnp.zeros_like(levels[:1]), jnp.squeeze(d) * psi))
        return extend((b * p[:, None, None]).sum(axis=0))
    return tau

tauM = surplus(2, ((0, 0), (1, 1)))
tauMS = surplus(3, ((0, 0), (1, 1), (2, 2), (2, 1), (1, 2)))

def scales(log_scales):
    one = jnp.ones(1, dtype=log_scales.dtype)
    a = jnp.concatenate((one, jnp.exp(log_scales)))
    return a, a

def model(tau, unpack):
    def evaluate(theta, t, d):
        levels, psi, log_scales = unpack(theta, jnp.squeeze(t))
        return tau(levels, psi, d), scales(log_scales)
    return evaluate

def callbacks(spec):
    return (lambda p, t, d, dev: spec(p, t, d)[0],
            lambda p, dev: spec(p, 0, 0)[1])

def calibration(rf):
    return lambda: -jnp.log(jnp.asarray(rf, dtype=jnp.float32))

psiM = calibration((1.1,)) # theoretical hazard ratio at cutoff
psiMS = calibration((0.046/0.035, 0.18/0.155, 1.0, 1.0))

M = model(tauM, lambda p, t: (p[:2], p[3:4], p[2:3]))
Mcal = model(tauM, lambda p, t: (p[:2], psiM(), p[2:3]))
MS = model(tauMS, lambda p, t: (p[:5], p[7:11], p[5:7]))
MScal = model(tauMS, lambda p, t: (p[:5], psiMS(), p[5:7]))
MStrend = model(tauMS, lambda p, t: (p[:5] + t*p[5:10], p[12:16], p[10:12]))
MScaltrend = model(tauMS, lambda p, t: (p[:5] + t*p[5:10], psiMS(), p[10:12]))

def tauMStri(par, t, d, dev):
    b_const = jnp.stack((
        mbasis(3, 0, 0, dev),
        mbasis(3, 1, 1, dev),
        mbasis(3, 2, 2, dev),
        mbasis(3, 2, 1, dev),
        d * mbasis(3, 0, 0, dev),
        d * mbasis(3, 1, 1, dev),
        d * mbasis(3, 2, 2, dev),
        d * mbasis(3, 2, 1, dev)))
    p_const = par[:8].reshape(-1, 1, 1)
    const = jnp.multiply(b_const, p_const).sum(axis = 0)
    return extend(const)

def tauKMS(par, t, d, dev):
    b_const = jnp.stack((
        mbasis(8, 0, 0, dev),
        mbasis(8, 1, 1, dev),
        mbasis(8, 2, 2, dev),
        mbasis(8, 3, 3, dev),
        mbasis(8, 4, 4, dev),
        mbasis(8, 5, 5, dev),
        mbasis(8, 5, 4, dev),
        mbasis(8, 6, 6, dev),
        mbasis(8, 7, 7, dev),
        mbasis(8, 7, 6, dev),
        d * mbasis(8, 7, 7, dev),
        d * mbasis(8, 7, 6, dev)))
    idcs = jnp.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    p_const = par[idcs].reshape(-1, 1, 1)
    const = jnp.multiply(b_const, p_const).sum(axis = 0)
    return extend(const)


# takes 4 parameters (old 3x3)
def tauMflex(par, dev):
    b = jnp.stack((
        mbasis(3, 0, 0, dev),
        mbasis(3, 1, 1, dev),
        mbasis(3, 0, 1, dev) + mbasis(3, 1, 0, dev),
        mbasis(3, 2, 2, dev))) # by convention the last one is (c, c)
    return extend(jnp.multiply(par.reshape(-1, 1, 1), b).sum(axis=0))

# takes 8 parameters (old 3x3)
def tauKMsimple(par, dev):
    b = jnp.stack((
        mbasis(6, 0, 0, dev), # znzn
        mbasis(6, 3, 3, dev), # knkn
        mbasis(6, 1, 1, dev), # zeze
        mbasis(6, 4, 4, dev), # keke
        mbasis(6, 0, 1, dev) + mbasis(6, 1, 0, dev), # znze,zezn
        mbasis(6, 3, 4, dev) + mbasis(6, 4, 3, dev), # knke,kekn
        mbasis(6, 2, 2, dev), # zczc
        mbasis(6, 5, 5, dev))) # kckc
    return extend(jnp.multiply(par.reshape(-1, 1, 1), b).sum(axis=0))



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

maskcKMS = \
[[True,  False, False, False, False, False, False, False, True],
 [False, True,  False, False, False, False, False, False, True],
 [False, False, True,  False, False, False, False, False, True],
 [False, False, False,  True, False, False, False, False, True],
 [False, False, False, False, True,  False, False, False, True],
 [False, False, False, False, True,  True,  False, False, True],
 [False, False, False, False, False, False, True,  False, True],
 [False, False, False, False, False, False, True,  True,  True],
 [True,  True,  True,  True,  True,  True,  True,  True,  False]]
masks0KMS = \
 [True,  True,  True,  True,  True,  True,  True,  True,  False]
masksKMS = (maskcKMS, masks0KMS)

# Function to minimize fb
def minfb(a, b):
    return a + b - jnp.sqrt(a**2 + b**2 + 1e-8)


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
    return a + b - jnp.sqrt(a**2 + b**2 + 1e-8)

