from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from .helpers import extend


def sinkhorn_knopp(A, row_margins, col_margins, iter):
    epsilon = 1e-12
    def project(_, A):
        row_sums = A.sum(axis=1, keepdims=True)
        A = A * (row_margins[:, None] / (row_sums + epsilon))
        col_sums = A.sum(axis=0, keepdims=True)
        A = A * (col_margins[None, :] / (col_sums + epsilon))
        return A
    return jax.lax.fori_loop(0, iter, project, A)


def margin_projection_inline(mu, M, F, iterations=20, epsilon=1e-8):
    mu = jnp.maximum(mu, epsilon)
    def project(_, mu):
        row_sums = mu[:-1].sum(axis=1)
        row_scaling = M[:-1] / row_sums
        mu = mu.at[:-1].set(mu[:-1] * row_scaling.reshape(-1, 1))
        col_sums = mu[:, :-1].sum(axis=0)
        col_scaling = F[:-1] / col_sums
        mu = mu.at[:, :-1].set(mu[:, :-1] * col_scaling.reshape(1, -1))
        return mu
    return jax.lax.fori_loop(0, iterations, project, mu)


def margin_projection(mu, M, F, iterations=20, epsilon=1e-8, tol=1e-6):
    current_mu = mu # jnp.maximum(mu, epsilon)
    def project(_, current_mu):
        # Row scaling for first n-1 rows
        rows = current_mu[:-1, :]
        row_sums = rows.sum(axis=1, keepdims=True)
        # Proper broadcasting for row scaling
        row_scaling = M[:-1, None] / (row_sums + epsilon)
        scaled_rows = rows * row_scaling
        # Rebuild tensor with scaled rows and original last row
        new_mu = jnp.concatenate([scaled_rows, current_mu[-1:, :]], axis=0)
        # Column scaling for first n-1 columns
        cols = new_mu[:, :-1]
        col_sums = cols.sum(axis=0, keepdims=True)
        # Proper broadcasting for column scaling
        col_scaling = F[None, :-1] / (col_sums + epsilon)
        scaled_cols = cols * col_scaling
        # Rebuild final tensor with scaled columns
        current_mu = jnp.concatenate([scaled_cols, new_mu[:, -1:]], axis=1)
        mproj = current_mu[:-1, :].sum(axis=1)
        fproj = current_mu[:, :-1].sum(axis=0)
        row_error = jnp.sum(jnp.square(mproj - M[:-1]))
        col_error = jnp.sum(jnp.square(fproj - F[:-1]))
        #if (row_error + col_error) < tol:
        #    break
        return current_mu

    current_mu = jax.lax.fori_loop(0, iterations, project, current_mu)
    return current_mu


class Sinkhorn(eqx.Module):
    ndim: int = eqx.field(static=True)
    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    theta_dim: int = eqx.field(static=True)
    ti: slice = eqx.field(static=True)
    m0i: slice = eqx.field(static=True)
    f0i: slice = eqx.field(static=True)
    tau: Callable = eqx.field(static=True)
    sinkhorn: Callable = eqx.field(static=True)
    extend: Callable = eqx.field(static=True)
    num_iterations: int = eqx.field(static=True)
    training: bool
    layers: eqx.nn.Sequential

    def __init__(self, tau, ndim, output_dim, thetadim, key,
                 hidden_layers=[192, 96], num_iterations=10):
        self.ndim = ndim
        self.input_dim = 2 * ndim + 2 # margins + time + treatment
        self.output_dim = output_dim
        self.theta_dim = thetadim
        self.ti = slice(0, self.theta_dim)
        self.m0i = slice(self.theta_dim, self.theta_dim + ndim)
        self.f0i = slice(self.theta_dim + ndim, self.theta_dim + 2*ndim)
        self.tau = tau
        self.sinkhorn = sinkhorn_knopp
        self.extend = extend
        self.num_iterations = num_iterations
        self.training = True
        self.layers = self._build_layers(hidden_layers, key)

    def _build_layers(self, hidden_layers, key):
        keys = jax.random.split(key, len(hidden_layers) + 1)
        layers = []
        prev_dim = self.input_dim
        for hidden_dim, layer_key in zip(hidden_layers, keys[:-1]):
            layers.append(eqx.nn.Linear(prev_dim, hidden_dim, dtype=jnp.float32, key=layer_key))
            layers.append(eqx.nn.Lambda(jax.nn.relu))
            prev_dim = hidden_dim
        layers.append(eqx.nn.Linear(prev_dim, self.output_dim, dtype=jnp.float32, key=keys[-1]))
        return eqx.nn.Sequential(layers)

    @property
    def params(self):
        return eqx.filter(self.layers, eqx.is_array)

    def forward(self, margins):
        return self.apply(self.params, margins)

    def __call__(self, margins):
        return self.forward(margins)

    def apply(self, params, margins):
        pass

    def with_params(self, params):
        return eqx.tree_at(lambda xi: xi.layers, self, eqx.combine(params, self.layers))

    def train(self, mode=True):
        return eqx.tree_at(lambda xi: xi.training, self, mode)

    def eval(self):
        return self.train(False)

class SinkhornGeneric(Sinkhorn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sinkhorn = margin_projection
        self.extend = extend
        self.num_iterations = 25

    def apply(self, params, margins_td):
        zs = jnp.zeros((margins_td.shape[0], 1), dtype=margins_td.dtype)
        M = jnp.concatenate((margins_td[:, :self.ndim], zs), axis=1)
        F = jnp.concatenate((margins_td[:, self.ndim:2*self.ndim], zs), axis=1)
        ts = margins_td[:,(2*self.ndim)]
        ds = margins_td[:,(2*self.ndim+1)]
        layers = eqx.combine(params, self.layers)
        pars = jax.vmap(layers)(margins_td)
        vmapper = jax.vmap(lambda p, t, d: self.tau(p, t, d, None),
                           in_axes=(0, 0, 0))
        muc = vmapper(jnp.exp(pars[:, self.ti]), ts, ds)  # pos, t/d!
        mucm0 = jnp.exp(pars[:, self.m0i])
        muc0f = jnp.exp(pars[:, self.f0i]) # positive!
        muc = muc.at[:, :(M.shape[1]-1), -1:].set(mucm0.reshape(-1, (M.shape[1]-1), 1))
        muc = muc.at[:, -1:, :(F.shape[1]-1)].set(muc0f.reshape(-1, 1, (F.shape[1]-1)))
        V = jnp.exp(pars[:, -1])
        stk = jnp.concatenate((muc, M[:, :, None], F[:, :, None]), axis=2)
        iter = self.num_iterations if self.training else 100
        mus = jax.vmap(lambda p: self.sinkhorn(p[:, :self.ndim+1],
                                              p[:,(self.ndim+1)],
                                              p[:,(self.ndim+2)],
                                              iter))(stk)
        return mus, V


# Should now be deprecated in favor of SinkhornGeneric
class SinkhornMS(Sinkhorn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sinkhorn = margin_projection
        self.extend = extend
        self.num_iterations = 25

    def apply(self, params, margins_td):
        zs = jnp.zeros((margins_td.shape[0], 1), dtype=margins_td.dtype)
        M = jnp.concatenate((margins_td[:, 0:4], zs), axis=1)
        F = jnp.concatenate((margins_td[:, 4:8], zs), axis=1)
        ts = margins_td[:,8]
        ds = margins_td[:,9]
        layers = eqx.combine(params, self.layers)
        pars = jax.vmap(layers)(margins_td)
        vmapper = jax.vmap(lambda p, t, d: self.tau(p, t, d, None),
                           in_axes=(0, 0, 0))
        muc = vmapper(jnp.exp(pars[:, self.ti]), ts, ds)  # pos, t/d-dep!
        mucm0 = jnp.exp(pars[:, self.m0i]) # pos!
        muc0f = jnp.exp(pars[:, self.f0i]) # pos!
        muc = muc.at[:, :(M.shape[1]-1), -1:].set(mucm0.reshape(-1, (M.shape[1]-1), 1))
        muc = muc.at[:, -1:, :(F.shape[1]-1)].set(muc0f.reshape(-1, 1, (F.shape[1]-1)))
        V = jnp.exp(pars[:, -1])
        stk = jnp.concatenate((muc, M[:, :, None], F[:, :, None]), axis=2)
        iter = self.num_iterations if self.training else 100
        mus = jax.vmap(lambda p: self.sinkhorn(p[:, 0:5],
                                              p[:, 5],
                                              p[:, 6],
                                              iter))(stk)
        return mus, V


# Old proportion of singles management, deprecated
class SinkhornM3x3(Sinkhorn):
    def apply(self, params, margins):
        M, F = margins[:, 0:3], margins[:, 3:6]
        layers = eqx.combine(params, self.layers)
        pars = jax.vmap(layers)(margins)
        muc = jax.vmap(lambda p: self.tau(p, None),
                        in_axes=0)(jnp.exp(pars[:, 0:4]))  # positive!
        sqs = SquashedSigmoid(0.02, 0.98)
        shm, shf = sqs(pars[:, 4:6]), sqs(pars[:, 6:8])
        V = jnp.exp(pars[:, -1])
        ones = jnp.ones((M.shape[0], 1), dtype=pars.dtype)
        shm0 = jnp.concatenate((shm, ones), axis=1)  # proportion of couples
        shf0 = jnp.concatenate((shf, ones), axis=1)
        mucm0, muc0f = M * shm0, F * shf0  # couples
        mum0, mu0f = M * (1 - shm0), F * (1 - shf0)  # singles
        stk = jnp.concatenate((muc, mucm0.reshape(-1, M.shape[1], 1),
                              muc0f.reshape(-1, F.shape[1], 1)), axis=2)
        iter = self.num_iterations if self.training else 1000
        muc = jax.vmap(lambda p: self.sinkhorn(p[:, 0:2],
                                              p[:, 2],
                                              p[:, 3],
                                              iter), in_axes=0)(stk)
        mus = jnp.concatenate((jnp.concatenate((muc, mum0.reshape(-1, M.shape[1], 1)),
                                               axis=2),
                               jnp.concatenate((mu0f.reshape(-1, 1, F.shape[1]),
                                                jnp.zeros((F.shape[0], 1, 1),
                                                          dtype=pars.dtype)),
                                               axis=2)
                               ), axis=1)
        return mus, V


class SquashedSigmoid(eqx.Module):
    low: float
    high: float

    def __init__(self, low=0.02, high=0.98):
        self.low = low
        self.high = high

    def forward(self, x):
        return self.low + (self.high - self.low) * jax.nn.sigmoid(x)

    def __call__(self, x):
        return self.forward(x)


def masked_log(tensor, mask):
    mask = jnp.asarray(mask, dtype=bool)
    # The original custom backward was zero outside the mask, even though its
    # forward pass returned the input there. Keep both behaviours explicitly.
    safe_input = jnp.where(mask, tensor, jnp.ones_like(tensor))
    return jnp.where(mask, jnp.log(safe_input + 1e-10),
                     jax.lax.stop_gradient(tensor))


def masked_log_list(tensor, mask_list):
    return masked_log(tensor, jnp.asarray(mask_list, dtype=bool))
