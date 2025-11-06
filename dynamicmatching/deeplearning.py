import torch
import torch.nn as nn
import math
from .helpers import extend

def sinkhorn_knopp(A, row_margins, col_margins, iter):
    epsilon = 1e-12
    for _ in range(iter):
        row_sums = A.sum(dim=1, keepdim=True)
        A = A * (row_margins.unsqueeze(1) / (row_sums + epsilon))
        col_sums = A.sum(dim=0, keepdim=True)
        A = A * (col_margins.unsqueeze(0) / (col_sums + epsilon))
    return A

def margin_projection_inline(mu, M, F, iterations=20, epsilon=1e-8):
    mu = torch.clamp(mu, min=epsilon)
    for _ in range(iterations):
        row_sums = mu[:-1].sum(dim=1)
        row_scaling = M[:-1] / row_sums
        mu[:-1] = mu[:-1] * row_scaling.view(-1, 1)
        col_sums = mu[:, :-1].sum(dim=0)
        col_scaling = F[:-1] / col_sums
        mu[:, :-1] = mu[:, :-1] * col_scaling.view(1, -1)
    return mu

def margin_projection(mu, M, F, iterations=20, epsilon=1e-8, tol=1e-6):
    current_mu = mu # torch.clamp(mu, min=epsilon)
    for _ in range(iterations):
        # Row scaling for first n-1 rows
        rows = current_mu[:-1, :]
        row_sums = rows.sum(dim=1, keepdim=True)
        # Proper broadcasting for row scaling
        row_scaling = M[:-1].unsqueeze(1) / (row_sums + epsilon)
        scaled_rows = rows * row_scaling
        # Rebuild tensor with scaled rows and original last row
        new_mu = torch.cat([scaled_rows, current_mu[-1:, :]], dim=0)
        # Column scaling for first n-1 columns
        cols = new_mu[:, :-1]
        col_sums = cols.sum(dim=0, keepdim=True)
        # Proper broadcasting for column scaling
        col_scaling = F[:-1].unsqueeze(0) / (col_sums + epsilon)
        scaled_cols = cols * col_scaling
        # Rebuild final tensor with scaled columns
        current_mu = torch.cat([scaled_cols, new_mu[:, -1:]], dim=1)
        mproj = current_mu[:-1, :].sum(dim=1)
        fproj = current_mu[:, :-1].sum(dim=0)
        row_error = torch.sum(torch.square(mproj - M[:-1]))
        col_error = torch.sum(torch.square(fproj - F[:-1]))
        #if (row_error + col_error) < tol:
        #    break
    return current_mu

class Sinkhorn(nn.Module):
    def __init__(self, tau, ndim, output_dim,
                 hidden_layers=[32, 16], num_iterations=10):
        super(Sinkhorn, self).__init__()
        self.input_dim = 2 * ndim + 2 # margins + time + treatment
        self.output_dim = output_dim
        self.tau = tau
        self.sinkhorn = sinkhorn_knopp
        self.num_iterations = num_iterations
        self.layers = self._build_layers(hidden_layers)
    def _build_layers(self, hidden_layers):
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, self.output_dim))
        return nn.Sequential(*layers)
    def forward(self, margins):
        raise NotImplementedError("Subclasses must implement forward method")
    def train(self, mode=True):
        super(Sinkhorn, self).train(mode)
        print(f"Training mode: training={self.training}")
        return self
    def eval(self):
        super(Sinkhorn, self).eval()
        print(f"Evaluation mode: training={self.training}")
        return self

class SinkhornMproto(Sinkhorn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sinkhorn = margin_projection
        self.extend = extend
        self.num_iterations = 25

    def forward(self, margins_td):
        zs = torch.zeros((margins_td.shape[0], 1),
                         device = margins_td.device)
        M = torch.cat((margins_td[:, 0:2], zs), dim=1)
        F = torch.cat((margins_td[:, 2:4], zs), dim=1)
        ts = margins_td[:,4]
        ds = margins_td[:,5]
        pars = self.layers(margins_td)
        vmapper = torch.vmap(lambda p, t, d: self.tau(p, t, d, pars.device),
                             in_dims=(0, 0, 0))
        muc = vmapper(torch.exp(pars[:, 0:5]), ts, ds)  # positive, t/d-dep!
        mucm0, muc0f = torch.exp(pars[:, 5:7]), torch.exp(pars[:, 7:9]) # positive!
        muc[:, :(M.shape[1]-1), -1:] = mucm0.view(-1, (M.shape[1]-1), 1)
        muc[:, -1:, :(F.shape[1]-1)] = muc0f.view(-1, 1, (F.shape[1]-1))
        V = torch.exp(pars[:, -1])
        stk = torch.cat((muc, M.unsqueeze(2), F.unsqueeze(2)), dim=2)
        iter = self.num_iterations if self.training else 100
        mus = torch.vmap(lambda p: self.sinkhorn(p[:, 0:3],
                                                 p[:, 3],
                                                 p[:, 4],
                                                 iter))(stk)
        return mus, V

class SinkhornMS(Sinkhorn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sinkhorn = margin_projection
        self.extend = extend
        self.num_iterations = 25

    def forward(self, margins_td):
        zs = torch.zeros((margins_td.shape[0], 1),
                         device = margins_td.device)
        M = torch.cat((margins_td[:, 0:4], zs), dim=1)
        F = torch.cat((margins_td[:, 4:8], zs), dim=1)
        ts = margins_td[:,8]
        ds = margins_td[:,9]
        pars = self.layers(margins_td)
        vmapper = torch.vmap(lambda p, t, d: self.tau(p, t, d, pars.device),
                             in_dims=(0, 0, 0))
        muc = vmapper(torch.exp(pars[:, 0:8]), ts, ds)  # pos, t/d-dep!
        mucm0, muc0f = torch.exp(pars[:, 8:12]), torch.exp(pars[:, 12:16]) # pos!
        muc[:, :(M.shape[1]-1), -1:] = mucm0.view(-1, (M.shape[1]-1), 1)
        muc[:, -1:, :(F.shape[1]-1)] = muc0f.view(-1, 1, (F.shape[1]-1))
        V = torch.exp(pars[:, -1])
        stk = torch.cat((muc, M.unsqueeze(2), F.unsqueeze(2)), dim=2)
        iter = self.num_iterations if self.training else 100
        mus = torch.vmap(lambda p: self.sinkhorn(p[:, 0:5],
                                                 p[:, 5],
                                                 p[:, 6],
                                                 iter))(stk)
        return mus, V


class SinkhornM(Sinkhorn):
    def forward(self, margins):
        M, F = margins[:, 0:3], margins[:, 3:6]
        pars = self.layers(margins)
        muc = torch.vmap(lambda p: self.tau(p, pars.device),
                         in_dims=0)(torch.exp(pars[:, 0:4]))  # positive!
        sqs = SquashedSigmoid(0.02, 0.98)
        shm, shf = sqs(pars[:, 4:6]), sqs(pars[:, 6:8])
        V = torch.exp(pars[:, -1])
        ones = torch.ones(M.shape[0], 1, device=pars.device)
        shm0 = torch.cat((shm, ones), dim=1)  # proportion of couples
        shf0 = torch.cat((shf, ones), dim=1)
        mucm0, muc0f = M * shm0, F * shf0  # couples
        mum0, mu0f = M * (1 - shm0), F * (1 - shf0)  # singles
        stk = torch.cat((muc, mucm0.view(-1, M.shape[1], 1),
                         muc0f.view(-1, F.shape[1], 1)), dim=2)
        iter = self.num_iterations if self.training else 1000
        muc = torch.vmap(lambda p: self.sinkhorn(p[:, 0:2],
                                                 p[:, 2],
                                                 p[:, 3],
                                                 iter), in_dims=0)(stk)
        mus = torch.cat((torch.cat((muc, mum0.view(-1, M.shape[1], 1)),
                                   dim=2),
                         torch.cat((mu0f.view(-1, 1, F.shape[1]),
                                    torch.zeros(F.shape[0], 1, 1,
                                                device=pars.device)),
                                   dim=2)
                         ), dim=1)
        return mus, V

class SinkhornKMsimple(Sinkhorn):
    def forward(self, margins):
        M, F = margins[:, 0:6], margins[:, 6:12]
        pars = self.layers(margins)
        muc = torch.vmap(lambda p: self.tau(p, pars.device),
                         in_dims=0)(torch.exp(pars[:, 0:8]))  # positive!
        sqs = SquashedSigmoid(0.02, 0.98)
        shm_f, shm_k = sqs(pars[:, 9:11]), sqs(pars[:, 11:13])
        shf_f, shf_k = sqs(pars[:, 13:15]), sqs(pars[:, 15:17])
        V = torch.exp(pars[:, -1])
        ones = torch.ones(M.shape[0], 1, device=pars.device)
        shm0 = torch.cat((torch.cat((shm_f, ones), dim=1),
                          torch.cat((shm_k, ones), dim=1)), dim=1)
        shf0 = torch.cat((torch.cat((shf_f, ones), dim=1),
                          torch.cat((shf_k, ones), dim=1)), dim=1)
        mucm0, muc0f = M * shm0, F * shf0  # couples
        mum0, mu0f = M * (1 - shm0), F * (1 - shf0)  # singles
        stk = torch.cat((muc, mucm0.view(-1, M.shape[1], 1),
                         muc0f.view(-1, F.shape[1], 1)), dim=2)
        iter = self.num_iterations if self.training else 1000
        muc = torch.vmap(lambda p: self.sinkhorn(p[:, 0:6],
                                                 p[:, 6],
                                                 p[:, 7],
                                                 iter), in_dims=0)(stk)
        mus = torch.cat((torch.cat((muc, mum0.view(-1, M.shape[1], 1)),
                                   dim=2),
                         torch.cat((mu0f.view(-1, 1, F.shape[1]),
                                    torch.zeros(F.shape[0], 1, 1,
                                                device=pars.device)), dim=2)
                         ), dim=1)
        return mus, V

class SquashedSigmoid(nn.Module):
    def __init__(self, low=0.02, high=0.98):
        super().__init__()
        self.low = low
        self.high = high
    def forward(self, x):
        return self.low + (self.high - self.low) * torch.sigmoid(x)

class MaskedLog(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask):
        ctx.save_for_backward(input, mask)
        epsilon = 1e-10
        result = torch.where(mask, torch.log(input + epsilon), input)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        input, mask = ctx.saved_tensors
        grad_input = torch.where(mask, grad_output / (input + 1e-10),
                                 torch.zeros_like(input))
        return grad_input, None

    #@staticmethod
    #def setup_context(ctx, inputs, output):
    #    input, mask = inputs
    #    ctx.save_for_backward(input, mask)


def masked_log_list(tensor, mask_list):
    mask = torch.tensor(mask_list, dtype=torch.bool, device=tensor.device)
    if mask.shape != tensor.shape:
        mask = mask.unsqueeze(0).expand_as(tensor)
    return MaskedLog.apply(tensor, mask)

def masked_log(tensor, mask):
    return MaskedLog.apply(tensor, mask)
