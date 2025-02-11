import torch
import torch.nn as nn
import math

def sinkhorn_knopp(A, row_margins, col_margins, iter):
    epsilon = 1e-12
    for _ in range(iter):
        row_sums = A.sum(dim=1, keepdim=True)
        A = A * (row_margins.unsqueeze(1) / (row_sums + epsilon))
        col_sums = A.sum(dim=0, keepdim=True)
        A = A * (col_margins.unsqueeze(0) / (col_sums + epsilon))
    return A

class Sinkhorn(nn.Module):
    def __init__(self, tau, ndim, output_dim,
                 hidden_layers=[64, 32, 16], num_iterations=10):
        super(Sinkhorn, self).__init__()
        self.input_dim = 2 * ndim
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
        muc = torch.vmap(lambda p: self.sinkhorn(p[:, 0:3],
                                                 p[:, 3],
                                                 p[:, 4],
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


def masked_log(tensor, mask_list):
    mask = torch.tensor(mask_list, dtype=torch.bool, device=tensor.device)
    if mask.shape != tensor.shape:
        mask = mask.unsqueeze(0).expand_as(tensor)
    return MaskedLog.apply(tensor, mask)

