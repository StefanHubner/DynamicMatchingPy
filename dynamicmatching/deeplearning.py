import torch
import torch.nn as nn
import math
import pdb

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

class SinkhornUnmatched(nn.Module):
    def __init__(self, tau, ndim=3, output_dim=9,
                 hidden_layers=[64, 32, 16],
                 temperature=0.1, num_iterations=10):
        super(SinkhornUnmatched, self).__init__()
        self.input_dim = 2 * ndim
        self.output_dim = output_dim
        self.temperature = temperature
        self.num_iterations = num_iterations
        self.tau = tau
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def sinkhorn(self, A, row_margins, col_margins):
        epsilon = 1e-10
        for _ in range(self.num_iterations):
            row_sums = A.sum(dim=1, keepdim=True)
            A = A * (row_margins.unsqueeze(1) / (row_sums + epsilon))
            col_sums = A.sum(dim=0, keepdim=True)
            A = A * (col_margins.unsqueeze(0) / (col_sums + epsilon))
        return A

    def forward(self, margins):
        M, F = margins[:,0:3], margins[:,3:6]
        pars = self.layers(margins)
        muc = torch.vmap(lambda p: self.tau(p, pars.device),
                         in_dims = 0)(torch.exp(pars[:,0:4])) # positive!
        shm, shf = torch.sigmoid(pars[:,4:6]), torch.sigmoid(pars[:,6:8])
        V = torch.exp(pars[:,-1])
        ones = torch.ones(M.shape[0], 1, device = pars.device)
        shm0 = torch.cat((shm, ones), dim=1) # proportion of couples
        shf0 = torch.cat((shf, ones), dim=1)
        mucm0, muc0f = M * shm0, F * shf0           # couples
        mum0, mu0f = M * (1 - shm0), F * (1 - shf0) # singles
        stk = torch.cat((muc,
                         mucm0.view(-1, M.shape[1], 1),
                         muc0f.view(-1, F.shape[1], 1)), dim = 2)
        muc = torch.vmap(lambda p: self.sinkhorn(p[:,0:3], p[:,3], p[:,4]),
                         in_dims = 0)(stk)
        mus = torch.cat((torch.cat((muc, mum0.view(-1, M.shape[1], 1)),
                                   dim = 2),
                         torch.cat((mu0f.view(-1, 1, F.shape[1]),
                                    torch.zeros(F.shape[0], 1, 1,
                                                device = pars.device)),
                                   dim = 2)),
                        dim=1)
        return (mus, V)


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

def masked_log(tensor, mask_list):
    mask = torch.tensor(mask_list, dtype=torch.bool, device=tensor.device)
    if mask.shape != tensor.shape:
        mask = mask.unsqueeze(0).expand_as(tensor)
    return MaskedLog.apply(tensor, mask)

