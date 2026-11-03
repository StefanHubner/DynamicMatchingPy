import torch
import torch.nn as nn

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
    def __init__(self, tau, ndim, output_dim, thetadim,
                 hidden_layers=[128, 64], num_iterations=10):
        super(Sinkhorn, self).__init__()
        self.ndim = ndim
        self.input_dim = 2 * ndim + 2 # margins + time + treatment
        self.output_dim = output_dim
        self.theta_dim = thetadim
        self.ti = range(self.theta_dim)
        self.m0i = range(self.theta_dim, self.theta_dim + ndim)
        self.f0i = range(self.theta_dim + ndim, self.theta_dim + 2*ndim)
        # 2026-11-03: Append signed male/female dual values and the fixed-flow continuation term.
        self.vi = self.theta_dim + 2*ndim
        self.wmi = range(self.vi + 1, self.vi + 1 + ndim)
        self.wfi = range(self.vi + 1 + ndim, self.vi + 1 + 2*ndim)
        self.tau = tau
        self.sinkhorn = margin_projection
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

class SinkhornGeneric(Sinkhorn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_iterations = 25

    def forward(self, margins_td):
        zs = torch.zeros((margins_td.shape[0], 1),
                         device = margins_td.device)
        M = torch.cat((margins_td[:, range(self.ndim)], zs), dim=1)
        F = torch.cat((margins_td[:, range(self.ndim, self.ndim*2)], zs), dim=1)
        ts = margins_td[:,(2*self.ndim)]
        ds = margins_td[:,(2*self.ndim+1)]
        pars = self.layers(margins_td)
        vmapper = torch.vmap(lambda p, t, d: self.tau(p, t, d, pars.device),
                             in_dims=(0, 0, 0))
        muc = vmapper(torch.exp(pars[:, self.ti]), ts, ds)  # pos, t/d!
        mucm0 = torch.exp(pars[:, self.m0i])
        muc0f = torch.exp(pars[:, self.f0i]) # positive!
        muc[:, :(M.shape[1]-1), -1:] = mucm0.view(-1, (M.shape[1]-1), 1)
        muc[:, -1:, :(F.shape[1]-1)] = muc0f.view(-1, 1, (F.shape[1]-1))
        V = torch.exp(pars[:, self.vi])
        wm, wf, J = pars[:, self.wmi], pars[:, self.wfi], pars[:, -1]
        stk = torch.cat((muc, M.unsqueeze(2), F.unsqueeze(2)), dim=2)
        iter = self.num_iterations if self.training else 100
        mus = torch.vmap(lambda p: self.sinkhorn(p[:,range(self.ndim+1)],
                                                 p[:,(self.ndim+1)],
                                                 p[:,(self.ndim+2)],
                                                 iter))(stk)
        return mus, V, wm, wf, J

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
