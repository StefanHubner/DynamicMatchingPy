import torch
import numpy as np

class NelderMeadOptimizer:
    def __init__(self, params, lr=1.0, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        self.param_groups = [{'params': list(params), 'lr': lr}]
        self.alpha, self.gamma, self.rho, self.sigma = alpha, gamma, rho, sigma
        self.shape_info = [(p.shape, p.numel()) for p in self.param_groups[0]['params']]
        self.dim = sum(n for _, n in self.shape_info)

        x0 = self._flatten_params().clone()
        self.simplex = [x0]
        for i in range(self.dim):
            x = x0.clone()
            x[i] += lr
            self.simplex.append(x)
        self.f_values = []

    def zero_grad(self):
        pass

    def _flatten_params(self):
        return torch.cat([p.detach().reshape(-1).clone() for p in self.param_groups[0]['params']])

    def _unflatten_params_(self, x):
        off = 0
        with torch.no_grad():
            for p, (shape, n) in zip(self.param_groups[0]['params'], self.shape_info):
                p.detach().view(-1).copy_(x[off:off+n])
                off += n
        return

    def _eval_at(self, x, closure):
        self._unflatten_params_(x)
        with torch.no_grad():
            val = closure()
            if torch.is_tensor(val):
                val = float(val.detach().cpu())
            else:
                val = float(val)
        return val

    def _sort_simplex_(self):
        idx = np.argsort(self.f_values)
        self.simplex = [self.simplex[i] for i in idx]
        self.f_values = [self.f_values[i] for i in idx]

    def step(self, closure):
        if len(self.f_values) < len(self.simplex):
            self.f_values = [self._eval_at(x, closure) for x in self.simplex]
        self._sort_simplex_()

        centroid = torch.stack(self.simplex[:-1]).mean(dim=0)
        worst = self.simplex[-1]

        reflected = centroid + self.alpha * (centroid - worst)
        f_ref = self._eval_at(reflected, closure)

        if self.f_values[0] <= f_ref < self.f_values[-2]:
            self.simplex[-1], self.f_values[-1] = reflected, f_ref
        elif f_ref < self.f_values[0]:
            expanded = centroid + self.gamma * (reflected - centroid)
            f_exp = self._eval_at(expanded, closure)
            if f_exp < f_ref:
                self.simplex[-1], self.f_values[-1] = expanded, f_exp
            else:
                self.simplex[-1], self.f_values[-1] = reflected, f_ref
        else:
            contracted = centroid + self.rho * (worst - centroid)  # inside contraction
            f_con = self._eval_at(contracted, closure)
            if f_con < self.f_values[-1]:
                self.simplex[-1], self.f_values[-1] = contracted, f_con
            else:
                best = self.simplex[0]
                for i in range(1, len(self.simplex)):
                    self.simplex[i] = best + self.sigma * (self.simplex[i] - best)
                self.f_values = [self.f_values[0]] + [None]*(len(self.simplex)-1)
                # re-evaluate all vertices after shrink
                self.f_values = [self._eval_at(x, closure) for x in self.simplex]

        self._sort_simplex_()
        self._unflatten_params_(self.simplex[0])
        return torch.tensor(self.f_values[0])

