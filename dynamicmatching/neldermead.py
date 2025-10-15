import torch
import numpy as np

class NelderMeadOptimizer:
    def __init__(self, params, lr=1.0, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        self.param_groups = [{'params': list(params), 'lr': lr}]
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma

        self.shape_info = [(p.shape, p.numel()) for p in self.param_groups[0]['params']]
        self.dim = sum(s[1] for s in self.shape_info)

        self.simplex = []
        self.f_values = []

        x0 = self._flatten_params()
        self.simplex.append(x0)

        for i in range(self.dim):
            x = x0.clone()
            x[i] += lr
            self.simplex.append(x)

    def zero_grad(self):
        pass

    def _flatten_params(self):
        return torch.cat([p.view(-1) for p in self.param_groups[0]['params']])

    def _unflatten_params(self, x):
        offset = 0
        for p, (shape, numel) in zip(self.param_groups[0]['params'], self.shape_info):
            p.data = x[offset:offset+numel].view(shape)
            offset += numel

    def _sort_simplex(self):
        idx = np.argsort(self.f_values)
        self.simplex = [self.simplex[i] for i in idx]
        self.f_values = [self.f_values[i] for i in idx]

    def step(self, closure):
        if len(self.f_values) < len(self.simplex):
            self.f_values = []
            for x in self.simplex:
                self._unflatten_params(x)
                loss = closure()
                self.f_values.append(loss.item() if torch.is_tensor(loss) else float(loss))

        self._sort_simplex()

        centroid = torch.stack(self.simplex[:-1]).mean(dim=0)
        worst = self.simplex[-1]

        reflected = centroid + self.alpha * (centroid - worst)
        self._unflatten_params(reflected)
        f_reflected = closure()
        f_reflected = f_reflected.item() if torch.is_tensor(f_reflected) else float(f_reflected)

        if self.f_values[0] <= f_reflected < self.f_values[-2]:
            self.simplex[-1] = reflected
            self.f_values[-1] = f_reflected
        elif f_reflected < self.f_values[0]:
            expanded = centroid + self.gamma * (reflected - centroid)
            self._unflatten_params(expanded)
            f_expanded = closure()
            f_expanded = f_expanded.item() if torch.is_tensor(f_expanded) else float(f_expanded)

            if f_expanded < f_reflected:
                self.simplex[-1] = expanded
                self.f_values[-1] = f_expanded
            else:
                self.simplex[-1] = reflected
                self.f_values[-1] = f_reflected
        else:
            contracted = centroid + self.rho * (worst - centroid)
            self._unflatten_params(contracted)
            f_contracted = closure()
            f_contracted = f_contracted.item() if torch.is_tensor(f_contracted) else float(f_contracted)

            if f_contracted < self.f_values[-1]:
                self.simplex[-1] = contracted
                self.f_values[-1] = f_contracted
            else:
                best = self.simplex[0]
                for i in range(1, len(self.simplex)):
                    self.simplex[i] = best + self.sigma * (self.simplex[i] - best)
                # re-evaluate all vertices after shrink to keep ordering consistent
                self.f_values = []
                for x in self.simplex:
                    self._unflatten_params(x)
                    loss = closure()
                    self.f_values.append(loss.item() if torch.is_tensor(loss) else float(loss))

        self._sort_simplex()
        self._unflatten_params(self.simplex[0])
        return torch.tensor(self.f_values[0])

