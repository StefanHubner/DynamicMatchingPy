import torch
import numpy as np

class NelderMeadOptimizer:
    def __init__(self, params, lr=0.5, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        self.param_groups = [{'params': list(params), 'lr': lr}]
        self.alpha = alpha  # reflection
        self.gamma = gamma  # expansion
        self.rho = rho      # contraction
        self.sigma = sigma  # shrink

        # Flatten parameters
        self.shape_info = [(p.shape, p.numel()) for p in self.param_groups[0]['params']]
        self.dim = sum(s[1] for s in self.shape_info)

        # Initialize simplex
        self.simplex = []
        self.f_values = []

        # Initial point
        x0 = self._flatten_params()
        self.simplex.append(x0)

        # Create simplex vertices
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

    def step(self, closure):
        # Evaluate all vertices if needed
        if len(self.f_values) < len(self.simplex):
            self.f_values = []
            for x in self.simplex:
                self._unflatten_params(x)
                loss = closure()
                self.f_values.append(loss.item() if torch.is_tensor(loss) else loss)

        # Sort vertices
        indices = np.argsort(self.f_values)
        self.simplex = [self.simplex[i] for i in indices]
        self.f_values = [self.f_values[i] for i in indices]

        # Calculate centroid (excluding worst point)
        centroid = torch.stack(self.simplex[:-1]).mean(dim=0)

        # Reflection
        worst = self.simplex[-1]
        reflected = centroid + self.alpha * (centroid - worst)
        self._unflatten_params(reflected)
        f_reflected = closure().item()

        if self.f_values[0] <= f_reflected < self.f_values[-2]:
            self.simplex[-1] = reflected
            self.f_values[-1] = f_reflected
        elif f_reflected < self.f_values[0]:
            # Expansion
            expanded = centroid + self.gamma * (reflected - centroid)
            self._unflatten_params(expanded)
            f_expanded = closure().item()

            if f_expanded < f_reflected:
                self.simplex[-1] = expanded
                self.f_values[-1] = f_expanded
            else:
                self.simplex[-1] = reflected
                self.f_values[-1] = f_reflected
        else:
            # Contraction
            contracted = centroid + self.rho * (worst - centroid)
            self._unflatten_params(contracted)
            f_contracted = closure().item()

            if f_contracted < self.f_values[-1]:
                self.simplex[-1] = contracted
                self.f_values[-1] = f_contracted
            else:
                # Shrink
                for i in range(1, len(self.simplex)):
                    self.simplex[i] = self.simplex[0] + self.sigma * (self.simplex[i] - self.simplex[0])
                self.f_values = [self.f_values[0]]  # Force re-evaluation

        # Set parameters to best vertex
        self._unflatten_params(self.simplex[0])

        return torch.tensor(self.f_values[0])
