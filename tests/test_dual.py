import unittest
from unittest.mock import patch

import torch

from dynamicmatching.bellman import choices, dual_residuals, matching_entropy, residuals
from dynamicmatching.deeplearning import SinkhornGeneric
from dynamicmatching.helpers import tauMcal, scaleMcal


class DynamicDualTests(unittest.TestCase):
    def setUp(self):
        # 2026-11-03: Check affine-flow dual accounting against an analytic feasible allocation.
        tensor = lambda values: torch.tensor(values, dtype=torch.float64)
        self.mask = torch.tensor([[True, False, True],
                                  [False, True, True],
                                  [True, True, False]])
        self.mu = tensor([[[.08, 0, .13], [0, .12, .10], [.17, .10, 0]],
                          [[.11, 0, .08], [0, .07, .17], [.15, .14, 0]]])
        total = (self.mu[:, :-1, :].sum((1, 2))
                 + self.mu[:, :, :-1].sum((1, 2)))
        self.mu = self.mu / total[:, None, None]
        self.s = torch.cat([self.mu[:, :-1, :].sum(2),
                            self.mu[:, :, :-1].sum(1)], dim=1)
        p0 = tensor([[[.8, .2], [0, 0], [.6, .4]],
                     [[0, 0], [.25, .75], [.1, .9]]])
        p1 = tensor([[[.7, .3], [0, 0], [.45, .55]],
                     [[0, 0], [.4, .6], [.3, .7]]])
        q0 = tensor([[[.55, .45], [0, 0]],
                     [[0, 0], [.35, .65]], [[.9, .1], [.2, .8]]])
        q1 = tensor([[[.65, .35], [0, 0]],
                     [[0, 0], [.15, .85]], [[.75, .25], [.4, .6]]])
        self.transitions = (None, None, p0, q0, p1, q1)
        self.d = tensor([[0], [1]])
        self.flow = tensor([.012, -.018, .005, -.009])
        self.netflow = lambda s: self.flow.to(s)
        self.beta = .87
        self.scales = (tensor([.7, 1.3]), tensor([1.1, .6]))
        wmnext = tensor([[.4, .9], [.2, .75]])
        wfnext = tensor([[.8, .3], [.65, .1]])
        Jnext = tensor([.13, -.08])
        self.next_values = (wmnext, wfnext, Jnext)
        c = 1 + self.flow.sum()
        self.phi = torch.zeros_like(self.mu)
        wm, wf = torch.zeros_like(wmnext), torch.zeros_like(wfnext)
        snext = torch.zeros_like(self.s)
        rewards = torch.zeros(2, dtype=torch.float64)
        for n, (p, q) in enumerate([(p0, q0), (p1, q1)]):
            pm = self.mu[n, :-1, :] / self.s[n, :2, None]
            pf = self.mu[n, :, :-1] / self.s[n, None, 2:]
            for i in range(2):
                wm[n, i] = (self.beta * p[i, -1].dot(wmnext[n]) / c
                            - self.scales[0][i] * pm[i, -1].log())
                wf[n, i] = (self.beta * q[-1, i].dot(wfnext[n]) / c
                            - self.scales[1][i] * pf[-1, i].log())
                self.phi[n, i, i] = (
                    wm[n, i] + wf[n, i]
                    + self.scales[0][i] * pm[i, i].log()
                    + self.scales[1][i] * pf[i, i].log()
                    - self.beta * (p[i, i].dot(wmnext[n])
                                   + q[i, i].dot(wfnext[n])) / c)
            snext[n, :2] = (self.mu[n, :-1, :, None] * p).sum((0, 1))
            snext[n, 2:] = (self.mu[n, :, :-1, None] * q).sum((0, 1))
            rewards[n] = (self.mu[n] * self.phi[n]).sum()
            for i in range(3):
                for j in range(3):
                    if not self.mask[i, j]:
                        continue
                    if i < 2:
                        rewards[n] -= self.mu[n, i, j] * self.scales[0][i] * pm[i, j].log()
                    if j < 2:
                        rewards[n] -= self.mu[n, i, j] * self.scales[1][j] * pf[i, j].log()
        snext = (snext + self.flow) / c
        self.assertTrue((snext > 0).all())
        wnext = torch.cat([wmnext, wfnext], dim=1)
        J = self.beta * ((self.flow / c * wnext).sum(1) + Jnext)
        vnext = (snext * wnext).sum(1) + Jnext
        self.values = (rewards + self.beta * vnext, wm, wf, J)

    def components(self, values=None, next_values=None, phi=None, d=None):
        return dual_residuals(
            self.mu, self.s, self.d if d is None else d,
            self.phi if phi is None else phi, self.scales,
            self.values if values is None else values,
            self.next_values if next_values is None else next_values,
            self.transitions, self.netflow, self.beta, self.mask)

    def loss(self, values=None, next_values=None, phi=None, d=None):
        dual, gap = self.components(values, next_values, phi, d)
        return dual.square().mean() + gap.square().mean()

    def test_affine_accounting_and_dynamic_budgets(self):
        self.assertLess(self.loss().item(), 1e-16)
        self.assertGreater(self.loss(d=1 - self.d).item(), 1e-4)

    def test_forbidden_cells_do_not_enter_the_dual(self):
        phi = self.phi.clone()
        phi[:, ~self.mask] = 1000
        torch.testing.assert_close(self.loss(phi=phi), self.loss())

    def test_every_value_head_is_constrained_and_differentiable(self):
        heads = self.values + self.next_values
        for index in range(len(heads)):
            changed = list(heads)
            changed[index] = changed[index] + .2
            self.assertGreater(self.loss(changed[:4], changed[4:]).item(), 1e-4)
        changed = [(head + .07 * (i + 1)).detach().requires_grad_()
                   for i, head in enumerate(heads)]
        gradients = torch.autograd.grad(self.loss(changed[:4], changed[4:]), changed)
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().max().item(), 1e-8)

    def test_dual_equations_and_value_gap_are_separate(self):
        dual, gap = self.components()
        values = (self.values[0] + .2,) + self.values[1:]
        changed_dual, changed_gap = self.components(values=values)
        torch.testing.assert_close(changed_dual, dual)
        torch.testing.assert_close(changed_gap, gap - .2)

        phi = self.phi.clone()
        phi[:, 0, 0] += .2
        changed_dual, changed_gap = self.components(phi=phi)
        self.assertGreater((changed_dual - dual).abs().max().item(), .1)
        torch.testing.assert_close(changed_gap, gap)

    def test_disabling_dual_skips_both_new_terms(self):
        with torch.random.fork_rng():
            torch.manual_seed(11)
            xi = SinkhornGeneric(tauMcal, 2, 13, 3, hidden_layers=[8])
            theta = torch.tensor([.1, .1, .1], requires_grad=True)
            transitions = tuple(p.float() if p is not None else p for p in self.transitions)
            with patch('dynamicmatching.bellman.dual_residuals',
                       side_effect=AssertionError("Disabled dual block was evaluated")) as dual:
                loss = residuals(4, xi, transitions, lambda s: s.new_zeros(4),
                                 self.beta, theta, tauMcal, scaleMcal,
                                 (self.mask, self.mask[-1]), torch.linspace(0, 1, 3),
                                 'cpu', use_dual=False)
            dual.assert_not_called()
            self.assertTrue(torch.isfinite(loss))
            gradients = torch.autograd.grad(loss, (theta,) + tuple(xi.parameters()))
            for gradient in gradients:
                self.assertTrue(torch.isfinite(gradient).all())

    def test_projected_primal_foc_uses_the_same_continuation(self):
        mu = self.mu.detach().requires_grad_()
        _, _, p0, q0, p1, q1 = self.transitions
        state = choices(mu, torch.zeros_like(self.d), self.d,
                        p0, q0, p1, q1, self.netflow, 1, mu.device)
        wmnext, wfnext, Jnext = self.next_values
        continuation = (state[:, :4] * torch.cat([wmnext, wfnext], dim=1)).sum(1) + Jnext
        flow = ((mu * self.phi).sum((1, 2))
                + matching_entropy(mu, self.s[:, :2], self.s[:, 2:], self.scales, self.mask))
        gradient = torch.autograd.grad((flow + self.beta * continuation).sum(), mu)[0]
        foc = gradient[:, :-1, :-1] - gradient[:, :-1, -1:] - gradient[:, -1:, :-1]
        self.assertLess(foc[:, self.mask[:-1, :-1]].abs().max().item(), 1e-8)


if __name__ == '__main__':
    unittest.main()
