from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

import torch

from dynamicmatching.bellman import choices, matching_entropy, residuals
from dynamicmatching.deeplearning import SinkhornGeneric
from dynamicmatching.helpers import (
    masksM, masksMS, tauMcal, tauMScal, tauMScaltrend,
    scaleMcal, scaleMScal, scaleMScaltrend,
)


def historical_residuals(ng0, xi, transitions, netflow, beta, theta,
                         tau, scale, masks, ts, dev):
    """Frozen numerical criterion from b96c01a, before the dual extension."""
    maskc, _ = masks
    _, _, p0, q0, p1, q1 = transitions
    ndim = p0.shape[0]
    distribution = torch.distributions.Dirichlet(torch.tensor([2.0] * ndim, device=dev))
    pm = .5 + (torch.rand((ng0, 1), device=dev) - .5) / 5
    s0 = torch.cat([distribution.sample((ng0,)) * pm,
                    distribution.sample((ng0,)) * (1 - pm)], dim=1)
    s = s0[torch.all(s0 > .001, dim=1)]
    ng = s.shape[0]
    t = ts[torch.randint(0, ts.numel(), (ng, 1))]
    d = torch.randint(0, 2, (ng, 1), device=dev)
    mu, value = xi(torch.cat([s, t, d], dim=1))[:2]
    next_state = choices(mu, t, d, p0, q0, p1, q1, netflow, ts[1] - ts[0], dev)
    next_value = xi(next_state)[1]
    phi = torch.vmap(lambda t, d: tau(theta, t, d, dev))(t, d)
    flow = ((mu * phi).sum((1, 2))
            + matching_entropy(mu, s[:, :ndim], s[:, ndim:], scale(theta, dev), maskc))
    gradient = torch.autograd.grad((flow + beta * next_value).sum(), mu, create_graph=True)[0]
    foc = gradient[:, :-1, :-1] - gradient[:, :-1, -1:] - gradient[:, -1:, :-1]
    r1 = torch.nn.functional.pad((foc * maskc[:-1, :-1]).square(), (0, 1, 0, 1)).reshape(ng, -1)
    r2 = (value - flow - beta * next_value).square().view(ng, 1)
    margins = torch.cat([mu[:, :-1, :].sum(2), mu[:, :, :-1].sum(1)], dim=1)
    r3 = 0.0 * (s - margins).square().view(ng, -1)
    return torch.cat([r1, r2, r3], dim=1).mean()


class PrimalEquivalenceTests(unittest.TestCase):
    specifications = [
        ('Mcal', 2, 3, tauMcal, scaleMcal, masksM),
        ('MScal', 3, 7, tauMScal, scaleMScal, masksMS),
        ('MScaltrend', 3, 12, tauMScaltrend, scaleMScaltrend, masksMS),
    ]

    def fixture(self, spec, negative_next_population=False):
        _, ndim, thetadim, tau, scale, masks = spec
        torch.manual_seed(170)
        xi = SinkhornGeneric(tau, ndim, thetadim + 4*ndim + 2, thetadim, hidden_layers=[8])
        with torch.no_grad():
            xi.layers[-1].weight.mul_(.1)
            xi.layers[-1].bias.zero_()
        theta = torch.full((thetadim,), .1, requires_grad=True)
        mask = tuple(torch.tensor(m, dtype=torch.bool) for m in masks)
        p0 = torch.rand(ndim, ndim + 1, ndim) + .2
        q0 = torch.rand(ndim + 1, ndim, ndim) + .2
        p0 = p0 / p0.sum(-1, keepdim=True) * mask[0][:-1, :, None]
        q0 = q0 / q0.sum(-1, keepdim=True) * mask[0][:, :-1, None]
        transitions = (None, None, p0, q0, p0.flip(-1), q0.flip(-1))
        flow = torch.zeros(2*ndim)
        if negative_next_population:
            flow[0], flow[1] = -1.2, 1.2
        netflow = lambda s: flow.to(s)
        args = (8, xi, transitions, netflow, .95, theta, tau, scale,
                mask, torch.linspace(0, 1, 5), 'cpu')
        return args, (theta,) + tuple(xi.parameters())

    def evaluate(self, function, args, parameters, seed):
        torch.manual_seed(seed)
        with redirect_stdout(StringIO()):
            loss = function(*args)
        gradients = torch.autograd.grad(loss, parameters)
        self.assertTrue(torch.isfinite(loss))
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
        return loss, gradients

    def assert_equivalent(self, spec, seed, negative_next_population=False):
        args, parameters = self.fixture(spec, negative_next_population)
        old_loss, old_gradients = self.evaluate(historical_residuals, args, parameters, seed)
        with patch('dynamicmatching.bellman.dual_residuals',
                   side_effect=AssertionError('Primal-only evaluation must skip the dual')):
            new_loss, new_gradients = self.evaluate(
                lambda *args: residuals(*args, use_dual=False), args, parameters, seed)
        torch.testing.assert_close(new_loss, old_loss, rtol=2e-6, atol=1e-7)
        for actual, expected in zip(new_gradients, old_gradients):
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)

    def test_disabled_dual_matches_historical_loss_and_gradients(self):
        # 2026-11-06: Protect the original primal criterion, including its padded averaging denominator.
        with torch.random.fork_rng():
            for spec in self.specifications:
                for seed in (3, 17):
                    with self.subTest(spec=spec[0], seed=seed):
                        self.assert_equivalent(spec, seed)

    def test_disabled_dual_preserves_historical_sampling_domain(self):
        with torch.random.fork_rng():
            for spec in self.specifications:
                with self.subTest(spec=spec[0]):
                    self.assert_equivalent(spec, 23, negative_next_population=True)

    def test_enabled_dual_adds_terms_without_rescaling_the_primal(self):
        def fixed_dual(mu, s, d, phi, scales, values, next_values,
                       transitions, netflow, beta, mask):
            equations = mu.new_full((len(mu), int(mask.sum()) + 1), .25)
            return equations, mu.new_full((len(mu),), .5)

        with torch.random.fork_rng():
            for spec in self.specifications:
                for mixed_next_populations in (False, True):
                    with self.subTest(spec=spec[0], mixed_next_populations=mixed_next_populations):
                        args, parameters = self.fixture(spec)
                        ndim, mask = spec[1], args[8][0]
                        if mixed_next_populations:
                            flow = torch.zeros(2*ndim)
                            flow[0], flow[1] = -.5/ndim, .5/ndim
                            args = args[:3] + (lambda s: flow.to(s),) + args[4:]
                        old_loss, old_gradients = self.evaluate(historical_residuals, args, parameters, 37)
                        with patch('dynamicmatching.bellman.dual_residuals', side_effect=fixed_dual) as dual:
                            new_loss, new_gradients = self.evaluate(residuals, args, parameters, 37)
                            dual.assert_called_once()
                            retained = len(dual.call_args.args[0])
                            if mixed_next_populations:
                                self.assertGreater(retained, 0)
                                self.assertLess(retained, args[0])
                            else:
                                self.assertEqual(retained, args[0])
                        denominator = (ndim + 1)**2 + 1 + 2*ndim
                        increment = ((int(mask.sum()) + 1) * .25**2 + .5**2) / denominator
                        torch.testing.assert_close(new_loss, old_loss + increment, rtol=2e-6, atol=1e-7)
                        for actual, expected in zip(new_gradients, old_gradients):
                            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
