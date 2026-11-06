from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch

from dynamicmatching.helpers import CF
from dynamicmatching.reporting import load_mus, prepare_plot_frames, MARGINAL_STATES
from dynamicmatching.graphs import plot_individual_welfare


def allocation(margins, fraction):
    ndim = margins.shape[-1] // 2
    male, female = margins[:ndim], margins[ndim:]
    couples = fraction * torch.minimum(male, female)
    mu = margins.new_zeros(ndim + 1, ndim + 1)
    mu[:ndim, :ndim] = torch.diag(couples)
    mu[:ndim, -1], mu[-1, :ndim] = male - couples, female - couples
    return mu


class RecordedValues(torch.nn.Module):
    """Encode population, date and regime in signed values, without fitted weights."""
    def __init__(self, ndim):
        super().__init__()
        self.ndim = ndim
        self.calls = []

    def forward(self, state):
        self.calls.append(state.detach().clone())
        n = self.ndim
        t, d = state[:, -2:-1], state[:, -1:]
        types = torch.arange(n, dtype=state.dtype)
        wm = -10 + state[:, :n] + .25 * types + 2 * t + 3 * d
        wf = -20 + state[:, n:2*n] + .5 * types + 4 * t + 5 * d
        J = (100 + 6 * t + 7 * d).flatten()
        mu = torch.stack([allocation(s[:2*n], .2) for s in state])
        return mu, (50 + t + d).flatten(), wm, wf, J


class IndividualWelfareTests(unittest.TestCase):
    years = range(2000, 2004)
    treatment = [1]
    scenarios = {'CFF': CF.None_, 'CF0': CF.HighCost, 'CF1': CF.LowCost}

    def run_path(self, ndim, cf, horizon=0, train0=False):
        weights = torch.arange(1, ndim + 1, dtype=torch.float32)
        margins = torch.cat([.45 * weights, .55 * weights.flip(0)]) / weights.sum()
        observed = allocation(margins, .08).repeat(len(self.years), 1, 1)
        p = torch.zeros(3, ndim, ndim + 1, ndim)
        q = torch.zeros(3, ndim + 1, ndim, ndim)
        for regime, rate_m, rate_f in [(0, .4, .25), (1, .7, .6), (2, .4, .25)]:
            for i in range(ndim):
                p[regime, i, :ndim, i] = 1 - rate_m
                p[regime, i, :ndim, (i + 1) % ndim] += rate_m
                p[regime, i, -1, i] = 1
                q[regime, :ndim, i, i] = 1 - rate_f
                q[regime, :ndim, i, (i - 1) % ndim] += rate_f
                q[regime, -1, i, i] = 1
        mask = torch.eye(ndim + 1, dtype=torch.bool)
        mask[-1, :-1], mask[:-1, -1], mask[-1, -1] = True, True, False
        xi = RecordedValues(ndim)
        with patch('dynamicmatching.bellman.minimise_inner',
                   side_effect=AssertionError('Plotting must not train')) as trainer:
            with redirect_stdout(StringIO()):
                result = load_mus(
                    xi, torch.zeros(1), p, q, observed, lambda s: s.new_zeros(2 * ndim),
                    1, 'cpu', lambda par, t, d, dev: observed.new_zeros(ndim + 1, ndim + 1),
                    lambda par, dev: (torch.ones(ndim), torch.ones(ndim)),
                    (mask, mask[-1]), self.treatment, self.years, cf, train0, 1, .8,
                    welfare_horizon=horizon)
            trainer.assert_not_called()
        return xi, result

    def test_values_follow_current_year_regime_and_the_matching_path(self):
        # 2026-11-03: Distinguish current-state values from the walker's next-state outputs.
        for ndim in (2, 3):
            for cf in self.scenarios.values():
                for train0 in (False, True):
                    with self.subTest(ndim=ndim, cf=cf, train0=train0):
                        xi, (observed, matches, _, _, _, individual) = self.run_path(ndim, cf, train0=train0)
                        wm, wf, J = individual
                        start, stop = (0 if train0 else 1), len(self.years) + 1
                        expected = observed[start].clone()
                        self.assertEqual(len(xi.calls), stop - start + 1)
                        for i in range(start, stop):
                            regime = {CF.None_: int(i in self.treatment), CF.HighCost: 0, CF.LowCost: 1}[cf]
                            t = i / (len(self.years) - 1)
                            male, female = expected[:-1, :].sum(1), expected[:, :-1].sum(0)
                            torch.testing.assert_close(matches[i], expected)
                            types = torch.arange(ndim)
                            torch.testing.assert_close(wm[i], -10 + male + .25 * types + 2*t + 3*regime)
                            torch.testing.assert_close(wf[i], -20 + female + .5 * types + 4*t + 5*regime)
                            torch.testing.assert_close(J[i], torch.tensor([100 + 6*t + 7*regime]))
                            couples = expected.diag()[:-1]
                            male = male + (.7 if regime else .4) * (couples.roll(1) - couples)
                            female = female + (.6 if regime else .25) * (couples.roll(-1) - couples)
                            expected = allocation(torch.cat([male, female]), .2)

    def test_discounted_flow_horizon_trims_but_does_not_discount_type_values(self):
        for ndim in (2, 3):
            for cf in self.scenarios.values():
                with self.subTest(ndim=ndim, cf=cf):
                    _, baseline = self.run_path(ndim, cf)
                    _, discounted = self.run_path(ndim, cf, horizon=3)
                    torch.testing.assert_close(discounted[1], baseline[1])
                    for actual, expected in zip(discounted[-1], baseline[-1]):
                        self.assertEqual(actual.shape[0], len(self.years) + 1)
                        torch.testing.assert_close(actual, expected)

    def test_gender_frames_and_signed_counterfactual_panels(self):
        for ndim in (2, 3):
            results = {name: self.run_path(ndim, cf)[1] for name, cf in self.scenarios.items()}
            frame = prepare_plot_frames(results, self.years, self.treatment, False, 1)['individual_welfare']
            self.assertEqual(frame.shape, (4, 6 * ndim))
            self.assertEqual(frame.index.tolist(), ['2001', '2002', '2003', '2004'])
            for scenario, result in results.items():
                for sex, values in zip(('M', 'F'), result[-1][:2]):
                    for index, (label, _) in enumerate(MARGINAL_STATES[ndim]):
                        torch.testing.assert_close(torch.tensor(frame[(scenario, sex, 'star', label)].values),
                                                   values[1:, index])
            for sex in ('M', 'F'):
                fig, selected = plot_individual_welfare(frame, sex, beyond=1, forecast_start=2004)
                try:
                    axes = [ax for ax in fig.axes if ax.get_visible()]
                    self.assertEqual(len(axes), ndim)
                    self.assertTrue(all(ax.get_ylim()[0] < 0 for ax in axes))
                    self.assertEqual(set(selected.columns.get_level_values('scenario')), {'CFF', 'CF1'})
                    self.assertEqual(set(selected.columns.get_level_values('sex')), {sex})
                    csv = pd.read_csv(StringIO(selected.to_csv()), header=[0, 1, 2, 3], index_col=0)
                    self.assertEqual(csv.columns.tolist(), selected.columns.tolist())
                finally:
                    plt.close(fig)


if __name__ == '__main__':
    unittest.main()
