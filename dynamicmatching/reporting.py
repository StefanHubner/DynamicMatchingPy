from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import jax.numpy as jnp
import numpy as np

from .bellman import (choices, conditional_kl_loss, match_moments,
                      incumbent_population)
from .entry import empirical_entry, entry_mass
from .graphs import create_heatmap, plot_cf_grid, plot_estimator_grid
from .helpers import TermColours


def format_training_step(step, max_steps, current_loss, best_loss, inner_loss,
                         current_theta, best_theta, max_mass_deviation,
                         *, conditional_diffs=None, steps_without_improvement=0,
                         patience=0, inner_min_step=None, inner_steps=None):
    """One coordinated outer-step report; details appear only for a new best."""
    def number(value):
        return "n/a" if value is None else f"{value:.10g}"

    def parameters(theta):
        if theta is None:
            return "n/a"
        return np.array2string(np.asarray(theta).flatten(), precision=3,
                               floatmode="fixed", suppress_small=True,
                               max_line_width=100000)

    cyan, green, yellow, grey, reset = (TermColours.CYAN, TermColours.GREEN,
                                       TermColours.YELLOW, TermColours.BRIGHT_BLACK,
                                       TermColours.RESET)
    patience_note = (f" | since best: {steps_without_improvement}/{patience}"
                     if patience else "")
    minimum_at = f"{inner_min_step}/{inner_steps}" if inner_min_step else "n/a"
    lines = [
        f"Step {step}/{max_steps} | total current: {cyan}{number(current_loss)}{reset}"
        f" | best: {green}{number(best_loss)}{reset}"
        f" | inner current: {cyan}{number(inner_loss)}{reset}{patience_note}",
        f"{yellow}Max mass deviation (all inner updates/trials): {max_mass_deviation:.3e}{reset}"
        f" | {TermColours.BLUE}Inner min at: {minimum_at}{reset}",
        f"Parameters current: {TermColours.WHITE}{parameters(current_theta)}{reset}"
        f" | best: {green}{parameters(best_theta)}{reset}",
    ]
    if conditional_diffs is not None:
        details = np.array2string(np.asarray(conditional_diffs), precision=4,
                                  floatmode="fixed", suppress_small=True,
                                  max_line_width=120)
        lines.append(f"{grey}New best: conditional differences (model - data, gender average)\n"
                     f"{details}{reset}")
    return "\n".join(lines)


def format_entry_losses(diagnostics):
    return (f"\nMatching KL: {diagnostics.get('matching_kl', np.nan):.10g}"
            f" | entry KL: {diagnostics.get('entry_kl', np.nan):.10g}")


def no_fit_text(diagnostics):
    return ''


def no_fit_report(results, years, start, beyond, prefix):
    return None


SERIES_NAMES = [("M", "hat"), ("M", "star"), ("F", "hat"), ("F", "star")]
COLUMN_NAMES = ["scenario", "sex", "estimator", "state"]
# 2026-11-03: Select plot states by the retained model's number of types.
CONDITIONAL_STATES = {
    2: [("U|U", (0, 0)), ("0|U", (0, 2)), ("C|C", (1, 1)), ("0|C", (1, 2))],
    3: [("U|U", (0, 0)), ("0|U", (0, 3)),
        ("CH|CH", (1, 1)), ("CW|CH", (1, 2)), ("0|CH", (1, 3)),
        ("CH|CW", (2, 1)), ("CW|CW", (2, 2)), ("0|CW", (2, 3))]
}
MARGINAL_STATES = {
    2: [("Unmarried", (0,)), ("Married", (1,))],
    3: [("Unmarried", (0,)), ("Home Production", (1,)), ("Market Work", (2,))]
}
JOINT_STATES = {
    2: [("Cohabiting", (0, 0)), ("Married", (1, 1)),
        ("Single Men", (0, 2)), ("Divorced Men", (1, 2)),
        ("Single Women", (2, 0)), ("Divorced Women", (2, 1))],
    3: [("Cohabiting", (0, 0)), ("Single Men", (0, 3)),
        ("Both Home Production", (1, 1)), ("Male Breadwinner", (2, 1)),
        ("Both Market Work", (2, 2)), ("Divorced Men (Market Work)", (2, 3)),
        ("Single Women", (3, 0)), ("Divorced Women (Home Production)", (3, 1)),
        ("Divorced Women (Market Work)", (3, 2))]
}

SANDBOX_LAYOUTS = {
    "M": {
        "hdmu": ['u', 'c', '0'],
        "hds": ['M_u', 'M_c', 'F_u', 'F_c', 't', 'd'],
        "cells": {"uu": (0, 0), "cc": (1, 1), "u0": (0, 2), "0u": (2, 0),
                  "c0": (1, 2), "0c": (2, 1)},
        "couples": ["uu", "cc"],
        "singles": ["u0", "0u", "c0", "0c"]
    },
    "MS": {
        "hdmu": ['u', 'ch', 'cw', '0'],
        "hds": ['M_u', 'M_{ch}', 'M_{cw}', 'F_u', 'F_{ch}', 'F_{cw}', 't', 'd'],
        "cells": {"uu": (0, 0), "chch": (1, 1), "chcw": (1, 2),
                  "cwch": (2, 1), "cwcw": (2, 2),
                  "u0": (0, 3), "0u": (3, 0),
                  "ch0": (1, 3), "0ch": (3, 1),
                  "cw0": (2, 3), "0cw": (3, 2)},
        "couples": ["uu", "chch", "chcw", "cwch", "cwcw"],
        "singles": ["u0", "0u", "ch0", "0ch", "cw0", "0cw"]
    },
    "KMS": {
        "hdmu": ['vuy', 'vuo', 'vcy', 'vco', 'kuh', 'kuw', 'kch', 'kcw', '0'],
        "hds": ['M_{vuy}', 'M_{vuo}', 'M_{vcy}', 'M_{vco}',
                'M_{kuh}', 'M_{kuw}', 'M_{kch}', 'M_{kcw}',
                'F_{vuy}', 'F_{vuo}', 'F_{vcy}', 'F_{vco}',
                'F_{kuh}', 'F_{kuw}', 'F_{kch}', 'F_{kcw}', 't', 'd'],
        "cells": {"vuyvuy": (0, 0), "vuovuo": (1, 1),
                  "vchvch": (2, 2), "vcwvcw": (3, 3),
                  "kuykuy": (4, 4), "kuokuo": (5, 5),
                  "kuokuy": (5, 4), "kchkch": (6, 6),
                  "kcwkcw": (7, 7), "kcwkch": (7, 6),
                  "vuy0": (0, 8), "0vuy": (8, 0),
                  "vuo0": (1, 8), "0vuo": (8, 1),
                  "vch0": (2, 8), "0vch": (8, 2),
                  "vcw0": (3, 8), "0vcw": (8, 3),
                  "kuy0": (4, 8), "0kuy": (8, 4),
                  "kuo0": (5, 8), "0kuo": (8, 5),
                  "kch0": (6, 8), "0kch": (8, 6),
                  "kcw0": (7, 8), "0kcw": (8, 7)},
        "couples": ["vuyvuy", "vuvouo", "vchvch", "vcwvcw", "vcwvch",
                    "kuykuy", "kukouo", "kchkch", "kcwkcw", "kcwkch"],
        "singles": ["vuy0", "0vuy", "vuo0", "0vuo", "vch0", "0vch", "vcw0", "0vcw",
                    "kuy0", "0kuy", "kuo0", "0kuo", "kch0", "0kch", "kcw0", "0kcw"]
    }
}


def discounted_welfare(flows, horizon, beta):
    """Sum each year's next horizon flow payoffs, including that year."""
    # 2026-10-22: Discount simulated flows at the model's annual beta, without an NN terminal value.
    weights = beta ** np.arange(horizon, dtype=np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(np.asarray(flows), horizon, axis=0)
    return (windows.astype(np.float64) * weights).sum(-1)


def load_mus(xi, theta, tPs, tQs, muh, netflow, ng, dev, tau,
             scale, masks, tis, years, cf, train0, beyond, beta,
             bellman_welfare=False, welfare_horizon=0, entry=empirical_entry,
             kl=conditional_kl_loss):
    tail = max(welfare_horizon - 1, 0)
    (_, muh1, mus, _, conds, margs, vs, _, _,
     diagnostics, logits) = match_moments(
        xi, theta, tPs, tQs, muh, netflow, ng,
        dev, tau, scale, masks, tis, years,
        skiptrain=True, cf=cf, train0=train0, beyond=beyond + tail,
        bellman_welfare=bellman_welfare, beta=beta, entry=entry, kl=kl)
    if welfare_horizon:
        start, stop = (0 if train0 else tis[0]), len(muh) + beyond
        mus = mus[:stop]
        vs = discounted_welfare(vs, welfare_horizon, beta)
        vs[:start] = 0
        matched = conditional_kl_loss(muh[start:] if beyond == 0 else mus[start:],
                                      mus[start:], masks)
        conds, margs = matched[1:5], matched[5:]
    logits = logits[:len(mus)]
    rates = diagnostics['entry_rates']
    return muh1, mus, conds, margs, vs, logits, rates


def _series_columns(scenario, sex, estimator, tensor, states):
    return {(scenario, sex, estimator, label):
            np.asarray(tensor[(slice(None),) + indices]).flatten()
            for label, indices in states}


def _plot_frame(series, labels):
    frame = pd.DataFrame(series, index=labels)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=COLUMN_NAMES)
    return frame


def prepare_plot_frames(results, years, treat_idcs, train0, beyond):
    """Turn the scenario outputs into the four tables used by the paper plots."""
    columns = {name: {} for name in ("conditional", "marginal", "joint", "welfare")}
    start = 0 if train0 else treat_idcs[0]
    labels = [str(year) for year in range(years.start + start, years.stop + beyond)]
    for scenario, (_, mu_star, conditional, marginal,
                   welfare, _, _) in results.items():
        ndim = mu_star.shape[1] - 1
        for name, tensors, states in (("conditional", conditional, CONDITIONAL_STATES[ndim]),
                                      ("marginal", marginal, MARGINAL_STATES[ndim])):
            for (sex, estimator), tensor in zip(SERIES_NAMES, tensors):
                columns[name].update(_series_columns(scenario, sex, estimator, tensor, states))
        columns["joint"].update(_series_columns(scenario, "both", "star",
                                               mu_star[start:], JOINT_STATES[ndim]))
        columns["welfare"].update(_series_columns(scenario, "both", "star",
                                                 welfare[start:], [("Welfare", (0,))]))

    return {name: _plot_frame(series, labels) for name, series in columns.items()}


def prepare_fit_frame(result, years, treat_idcs, train0, masks):
    mu_hat, mu_star, _, _, _, _, _ = result
    start = 0 if train0 else treat_idcs[0]
    # 2026-10-12: Compare historical observed and model matches, excluding forecast alignment copies.
    conditional = conditional_kl_loss(mu_hat[start:], mu_star[start:len(mu_hat)], masks)[1:5]
    series = {}
    for (sex, estimator), tensor in zip(SERIES_NAMES, conditional):
        series.update(_series_columns("CFF", sex, estimator, tensor,
                                       CONDITIONAL_STATES[mu_star.shape[1] - 1]))
    return _plot_frame(series, [str(year) for year in years[start:len(mu_hat)]])


def prepare_entry_fit(result):
    _, _, _, _, _, _, rates = result
    series = {('CFF', 'both', label, 'Marriage entry probability'): rates[:, i]
              for i, label in enumerate(('hat', 'star'))}
    return _plot_frame(series, ['2001–2008', '2009–2020'])


def report_entry(results, years, start, beyond, prefix):
    series = {}
    for scenario, (_, mus, _, _, _, logits, _) in results.items():
        logits = np.asarray(logits)[start:, 0]
        logp = -np.logaddexp(0, -logits)
        p = np.exp(logp)
        flows = p * np.asarray(mus)[start:, 0, 0]
        series.update({
            (scenario, "both", "star", "Marriage entry probability"): p,
            (scenario, "both", "star", "Marriage entry flow"): flows})
    labels = [str(y) for y in range(years.start + start, years.stop + beyond)]
    frame = _plot_frame(series, labels)
    save_plot(plot_cf_grid, frame, f"{prefix}_entry_cf.pdf", csv=True,
              sex="both", dim=(1, 2), beyond=beyond, forecast_start=years.stop)
    fitted = prepare_entry_fit(results["CFF"])
    save_plot(plot_estimator_grid, fitted, f"{prefix}_entry_fit.pdf", csv=True,
              sex="both", nrows=1, ncols=1)
    _, _, _, _, _, _, rates = results["CFF"]
    table = pd.DataFrame(rates, index=fitted.index,
                         columns=['Entry observed', 'Entry fitted'])
    print(table)


def save_plot(plotter, frame, filename, csv=False, **kwargs):
    """Draw and export a plot, closing its figure after saving."""
    filename = Path(filename)
    fig, selected = plotter(frame, **kwargs)
    try:
        fig.savefig(filename, bbox_inches="tight")
        if csv:
            selected.to_csv(filename.with_suffix(".csv"))
    finally:
        plt.close(fig)


def sandbox_controls(spec, years):
    """Build the existing sidebar controls and expose its state and labels."""
    s = st.session_state
    if 'currentyear' not in s: s.currentyear = years[0]
    if 't' not in s: s.t = jnp.array(0.0).reshape(1, 1)
    dt = 1.0 / (years[-1] - years[0])

    def update_year():
        s.t = jnp.array((s.currentyear - years[0]) * dt).reshape(1, 1)

    st.sidebar.slider('Year', years[0], years[-1], step=1,
                      key='currentyear', on_change=update_year)
    if 'mu' not in s: s.mu = 0.25
    if 'vu' not in s: s.vu = 0.25

    def update_c():
        s.mc = 0.5 - s.mu
        s.vc = 0.5 - s.vu

    for label, key in [('$M_{u}$', 'mu'), ('$F_{u}$', 'vu')]:
        st.sidebar.slider(label, 0.0, 0.5, step=0.01, key=key, on_change=update_c)
    update_c()

    if spec == "M":
        ss = lambda d: jnp.array([[s.mu, s.mc, s.vu, s.vc, float(s.t.item()), d]])

    if spec == "MS":
        for key in ('mhc', 'vhc'):
            if key not in s: s[key] = 0.5

        def update_wc():
            s.mwc = 1 - s.mhc
            s.fwc = 1 - s.vhc

        for label, key, callback in [('$M_{h|c}$', 'mhc', update_wc),
                                     ('$F_{h|c}$', 'vhc', update_wc)]:
            st.sidebar.slider(label, 0.0, 1.0, step=0.01, key=key, on_change=callback)
            callback()

        ss = lambda d: jnp.array([[
                           s.mu, s.mc * s.mhc, s.mc * (1 - s.mhc),
                           s.vu, s.vc * s.vhc, s.vc * (1 - s.vhc),
                           float(s.t.item()), d]])

    return dict(SANDBOX_LAYOUTS[spec], ss=ss, t=s.t)


def show_sandbox(xi, theta, tau, ndim, controls, transitions, netflow, dt,
                 entry=empirical_entry):
    ss, t = controls["ss"], controls["t"]
    hdmu, hds = controls["hdmu"], controls["hds"]
    _, _, tP0, tQ0, tP1, tQ1 = transitions
    population, _, _, _ = entry(theta, transitions, incumbent_population)
    zero = jnp.array(0.0).reshape(1, 1)
    matches, values, next_states = [], [], []
    for regime, d in enumerate((zero, zero + 1)):
        state = ss(regime)
        mus, v, logits = xi.policy(xi.params, state)
        matches.append(mus)
        values.append(v)
        e = entry_mass(matches[0], logits)
        stnext = choices(matches[0], t, d, population, e, netflow, dt, "cpu")
        next_states.append(stnext)

    ps = np.asarray(theta)
    st.write("Parameters: " + ", ".join(["{:.2f}".format(p) for p in ps]))
    cols = st.columns(2)
    for col, d in zip(cols, [zero, zero + 1]):
        with col:
            st.subheader('$\\Phi_{{t}}({})$'.format(int(d.item())))
            st.dataframe(pd.DataFrame(np.asarray(tau(theta, t, d, "cpu")),
                                      index=hdmu, columns=hdmu))

    st.subheader('$\\mu_1$ in %')
    fmarg = np.asarray(ss(1)[0, ndim:2*ndim]).tolist()
    mmarg = np.asarray(ss(1)[0, 0:ndim]).tolist()
    fig = create_heatmap(matches[1][0]*100,
                         ["{:.1f}".format(f*100) for f in fmarg + [0]],
                         ["{:.1f}".format(m*100) for m in mmarg + [0]], hdmu)
    try:
        st.pyplot(fig, use_container_width=False)
    finally:
        plt.close(fig)
    for col, ssnext, v in zip(cols, next_states, values):
        with col:
            st.subheader('($M_{{t+1}}^*, F_{{t+1}}^*) $')
            st.dataframe(pd.DataFrame(np.asarray(ssnext), columns=hds), hide_index=True)
            st.subheader('$V$')
            st.write(np.asarray(v))


def show_raw_matches(mu_hat, mu_star, years, labels):
    for col, mu, title in zip(st.columns(2), [mu_hat, mu_star],
                              ["Observed: $\\widehat{\\mu}_t$", "Model: $\\mu^*_t$"]):
        with col:
            st.header(title)
            for i in range(mu.shape[0]):
                st.subheader("{}".format(years[i]))
                st.dataframe(pd.DataFrame(np.asarray(mu[i]), columns=labels, index=labels))
                st.dataframe(pd.DataFrame({
                    'M': np.asarray(mu[i, :-1, :].sum(axis=1)),
                    'F': np.asarray(mu[i, :, :-1].sum(axis=0))
                }, index=labels[:-1]))
