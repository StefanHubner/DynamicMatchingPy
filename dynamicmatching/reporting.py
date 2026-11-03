from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from .bellman import choices, conditional_kl_loss, match_moments
from .graphs import create_heatmap


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
    }
}


def discounted_welfare(flows, horizon, beta):
    """Sum each year's next horizon flow payoffs, including that year."""
    if not torch.isfinite(flows).all():
        raise ValueError("Non-finite flow welfare in the extended rollout.")
    # 2026-10-22: Discount simulated flows at the model's annual beta, without an NN terminal value.
    weights = beta ** torch.arange(horizon, dtype=torch.float64, device=flows.device)
    return (flows.unfold(0, horizon, 1).double() * weights).sum(-1)


@torch.no_grad()
def load_mus(xi, theta, tPs, tQs, muh, netflow, ng, dev, tau,
             scale, masks, tis, years, cf, train0, beyond, beta,
             bellman_welfare=False, welfare_horizon=0):
    if welfare_horizon < 0 or (bellman_welfare and welfare_horizon):
        raise ValueError("Choose NN welfare or a nonnegative discounted-flow horizon.")
    tail = max(welfare_horizon - 1, 0)
    _, muh1, mus, _, conds, margs, vs, individual = match_moments(xi, theta,
                                        tPs, tQs, muh, netflow, ng,
                                        dev, tau, scale, masks, tis, years,
                                        skiptrain = True, cf = cf,
                                        train0 = train0, beyond = beyond + tail,
                                        bellman_welfare = bellman_welfare, beta=beta)
    if welfare_horizon:
        start, stop = (0 if train0 else tis[0]), len(muh) + beyond
        mus = mus[:stop]
        individual = tuple(values[:stop] for values in individual)
        vs = discounted_welfare(vs, welfare_horizon, beta)
        vs[:start] = 0
        matched = conditional_kl_loss(muh[start:] if beyond == 0 else mus[start:],
                                      mus[start:], masks)
        conds, margs = matched[1:5], matched[5:]
    return muh1, mus, conds, margs, vs, individual


def _series_columns(scenario, sex, estimator, tensor, states):
    return {(scenario, sex, estimator, label):
            tensor[(slice(None),) + indices].flatten().detach().numpy()
            for label, indices in states}


def _plot_frame(series, labels):
    frame = pd.DataFrame(series, index=labels)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=COLUMN_NAMES)
    return frame


def prepare_plot_frames(results, years, treat_idcs, train0, beyond):
    """Turn the scenario outputs into the tables used by the paper plots."""
    columns = {name: {} for name in ("conditional", "marginal", "joint", "welfare",
                                    "individual_welfare")}
    start = 0 if train0 else treat_idcs[0]
    labels = [str(year) for year in range(years.start + start, years.stop + beyond)]
    for scenario, (_, mu_star, conditional, marginal, welfare, individual) in results.items():
        ndim = mu_star.shape[1] - 1
        for name, tensors, states in (("conditional", conditional, CONDITIONAL_STATES[ndim]),
                                      ("marginal", marginal, MARGINAL_STATES[ndim])):
            for (sex, estimator), tensor in zip(SERIES_NAMES, tensors):
                columns[name].update(_series_columns(scenario, sex, estimator, tensor, states))
        columns["joint"].update(_series_columns(scenario, "both", "star",
                                               mu_star[start:], JOINT_STATES[ndim]))
        columns["welfare"].update(_series_columns(scenario, "both", "star",
                                                 welfare[start:], [("Welfare", (0,))]))
        # 2026-11-03: Report type values without population weights or allocating the fixed-flow term J.
        wm, wf, _ = individual
        for sex, values in (("M", wm), ("F", wf)):
            columns["individual_welfare"].update(_series_columns(
                scenario, sex, "star", values[start:], MARGINAL_STATES[ndim]))

    return {name: _plot_frame(series, labels) for name, series in columns.items()}


def prepare_fit_frame(result, years, treat_idcs, train0, masks):
    mu_hat, mu_star, _, _, _, _ = result
    start = 0 if train0 else treat_idcs[0]
    # 2026-10-12: Compare historical observed and model matches, excluding forecast alignment copies.
    conditional = conditional_kl_loss(mu_hat[start:], mu_star[start:len(mu_hat)], masks)[1:5]
    series = {}
    for (sex, estimator), tensor in zip(SERIES_NAMES, conditional):
        series.update(_series_columns("CFF", sex, estimator, tensor,
                                       CONDITIONAL_STATES[mu_star.shape[1] - 1]))
    return _plot_frame(series, [str(year) for year in years[start:len(mu_hat)]])


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
    if 't' not in s: s.t = torch.tensor(0.0).view(1, 1)
    dt = 1.0 / (years[-1] - years[0])

    def update_year():
        s.t = torch.tensor((s.currentyear - years[0]) * dt).view(1, 1)

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
        ss = lambda d: torch.tensor([[s.mu, s.mc, s.vu, s.vc, s.t, d]], device="cpu")

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

        ss = lambda d: torch.tensor([[
                           s.mu, s.mc * s.mhc, s.mc * (1 - s.mhc),
                           s.vu, s.vc * s.vhc, s.vc * (1 - s.vhc),
                           s.t, d]], device="cpu")

    return dict(SANDBOX_LAYOUTS[spec], ss=ss, t=s.t)


def show_sandbox(xi, theta, tau, ndim, controls, transitions, netflow, dt):
    ss, t = controls["ss"], controls["t"]
    hdmu, hds = controls["hdmu"], controls["hds"]
    _, _, tP0, tQ0, tP1, tQ1 = transitions
    torch0 = torch.tensor(0.0, device="cpu").view(1, 1)
    matches, values, next_states = [], [], []
    for regime, d in enumerate((torch0, torch0 + 1)):
        mus, v, _, _, _ = xi(ss(regime))
        matches.append(mus)
        values.append(v)
        next_states.append(choices(matches[0], t, d, tP0, tQ0, tP1, tQ1, netflow, dt, "cpu"))

    ps = theta.detach().numpy()
    st.write("Parameters: " + ", ".join(["{:.2f}".format(p) for p in ps]))
    cols = st.columns(2)
    for col, d in zip(cols, [torch0, torch0 + 1]):
        with col:
            st.subheader('$\\Phi_{{t}}({})$'.format(int(d.item())))
            st.dataframe(pd.DataFrame(tau(theta, t, d, "cpu").detach().numpy(),
                                      index=hdmu, columns=hdmu))

    st.subheader('$\\mu_1$ in %')
    fmarg = ss(1)[0, ndim:2*ndim].detach().numpy().tolist()
    mmarg = ss(1)[0, 0:ndim].detach().numpy().tolist()
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
            st.dataframe(pd.DataFrame(ssnext.detach().numpy(), columns=hds), hide_index=True)
            st.subheader('$V$')
            st.write(v.detach().numpy())


def show_raw_matches(mu_hat, mu_star, years, labels):
    for col, mu, title in zip(st.columns(2), [mu_hat, mu_star],
                              ["Observed: $\\widehat{\\mu}_t$", "Model: $\\mu^*_t$"]):
        with col:
            st.header(title)
            for i in range(mu.shape[0]):
                st.subheader("{}".format(years[i]))
                st.dataframe(pd.DataFrame(mu[i].detach().numpy(), columns=labels, index=labels))
                st.dataframe(pd.DataFrame({
                    'M': mu[i, :-1, :].sum(axis=1).detach().numpy(),
                    'F': mu[i, :, :-1].sum(axis=0).detach().numpy()
                }, index=labels[:-1]))
