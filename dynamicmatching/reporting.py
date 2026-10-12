from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from .bellman import choices, match_moments
from .graphs import create_heatmap


SERIES_NAMES = [("M", "hat"), ("M", "star"), ("F", "hat"), ("F", "star")]
COLUMN_NAMES = ["scenario", "sex", "estimator", "state"]
CONDITIONAL_STATES = [("U|U", (0, 0)), ("0|U", (0, 3)),
                      ("CH|CH", (1, 1)), ("CW|CH", (1, 2)), ("0|CH", (1, 3)),
                      ("CH|CW", (2, 1)), ("CW|CW", (2, 2)), ("0|CW", (2, 3))]
MARGINAL_STATES = [("Unmarried", (0,)), ("Home Production", (1,)),
                   ("Market Work", (2,))]
JOINT_STATES = [("Cohabiting", (0, 0)), ("Single Men", (0, 3)),
                ("Both Home Production", (1, 1)), ("Male Breadwinner", (2, 1)),
                ("Both Market Work", (2, 2)), ("Divorced Men (Market Work)", (2, 3)),
                ("Single Women", (3, 0)), ("Divorced Women (Home Production)", (3, 1)),
                ("Divorced Women (Market Work)", (3, 2))]

SANDBOX_LAYOUTS = {
    "M": {
        "hdmu": ['u', 'c', '0'],
        "hds": ['M_u', 'M_c', 'F_u', 'F_c', 't', 'd'],
        "cells": {"uu": (0, 0), "cc": (1, 1), "u0": (0, 2), "0u": (2, 0),
                  "c0": (1, 2), "0c": (2, 1)},
        "couples": ["uu", "cc"],
        "singles": ["u0", "0u", "c0", "0c"]
    },
    # 2026-10-12: Retain the legacy four-type MS sandbox separately from the three-type plots.
    "MS": {
        "hdmu": ['uy', 'uo', 'ch', 'cw', '0'],
        "hds": ['M_{uy}', 'M_{uo}', 'M_{ch}', 'M_{cw}',
                'F_{uy}', 'F_{uu}', 'F_{ch}', 'F_{cw}', 't', 'd'],
        "cells": {"uyuy": (0, 0), "uouo": (1, 1),
                  "chch": (2, 2), "cwcw": (3, 3),
                  "uouy": (1, 0), "cwch": (3, 2),
                  "uy0": (0, 4), "0uy": (4, 0),
                  "uo0": (1, 4), "0uo": (4, 1),
                  "ch0": (2, 4), "0ch": (4, 2),
                  "cw0": (3, 4), "0cw": (4, 3)},
        "couples": ["uyuy", "uouo", "uouy", "chch", "cwcw", "cwch"],
        "singles": ["uy0", "0uy", "uo0", "0uo", "ch0", "0ch", "cw0", "0cw"]
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


def load_mus(xi, theta, tPs, tQs, muh, netflow, ng, dev, tau,
             scale, masks, tis, years, cf, train0, beyond):
    _, muh1, mus, _, conds, margs, vs = match_moments(xi, theta,
                                        tPs, tQs, muh, netflow, ng,
                                        dev, tau, scale, masks, tis, years,
                                        skiptrain = True, cf = cf,
                                        train0 = train0, beyond = beyond)
    return muh1, mus, conds, margs, vs


def _series_columns(scenario, sex, estimator, tensor, states):
    return {(scenario, sex, estimator, label):
            tensor[(slice(None),) + indices].flatten().detach().numpy()
            for label, indices in states}


def prepare_plot_frames(results, years, treat_idcs, train0, beyond):
    """Turn the scenario outputs into the four tables used by the paper plots."""
    columns = {name: {} for name in ("conditional", "marginal", "joint", "welfare")}
    start = 0 if train0 else treat_idcs[0]
    labels = [str(year) for year in range(years.start + start, years.stop + beyond)]
    for scenario, (_, mu_star, conditional, marginal, welfare) in results.items():
        for name, tensors, states in (("conditional", conditional, CONDITIONAL_STATES),
                                      ("marginal", marginal, MARGINAL_STATES)):
            for (sex, estimator), tensor in zip(SERIES_NAMES, tensors):
                columns[name].update(_series_columns(scenario, sex, estimator, tensor, states))
        columns["joint"].update(_series_columns(scenario, "both", "star",
                                               mu_star[start:], JOINT_STATES))
        columns["welfare"].update(_series_columns(scenario, "both", "star",
                                                 welfare[start:], [("Welfare", (0,))]))

    frames = {}
    for name, series in columns.items():
        frame = pd.DataFrame(series, index=labels)
        frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=COLUMN_NAMES)
        frames[name] = frame
    return frames


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
        for key in ('myu', 'mhc', 'vyu', 'vhc'):
            if key not in s: s[key] = 0.5

        def update_ou():
            s.mou = 1 - s.myu
            s.fou = 1 - s.vyu

        def update_wc():
            s.mwc = 1 - s.mhc
            s.fwc = 1 - s.vhc

        for label, key, callback in [('$M_{y|u}$', 'myu', update_ou),
                                     ('$M_{h|c}$', 'mhc', update_wc),
                                     ('$F_{y|u}$', 'vyu', update_ou),
                                     ('$F_{h|c}$', 'vhc', update_wc)]:
            st.sidebar.slider(label, 0.0, 1.0, step=0.01, key=key, on_change=callback)
            callback()

        ss = lambda d: torch.tensor([[
                           s.mu * s.myu, s.mu * (1 - s.myu),
                           s.mc * s.mhc, s.mc * (1 - s.mhc),
                           s.vu * s.vyu, s.vu * (1 - s.vyu),
                           s.vc * s.vhc, s.vc * (1 - s.vhc),
                           s.t, d]], device="cpu")

    return dict(SANDBOX_LAYOUTS[spec], ss=ss, t=s.t)


def show_sandbox(xi, theta, tau, ndim, controls, transitions, netflow, dt):
    ss, t = controls["ss"], controls["t"]
    hdmu, hds = controls["hdmu"], controls["hds"]
    _, _, tP0, tQ0, tP1, tQ1 = transitions
    torch0 = torch.tensor(0.0, device="cpu").view(1, 1)
    matches, values, next_states = [], [], []
    for regime, d in enumerate((torch0, torch0 + 1)):
        mus, v = xi(ss(regime))
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
