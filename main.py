#!/usr/bin/env python

import argparse
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import streamlit as st
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from dynamicmatching import create_closure, create_netflow, overallPQ
from dynamicmatching import M, MS, MStrend, Mcal, MScal, MScaltrend, callbacks
from dynamicmatching import psiM, psiMS, masksM, masksMS, CF
from dynamicmatching import matched_process_plot, plot_cf_grid, plot_estimator_grid, plot_margin_counterfactuals
from dynamicmatching import SinkhornGeneric
from dynamicmatching import NelderMeadOptimizer
from dynamicmatching.reporting import load_mus, prepare_plot_frames, prepare_fit_frame, save_plot
from dynamicmatching.reporting import sandbox_controls, show_sandbox, show_raw_matches, format_training_step


# 2026-10-22: Set the annual discount factor once for estimation and welfare reporting.
beta = 0.95

st.set_page_config(page_title="Dynamic Matching")

# for debug
# noload, neldermead, ng0 = True, True, 128


# JAX devices need an explicit cache key; keep data cached separately per device.
@st.cache_resource(hash_funcs={jax.Device: lambda dev: (dev.platform, dev.id, dev.process_index)})
def load_data(name, dev):
    # 2026-10-02: Reuse HF_TOKEN or the cached login without prompting again.
    data = load_dataset("StefanHubner/DivorceData", token=True)[name]
    tPs = jax.device_put(jnp.asarray(data["p"][0]), dev)
    tQs = jax.device_put(jnp.asarray(data["q"][0]), dev)
    tMuHat = jax.device_put(jnp.asarray(data["couplings"][0]), dev)
    ee = jax.device_put(jnp.asarray(data["entryexit"][0]), dev)
    return tPs, tQs, tMuHat, ee


def main(train = False, noload = False,
         neldermead = True, matchingplot = False, patience = 150):

    np.set_printoptions(precision=4, suppress=True)
    figures = Path("figures")
    figures.mkdir(exist_ok=True)
    dev = jax.devices("cpu")[0] if matchingplot else jax.devices()[0]
    key, network_key = jax.random.split(jax.random.key(0))

    # Define specification and load appropriate dataset
    # outdim = network coefficient count + # of single types + value vct
    # (data, state dim, masks, model, par dim, initial pars, years, train0)
    current = args.spec
    theta_ms = jnp.concatenate((jnp.full(7, 0.1), psiMS()))

    spec  = { "M":
                ("M", 2, masksM, M, 4,
                 jnp.concatenate((jnp.full(3, 0.1), psiM())),
                 range(1999, 2021), False),
              "MS":
                ("MS", 3, masksMS, MS, 11,
                 theta_ms, range(1999, 2021), False),
              "MStrend":
                ("MS", 3, masksMS, MStrend, 16,
                 jnp.concatenate((theta_ms[:5], jnp.zeros(5), theta_ms[5:])),
                 range(1999, 2021), False),
              "Mcal":
                ("M", 2, masksM, Mcal, 2+1, None,
                 range(1999, 2021), False),
              "MScal":
                # 2026-10-02: One fixed reference scale leaves two estimated type scales.
                ("MS", 3, masksMS, MScal, 5+2, None,
                 range(1999, 2021), False),
              "MScaltrend":
                # 2026-10-06: Fresh coefficients and zero initial slopes; all trends remain free.
                ("MS", 3, masksMS, MScaltrend, 10+2,
                 5 * [0.1] + 5 * [0.0] + 2 * [0.1],
                 range(1999, 2021), False)
             }[current]
    vars, ndim, (maskc, mask0), evaluate, thetadim, theta0, years, train0 = spec
    tau, scale = callbacks(evaluate)
    net_model, netdim = {"M": (Mcal, 3), "MS": (MScal, 7),
                         "MStrend": (MScaltrend, 12)}.get(current, (evaluate, thetadim))
    net_tau, _ = callbacks(net_model)
    outdim = netdim + 2 * ndim + 1
    tPs, tQs, tMuHat, ee = load_data(vars, dev)
    treat_idcs = [i for i, t in enumerate(years) if 2001 <= t <= 2008]
    fixedflows = False
    netflow = create_netflow(ee, tMuHat, tPs, tQs, treat_idcs, train0, fixed=fixedflows)
    masks = (jax.device_put(jnp.asarray(maskc, dtype=bool), dev),
             jax.device_put(jnp.asarray(mask0, dtype=bool), dev))

    load = not noload
    if load:
        repo = "StefanHubner/DutchDivorce"
        f = lambda n: hf_hub_download(
            repo_id = repo,
            filename = n + current + (".eqx" if n == "xi" else ".npz"),
            cache_dir = ".hfcache"
        )
        with np.load(f("theta")) as checkpoint:
            theta = jax.device_put(jnp.asarray(checkpoint["theta"]), dev)
    else:
        theta0 = thetadim * [0.1] if theta0 is None else theta0
        theta = jax.device_put(jnp.asarray(theta0, dtype=jnp.float32), dev)

    network = SinkhornGeneric(net_tau, ndim, outdim, netdim, key=network_key)
    if load:
        network = eqx.tree_deserialise_leaves(f("xi"), network)
    xi = network.with_params(jax.device_put(network.params, dev))
    ng = 2**14 # 16,384 Dirichlet state draws per inner update
    xihat, thetahat, losshat = xi, theta, float("inf")

    optim = NelderMeadOptimizer([theta], lr = 0.3)
    num_epochs = 1000

    if train:

        closure, add_outputs = create_closure(xi, theta,
                                              tPs, tQs, tMuHat, netflow,
                                              ng, dev, tau, scale, masks,
                                              treat_idcs, years, optim,
                                              CF.None_, train0, beta, key=key)
        np.set_printoptions(precision=5, suppress=True)

        columns = (['loss', 'l'] + [f'theta{i}' for i in range(thetadim)]
                   + ['inner_min_step', 'inner_steps'])
        history = pd.DataFrame(index=np.arange(1, num_epochs+1),
                               columns=columns)

        hfpath = "./hfdd/"
        best_epoch = 0
        for epoch in range(1, num_epochs + 1):
            # 2026-10-02: Compare this step's evaluated pair with the separately saved best.
            add_outputs[6] = None
            add_outputs[7] = 0.0
            add_outputs[8] = None
            optim.step(closure)
            evaluated = add_outputs[6]
            if evaluated is None:
                # Do not pair a retained simplex score with unrelated/invalid outputs.
                curloss, l = np.nan, np.nan
                par = np.full(thetadim, np.nan)
            else:
                curloss, eval_theta, eval_xi, outputs = evaluated
                mush, muss, l, conds, margs = outputs
                cond_m_hat, cond_m_star, cond_f_hat, cond_f_star = conds
                par = np.asarray(eval_theta).flatten()

            inner_diagnostics = add_outputs[8] or {}
            inner_min_step = inner_diagnostics.get('inner_min_step')
            inner_steps = inner_diagnostics.get('inner_steps')

            improved = evaluated is not None and curloss < losshat
            if improved:
                losshat = curloss
                thetahat = eval_theta
                xihat = eval_xi
                best_epoch = epoch
                np.savez(hfpath + "theta" + current + ".npz", theta=np.asarray(thetahat))
                eqx.tree_serialise_leaves(hfpath + "xi" + current + ".eqx", xihat)

            diffs = (0.5 * (cond_m_star - cond_m_hat) + 0.5 * (cond_f_star - cond_f_hat)
                     if improved else None)
            print(format_training_step(
                epoch, num_epochs, curloss if evaluated is not None else None,
                losshat if best_epoch else None, float(l) if evaluated is not None else None,
                par if evaluated is not None else None, thetahat if best_epoch else None,
                add_outputs[7], conditional_diffs=diffs,
                steps_without_improvement=epoch - best_epoch, patience=patience,
                inner_min_step=inner_min_step, inner_steps=inner_steps), flush=True)

            history.loc[epoch] = ([curloss, float(l)] + par.tolist()
                                  + [inner_min_step or np.nan, inner_steps or np.nan])
            history.iloc[:epoch].to_csv(figures / 'training_history.csv')

            if not best_epoch:
                raise RuntimeError("No finite objective value in the first outer step; no new checkpoint saved.")
            if patience and epoch - best_epoch >= patience:
                print(f"Stopped after {patience} outer steps without a new best. "
                      f"Saved best is from step {best_epoch}.", flush=True)
                break
        else:
            print(f"Reached the {num_epochs}-step limit. Saved best is from step {best_epoch}.", flush=True)


    if not train:

        cpu = jax.devices("cpu")[0]
        xi = xihat.with_params(jax.device_put(xihat.params, cpu))
        xi = xi.eval()
        theta = jax.device_put(thetahat, cpu)
        tPs, tQs = jax.device_put(tPs, cpu), jax.device_put(tQs, cpu)
        mu_hat = jax.device_put(tMuHat, cpu)
        ee = jax.device_put(ee, cpu)
        masks = jax.device_put(masks, cpu)
        netflow = create_netflow(ee, mu_hat, tPs, tQs, treat_idcs, train0, fixed=fixedflows)

        # 2026-10-12: Keep the reporting workflow here; preparation and display details live in reporting.py.
        # 2026-10-13: Keep shorter fixed-flow windows; proportional flows use the full horizon.
        beyond = {"MScal": 12, "MScaltrend": 16}.get(current, 20) if fixedflows else 20
        scenarios = {"CFF": CF.None_, "CF0": CF.HighCost, "CF1": CF.LowCost}
        welfare_measures = [("flow", 0, "", "Welfare")]
        if not fixedflows:
            welfare_measures.append(("discounted", 1000, "_discounted", "Discounted welfare"))
        results_by_welfare = {}
        for measure, horizon, _, _ in welfare_measures:
            results_by_welfare[measure] = {
                name: load_mus(xi, theta, tPs, tQs, mu_hat, netflow,
                               ng, "cpu", tau, scale, masks,
                               treat_idcs, years, cf, train0, beyond, beta,
                               welfare_horizon=horizon)
                for name, cf in scenarios.items()}

        frames_by_welfare = {measure: prepare_plot_frames(result, years, treat_idcs, train0, beyond)
                             for measure, result in results_by_welfare.items()}
        results, frames = results_by_welfare["flow"], frames_by_welfare["flow"]
        conditional = frames["conditional"].iloc[1:, :]
        fitted = prepare_fit_frame(results["CFF"], years, treat_idcs, train0, masks).iloc[1:, :]
        # 2026-10-12: Prefix plot and CSV exports with their specification.
        for sex in ("M", "F"):
            save_plot(plot_cf_grid, conditional, figures / f"{current}_{sex}_cf1_grid.pdf",
                      sex=sex, dim=(2 * (ndim - 1), 2))
            save_plot(plot_estimator_grid, fitted, figures / f"{current}_star_hat_{sex}_grid.pdf",
                      sex=sex, scenario="CFF", nrows=2 * (ndim - 1))

        save_plot(plot_cf_grid, frames["joint"].iloc[1:, :], figures / f"{current}_joint1_cf1_grid.pdf",
                  csv=True, sex="both", dim=(3, ndim), beyond=beyond,
                  forecast_start=years.stop)
        save_plot(plot_margin_counterfactuals, frames["marginal"], figures / f"{current}_margins_CF.pdf",
                  csv=True, estimator="star", scenarios=("CFF", "CF1"),
                  beyond=beyond, forecast_start=years.stop)
        # 2026-10-22: Keep contemporaneous welfare and export discounted path welfare separately.
        for measure, _, suffix, title in welfare_measures:
            welfare = frames_by_welfare[measure]["welfare"].rename(columns={"Welfare": title}, level="state")
            save_plot(plot_cf_grid, welfare, figures / f"{current}_V{suffix}_cf.pdf",
                      csv=True, sex="both", dim=(1, 1), beyond=beyond,
                      forecast_start=years.stop)

        if matchingplot:
            return

        pre = range(0, treat_idcs[0])
        post = range(treat_idcs[-1] + 1, len(mu_hat))
        transitions = overallPQ(tPs, tQs, int(train0)*len(pre),
                                len(treat_idcs), len(post))
        dt = 1.0 / (years[-1] - years[0])
        controls = sandbox_controls(vars, years)
        mu_star = results["CFF"][1]
        sandbox, process, raw = st.tabs(["Sandbox", "Matched Processes", "Raw"])

        with sandbox:
            show_sandbox(xi, theta, tau, ndim, controls, transitions, netflow, dt)

        with process:
            myears = np.asarray([train0 for _ in range(treat_idcs[0])]
                                + [True for _ in years[treat_idcs[0]:]])
            ys = np.asarray(years)[myears]
            fig, ax1, ax2 = matched_process_plot(mu_hat[myears, :, :],
                                                 mu_star[:len(years)][myears, :, :],
                                                 ys, controls["cells"],
                                                 controls["couples"], controls["singles"],
                                                 policy_years=(years[treat_idcs[0]],
                                                               years[treat_idcs[-1]]))
            st.pyplot(fig, use_container_width=False)

        with raw:
            show_raw_matches(mu_hat, mu_star[:len(years)], years,
                             controls["hdmu"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="options")
    parser.add_argument("--spec",
                        type=str,
                        default="MS",
                        help="Specification to estimate or plot.")
    parser.add_argument("--train",
                        action="store_true",
                        help="Set to train the network.")
    parser.add_argument("--noload",
                        action="store_true",
                        help="Start from fresh coefficients and network weights.")
    parser.add_argument("--neldermead",
                        action="store_true", default=True,
                        help="Use Nelder-Mead for outer estimation (the default).")
    parser.add_argument("--matchingplot",
                        action="store_true",
                        help="Run this script only to generate plots.")
    args = parser.parse_args()
    try:
        main(args.train, args.noload, args.neldermead, args.matchingplot)
    except KeyboardInterrupt:
        print("\nStopped by Ctrl-C.", flush=True)
