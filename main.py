#!/usr/bin/env python

import argparse
import torch
import pandas as pd
import numpy as np
import streamlit as st
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from copy import deepcopy

from dynamicmatching import create_closure, overallPQ
from dynamicmatching import tauMcal, tauMS, tauMStri, tauMScal, tauMStrend, tauKMS, masksM, masksMS, masksKMS, TermColours, CF
from dynamicmatching import scaleMcal, scaleMScal, scaleMScaltrend, tauMScaltrend
from dynamicmatching import matched_process_plot, plot_cf_grid, plot_estimator_grid, plot_margin_counterfactuals
from dynamicmatching import SinkhornGeneric
from dynamicmatching import NelderMeadOptimizer
from dynamicmatching.reporting import load_mus, prepare_plot_frames, save_plot
from dynamicmatching.reporting import sandbox_controls, show_sandbox, show_raw_matches


st.set_page_config(page_title="Dynamic Matching")

# for debug
# noload, lbfgs, neldermead, ng0 = True, False, True, 128


@st.cache_resource
def load_data(name, dev):
    # 2026-10-02: Reuse HF_TOKEN or the cached login without prompting again.
    data = load_dataset("StefanHubner/DivorceData", token=True)[name]
    tPs = torch.tensor(data["p"][0], device=dev)
    tQs = torch.tensor(data["q"][0], device=dev)
    tMuHat = torch.tensor(data["couplings"][0], device=dev)
    ee = torch.tensor(data["entryexit"][0], device=dev)
    return tPs, tQs, tMuHat, ee


def main(train = False, noload = False, lbfgs = False,
         neldermead = False, matchingplot = False):

    torch.set_printoptions(precision=4, sci_mode=False)
    if torch.cuda.is_available() and not matchingplot:
        dev = "cuda"
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("Memory Allocated {}, memory reserved {}".format(torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
    else:
        dev = "cpu"
    torch.autograd.set_detect_anomaly(True)

    # Define specification and load appropriate dataset
    # outdim = par dim + # of single types + value vct
    # (name, state dim, net class, masks, basis, par dim, ys, train0, name)
    current = args.spec
    tpl = lambda a: (a, a)
    scale1 = lambda d: lambda _, dev: tpl(torch.ones(d, device=dev))

    spec  = { "Mcal1":
                ("M", 2, masksM, tauMcal, scale1(2), 2, None,
                 range(1999, 2021), False),
              "Mcal":
                ("M", 2, masksM, tauMcal, scaleMcal, 2+1, None,
                 range(1999, 2021), False),
              "MS":
                ("MS", 3, masksMS, tauMS, scale1(3), 10, None,
                 range(1999, 2021), False),
              "MStrend":
                ("MS", 3, masksMS, tauMStrend, scale1(3), 14, None,
                 range(1999, 2021), False),
              "MScal1":
                ("MS", 3, masksMS, tauMScal, scale1(3), 5, None,
                 range(1999, 2021), False),
              "MScal":
                # 2026-10-02: One fixed reference scale leaves two estimated type scales.
                ("MS", 3, masksMS, tauMScal, scaleMScal, 5+2, None,
                 range(1999, 2021), False),
              "MScalold":
                # 2026-10-02: Retain the legacy eight-parameter model with three free scales.
                ("MS", 3, masksMS, tauMScal,
                 lambda par, dev: tpl(torch.exp(par[5:])), 5+3, None,
                 range(1999, 2021), False),
              "MScaltrend":
                # 2026-10-06: Fresh coefficients and zero initial slopes; all trends remain free.
                ("MS", 3, masksMS, tauMScaltrend, scaleMScaltrend, 10+2,
                 5 * [0.1] + 5 * [0.0] + 2 * [0.1],
                 range(1999, 2021), False),
              "MScaltrend2":
                ("MS", 3, masksMS, tauMScaltrend, scaleMScaltrend, 10+2, None,
                 range(1999, 2021), False),
              "MStri":
                ("MS", 3, masksMS, tauMStri, scale1(3), 8, None,
                 range(1999, 2021), False),
              "KMS":
                ("KMS", 8, masksKMS, tauKMS, scale1(8), 12, None,
                 range(1999, 2021), False),
              "KMSclosed":
                ("KMS", 8, masksKMS, tauKMS, scale1(8), 12, None,
                 range(1999, 2021), False)
             }[current]
    vars, ndim, (maskc, mask0), tau, scale, thetadim, theta0, years, train0 = spec
    outdim = thetadim + 2 * ndim + 1
    tPs, tQs, tMuHat, ee = load_data(vars, dev)
    netflow = (ee * torch.tensor([1,-1], device = dev).unsqueeze(0)).sum(1)
    masks = (torch.tensor(maskc, dtype=torch.bool, device=dev),
             torch.tensor(mask0, dtype=torch.bool, device=dev))

    load = not noload
    if load:
        repo = "StefanHubner/DutchDivorce"
        f = lambda n: hf_hub_download(
            repo_id = repo,
            filename = n + ("MScal" if current == "MScalold" else current) + ".pt",
            # 2026-10-02: Pin legacy files so future MScal checkpoints cannot replace them.
            revision = "ba7727a7e639833895967f9fa4241b385bbcde91" if current == "MScalold" else None,
            cache_dir = ".hfcache"
        )
        theta = torch.load(f("theta"),
                            weights_only = False, map_location=torch.device(dev))
        xi_sd = torch.load(f("xi"),
                            weights_only = False, map_location=torch.device(dev))
    else:
        theta0 = thetadim * [0.1] if theta0 is None else theta0
        theta = torch.tensor(theta0, dtype=torch.float32,
                             device=dev, requires_grad = True)

    network = SinkhornGeneric(tau, ndim, outdim, thetadim)
    if load:
        network.load_state_dict(xi_sd) # mutable operation
    xi = network.to(dev)
    ng = 2**12 # max 2**19 number of draws (uniform gridpoints)
    treat_idcs = [i for i, t in enumerate(years) if 2001 <= t <= 2008]
    xihat, thetahat, losshat = deepcopy(xi), theta.detach().clone(), 10e30

    if lbfgs:
        optim = torch.optim.LBFGS([theta],
                                  lr=.1, max_iter=100,
                                  line_search_fn = 'strong_wolfe')
        num_epochs = 1
    elif neldermead:
        optim = NelderMeadOptimizer([theta], lr = 0.3)
        num_epochs = 1000
    else:
        optim = torch.optim.Adam([theta], lr = .1)
        num_epochs = 2000

    if train:

        closure, add_outputs = create_closure(xi, theta,
                                              tPs, tQs, tMuHat, netflow,
                                              ng, dev, tau, scale, masks,
                                              treat_idcs, years, optim,
                                              CF.None_, train0, not neldermead)
        best_evaluation = None
        # 2026-10-02: Snapshot each evaluated pair before the outer optimizer changes theta or xi.
        def tracked_closure():
            nonlocal best_evaluation
            loss = closure()
            value = loss.detach().item()
            if value < (best_evaluation[0] if best_evaluation else float("inf")):
                best_evaluation = (value, theta.detach().clone(),
                                   deepcopy(xi), deepcopy(add_outputs))
            return loss
        step_closure = tracked_closure
        torch.set_printoptions(precision = 5, sci_mode=False)

        columns = ['loss', 'l'] + [f'theta{i}' for i in range(thetadim)]
        history = pd.DataFrame(index=np.arange(1, num_epochs+1),
                               columns=columns)

        hfpath = "./hfdd/"
        for epoch in range(1, num_epochs + 1):
            # 2026-10-02: Compare this step's evaluated pair with the separately saved best.
            best_evaluation = None
            loss = optim.step(step_closure)
            curloss, eval_theta, eval_xi, outputs = best_evaluation or (
                loss.cpu().detach().clone().item(), theta, xi, add_outputs)
            mush, muss, l, conds, margs = outputs
            cond_m_hat, cond_m_star, cond_f_hat, cond_f_star = conds
            par = eval_theta.cpu().detach().numpy().flatten()
            print(f"theta_t (best trial this step): {TermColours.CYAN}{par}{TermColours.RESET}")
            if curloss < losshat:
                losshat = deepcopy(curloss)  # force value assignment
                thetahat = eval_theta.detach().clone()
                xihat = eval_xi if best_evaluation is not None else deepcopy(eval_xi)
                print("Saving tensors")
                torch.save(thetahat, hfpath + "theta" + current + ".pt")
                torch.save(xihat.state_dict(),
                           hfpath + "xi" + current + ".pt")
            else:
                print("No improvement: step loss {}, best loss {}".format(curloss, losshat))

            perc = int((epoch / num_epochs) * 100)
            diffs = 0.5 * (cond_m_star - cond_f_hat) + 0.5 * (cond_f_star - cond_m_hat)
            print(f"{TermColours.BRIGHT_RED}{perc}% : best loss {losshat:.4f} : \
                    best theta {thetahat}: \r\n \
                    {TermColours.GREEN}{diffs} \
                    {TermColours.RESET}",
                  end='\t', flush=True)

            history.loc[epoch] = [curloss, l.item()] + par.tolist()
            history.iloc[:epoch].to_csv('training_history.csv')

        print(thetahat)
        print("Done.")


    if not train:

        xi = xihat.cpu()
        xi.eval()
        theta = thetahat.cpu()
        tPs, tQs = tPs.cpu(), tQs.cpu()
        mu_hat = tMuHat.cpu()

        # 2026-10-12: Keep the reporting workflow here; preparation and display details live in reporting.py.
        beyond = 20
        scenarios = {"CFF": CF.None_, "CF0": CF.HighCost, "CF1": CF.LowCost}
        results = {}
        for name, cf in scenarios.items():
            results[name] = load_mus(xi, theta, tPs, tQs, mu_hat, netflow,
                                     ng, "cpu", tau, scale, masks,
                                     treat_idcs, years, cf, train0, beyond)

        frames = prepare_plot_frames(results, years, treat_idcs, train0, beyond)
        conditional = frames["conditional"].iloc[1:, :]
        for sex in ("M", "F"):
            save_plot(plot_cf_grid, conditional, f"{sex}_cf1_grid.pdf", sex=sex)
            save_plot(plot_estimator_grid, conditional, f"star_hat_{sex}_grid.pdf",
                      sex=sex, scenario="CFF")

        save_plot(plot_cf_grid, frames["joint"].iloc[1:, :], "joint1_cf1_grid.pdf",
                  csv=True, sex="both", dim=(3, 3), beyond=beyond,
                  forecast_start=years.stop)
        save_plot(plot_margin_counterfactuals, frames["marginal"], "margins_CF.pdf",
                  csv=True, estimator="star", scenarios=("CFF", "CF1"),
                  beyond=beyond, forecast_start=years.stop)
        save_plot(plot_cf_grid, frames["welfare"], "V_cf.pdf",
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
            myears = torch.tensor([train0 for _ in range(treat_idcs[0])]
                                + [True for _ in years[treat_idcs[0]:]])
            ys = torch.tensor(years)[myears].numpy()
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
                        help="Specification, one of {'Mtrend', 'MS'}.")
    parser.add_argument("--train",
                        action="store_true",
                        help="Set to train the network.")
    parser.add_argument("--noload",
                        action="store_true",
                        help="Set to overwrite pars from HuggingFace.")
    parser.add_argument("--lbfgs",
                        action="store_true",
                        help="Use LBFGS optim instead of Adam for outer.")
    parser.add_argument("--neldermead",
                        action="store_true",
                        help="Use Nelder-Mead optim instead of Adam for outer.")
    parser.add_argument("--matchingplot",
                        action="store_true",
                        help="Run this script only to generate plots.")
    args = parser.parse_args()
    main(args.train, args.noload, args.lbfgs, args.neldermead, args.matchingplot)
