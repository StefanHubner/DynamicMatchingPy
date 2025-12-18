#!/usr/bin/env python

import argparse
import os
import torch
import pandas as pd
import numpy as np
import streamlit as st
from datasets import load_dataset
from huggingface_hub import login, hf_hub_download

from dynamicmatching import match_moments, create_closure, choices, overallPQ
from dynamicmatching import tauMcal, tauMS, tauMStri, tauMScal, tauMStrend, tauKMS, masksM, masksMS, masksKMS, TermColours, CF
from dynamicmatching import scaleMcal, scaleMScal, scaleMScaltrend, tauMScaltrend
from dynamicmatching import matched_process_plot, create_heatmap, plot_cf_grid, plot_estimator_grid, plot_margin_counterfactuals
from dynamicmatching import SinkhornGeneric
from dynamicmatching import NelderMeadOptimizer


st.set_page_config(page_title="Dynamic Matching")

# for debug
# noload, lbfgs, neldermead, ng0 = True, False, True, 128


@st.cache_resource
def load_data(name, dev):
    token = os.environ.get("HF_TOKEN")  # HF_TOKEN is used by default
    login(token=token, add_to_git_credential=True)
    data = load_dataset("StefanHubner/DivorceData")[name]
    tPs = torch.tensor(data["p"][0], device=dev)
    tQs = torch.tensor(data["q"][0], device=dev)
    tMuHat = torch.tensor(data["couplings"][0], device=dev)
    ee = torch.tensor(data["entryexit"][0], device=dev)
    return tPs, tQs, tMuHat, ee


def load_mus(xi, theta, tPs, tQs, muh, netflow, ng, dev, tau,
             scale, masks, tis, years, cf, train0):
    _, muh1, mus, _, conds, margs = match_moments(xi, theta,
                                    tPs, tQs, muh, netflow, ng,
                                    dev, tau, scale, masks, tis, years,
                                    skiptrain = True, cf = cf,
                                    train0 = train0)
    return muh1, mus, conds, margs

def main(train = False, noload = False, lbfgs = False,
         neldermead = False, matchingplot = False):

    torch.set_printoptions(precision=4, sci_mode=False)
    if torch.cuda.is_available():
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
                ("MS", 3, masksMS, tauMScal, scaleMScal, 5+2, None,
                 range(1999, 2021), False),
              "MScaltrend":
                ("MS", 3, masksMS, tauMScaltrend, scaleMScaltrend, 10+2,
                [-1.13, 1.64, 1.90, 1.47, 0.55, 0, 0, 0, 0, 0, -1.37, -1.26],
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
            filename = n + current + ".pt",
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
    treat_idcs = [i for i,t in enumerate(years) if 2001 <= t <= 2008]
    xihat, thetahat = xi, theta

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
        torch.set_printoptions(precision = 5, sci_mode=False)
        columns = ['loss', 'l']
        for i in range(theta.shape[0]):
            columns.append(f'theta{i}')
        history = pd.DataFrame(index=np.arange(1, num_epochs+1),
                               columns=columns)

        losshat = 10e30 #torch.tensor(10e30, device=dev)
        hfpath = "./hfdd/"
        for epoch in range(1, num_epochs + 1):
            loss = optim.step(closure)
            curloss = loss.item()
            mush, muss, l, conds, margs = add_outputs
            cond_m_hat, cond_m_star, cond_f_hat, cond_f_star = conds
            record = [curloss, l.item()]
            par = theta.cpu().detach().numpy().flatten()
            print("theta_t: {}".format(par))
            if curloss < losshat:
                losshat = curloss
                xihat, thetahat = xi, theta
                print("Saving tensors")
                torch.save(thetahat, hfpath + "theta" + current + ".pt")
                torch.save(xihat.state_dict(),
                           hfpath + "xi" + current + ".pt")
            else:
                print("previous loss: {} < loss: {}".format(losshat, curloss))
            record.extend(par)
            history.loc[epoch] = record
            if True: # loss <= losshat:
                perc = int((epoch / num_epochs) * 100)
                muss = muss.cpu().detach().numpy()
                muhat = tMuHat.cpu().detach().numpy()
                diffs = 0.5 * (cond_m_star - cond_f_hat) + 0.5 * (cond_f_star - cond_m_hat)
                print(f"{TermColours.BRIGHT_RED}{perc}% : {loss.item():.4f} : \
                        {thetahat}: \
                        {TermColours.GREEN}{diffs} \
                        {TermColours.RESET}",
                      end='\t', flush=True)
            history.iloc[:epoch].to_csv('training_history.csv')

        print(thetahat)
        print("Done.")


    if not train:

        xi = xihat.cpu()
        xi.eval()
        theta = thetahat.cpu()
        tPs, tQs = tPs.cpu(), tQs.cpu()
        pre = range(0, treat_idcs[0])
        post = range(treat_idcs[-1] + 1, len(tMuHat))
        transitions = overallPQ(tPs, tQs, int(train0)*len(pre),
                                len(treat_idcs), len(post))
        tP, tQ, tP0, tQ0, tP1, tQ1 = transitions
        mu_hat = tMuHat.cpu()
        sandbox, process, raw, net = st.tabs(["Sandbox",
                                              "Matched Processes",
                                              "Raw",
                                              "Network"])
        mu_stars  = {}
        condss = {}
        margss = {}
        df = pd.DataFrame()
        df1 = pd.DataFrame()

        condss_names = [("M", "hat"), ("M", "star"), ("F", "hat"), ("F", "star")]
        vars = [("U|U",   (0,0)), ("0|U",  (0,3)),
                ("CH|CH", (1,1)), ("CW|CH", (1,2)), ("0|CH", (1,3)),
                ("CH|CW", (2,1)), ("CW|CW", (2,2)), ("0|CW", (2,3))]
        for n, cf in zip(["CFF", "CF0", "CF1"], [CF.None_, CF.HighCost, CF.LowCost]):
            _, mu_stars[cf], condss[cf], margss[cf] = load_mus(xi, theta, tPs, tQs, 
                                                  mu_hat, netflow, ng,
                                                 "cpu", tau, scale, masks, treat_idcs, years,
                                                  cf = cf, train0 = train0)
            for ((s, e), t) in zip(condss_names, condss[cf]):
                for (cond, (i, j)) in vars:
                    df[(n, s, e, cond)] = t[:, i, j].detach().numpy()
            for ((s, e), t) in zip(condss_names, margss[cf]):
                for (mg, i) in zip(["U", "CH", "CW"], [0, 1, 2]):
                    df1[(n, s, e, mg)] = t[:, i].flatten().detach().numpy()
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=["scenario", "sex", "estimator", "state"])
        df.index = [f"{l}" for l in np.array(list(years))[pre if train0 else [] + treat_idcs + list(post)].tolist()]
        df1.columns = pd.MultiIndex.from_tuples(df1.columns, names=["scenario", "sex", "estimator", "state"])
        df1.index = [f"{l}" for l in np.array(list(years))[pre if train0 else [] + treat_idcs + list(post)].tolist()]

        # this doesn't account for outflow of divorce state
        mu_minus = mu_hat[treat_idcs[-1]]
        zero, one = torch.tensor([0.0, 0.42]), torch.tensor([1.0, 0.37])
        s_minus = torch.concatenate([mu_minus.sum(dim=1)[:-1], mu_minus.sum(dim=0)[:-1], one])
        mu_plus = mu_hat[post[0]]
        s_plus = torch.concatenate([mu_plus.sum(dim=1)[:-1], mu_plus.sum(dim=0)[:-1], zero])

        mustar_minus, _ = xi(s_minus.unsqueeze(0))
        mustar_plus, _ = xi(s_plus.unsqueeze(0))

        s_minus_star = choices(mustar_minus, one[1].view(1,1), one[0].view(1,1),
                              tP0, tQ0, tP1, tQ1, netflow, zero[1]-one[1], "cpu")
        s_plus_star = choices(mustar_plus, zero[1].view(1,1), zero[0].view(1,1),
                              tP0, tQ0, tP1, tQ1, netflow, zero[1]-one[1], "cpu")

        mustar_minus_next, _ = xi(s_minus_star)
        mustar_plus_next, _ = xi(s_plus_star)

        flow_minus = (mustar_minus_next - mustar_minus)[0, [1,2], 3].sum() + (mustar_minus_next - mustar_minus)[0, 3, [1,2]].sum()
        flow_plus = (mustar_plus_next - mustar_plus)[0, [1,2], 3].sum() + (mustar_plus_next - mustar_plus)[0, 3, [1,2]].sum()

        lambda_minus = flow_minus / mustar_minus[0, 1:3, 1:3].sum()
        lambda_plus = flow_plus / mustar_plus[0, 1:3, 1:3].sum()
        lambda_plus/lambda_minus
        # end attempt to further calibrate psi beyond log(1.1) which is theoretically correct

        df.to_csv("conditional_distr.csv")

        fig = plot_cf_grid(df.iloc[1:,:], sex="M")
        fig.savefig("M_cf1_grid.pdf", bbox_inches="tight")
        fig = plot_cf_grid(df.iloc[1:,:], sex="F")
        fig.savefig("F_cf1_grid.pdf", bbox_inches="tight")

        fig2 = plot_estimator_grid(df.iloc[1:,:], sex="M", scenario="CFF")
        fig2.savefig("star_hat_M_grid.pdf", bbox_inches="tight")
        fig2 = plot_estimator_grid(df.iloc[1:,:], sex="F", scenario="CFF")
        fig2.savefig("star_hat_F_grid.pdf", bbox_inches="tight")

        fig = plot_margin_counterfactuals(df1, estimator="star", scenarios=("CFF", "CF1"))
        fig.savefig("margins_CF.pdf", bbox_inches="tight")

        if matchingplot:
            return

        mu_star = mu_stars[CF.None_]
        cond_m_hat, cond_m_star, cond_f_hat, cond_f_star = condss[CF.None_]


        s = st.session_state
        if 'currentyear' not in s: s.currentyear = years[0]
        if 't' not in s: s.t = torch.tensor(0.0).view(1,1)
        dt = 1.0 / (years[-1] - years[0])
        def update_year():
            s.t = torch.tensor((s.currentyear - years[0]) * dt).view(1,1)

        year = st.sidebar.slider('Year', years[0], years[-1], step=1,
                                  key='currentyear', on_change=update_year)

        if 'mu' not in s: s.mu = 0.25
        if 'vu' not in s: s.vu = 0.25

        def update_c():
            s.mc = 0.5 - s.mu
            s.vc = 0.5 - s.vu

        mu = st.sidebar.slider('$M_{u}$', 0.0, 0.5, step=0.01,
                               key='mu', on_change=update_c)
        vu = st.sidebar.slider('$F_{u}$', 0.0, 0.5, step=0.01,
                               key='vu', on_change=update_c)
        update_c()

        if vars == "M":
            ss = lambda d: torch.tensor([[s.mu, s.mc, s.vu, s.vc, s.t, d]],
                                        device="cpu")
            hdmu = ['u', 'c', '0']
            hds = ['M_u', 'M_c', 'F_u', 'F_c', 't', 'd']
            cells = {"uu": (0,0), "cc": (1,1), "u0": (0,2), "0u": (2,0),
                     "c0": (1,2), "0c": (2,1)}
            couples = ["uu", "cc"]
            singles = ["u0", "0u", "c0", "0c"]


        if vars == "MS":
            if 'myu' not in s: s.myu = 0.5
            if 'mhc' not in s: s.mhc = 0.5
            if 'vyu' not in s: s.vyu = 0.5
            if 'vhc' not in s: s.vhc = 0.5

            def update_ou():
                s.mou = 1 - s.myu
                s.fou = 1 - s.vyu
            def update_wc():
                s.mwc = 1 - s.mhc
                s.fwc = 1 - s.vhc

            st.sidebar.slider('$M_{y|u}$', 0.0, 1.0, step=0.01,
                              key='myu', on_change=update_ou)
            update_ou()
            st.sidebar.slider('$M_{h|c}$', 0.0, 1.0, step=0.01,
                              key='mhc', on_change=update_wc)
            update_wc()

            update_c()
            st.sidebar.slider('$F_{y|u}$', 0.0, 1.0, step=0.01,
                              key='vyu', on_change=update_ou)
            update_ou()
            st.sidebar.slider('$F_{h|c}$', 0.0, 1.0, step=0.01,
                              key='vhc', on_change=update_wc)
            update_wc()

            ss = lambda d: torch.tensor([[
                               s.mu * s.myu, s.mu * (1 - s.myu),
                               s.mc * s.mhc, s.mc * (1 - s.mhc),
                               s.vu * s.vyu, s.vu * (1 - s.vyu),
                               s.vc * s.vhc, s.vc * (1 - s.vhc),
                               s.t, d]], device="cpu")

            hdmu = ['uy', 'uo', 'ch', 'cw', '0']
            hds = ['M_{uy}', 'M_{uo}', 'M_{ch}', 'M_{cw}',
                   'F_{uy}', 'F_{uu}', 'F_{ch}', 'F_{cw}', 't', 'd']

            cells = {"uyuy": (0,0), "uouo": (1,1),
                     "chch": (2,2), "cwcw": (3,3),
                     "uouy": (1,0), "cwch": (3,2),
                     "uy0": (0,4), "0uy": (4,0),
                     "uo0": (1,4), "0uo": (4,1),
                     "ch0": (2,4), "0ch": (4,2),
                     "cw0": (3,4), "0cw": (4,3)}

            couples = ["uyuy", "uouo", "uouy", "chch", "cwcw", "cwch"]
            singles = ["uy0", "0uy", "uo0", "0uo", "ch0", "0ch", "cw0", "0cw"]

        if vars == "KMS":
            hdmu = ['vuy', 'vuo', 'vcy', 'vco',
                    'kuh', 'kuw', 'kch', 'kcw', '0']
            hds = ['M_{vuy}', 'M_{vuo}', 'M_{vcy}', 'M_{vco}',
                   'M_{kuh}', 'M_{kuw}', 'M_{kch}', 'M_{kcw}',
                   'F_{vuy}', 'F_{vuo}', 'F_{vcy}', 'F_{vco}',
                   'F_{kuh}', 'F_{kuw}', 'F_{kch}', 'F_{kcw}',
                   't', 'd']

            cells = {"vuyvuy": (0,0), "vuovuo": (1,1),
                     "vchvch": (2,2), "vcwvcw": (3,3),
                     "kuykuy": (4,4), "kuokuo": (5,5),
                     "kuokuy": (5,4), "kchkch": (6,6),
                     "kcwkcw": (7,7), "kcwkch": (7,6),
                     "vuy0": (0,8), "0vuy": (8,0),
                     "vuo0": (1,8), "0vuo": (8,1),
                     "vch0": (2,8), "0vch": (8,2),
                     "vcw0": (3,8), "0vcw": (8,3),
                     "kuy0": (4,8), "0kuy": (8,4),
                     "kuo0": (5,8), "0kuo": (8,5),
                     "kch0": (6,8), "0kch": (8,6),
                     "kcw0": (7,8), "0kcw": (8,7)}

            couples = ["vuyvuy", "vuvouo", "vchvch", "vcwvcw", "vcwvch",
                       "kuykuy", "kukouo", "kchkch", "kcwkcw", "kcwkch"]
            singles = ["vuy0", "0vuy", "vuo0", "0vuo",
                       "vch0", "0vch", "vcw0", "0vcw",
                       "kuy0", "0kuy", "kuo0", "0kuo",
                       "kch0", "0kch", "kcw0", "0kcw"]

        torch0 = torch.tensor(0.0, device="cpu").view(1,1)
        mus0, v0 = xi(ss(0))
        ssnext0 = choices(mus0, s.t, torch0, tP0, tQ0, tP1, tQ1, netflow, dt, "cpu")
        mus1, v1 = xi(ss(1))
        ssnext1 = choices(mus0, s.t, torch0 + 1, tP0, tQ0, tP1, tQ1, netflow, dt, "cpu")
        with sandbox:
            ps = theta.detach().numpy()
            st.write("Parameters: " +
                     ", ".join(["{:.2f}".format(p) for p in ps]))

            cols = st.columns(2)
            columns_data = zip(cols, [torch0, torch0 + 1])
            for col, d in columns_data:
                with col:
                    st.subheader('$\\Phi_{{t}}({})$'.format(int(d.item())))
                    df = pd.DataFrame(
                        tau(theta, s.t, d, "cpu").detach().numpy(),
                        index=hdmu,
                        columns=hdmu
                    )
                    st.dataframe(df)

            st.subheader('$\\mu_1$ in %')
            fmarg = ss(1)[0, ndim:2*ndim].detach().numpy().tolist()
            mmarg = ss(1)[0, 0:ndim].detach().numpy().tolist()
            st.pyplot(create_heatmap(mus1[0]*100,
                       ["{:.1f}".format(f*100) for f in fmarg + [0]],
                       ["{:.1f}".format(m*100) for m in mmarg + [0]]
                                     , hdmu), use_container_width=False)
            columns_data = zip(cols, [ssnext0, ssnext1], [v0, v1], range(2))
            for col, ssnext, v, idx in columns_data:
                with col:
                    st.subheader('($M_{{t+1}}^*, F_{{t+1}}^*) $')
                    df = pd.DataFrame(
                        ssnext.detach().numpy(),
                        columns=hds
                    )
                    st.dataframe(df, hide_index=True)
                    st.subheader('$V$')
                    st.write(v.detach().numpy())

        with process:
            myears = torch.tensor([train0 for _ in range(treat_idcs[0])]
                                + [True for _ in years[treat_idcs[0]:]])
            ys = torch.tensor(years)[myears].numpy()
            fig, ax1, ax2 = matched_process_plot(mu_hat[myears,:,:],
                                                 mu_star[myears,:,:],
                                                 ys, cells, couples, singles)
            st.pyplot(fig, use_container_width=False)
            #fig.savefig("matched_processes.pdf", format="pdf",
            #            bbox_inches="tight")
        with raw:
            col1, col2 = st.columns(2)
            with col1:
                st.header("Observed: $\\widehat{\\mu}_t$")
            with col2:
                st.header("Model: $\\mu^*_t$")
            for (col, mu) in [(col1, mu_hat), (col2, mu_star)]:
                for i in range(mu.shape[0]):
                    if vars == "KM":
                        couples = mu[i,:-1,:-1].reshape(2, 3, 2, 3).sum(axis=(0,2))
                    else:
                        couples = mu[i,:-1,:-1]
                    with col:
                        pcnc = 2 * couples[:-1,:-1].sum()
                        pcc = 2 * couples[-1,-1].sum()
                        pf, pm = mu[i,-1,:].sum(), mu[i,:,-1].sum()
                        st.subheader("{}".format(years[i]))
                        #st.write(("$\\underbrace{{{:.3f}}}_{{P(C_{{\\neg m}})}} + " +
                        #          " \\underbrace{{{:.3f}}}_{{P(C_m)}} + " +
                        #          " \\underbrace{{{:.3f}}}_{{P(M_0)}} + " +
                        #          " \\underbrace{{{:.3f}}}_{{P(F_0)}} = " +
                        #          " {{{:.3f}}} $").format(
                        #                pcnc, pcc, pm, pf,
                        #                pcnc + pcc + pm + pf))
                        df = pd.DataFrame(mu[i].detach().numpy(),
                                          columns = hdmu, index = hdmu)
                        #mu[i,:-1,:] * tP[0,:,:].T # mass going to uy
                        #mu[i,:,:-1] * tQ[:,:,0] # mass going to uy
                        st.dataframe(df)
                        df = pd.DataFrame({
                            'M': mu[i,:-1,:].sum(axis=1).detach().numpy(),
                            'F': mu[i,:,:-1].sum(axis=0).detach().numpy()
                            }, index = hdmu[:-1])
                        st.dataframe(df)

        with net:
            st.header("Network 0")
            #dot = make_dot(xi(ss),
            #               params = dict(xi.named_parameters()),
            #               show_attrs=True, show_saved=True)
            #png_data = dot.pipe(format='png')
            #image = Image.open(io.BytesIO(png_data))
            #st.image(image, use_column_width=True)


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
