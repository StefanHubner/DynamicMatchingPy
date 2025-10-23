#!/usr/bin/env python

import argparse
import io
import torch
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datasets import load_dataset
from huggingface_hub import login, whoami
from torchviz import make_dot
from PIL import Image

from dynamicmatching.bellman import match_moments, create_closure
from dynamicmatching.helpers import tauMflex, tauKMsimple, tauMproto, tauMtrend, masksM, masksKM, masksMproto, TermColours, CF
from dynamicmatching.graphs import matched_process_plot, create_heatmap, svg_to_data_url
from dynamicmatching.bellman import minimise_inner, choices
from dynamicmatching.deeplearning import SinkhornM, SinkhornKMsimple, SinkhornMproto, masked_log
from dynamicmatching.neldermead import NelderMeadOptimizer

st.set_page_config(page_title = "Dynamic Matching")

# for debug
# noload, lbfgs, neldermead, ng0 = True, False, True, 128

@st.cache_resource
def load_data(name, dev):
    import os
    token = os.environ.get("HF_TOKEN") # I think HF_TOKEN is used by default
    login(token=token, add_to_git_credential=True) 
    data = load_dataset("StefanHubner/DivorceData")[name]
    tPs = torch.tensor(data["p"][0], device = dev)
    tQs = torch.tensor(data["q"][0], device = dev)
    tMuHat = torch.tensor(data["couplings"][0], device = dev)
    return tPs, tQs, tMuHat

def load_mus(xi, theta, tPs, tQs, muh, ng, dev, tau,
             masks, tis, years, cf, train0):
    _, muh1, mus, _ = match_moments(xi, theta,
                                    tPs, tQs, muh, ng,
                                    dev, tau, masks, tis, years,
                                    skiptrain = True, cf = cf,
                                    train0 = train0)
    return muh1, mus

def main(train = False, noload = False, lbfgs = False,
         neldermead = False, matchingplot = True):

    torch.set_printoptions(precision=4, sci_mode=False)
    if torch.cuda.is_available():
        dev = "cuda"
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("Memory Allocated {}, memory reserved {}".format(torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
    else:
        dev = "cpu"
    torch.autograd.set_detect_anomaly(True)

    #Define specification and load appropriate dataset
    # outdim = par dim + share of unrest. singles for both sx's + value fct
    # (name, state dim, net definition, outdim, masks, basis, parameter dim)
    #spec = ("M", 3, SinkhornM, 9, masksM, tauMflex, 4)
    #spec = ("KM", 6, SinkhornKMsimple, 17, masksKM, tauKMsimple, 8)
    #spec = ("M", 2, SinkhornMproto, 7, masksMproto, tauMproto, 2, range(1999, 2021), False, "M")
    spec = ("M", 2, SinkhornMproto, 10, masksMproto, tauMtrend, 5, range(1999, 2021), False, "Mtrend")
    vars, ndim, NN, outdim, (maskc, mask0), tau, thetadim, years, train0, current = spec
    # current = NN.__name__
    tPs, tQs, tMuHat = load_data(vars, dev)
    masks = (torch.tensor(maskc, dtype=torch.bool, device=dev),
             torch.tensor(mask0, dtype=torch.bool, device=dev))

    hfpath = "./hfdd/"
    load = not noload
    if load:
        theta = torch.load(hfpath + "theta" + current + ".pt",
                            weights_only = False, map_location=torch.device(dev))
        xi_sd = torch.load(hfpath + "xi" + current + ".pt",
                            weights_only = False, map_location=torch.device(dev))
    else:
        theta = torch.tensor(thetadim * [0.1], dtype=torch.float32,
                             device=dev, requires_grad = True)

    network = NN(tau, ndim, outdim)
    if load:
        network.load_state_dict(xi_sd)
    xi = network.to(dev)

    if lbfgs:
        optim = torch.optim.LBFGS([theta],
                                  lr=.1, max_iter=100,
                                  line_search_fn = 'strong_wolfe')
        num_epochs = 1
    elif neldermead:
        optim = NelderMeadOptimizer([theta], lr = 1.0)
        num_epochs = 1000
    else:
        optim = torch.optim.Adam([theta], lr = .1)
        num_epochs = 2000

    ng = 2**19 # max 2**19 number of draws (uniform gridpoints)
    treat_idcs = [i for i,t in enumerate(years) if 2001 <= t <= 2008]
    xihat, thetahat = xi, theta

    if train:

        closure, add_outputs = create_closure(xi, theta,
                                              tPs, tQs, tMuHat, ng,
                                              dev, tau, masks,
                                              treat_idcs, years, optim,
                                              CF.None_, train0, not neldermead)
        torch.set_printoptions(precision = 5, sci_mode=False)
        columns = ['loss', 'l']
        for i in range(theta.shape[0]):
            columns.append(f'theta{i}')
        history = pd.DataFrame(index=np.arange(1, num_epochs+1),
                               columns=columns)

        losshat = torch.tensor(10e30, device=dev)
        for epoch in range(1, num_epochs + 1):
            loss = optim.step(closure)
            mush, muss, l = add_outputs
            record = [loss.item(), l.item()]
            par = theta.cpu().detach().numpy().flatten()
            print("theta_t: {}".format(par))
            if loss < losshat:
                losshat = loss
                xihat, thetahat = xi, theta
                print("Saving tensors")
                torch.save(thetahat, hfpath + "theta" + current + ".pt")
                torch.save(xihat.state_dict(),
                           hfpath + "xi" + current + ".pt")
            else:
                print("previous loss: {} < loss: {}".format(losshat, loss))
            record.extend(par)
            history.loc[epoch] = record
            if True: # loss <= losshat:
                perc = int((epoch / num_epochs) * 100)
                muss = muss.cpu().detach().numpy()
                muhat = tMuHat.cpu().detach().numpy()
                print(f"{TermColours.BRIGHT_RED}{perc}% : {loss.item():.4f} : \
                        {thetahat}: \
                        {TermColours.GREEN}{muss - muhat} \
                        {TermColours.RESET}",
                      end='\t', flush=True)
            history.to_csv('training_history.csv')

        print(thetahat)
        print("Done.")


    if not train:

        xi = xihat.cpu()
        xi.eval()
        theta = thetahat.cpu()
        tPs, tQs = tPs.cpu(), tQs.cpu()
        mu_hat = tMuHat.cpu()
        sandbox, process, raw, net = st.tabs(["Sandbox",
                                              "Matched Processes",
                                              "Raw",
                                              "Network"])
        _, mu_star = load_mus(xi, theta, tPs, tQs, mu_hat, ng,
                              "cpu", tau, masks, treat_idcs, years,
                              cf = CF.None_, train0 = train0)

        s = st.session_state
        if 'currentyear' not in s: s.currentyear = years[0]
        if 't' not in s: s.t = torch.tensor(0.0).view(1,1)
        def update_year():
            s.t = torch.tensor((s.currentyear - years[0]) / (years[-1] - years[0])).view(1,1)

        year = st.sidebar.slider('Year', years[0], years[-1], step=1,
                                  key='currentyear', on_change=update_year)
        if vars == "KM":
            if 'mn' not in s: s.mn = 0.20
            if 'me' not in s: s.me = 0.03
            if 'zn' not in s: s.zn = 0.16
            if 'ze' not in s: s.ze = 0.03
            if 'pkc' not in s: s.pkc = 0.7
            if 'pknc' not in s: s.pknc = 0.1
            def update_mc():
                s.mc = 0.5 - s.mn - s.me

            def update_zc():
                s.zc = s.mc

            def update_ze():
                s.ze = 0.5 - s.zn - s.zc

            mn = st.sidebar.slider('$M_n$', 0.0, 0.5, step=0.01,
                                   key='mn', on_change=update_mc)
            me = st.sidebar.slider('$M_e$', 0.0, 0.5, step=0.01,
                                   key='me', on_change=update_mc)
            update_mc()
            st.sidebar.slider('$M_c$', 0.0, 0.5, step=0.01,
                              key='mc', disabled=True, on_change=update_zc)
            update_zc()
            zn = st.sidebar.slider('$F_n$', 0.0, 0.5, step=0.01,
                                   key='zn', on_change=update_ze)
            update_ze()
            st.sidebar.slider('$F_e$', 0.0, 0.5, step=0.01,
                              key='ze', disabled=True)
            st.sidebar.slider('$F_c$', 0.0, 0.5, step=0.01,
                              key='zc', disabled=True)

            st.sidebar.slider('$P(k|c)$', 0.0, 1.0, step=0.01,
                              key='pkc', disabled=False)
            st.sidebar.slider('$P(k|\\neg c)$', 0.0, 1.0, step=0.01,
                              key='pknc', disabled=False)
        elif vars == "Mproto" or vars == "M":
            if 'mu' not in s: s.mu = 0.25
            if 'fu' not in s: s.fu = 0.25

            def update_mc():
                s.mc = 0.5 - s.mu
            def update_fc():
                s.fc = 0.5 - s.fu

            mu = st.sidebar.slider('$M_u$', 0.0, 0.5, step=0.01,
                                   key='mu', on_change=update_mc)
            update_mc()
            fu = st.sidebar.slider('$F_u$', 0.0, 0.5, step=0.01,
                                   key='fu', on_change=update_fc)
            update_fc()
            ss = lambda d: torch.tensor([[s.mu, s.mc, s.fu, s.fc, s.t, d]], device="cpu")
            hdmu = ['u', 'c', '0']
            hds = ['M_u', 'M_c', 'F_u', 'F_c', 't', 'd']
            cells = {"uu": (0,0), "cc": (1,1), "u0": (0,2), "0u": (2,0),
                     "c0": (1,2), "0c": (2,1)}
            couples = ["uu", "cc"]
            singles = ["u0", "0u", "c0", "0c"]

        if vars == "KM":
            pk, pz = s.pknc, 1 - s.pknc
            ss = torch.tensor([[mn*pz, me*pz, s.mc*(1-s.pkc),
                                mn*pk, me*pk, s.mc*s.pkc,
                                zn*pz, s.ze*pz, s.zc*(1-s.pkc),
                                zn*pk, s.ze*pk, s.zc*s.pkc]],
                              device="cpu")
            #ss = torch.kron(torch.ones(1, 2, device = "cpu"), ss)
            #hdmu = ['zn','kn', 'ze', 'ke','zc', 'kc','0']
            hdmu = ['zn','ze', 'zc', 'kn','ke', 'kc','0']
            hds = ['M_{zn}','M_{ze}','M_{zc}','M_{kn}',
                   'M_{ke}','M_{kc}','F_{zn}','F_{ze}',
                   'F_{zc}','F_{kn}','F_{ke}','F_{kc}']
            cells = {"znzn":  (0,0), "zeze": (1,1,), "zczc": (2,2),
                     "kczc": (5,2), "zckc": (2,5), "kckc": (5,5),
                     "zn0": (0,6), "0zn": (6,0),
                     "knkn": (3,3), "keke": (4,4)}
            couples = ["znzn", "zeze", "zczc", "knkn", "keke", "kckc"]
            singles = ["zn0", "0zn"]

        torch0 = torch.tensor(0.0, device="cpu").view(1,1)
        mus0, v0 = xi(ss(0))
        ssnext0 = choices(mus0, s.t, torch0, tPs[2], tQs[2], "cpu")
        mus1, v1 = xi(ss(1))
        ssnext1 = choices(mus0, s.t, torch0 + 1, tPs[1], tQs[1], "cpu")
        with sandbox:
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
                        st.write(("$\\underbrace{{{:.3f}}}_{{P(C_{{\\neg m}})}} + " +
                                  " \\underbrace{{{:.3f}}}_{{P(C_m)}} + " +
                                  " \\underbrace{{{:.3f}}}_{{P(M_0)}} + " +
                                  " \\underbrace{{{:.3f}}}_{{P(F_0)}} = " +
                                  " {{{:.3f}}} $").format(
                                        pcnc, pcc, pm, pf,
                                        pcnc + pcc + pm + pf))
                        df = pd.DataFrame(mu[i].detach().numpy(),
                                          columns = hdmu, index = hdmu)
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
                        help="Run this script in streamlit.")
    args = parser.parse_args()
    main(args.train, args.noload, args.lbfgs, args.neldermead, args.matchingplot)
