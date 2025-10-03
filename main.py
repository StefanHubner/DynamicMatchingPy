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
from dynamicmatching.helpers import tauMflex, tauKMsimple, tauMproto, masksM, masksKM, masksMproto, TermColours, CF
from dynamicmatching.graphs import matched_process_plot, create_heatmap, svg_to_data_url
from dynamicmatching.bellman import minimise_inner, choices
from dynamicmatching.deeplearning import SinkhornM, SinkhornKMsimple, SinkhornMproto, masked_log
from dynamicmatching.neldermead import NelderMeadOptimizer

st.set_page_config(page_title = "Dynamic Matching")


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

def load_mus(xi0, xi1, xi2, t0, t1, t2, tPs, tQs, muh, ng, dev, tau,
             masks, tis, cf, train0):
    _, muh1, mus, _, _, _ = match_moments(xi0, xi1, xi2, t0, t1, t2,
                                          tPs, tQs, muh, ng,
                                          dev, tau, masks, tis,
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
    spec = ("M", 2, SinkhornMproto, 7, masksMproto, tauMproto, 2, range(1999, 2021), False) # should be this, but used cached data 
    #spec = ("Mproto", 2, SinkhornMproto, 7, masksMproto, tauMproto, 2, range(1995, 2021), False)
    vars, ndim, NN, outdim, (maskc, mask0), tau, thetadim, years, train0 = spec
    current = NN.__name__
    tPs, tQs, tMuHat = load_data(vars, dev)

    masks = (torch.tensor(maskc, dtype=torch.bool, device=dev),
             torch.tensor(mask0, dtype=torch.bool, device=dev))

    hfpath = "./hfdd/"
    load = not noload
    if load:
        theta0 = torch.load(hfpath + "theta0" + current + ".pt",
                            weights_only = True, map_location=torch.device(dev))
        theta1 = torch.load(hfpath + "theta1" + current + ".pt",
                            weights_only = True, map_location=torch.device(dev))
        theta2 = torch.load(hfpath + "theta2" + current + ".pt",
                            weights_only = True, map_location=torch.device(dev))
        xi0_sd = torch.load(hfpath + "xi0" + current + ".pt",
                            weights_only = True, map_location=torch.device(dev))
        xi1_sd = torch.load(hfpath + "xi1" + current + ".pt",
                            weights_only = True, map_location=torch.device(dev))
        xi2_sd = torch.load(hfpath + "xi2" + current + ".pt",
                            weights_only = True, map_location=torch.device(dev))
    else:
        theta0 = torch.tensor(thetadim * [0.0], dtype=torch.float32,
                              device=dev, requires_grad = True)
        theta1 = torch.tensor(thetadim * [0.0], dtype=torch.float32,
                              device=dev, requires_grad = True)
        theta2 = torch.tensor(thetadim * [0.0], dtype=torch.float32,
                              device=dev, requires_grad = True)
        if vars == "KM": # this is from GS09 back of envelope of the mean
            boe = [0.64, 7.17, 2.62, 3.75, 0.32, 4.10, 10.0, 20.0]
            boe1 = [0.77, 6.47, 2.55, 4.19, 0.46, 4.30, 10.0, 20.0]
            boe2 = [0.77, 7.43, 2.56, 3.84, 0.42, 4.50, 10.0, 20.0]
            boe3 = [0.50, 7.13, 2.67, 3.56, 0.30, 4.05, 10.0, 20.0]
            theta0 = torch.tensor(boe1, device=dev, requires_grad = True)
            theta1 = torch.tensor(boe2, device=dev, requires_grad = True)
            theta2 = torch.tensor(boe3, device=dev, requires_grad = True)
        elif vars == "Mproto":
            #theta0 = torch.tensor([-1.185, 4.094], device=dev, requires_grad=True)
            #theta1 = torch.tensor([-1.185, 4.094], device=dev, requires_grad=True)
            #theta2 = torch.tensor([-1.185, 4.094], device=dev, requires_grad=True)
            theta0 = torch.tensor([0.0, 0.0], device=dev, requires_grad=True)
            theta1 = torch.tensor([0.0, 0.0], device=dev, requires_grad=True)
            theta2 = torch.tensor([0.0, 0.0], device=dev, requires_grad=True)

    network0 = NN(tau, ndim, outdim)
    network1 = NN(tau, ndim, outdim)
    network2 = NN(tau, ndim, outdim)

    if load:
        network0.load_state_dict(xi0_sd)
        network1.load_state_dict(xi1_sd)
        network2.load_state_dict(xi2_sd)
    xi0 = network0.to(dev)
    xi1 = network1.to(dev)
    xi2 = network2.to(dev)
    if lbfgs:
        optim = torch.optim.LBFGS([theta0, theta1, theta2],
                                  lr=.1, max_iter=100,
                                  line_search_fn = 'strong_wolfe')
        num_epochs = 1
    elif neldermead:
        optim = NelderMeadOptimizer([theta0, theta1, theta2], lr = 1.0)
        num_epochs = 100
    else:
        optim = torch.optim.Adam([theta0, theta1, theta2], lr = .1)
        num_epochs = 2000

    ng = 2**19 # max 2**19 number of draws (uniform gridpoints)
    treat_idcs = [i for i,t in enumerate(years) if 2001 <= t <= 2008]
    xi0hat, xi1hat, xi2hat = xi0, xi1, xi2
    theta0hat, theta1hat, theta2hat = theta0, theta1, theta2

    if train:

        closure, add_outputs = create_closure(xi0, xi1, xi2, theta0, theta1, theta2,
                                              tPs, tQs, tMuHat, ng,
                                              dev, tau, masks,
                                              treat_idcs, optim, CF.None_, train0)
        torch.set_printoptions(precision = 5, sci_mode=False)
        columns = ['loss', 'l0', 'l1', 'l2']
        for i in range(theta0.shape[0]):
            columns.append(f'theta0{i}')
        for i in range(theta1.shape[0]):
            columns.append(f'theta1{i}')
        for i in range(theta2.shape[0]):
            columns.append(f'theta2{i}')
        history = pd.DataFrame(index=np.arange(1, num_epochs+1),
                               columns=columns)

        losshat = torch.tensor(10e30, device=dev)
        for epoch in range(1, num_epochs + 1):
            loss = optim.step(closure)
            mush, muss, l0, l1, l2 = add_outputs
            record = [loss.item(), l0.item(), l1.item(), l2.item()]
            par0 = theta0.cpu().detach().numpy().flatten()
            par1 = theta1.cpu().detach().numpy().flatten()
            par2 = theta2.cpu().detach().numpy().flatten()
            if loss < losshat:
                losshat = loss
                xi0hat = xi0
                xi1hat = xi1
                xi2hat = xi2
                theta0hat = theta0
                theta1hat = theta1
                theta2hat = theta2
                print("Saving tensors")
                torch.save(theta0hat, hfpath + "theta0" + current + ".pt")
                torch.save(theta1hat, hfpath + "theta1" + current + ".pt")
                torch.save(theta2hat, hfpath + "theta2" + current + ".pt")
                torch.save(xi0hat.state_dict(),
                           hfpath + "xi0" + current + ".pt")
                torch.save(xi1hat.state_dict(),
                           hfpath + "xi1" + current + ".pt")
                torch.save(xi2hat.state_dict(),
                           hfpath + "xi2" + current + ".pt")
            else:
                print("previous loss: {} < loss: {}".format(losshat, loss))
            record.extend(par0)
            record.extend(par1)
            record.extend(par2)
            history.loc[epoch] = record
            if True: # loss <= losshat:
                perc = int((epoch / num_epochs) * 100)
                muss = muss.cpu().detach().numpy()
                print(f"{TermColours.BRIGHT_RED}{perc}% : {loss.item():.4f} : \
                        {theta0hat} {theta1hat} {theta2hat}: \
                        {TermColours.GREEN}{muss} \
                        {TermColours.RESET}",
                      end='\t', flush=True)
            history.to_csv('training_history.csv')

        print(theta0hat, theta1hat, theta2hat)
        print("Done.")


    if not train:

        xi0, xi1, xi2 = xi0hat.cpu(), xi1hat.cpu(), xi2hat.cpu()
        xi0.eval()
        xi1.eval()
        xi2.eval()
        theta0, theta1, theta2 = theta0hat.cpu(), theta1hat.cpu(), theta2hat.cpu()
        tPs, tQs = tPs.cpu(), tQs.cpu()
        mu_hat = tMuHat.cpu()
        sandbox, process, raw, net = st.tabs(["Sandbox",
                                              "Matched Processes",
                                              "Raw",
                                              "Network"])
        _, mu_star = load_mus(xi0, xi1, xi2, theta0, theta1, theta2,
                              tPs, tQs, mu_hat, ng,
                              "cpu", tau, masks, treat_idcs,
                              cf = CF.None_, train0 = train0)

        s = st.session_state
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
            ss = torch.tensor([[s.mu, s.mc, s.fu, s.fc]], device="cpu")
            hdmu = ['u', 'c', '0']
            hds = ['M_u', 'M_c', 'F_u', 'F_c']
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

        mus0, v0 = xi0(ss)
        ssnext0 = choices(mus0, tPs[0], tQs[0], "cpu")
        mus1, v1 = xi1(ss)
        ssnext1 = choices(mus1, tPs[1], tQs[1], "cpu")
        mus2, v2 = xi2(ss)
        ssnext2 = choices(mus2, tPs[2], tQs[2], "cpu")
        phi = lambda theta: torch.cat((
                                  torch.cat((
                                        tau(theta, "cpu"),
                                        torch.zeros(ndim, 1)), dim=1),
                                  torch.zeros(1, ndim+1)), dim=0)
        with sandbox:
            cols = st.columns(3)
            columns_data = zip(cols, [theta0, theta1, theta2], range(3))
            for col, theta, idx in columns_data:
                with col:
                    st.subheader('$\\Phi(\\widehat{{\\theta}}_{})$'.format(idx))
                    df = pd.DataFrame(
                        phi(theta).detach().numpy(),
                        index=hdmu,
                        columns=hdmu
                    )
                    st.dataframe(df)

            st.subheader('$\\mu_1$ in %')
            fmarg = ss[0, ndim:2*ndim].detach().numpy().tolist()
            mmarg = ss[0, 0:ndim].detach().numpy().tolist()
            st.pyplot(create_heatmap(mus1[0]*100,
                       ["{:.1f}".format(f*100) for f in fmarg + [0]],
                       ["{:.1f}".format(m*100) for m in mmarg + [0]]
                                     , hdmu), use_container_width=False)
            columns_data = zip(cols, [ssnext0, ssnext1, ssnext2], [v0, v1, v2], range(3))
            for col, ssnext, v, idx in columns_data:
                with col:
                    st.subheader('($M_{{t+1}}^*, F_{{t+1}}^*)_{} $'.format(idx))
                    df = pd.DataFrame(
                        ssnext.detach().numpy(),
                        columns=hds
                    )
                    st.dataframe(df, hide_index=True)
                    st.subheader('$V_{}$'.format(idx))
                    st.write(v.detach().numpy())

        with process:
            fig, ax1, ax2 = matched_process_plot(mu_hat, mu_star, years,
                                                 cells, couples, singles)
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
