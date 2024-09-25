import argparse
import io
import torch
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datasets import load_dataset
from torchviz import make_dot
from PIL import Image

from dynamicmatching.bellman import match_moments, create_closure
from dynamicmatching.helpers import tauMflex, TermColours
from dynamicmatching.graphs import matched_process_plot, create_heatmap, svg_to_data_url
from dynamicmatching.bellman import minimise_inner, choices
from dynamicmatching.deeplearning import Perceptron, SinkhornUnmatched, masked_log

st.set_page_config(page_title = "Dynamic Matching")

@st.cache_resource
def load_data(name, dev):
    data = load_dataset("StefanHubner/DivorceData")[name]
    tPs = torch.tensor(data["p"][0], device = dev)
    tQs = torch.tensor(data["q"][0], device = dev)
    tMuHat = torch.tensor(data["couplings"][0], device = dev)
    return tPs, tQs, tMuHat

def load_mus(xi0, xi1, t0, t1, tPs, tQs, muh, ng, dev, tau, masks, tis):
        _, muh1, mus, _, _ = match_moments(xi0, xi1, t0, t1,
                                           tPs, tQs, muh, ng,
                                           dev, tau, masks, tis,
                                           skiptrain = True)
        return muh1, mus

def main(train = False, noload = False, lbfgs = False, matchingplot = True):

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the dataset
    current, ndim = ("M", 3)
    tPs, tQs, tMuHat = load_data(current, dev)

    hfpath = "../../../../HFModels/DutchDivorce/"
    load = not noload
    if load:
        theta0 = torch.load(hfpath + "theta0" + current + ".pt",
                            weights_only = True)
        theta1 = torch.load(hfpath + "theta1" + current + ".pt",
                            weights_only = True)
        xi0_sd = torch.load(hfpath + "xi0" + current + ".pt",
                            weights_only = True)
        xi1_sd = torch.load(hfpath + "xi1" + current + ".pt",
                            weights_only = True)
    else:
        theta0 = torch.tensor([10, 10, 5 , 10], dtype=torch.float32,
                              device=dev, requires_grad = True)
        theta1 = torch.tensor([10, 5, 5, 10], dtype=torch.float32,
                              device=dev, requires_grad = True)

    #network0 = Perceptron([64, 64, 64], ndim, len(theta0), llb = -2)
    #network1 = Perceptron([64, 64, 64], ndim, len(theta1), llb = -2)
    maskc = [[True, True, False, False],
             [True, True, False, False],
             [False, False, True, False],
             [False, False, False, False]]
    mask0 = [True, True, False, False]
    masks = (maskc, mask0)
    # If masks are provided, log-domain is used. provide None otherwise
    network0 = SinkhornUnmatched(tauMflex, ndim, 9, masks = None)
    network1 = SinkhornUnmatched(tauMflex, ndim, 9, masks = None)

    if load:
        network0.load_state_dict(xi0_sd)
        network1.load_state_dict(xi1_sd)
    xi0 = network0.to(dev)
    xi1 = network1.to(dev)
    if lbfgs:
        optim = torch.optim.LBFGS([theta0, theta1],
                                  lr=1, max_iter=100,
                                  line_search_fn = 'strong_wolfe')
        num_epochs = 100
    else:
        optim = torch.optim.Adam([theta0, theta1], lr = 1)
        num_epochs = 3000
    ng = 2**17 # max 2**19 number of draws (uniform gridpoints)
    treat_idcs = [i for i,t in enumerate(range(1996, 2020))
                   if 2001 <= t <= 2008]


    xi0hat, xi1hat, theta0hat, theta1hat = xi0, xi1, theta0, theta1

    if train:

        closure, add_outputs = create_closure(xi0, xi1, theta0, theta1,
                                              tPs, tQs, tMuHat, ng,
                                              dev, tauMflex, masks,
                                              treat_idcs, optim)
        torch.set_printoptions(precision = 5, sci_mode=False)
        columns = ['loss', 'l0', 'l1']
        for i in range(theta0.shape[0]):
            columns.append(f'theta0{i}')
        for i in range(theta1.shape[0]):
            columns.append(f'theta1{i}')
        history = pd.DataFrame(index=np.arange(1, num_epochs+1),
                               columns=columns)

        losshat = torch.tensor(10e30, device=dev)
        for epoch in range(1, num_epochs + 1):
            loss = optim.step(closure)
            mush, muss, l0, l1 = add_outputs
            record = [loss.item(), l0.item(), l1.item()]
            par0 = theta0.cpu().detach().numpy().flatten()
            par1 = theta1.cpu().detach().numpy().flatten()
            if loss < losshat:
                losshat = loss
                xi0hat = xi0
                xi1hat = xi1
                theta0hat = theta0
                theta1hat = theta1
                torch.save(theta0hat, hfpath + "theta0" + current + ".pt")
                torch.save(theta1hat, hfpath + "theta1" + current + ".pt")
                torch.save(xi0hat.state_dict(),
                           hfpath + "xi0" + current + ".pt")
                torch.save(xi1hat.state_dict(),
                           hfpath + "xi1" + current + ".pt")
            record.extend(par0)
            record.extend(par1)
            history.loc[epoch] = record
            if True: # loss <= losshat:
                perc = int((epoch / num_epochs) * 100)
                muss = muss.cpu().detach().numpy()
                print(f"{TermColours.RED}{perc}% : {loss.item():.4f} : \
                        {theta0hat} {theta1hat} : {TermColours.GREEN}{muss} \
                        {TermColours.RESET}",
                      end='\t', flush=True)

        history.to_csv('training_history.csv')
        print(theta0hat, theta1hat)
        print("Done.")


    if not train:
        years = range(1996, 2021)
        xi0, xi1 = xi0hat.cpu(), xi1hat.cpu()
        xi0.eval()
        xi1.eval()
        theta0, theta1 = theta0hat.cpu(), theta1hat.cpu()
        tPs, tQs = tPs.cpu(), tQs.cpu()
        mu_hat = tMuHat.cpu()
        sandbox, process, raw, net = st.tabs(["Sandbox",
                                              "Matched Processes",
                                              "Raw",
                                              "Network"])
        s = st.session_state
        _, mu_star = load_mus(xi0, xi1, theta0, theta1,
                                   tPs, tQs, mu_hat, ng,
                                   "cpu", tauMflex, masks, treat_idcs)
        if 'mn' not in s: s.mn = 0.16
        if 'me' not in s: s.me = 0.16
        if 'fn' not in s: s.fn = 0.16
        if 'fe' not in s: s.fe = 0.16
        def update_mc():
            s.mc = 0.5 - s.mn - s.me

        def update_fc():
            s.fc = s.mc

        def update_fe():
            s.fe = 0.5 - s.fn - s.fc

        mn = st.sidebar.slider('$M_n$', 0.0, 0.5, step=0.01,
                               key='mn', on_change=update_mc)
        me = st.sidebar.slider('$M_e$', 0.0, 0.5, step=0.01,
                               key='me', on_change=update_mc)
        update_mc()
        st.sidebar.slider('$M_c$', 0.0, 0.5, step=0.01,
                          key='mc', disabled=True, on_change=update_fc)
        update_fc()
        fn = st.sidebar.slider('$F_n$', 0.0, 0.5, step=0.01,
                               key='fn', on_change=update_fe)
        update_fe()
        st.sidebar.slider('$F_e$', 0.0, 0.5, step=0.01,
                          key='fe', disabled=True)
        st.sidebar.slider('$F_c$', 0.0, 0.5, step=0.01,
                          key='fc', disabled=True)

        ss = torch.tensor([[mn, me, s.mc, fn, s.fe, s.fc]], device="cpu")
        mus0, v0 = xi0(ss)
        ssnext0 = choices(mus0, tPs[1], tQs[1], "cpu")
        mus1, v1 = xi1(ss)
        ssnext1 = choices(mus1, tPs[1], tQs[1], "cpu")
        phi = lambda theta: torch.cat((
                                  torch.cat((
                                        tauMflex(theta, "cpu"),
                                        torch.zeros(3, 1)), dim=1),
                                  torch.zeros(1, 4)), dim=0)
        with sandbox:
            col1, col2 = st.columns(2)
            hds = ['nt','e','c','0']
            with col1:
                st.subheader('$\\Phi(\\widehat{\\theta}_0)$')
                st.write()
                df = pd.DataFrame(phi(theta0).detach().numpy(),
                                  index=hds, columns=hds)
                st.dataframe(df)
            with col2:
                st.subheader('$\\Phi(\\widehat{\\theta}_1)$')
                df = pd.DataFrame(phi(theta1).detach().numpy(),
                                  index=hds, columns=hds)
                st.dataframe(df)
            st.subheader('$\\mu_0$ in %')
            st.pyplot(create_heatmap(mus0[0]*100,
                       ["{:.1f}".format(f*100) for f in [fn, s.fe, s.fc, 0]],
                       ["{:.1f}".format(m*100) for m in [mn, me, s.mc, 0]]
                                     ), use_container_width=False)
            col1, col2 = st.columns(2)
            hds = ['M_n', 'M_e', 'M_c', 'F_n', 'F_e', 'F_c']
            with col1:
                st.subheader('($M_{t+1}^*, F_{t+1}^*)_0 $')
                df = pd.DataFrame(ssnext0.detach().numpy(),
                                  columns=hds)
                st.dataframe(df, hide_index = True)
                st.subheader('$V_0$')
                st.write(v0.detach().numpy())
            with col2:
                st.subheader('($M_{t+1}^*, F_{t+1}^*)_1 $')
                #st.write(ssnext1.detach().numpy())
                df = pd.DataFrame(ssnext1.detach().numpy(),
                                  columns=hds)
                st.dataframe(df, hide_index = True)
                st.subheader('$V_1$')
                st.write(v1.detach().numpy())
        with process:
            fig, ax1, ax2 = matched_process_plot(mu_hat, mu_star, years)
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
                    act = mu[i].detach().numpy()
                    with col:
                        pcnc = 2 * act[:-2,:-2].sum()
                        pcc = 2 * act[-2,-2].sum()
                        pm, pf = act[-1,:].sum(), act[:,-1].sum()
                        st.subheader("Year {}".format(years[i]))
                        st.write(("$\\underbrace{{{:.3f}}}_{{P(C_{{\\neg m}})}} + " +
                                  " \\underbrace{{{:.3f}}}_{{P(C_m)}} + " +
                                  " \\underbrace{{{:.3f}}}_{{P(M_0)}} + " +
                                  " \\underbrace{{{:.3f}}}_{{P(F_0)}} + " +
                                  " = {{{:.3f}}} $").format(
                                        pcnc, pcc, pm, pf,
                                        pcnc + pcc + pm + pf))
                        st.write(act)
                        df = pd.DataFrame(act)
                        st.dataframe(df, hide_index = True)

        with net:
            st.header("Network 0")
            dot = make_dot(xi0hat(ss),
                           params = dict(xi0hat.named_parameters()),
                           show_attrs=True, show_saved=True)
            png_data = dot.pipe(format='png')
            image = Image.open(io.BytesIO(png_data))
            st.image(image, use_column_width=True)


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
    parser.add_argument("--matchingplot",
                        action="store_true",
                        help="Run this script in streamlit.")
    args = parser.parse_args()
    main(args.train, args.noload, args.lbfgs, args.matchingplot)

