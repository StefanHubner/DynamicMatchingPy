import argparse
import io
import torch
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datasets import load_dataset

from dynamicmatching.bellman import match_moments, create_closure
from dynamicmatching.helpers import tauMflex, TermColours
from dynamicmatching.graphs import matched_process_plot, create_heatmap
from dynamicmatching.bellman import minimise_inner, choices
from dynamicmatching.deeplearning import Perceptron, SinkhornUnmatched, masked_log

@st.cache_resource
def load_data(name, dev):
    data = load_dataset("StefanHubner/DivorceData")[name]
    tPs = torch.tensor(data["p"][0], device = dev)
    tQs = torch.tensor(data["q"][0], device = dev)
    tMuHat = torch.tensor(data["couplings"][0], device = dev)
    return tPs, tQs, tMuHat

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
        theta0 = torch.tensor([15, 15, 0, 12], dtype=torch.float32,
                              device=dev, requires_grad = True)
        theta1 = torch.tensor([18, 15, 0, 15], dtype=torch.float32,
                              device=dev, requires_grad = True)

    #network0 = Perceptron([64, 64, 64], ndim, len(theta0), llb = -2)
    #network1 = Perceptron([64, 64, 64], ndim, len(theta1), llb = -2)
    network0 = SinkhornUnmatched(tauMflex, ndim, 9)
    network1 = SinkhornUnmatched(tauMflex, ndim, 9)
    maskc = [[True, True, False, False],
             [True, True, False, False],
             [False, False, True, False],
             [False, False, False, False]]
    mask0 = [True, True, False, False]
    masks = (maskc, mask0)


    if load:
        network0.load_state_dict(xi0_sd)
        network1.load_state_dict(xi1_sd)
    xi0 = network0.to(dev)
    xi1 = network1.to(dev)
    if lbfgs:
        optim = torch.optim.LBFGS([theta0, theta1],
                                  lr=1, max_iter=100,
                                  line_search_fn = 'strong_wolfe')
        num_epochs = 1
    else:
        optim = torch.optim.Adam([theta0, theta1], lr=0.01)
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
            ssh, sss, l0, l1 = add_outputs
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
                sss = sss.cpu().detach().numpy()
                print(f"{TermColours.RED}{perc}% : {loss.item():.4f} : \
                        {theta0hat} {theta1hat} : {TermColours.GREEN}{sss} \
                        {TermColours.RESET}",
                      end='\t', flush=True)

        history.to_csv('training_history.csv')
        print(theta0hat, theta1hat)
        print("Done.")

    if matchingplot:
        resid, ss_hat, ss_star, _, _ = match_moments(xi0hat, xi1hat,
                                                 theta0hat, theta1hat,
                                                 tPs, tQs, tMuHat, ng,
                                                 dev, tauMflex, treat_idcs,
                                                 skiptrain = True)
        fig = matched_process_plot(ss_hat, ss_star)
        fig.savefig("matched_processes.pdf",
                    format="pdf", bbox_inches="tight")
        plt.show()


    if not matchingplot and not train:
        xi = xi0hat.cpu()
        theta0 = theta0hat.cpu()
        tPs, tQs = tPs.cpu(), tQs.cpu()
        s = st.session_state
        if 'mn' not in s: s.mn = 0.16
        if 'me' not in s: s.me = 0.16
        if 'fn' not in s: s.fn = 0.16
        if 'fe' not in s: s.fe = 0.16
        def update_mc(): s.mc = 0.5 - s.mn - s.me
        def update_fc(): s.fc = 0.5 - s.fn - s.fe

        mn = st.sidebar.slider('$M_N$', 0.0, 0.5,
                               step=0.01, key='mn', on_change=update_mc)
        me = st.sidebar.slider('$M_E$', 0.0, 0.5,
                               step=0.01, key='me', on_change=update_mc)
        update_mc()
        st.sidebar.slider('$M_C$', 0.0, 0.5, step=0.01,
                                   key='mc', disabled=True)
        fn = st.sidebar.slider('$F_N$', 0.0, 0.5,
                               step=0.01, key='fn', on_change=update_fc)
        fe = st.sidebar.slider('$F_E$', 0.0, 0.5,
                               step=0.01, key='fe', on_change=update_fc)
        update_fc()
        st.sidebar.slider('$F_C$', 0.0, 0.5, step=0.01,
                                   key='fc', disabled=True)

        ss = torch.tensor([[mn, me, s.mc, fn, fe, s.fc]], device="cpu")
        mus, v = xi(ss)
        ssnext = choices(mus, tPs[1], tQs[1], "cpu")
        phi = torch.cat((torch.cat((tauMflex(theta0, "cpu"),
                                    torch.zeros(3, 1)), dim=1),
                        torch.zeros(1, 4)), dim=0)
        st.subheader('$\\Phi(\\widehat{\\theta})$')
        st.write(phi.detach().numpy())
        #st.pyplot(create_heatmap(phi), use_container_width=False)
        st.subheader('$\\mu$')
        #st.write(mus[0].detach().numpy())
        st.pyplot(create_heatmap(mus*100,
                    ["{:.1f}".format(f*100) for f in [fn, fe, s.fc, 0]],
                    ["{:.1f}".format(m*100) for m in [mn, me, s.mc, 0]]
                                 ), use_container_width=False)
        st.subheader('($M_{t+1}^*, F_{t+1}^*) $')
        st.write(ssnext.detach().numpy())
        st.subheader('$V$')
        st.write(v.detach().numpy())


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

