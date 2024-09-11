import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset

from dynamicmatching.bellman import match_moments
from dynamicmatching.helpers import tauMflex, TermColours

# only for debugging:
from dynamicmatching.bellman import minimise_inner, dr
from dynamicmatching.deeplearning import Perceptron

def main(train = False, noload = False):

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the dataset
    data = load_dataset('StefanHubner/DivorceData')
    current = "M"

    # Transition matrices are 3-tensors (0: pre, 1: treat, 2: post, f/m, mf)
    tPs = torch.tensor(data[current]["p"][0], device = dev)
    tQs = torch.tensor(data[current]["q"][0], device = dev)
    tMuHat = torch.tensor(data[current]["couplings"][0], device = dev)

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
        theta0 = torch.tensor([-7.7,-7.9,-10.1,-9.4], dtype=torch.float16,
                              device=dev, requires_grad = True)
        theta1 = torch.tensor([-7.7,-7.9,-10.1,-9.4], dtype=torch.float16,
                              device=dev, requires_grad = True)

    network0 = Perceptron([64, 64], tPs.shape[1], llb = -8)
    network1 = Perceptron([64, 64], tPs.shape[1], llb = -8)
    if load:
        network0.load_state_dict(xi0_sd)
        network1.load_state_dict(xi1_sd)
    xi0 = network0.to(dev)
    xi1 = network1.to(dev)
    optimizer_outer = torch.optim.SGD([theta0, theta1], lr=0.001)
    num_epochs = 3000
    ng = 2**14 # max 2**19 number of draws (uniform gridpoints)
    treat_idcs = [i for i,t in enumerate(range(1996, 2020))
                   if 2001 <= t <= 2008]

    # Define the loss calculation function
    def calc_loss():
        optimizer_outer.zero_grad()
        resid, ssh, sss, l0, l1 = match_moments(xi0, xi1, theta0, theta1,
                                                tPs, tQs,
                                                tMuHat, ng, dev,
                                                tauMflex, treat_idcs,
                                                skiptrain = False)
        resid.backward()
        return resid, ssh, sss, l0, l1

    # loss = optimizer_outer.step(calc_loss)

    xi0hat, xi1hat, theta0hat, theta1hat = xi0, xi1, theta0, theta1

    if train:
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
            loss, ssh, sss, l0, l1 = optimizer_outer.step(calc_loss)
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
            if loss <= losshat:
                perc = int((epoch / num_epochs) * 100)
                sss = sss.cpu().detach().numpy()
                print(f"{TermColours.RED}{perc}% : {loss.item():.4f} : \
                        {theta0hat} {theta1hat} : {TermColours.GREEN}{sss} \
                        {TermColours.RESET}",
                      end='\t', flush=True)

        history.to_csv('training_history.csv')
        print("Done.")

    resid, ss_hat, ss_star, _, _ = match_moments(xi0hat, xi1hat,
                                                 theta0hat, theta1hat,
                                                 tPs, tQs, tMuHat, ng,
                                                 dev, tauMflex, treat_idcs,
                                                 skiptrain = True)

    colors = ['b', 'g', 'r', 'c', 'm']
    sh = ss_hat.cpu().detach().numpy()
    ss = ss_star.cpu().detach().numpy()
    for i in range(5):
        plt.plot(sh[:, i], label=f'Shat{i+1}',
                 color=colors[i], linestyle='-')
        plt.plot(ss[:, i], label=f'Sstar{i+1}',
                 color=colors[i], linestyle='--')
    plt.title('Line Plots of Tensor Columns')
    plt.legend()
    plt.show()
    plt.savefig('matched_process.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="options")
    parser.add_argument("--train",
                        action="store_true",
                        help="Set to train the network.")
    parser.add_argument("--noload",
                        action="store_true",
                        help="Set to overwrite pars from HuggingFace.")
    args = parser.parse_args()
    main(args.train, args.noload)


