#!/usr/bin/env python

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
import json
import torch

file_path = './revised_data_november.xls'

def couplings():
    return pd.read_excel(file_path,
                         sheet_name = 'statics_combined',
                         header=None,
                         index_col=[0]
                        ).fillna(0.0)

def trans(sex, i):
    trans = pd.read_excel(file_path,
                          sheet_name = sex + '_' + str(i),
                          index_col = [0],
                          header  = None)
    return trans.fillna(0.0)

def offset(lst, offs):
    return [(x + offs) for x in lst]

def reduce_to(d1, so, ntypes):
    n = d1.shape[1]
    dims = [2, 2, 2, 3]
    t1 = torch.tensor(d1[:-1, :-1]).flatten().view(dims + dims)
    scol = torch.tensor(d1[:np.prod(dims), -1]).view(dims)
    srow = torch.tensor(d1[-1, :np.prod(dims)]).view(dims)
    cs = torch.sum(t1, dim=so + offset(so, len(dims))
                   ).view([ntypes, ntypes])
    sf = torch.sum(scol, dim=so).view([ntypes, 1])
    sm = torch.sum(srow, dim=so).view([1, ntypes])
    byT = torch.cat([
        torch.cat([cs, sf], dim=1),
        torch.cat([sm, torch.tensor(0).view(1, 1)], dim=1)
    ], dim=0)
    return byT.detach().numpy()

def reduce_trans_to(ts, so, ntypes):
    if isinstance(ts, list):
        dt1 = np.column_stack(ts)
    else:
        dt1 = ts
    m, n = dt1.shape
    if m != n * (n + 1):
        raise ValueError(f"""Dimensions of matrix are {m} {n}.
                             Should be m times m(m+1)!""")
    dims = [2, 2, 2, 3]
    tt1 = torch.tensor(dt1[:-(np.prod(dims)), :]).flatten(
                       ).view(dims + dims + dims)
    sm = torch.tensor(dt1[-(np.prod(dims)):, :]
                      ).view(dims + dims)
    cs1 = torch.sum(tt1,
                    dim = so + offset(so, len(dims)) +
                               offset(so, 2 * len(dims))
                    ).view([ntypes ** 2, ntypes])
    ss1 = torch.sum(sm,
                    dim= so + offset(so, len(dims))
                    ).view([ntypes, ntypes])
    stacked = torch.cat([cs1, ss1], dim=0).detach().numpy()
    ps = stacked / np.sum(stacked, axis=1, keepdims=True)
    ps[np.isnan(ps)] = 0
    return ps

reduce_to_imm_educ_kids = lambda d1: reduce_to(d1, [3], 8)
reduce_trans_to_imm_educ_kids = lambda ts: reduce_trans_to(ts, [3], 8)

reduce_to_educ_kids = lambda d1: reduce_to(d1, [0, 3], 4)
reduce_trans_to_educ_kids = lambda ts: reduce_trans_to(ts, [0, 3], 4)

reduce_to_educ_marriage = lambda d1: reduce_to(d1, [0, 2], 6)
reduce_trans_to_educ_marriage = lambda ts: reduce_trans_to(ts, [0, 2], 6)

reduce_to_kids = lambda d1: reduce_to(d1, [0, 1, 3], 2)
reduce_trans_to_kids = lambda ts: reduce_trans_to(ts, [0, 1, 3], 2)

reduce_to_marriage = lambda d1: reduce_to(d1, [0, 1, 2], 3)
reduce_trans_to_marriage = lambda ts: reduce_trans_to(ts, [0, 1, 2], 3)

reduce_to_kids_marriage = lambda d1: reduce_to(d1, [0, 1], 6)
reduce_trans_to_kids_marriage = lambda ts: reduce_trans_to(ts, [0, 1], 6)


# i is for pre, post, during
def tt(i, ntypes, reducer):
    ncouples = (ntypes + 1) * ntypes  # +1 for single/unmatched types
    P_ = reducer(torch.tensor(trans('men', i).values, dtype=torch.float32))
    P = torch.tensor(P_).view(ntypes + 1, ntypes, ntypes
                              ).transpose(0, 1
                              ).flatten(
                              ).view(ncouples, ntypes
                              ).t()
    Q_ = reducer(torch.tensor(trans('women', i).values, dtype=torch.float32))
    Q = Q_.T
    return (P, Q)

def tt_tensor(i, ntypes, reducer):
    ncouples = (ntypes + 1) * ntypes  # +1 for single/unmatched types
    P_ = reducer(torch.tensor(trans('men', i).values, dtype=torch.float32))
    P = torch.tensor(P_).view(ntypes + 1, ntypes, ntypes).transpose(0, 1)
    Q_ = reducer(torch.tensor(trans('women', i).values, dtype=torch.float32))
    Q = torch.tensor(Q_).view(ntypes + 1, ntypes, ntypes)
    return (P, Q)


def normalize_tensor(tensor):
    (_, m, l) = tensor.shape
    muhat_mf = tensor[:, :(m-1), :(l-1)]
    muhat_m0 = tensor[:, :(m-1), -1]
    muhat_0f = tensor[:, -1, :(l-1)]
    nhat = 2 * torch.sum(muhat_mf, dim=(1, 2)) + torch.sum(muhat_m0, dim=1) + torch.sum(muhat_0f, dim=1)
    nhat = nhat.view(-1, 1, 1)
    return tensor / nhat

def data(red, tsred, ntypes):
    block = couplings()
    matchhat = torch.tensor(block.values).reshape(26, 25, 25)
    redmhat = torch.stack([torch.tensor(red(m)) for m in matchhat])
    muhat = normalize_tensor(redmhat)
    p_pre, q_pre   = tt_tensor(1, ntypes, tsred)
    p, q           = tt_tensor(2, ntypes, tsred)
    p_post, q_post = tt_tensor(3, ntypes, tsred)
    return Dataset.from_dict({'p': [np.stack([p_pre, p, p_post])],
                              'q': [np.stack([q_pre, q, q_post])],
                              'couplings': [muhat]})

# this won't work because  kc, zc are individual types not market outcomes
def dataKM():
    red = reduce_to_kids_marriage
    tsred = reduce_trans_to_kids_marriage
    ntypes = 6
    block = couplings()
    matchhat = torch.tensor(block.values).reshape(26, 25, 25)
    redmhat = torch.stack([torch.tensor(red(m)) for m in matchhat])
    muhat = normalize_tensor(redmhat)
    mask_from = torch.arange(muhat.size(1)) != 2 # selects the zc column
    allbut = muhat[:,mask_from,:][:,:,mask_from]
    mass = muhat[:,mask_from.logical_not(),:][:,:,mask_from.logical_not()]
    mask_to = torch.arange(allbut.size(1)) == 4 # new kc column (outs. opt.)
    allbut[:,mask_to,-1] = mass.reshape(26,1) / 2
    allbut[:,-1,mask_to] = mass.reshape(26,1) / 2
    p_pre, q_pre   = tt_tensor(1, ntypes, tsred)
    p, q           = tt_tensor(2, ntypes, tsred)
    p_post, q_post = tt_tensor(3, ntypes, tsred)
    # TODO: adjust the transition matrices here!

def prototypeM():
    red = reduce_to_marriage
    tsred = reduce_trans_to_marriage
    ntypes = 3
    block = couplings()
    matchhat = torch.tensor(block.values).reshape(26, 25, 25)
    redmhat = torch.stack([torch.tensor(red(m)) for m in matchhat])
    muhat = normalize_tensor(redmhat)
    muhat[:,-1,2] = muhat[:,1,-1]
    muhat[:,2,-1] = muhat[:,-1,1]
    muhat[:,1,-1] = muhat[:,-1,1] = 0
    op = torch.tensor([[1.0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1] ])
    muhat2 = torch.stack([op @ m @ op.T for m in muhat])
    nmemb = torch.tensor([[2,2,1],[2,2,1],[1,1,0.0]])
    pn, pe, pc =  (muhat2*nmemb).mean(0).sum(1) # stationary uncond. distributions
    qn, qe, qc =  (muhat2*nmemb).mean(0).sum(0) # TODO: perhaps split up by regs
    p_pre, q_pre   = tt_tensor(1, ntypes, tsred)
    p, q           = tt_tensor(2, ntypes, tsred)
    p_post, q_post = tt_tensor(3, ntypes, tsred)
    return Dataset.from_dict({'p': [np.stack([uc_p(p_pre, pn, pe),
                                              uc_p(p, pn, pe),
                                              uc_p(p_post, pn, pe)])],
                              'q': [np.stack([uc_q(q_pre, qn, qe),
                                              uc_q(q, qn, qe),
                                              uc_q(q_post, qn, qe)])],
                              'couplings': [muhat2]})



torch.set_printoptions(sci_mode=False, edgeitems=4, precision=2)

# Combine datasets into a DatasetDict
# they are really all just functions of muhat with the J_m, J_f projections
dataset_dict = DatasetDict({
    'K': data(reduce_to_kids, reduce_trans_to_kids, 2),
    'M': data(reduce_to_marriage, reduce_trans_to_marriage, 3),
    'KM': data(reduce_to_kids_marriage, reduce_trans_to_kids_marriage, 6),
    'Mproto': prototypeM()
})

hf_rwtoken = "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"
dataset_dict.push_to_hub("StefanHubner/DivorceData", token = hf_rwtoken)

class TensorModule(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.register_buffer("t", tensor)
    def forward(self):
        return self.t

# Assuming tMuHat, tPs, and tQs are already defined tensors.
for filename, tensor in [("/tmp/tMuHat.pt", tMuHat),
                         ("/tmp/tPs.pt", tPs),
                         ("/tmp/tQs.pt", tQs)]:
    module = TensorModule(tensor)
    script_module = torch.jit.script(module)
    torch.jit.save(script_module, filename)


def uc_p(p, p_n, p_e):
    # Initialize new tensor of shape [2,3,2]
    pnew = torch.zeros((2, 3, 2))
    # Normalization factor for aggregated state u
    p_u = p_n + p_e
    # Case 1: For transitions with no aggregation in m or m'
    pnew[1, 1, 1] = p[2, 2, 2]  # Q(c|cc) = P(c|cc)
    pnew[1, 2, 1] = p[2, 3, 2]  # Q(c|c0) = P(c|c0)
    # Case 2: For transitions to aggregated state u
    pnew[1, 1, 0] = p[2, 2, 0] + p[2, 2, 1]  # Q(u|cc) = P(n|cc) + P(e|cc)
    pnew[1, 2, 0] = p[2, 3, 0] + p[2, 3, 1]  # Q(u|c0) = P(n|c0) + P(e|c0)
    # Case 3: For transitions from aggregated state u
    pnew[0, 1, 1] = (p[0, 2, 2] * p_n + p[1, 2, 2] * p_e) / p_u  # Q(c|uc)
    pnew[0, 2, 1] = (p[0, 3, 2] * p_n + p[1, 3, 2] * p_e) / p_u  # Q(c|u0)
    # Case 4: For transitions with aggregated f=u
    pnew[1, 0, 1] = (p[2, 0, 2] * p_n + p[2, 1, 2] * p_e) / p_u  # Q(c|cu)
    # Case 5: For transitions with both m and f aggregated
    numerator = (p[0, 0, 2] * p_n * p_n +
                p[0, 1, 2] * p_n * p_e +
                p[1, 0, 2] * p_e * p_n +
                p[1, 1, 2] * p_e * p_e)
    pnew[0, 0, 1] = numerator / (p_u * p_u)  # Q(c|uu)
    # Fill remaining entries using complementary probability
    pnew[0, 1, 0] = 1 - pnew[0, 1, 1]  # Q(u|uc) = 1 - Q(c|uc)
    pnew[0, 2, 0] = 1 - pnew[0, 2, 1]  # Q(u|u0) = 1 - Q(c|u0)
    pnew[1, 0, 0] = 1 - pnew[1, 0, 1]  # Q(u|cu) = 1 - Q(c|cu)
    pnew[0, 0, 0] = 1 - pnew[0, 0, 1]  # Q(u|uu) = 1 - Q(c|uu)
    return pnew



def uc_q(q, q_n, q_e):
    # Initialize new tensor of shape [3,2,2]
    qnew = torch.zeros((3, 2, 2))
    # Normalization factor for aggregated state u
    q_u = q_n + q_e
    # Case 1: m=k, f=k, f'=k (no aggregation needed)
    qnew[1, 1, 1] = q[2, 2, 2]
    # Case 2: m=k, f=k, f'=u (aggregate f')
    qnew[1, 1, 0] = q[2, 2, 0] + q[2, 2, 1]
    # Case 3: m=k, f=u, f'=k (aggregate f)
    qnew[1, 0, 1] = (q[2, 0, 2] * q_n + q[2, 1, 2] * q_e) / q_u
    # Case 4: m=k, f=u, f'=u (aggregate f and f')
    qnew[1, 0, 0] = 1 - qnew[1, 0, 1]
    # Case 5: m=u, f=k, f'=k (aggregate m)
    qnew[0, 1, 1] = (q[0, 2, 2] * q_n + q[1, 2, 2] * q_e) / q_u
    # Case 6: m=u, f=k, f'=u (aggregate m and f')
    qnew[0, 1, 0] = (q[0, 2, 0] * q_n + q[0, 2, 1] * q_n + 
                     q[1, 2, 0] * q_e + q[1, 2, 1] * q_e) / q_u
    # Case 7: m=u, f=u, f'=k (aggregate m and f)
    numerator_nk = (q[0, 0, 2] * q_n * q_n + q[0, 1, 2] * q_n * q_e + 
                   q[1, 0, 2] * q_e * q_n + q[1, 1, 2] * q_e * q_e)
    qnew[0, 0, 1] = numerator_nk / (q_u * q_u)
    # Case 8: m=u, f=u, f'=u (aggregate m, f, and f')
    qnew[0, 0, 0] = 1 - qnew[0, 0, 1]
    # Case 9: m=0, f=k, f'=k (no aggregation)
    qnew[2, 1, 1] = q[3, 2, 2]
    # Case 10: m=0, f=k, f'=u (aggregate f')
    qnew[2, 1, 0] = q[3, 2, 0] + q[3, 2, 1]
    # Case 11: m=0, f=u, f'=k (aggregate f)
    qnew[2, 0, 1] = (q[3, 0, 2] * q_n + q[3, 1, 2] * q_e) / q_u
    # Case 12: m=0, f=u, f'=u (aggregate f and f')
    qnew[2, 0, 0] = 1 - qnew[2, 0, 1]
    return qnew
