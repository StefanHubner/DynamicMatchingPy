#!/usr/bin/env python

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
import json
import torch

file_path = ''
file_path = './250710_0500__RES_07_09_for_export_aangepast.xls'

def couplings_old():
    return pd.read_excel('./revised_data_november.xls',
                         sheet_name='statics_combined',
                         header=None,
                         index_col=[0]
                        ).fillna(0.0)

def trans_old(sex, i):
    trans = pd.read_excel('./revised_data_november.xls',
                          sheet_name = sex + '_' + str(i),
                          index_col = [0],
                          header  = None)
    return trans.fillna(0.0)


def couplings() -> pd.DataFrame:
    return pd.concat([couplings_regime(1),
                      couplings_regime(2),
                      couplings_regime(3)])


def couplings_regime(i: int) -> pd.DataFrame:
    n = f'statics_{i}'
    index_names = pd.read_excel(
        file_path,
        sheet_name=n,
        header=None,
        skiprows=7,
        nrows=1,
        usecols="A:F"
    ).iloc[0].tolist()
    df = pd.read_excel(
        file_path,
        sheet_name = n,
        header=[1, 2, 3, 4, 5, 6],  # Multi-header rows
        index_col=[0, 1, 2, 3, 4, 5],  # Multi-index from first 6 columns
        skiprows=1 # Skip top 2 empty rows
    ).fillna(0.0)
    df.index.names = index_names
    return df

c = couplings()
c.index
c.columns

def trans(sex: str, i: int) -> pd.DataFrame:
    n = f'{sex}_{i}'
    index_names = pd.read_excel(
        file_path,
        sheet_name=n,
        header=None,
        skiprows=1,
        nrows=1,
        usecols="A:K"
    ).iloc[0].tolist()
    df = pd.read_excel(
        file_path,
        sheet_name = n,
        header=[1, 2, 3, 4],  # Multi-header rows
        index_col=list(range(11)),  # Multi-index from first 6 columns
        skiprows=1
    ).fillna(0.0)
    df.index.names = index_names
    return df

t = trans("men", 1)
t.index
t.columns
t

def trans_tensor(sex: str, i: int) -> torch.Tensor:
    raw = trans(sex, i).values
    diag = block_diagonal(ts.reshape((33, 32, 8))).reshape(33*32, 32)
    return torch.tensor(diag, dtype=torch.float32)


def trans_tensor_old(sex: str, i: int) -> torch.Tensor:
    return torch.tensor(trans_old(sex, i).values, dtype=torch.float32)

def offset(lst, offs):
    return [(x + offs) for x in lst]

def reduce_to_ho(dims, d1, so, ntypes):
    n = d1.shape[1]
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

def reduce_trans_to_ho(dims, ts, so, ntypes):
    if isinstance(ts, list):
        dt1 = np.column_stack(ts)
    else:
        dt1 = ts
    m, n = dt1.shape
    if m != n * (n + 1):
        raise ValueError(f"""Dimensions of matrix are {m} {n}.
                             Should be m times m(m+1)!""")
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

# old reducers
reduce_to_old = lambda d1, so, ntypes: reduce_to_ho([2, 2, 2, 3], d1, so, ntypes)
reduce_trans_to_old = lambda d1, so, ntypes: reduce_trans_to_ho([2, 2, 2, 3], d1, so, ntypes)
reduce_to_imm_educ_kids = lambda d1: reduce_to_old(d1, [3], 8)
reduce_trans_to_imm_educ_kids = lambda ts: reduce_trans_to_old(ts, [3], 8)
reduce_to_educ_kids = lambda d1: reduce_to_old(d1, [0, 3], 4)
reduce_trans_to_educ_kids = lambda ts: reduce_trans_to_old(ts, [0, 3], 4)
reduce_to_marriage_old = lambda d1: reduce_to_old(d1, [0, 1, 2], 3)
reduce_trans_to_marriage_old = lambda ts: reduce_trans_to_old(ts, [0, 1, 2], 3)
reduce_to_educ_marriage = lambda d1: reduce_to_old(d1, [0, 2], 6)
reduce_trans_to_educ_marriage = lambda ts: reduce_trans_to_old(ts, [0, 2], 6)
reduce_to_kids = lambda d1: reduce_to_old(d1, [0, 1, 3], 2)
reduce_trans_to_kids = lambda ts: reduce_trans_to_old(ts, [0, 1, 3], 2)
reduce_to_kids_marriage = lambda d1: reduce_to_old(d1, [0, 1], 6)
reduce_trans_to_kids_marriage = lambda ts: reduce_trans_to_old(ts, [0, 1], 6)

# new reducers
reduce_to = lambda d1, so, ntypes: reduce_to_ho([2] * 5, d1, so, ntypes)
reduce_trans_to = lambda d1, so, ntypes: reduce_trans_to_ho([2] * 5, d1, so, ntypes)
reduce_to_marriage = lambda d1: reduce_to(d1, [0, 1, 2, 4], 2)
reduce_trans_to_marriage = lambda d1: reduce_trans_to(d1, [0, 1, 2, 4], 2)
reduce_to_marriage_spec = lambda d1: reduce_to(d1, [0, 1, 2], 4)
reduce_trans_to_marriage_spec = lambda d1: reduce_trans_to(d1, [0, 1, 2], 4)
reduce_to_kids_marriage_spec = lambda d1: reduce_to(d1, [0, 1], 8)
reduce_trans_to_kids_marriage_spec = lambda d1: reduce_trans_to(d1, [0, 1], 8)

# for debug
red = reduce_to_kids_marriage_spec
tsred = reduce_trans_to_kids_marriage_spec

def block_diagonal(tensor):
    i, j, k = tensor.shape  # (33, 32, 8)
    output = torch.zeros(i, 32, 32,
                         dtype=tensor.float32,
                         device=tensor.device)
    blocks = tensor.view(i, 4, 8, 8)
    idx = torch.arange(4)
    start_idx = idx * 8
    for b in range(4):
        output[:, b*8:(b+1)*8, b*8:(b+1)*8] = blocks[:, b]
    return output

def tt_tensor_ho(loader, i, ntypes, reducer):
    ncouples = (ntypes + 1) * ntypes  # +1 for single/unmatched types
    P_ = reducer(loader('men', i))
    P = torch.tensor(P_).view(ntypes + 1, ntypes, ntypes).transpose(0, 1)
    Q_ = reducer(loader('women', i))
    Q = torch.tensor(Q_).view(ntypes + 1, ntypes, ntypes)
    return (P, Q)

tt_tensor = lambda i, n, r: tt_tensor_ho(trans_tensor, i, n, r)
tt_tensor_old = lambda i, n, r: tt_tensor_ho(trans_tensor_old, i, n, r)

def normalise_tensor(tensor):
    (_, m, l) = tensor.shape
    muhat_mf = tensor[:, :(m-1), :(l-1)]
    muhat_m0 = tensor[:, :(m-1), -1]
    muhat_0f = tensor[:, -1, :(l-1)]
    nhat = 2 * torch.sum(muhat_mf, dim=(1, 2)) + torch.sum(muhat_m0, dim=1) + torch.sum(muhat_0f, dim=1)
    nhat = nhat.view(-1, 1, 1)
    return tensor / nhat

def data(red, tsred, ntypes):
    block = couplings()
    matchhat = torch.tensor(block.values, dtype=torch.float32).reshape(22, 33, 33)
    redmhat = torch.stack([torch.tensor(red(m)) for m in matchhat])
    muhat = normalise_tensor(redmhat)
    p_pre, q_pre   = tt_tensor(1, ntypes, tsred)
    p, q           = tt_tensor(2, ntypes, tsred)
    p_post, q_post = tt_tensor(3, ntypes, tsred)
    return Dataset.from_dict({'p': [np.stack([p_pre, p, p_post])],
                              'q': [np.stack([q_pre, q, q_post])],
                              'couplings': [muhat]})

def data_old(red, tsred, ntypes):
    block = couplings_old()
    matchhat = torch.tensor(block.values).reshape(26, 25, 25)
    redmhat = torch.stack([torch.tensor(red(m)) for m in matchhat])
    muhat = normalise_tensor(redmhat)
    p_pre, q_pre   = tt_tensor_old(1, ntypes, tsred)
    p, q           = tt_tensor_old(2, ntypes, tsred)
    p_post, q_post = tt_tensor_old(3, ntypes, tsred)
    return Dataset.from_dict({'p': [np.stack([p_pre, p, p_post])],
                              'q': [np.stack([q_pre, q, q_post])],
                              'couplings': [muhat]})


torch.set_printoptions(sci_mode=False, edgeitems=4, precision=2)

# Combine datasets into a DatasetDict
# they are really all just functions of muhat with the J_m, J_f projections
dataset_dict = DatasetDict({
    'K': data_old(reduce_to_kids, reduce_trans_to_kids, 2),
    'Mold': data_old(reduce_to_marriage_old, reduce_trans_to_marriage_old, 3),
    'KMold': data_old(reduce_to_kids_marriage, reduce_trans_to_kids_marriage, 6),
    'M': data(reduce_to_marriage, reduce_trans_to_marriage, 2),
    'MS': data(reduce_to_marriage_spec, reduce_trans_to_marriage_spec, 4),
    'KMS': data(reduce_to_kids_marriage_spec, reduce_trans_to_kids_marriage_spec, 8),
})


from dotenv import load_dotenv
from huggingface_hub import login
import os

load_dotenv()
hf_token = os.getenv('HF_TOKEN')
from huggingface_hub import login
login(token=hf_rwtoken)
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

