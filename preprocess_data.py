#!/usr/bin/env python

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
import json
import torch

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

p = trans("men", 1)
q = trans("women", 1)
p.index
p.columns
p

def entryexit() -> pd.DataFrame:
    n = 'demographics'
    index_names = pd.read_excel(
        file_path,
        sheet_name=n,
        header=None,
        skiprows=1,
        nrows=1,
        usecols="A:E"
    ).iloc[0].tolist()
    df = pd.read_excel(
        file_path,
        sheet_name = n,
        header=[0],  # Multi-header rows
        index_col=[0, 1, 2, 3, 4],  # Multi-index from first 5 columns
        skiprows=1 # Skip top 1 empty rows
    ).fillna(0.0)
    df.index.names = index_names
    return df

ee = entryexit()
ee.index
ee.columns
ee


def trans_tensor(sex: str, i: int) -> torch.Tensor:
    raw = torch.tensor(trans(sex, i).values, dtype=torch.float32)
    diag = block_diagonal(raw.reshape((33, 32, 8))).reshape(33*32, 32)
    return diag

def trans_tensor_old(sex: str, i: int) -> torch.Tensor:
    return torch.tensor(trans_old(sex, i).values, dtype=torch.float32)

def offset(lst, offs):
    return [(x + offs) for x in lst]

def reduce_ee_to_ho(dims, d1, so, ntypes):
    n = d1.shape[1]
    srow = torch.tensor(d1[:np.prod(dims), :]).view([*dims, 2]) #2 for ee
    s = torch.sum(srow, dim=so).view((ntypes, 2)) # .view([2, ntypes])
    return s.detach().numpy()

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

# new reducers (NEKMS)
reduce_to = lambda d1, so, ntypes: reduce_to_ho([2] * 5, d1, so, ntypes)
reduce_trans_to = lambda d1, so, ntypes: reduce_trans_to_ho([2] * 5, d1, so, ntypes)
reduce_ee_to = lambda d1, so, ntypes: reduce_ee_to_ho([2] * 5, d1, so, ntypes)

reduce_to_marriage = lambda d1: reduce_to(d1, [0, 1, 2, 4], 2)
reduce_trans_to_marriage = lambda d1: reduce_trans_to(d1, [0, 1, 2, 4], 2)
reduce_ee_to_marriage = lambda d1: reduce_ee_to(d1, [0, 1, 2, 4], 2)
reduce_to_marriage_spec = lambda d1: reduce_to(d1, [0, 1, 2], 4)
reduce_trans_to_marriage_spec = lambda d1: reduce_trans_to(d1, [0, 1, 2], 4)
reduce_ee_to_marriage_spec = lambda d1: reduce_ee_to(d1, [0, 1, 2], 4)
reduce_to_kids_marriage = lambda d1: reduce_to(d1, [0, 1, 4], 4)
reduce_trans_to_kids_marriage= lambda d1: reduce_trans_to(d1, [0, 1, 4], 4)
reduce_ee_to_kids_marriage= lambda d1: reduce_ee_to(d1, [0, 1, 4], 4)
reduce_to_kids_marriage_spec = lambda d1: reduce_to(d1, [0, 1], 8)
reduce_trans_to_kids_marriage_spec = lambda d1: reduce_trans_to(d1, [0, 1], 8)
reduce_ee_to_kids_marriage_spec = lambda d1: reduce_ee_to(d1, [0, 1], 8)

# for debug
red = reduce_to_marriage_spec
tsred = reduce_trans_to_marriage_spec
eered = reduce_ee_to_marriage_spec

def block_diagonal(tensor):
    i, j, k = tensor.shape  # (33, 32, 8)
    output = torch.zeros(i, 32, 32,
                         dtype=torch.float32,
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
    return tensor / nhat, nhat.flatten()

def data(red, tsred, eered, ntypes):
    block = couplings()
    ee = entryexit()
    matchhat = torch.tensor(block.values, dtype=torch.float32).reshape(22, 33, 33)
    redmhat = torch.stack([torch.tensor(red(m)) for m in matchhat])
    muhat, nhat = normalise_tensor(redmhat)
    eehat = torch.tensor(eered(ee.values)).to(dtype=torch.float32) / nhat.mean()
    ng = nhat[1:] / nhat[:-1] - 1
    p_pre, q_pre   = tt_tensor(1, ntypes, tsred)
    p, q           = tt_tensor(2, ntypes, tsred)
    p_post, q_post = tt_tensor(3, ntypes, tsred)
    return Dataset.from_dict({'p': [np.stack([p_pre, p, p_post])],
                              'q': [np.stack([q_pre, q, q_post])],
                              'couplings': [muhat],
                              'entryexit': [eehat]})

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
                              'couplings': [muhat],
                              'entryexit': []})


torch.set_printoptions(sci_mode=False, edgeitems=4, precision=2)

class TensorModule(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.register_buffer("t", tensor)
    def forward(self):
        return self.t

# Assuming tMuHat, tPs, and tQs are already defined tensors.
# for filename, tensor in [("/tmp/tMuHat.pt", tMuHat),
#                          ("/tmp/tPs.pt", tPs),
#                          ("/tmp/tQs.pt", tQs)]:
#    module = TensorModule(tensor)
#    script_module = torch.jit.script(module)
#    torch.jit.save(script_module, filename)


sri = c.index.get_level_values('imigration') == 'single'
sci = c.columns.get_level_values('code') == '30000'

scpl = c.iloc[~sri, ~sci]
srow = c.iloc[sri,~sci].reset_index(level=[1,2,3,4,5], drop=True)
scol = c.iloc[:, sci].T.reset_index(level=[0,2,3,4,5], drop=True).T

agg = (
       scpl
       .groupby(level=['year', 'children', 'marital_st', 'whyo'], axis=0).sum()
       .groupby(level=['children', 'marital_st', 'whyo'], axis=1).sum()
      )

agg_srow = srow.groupby(level=['children', 'marital_st', 'whyo'], axis=1).sum()
agg_scol = scol.groupby(level=['year', 'children', 'marital_st', 'whyo'], axis=0).sum()

eeagg = ee.groupby(level=['children', 'marital_st', 'whyo'], axis=0).sum()

def map_state(children: str, marital: str, whyo: str) -> str:
    if marital == "current":
        if children == "yes":
            if whyo == "home":
                return "curhome"
            elif whyo == "work":
                return "curwork"
        elif children == "no":
            return "curwork"
    elif marital == "unmarried":
        return "unmarried"

def remap_col(col, years: bool = False, trans: bool = False):
    if years:
        year, children, marital, whyo = col
    elif trans:
        child_p, mar_p, whyo_p, children, marital, whyo = col
        state_p = map_state(child_p, mar_p, whyo_p)
        year = None
    else:
        children, marital, whyo = col
        year = None
    state = map_state(children, marital, whyo)
    return (year, state) if years else ((state,) if not trans else (state_p, state))

def new_cols(cols, years: bool = False, trans: bool = False) -> pd.MultiIndex:
    names = ["year", "state"] if years else (["state"] if not trans else ["state_p", "state"])
    return pd.MultiIndex.from_tuples(
        [remap_col(c, years=years, trans=trans) for c in cols],
        names=names,
    )

state_order = ["unmarried", "curhome", "curwork"]

eeagg2 = eeagg.T.copy()
eeagg2.columns = new_cols(eeagg2.columns, years=False)
eeagg3 = eeagg2.groupby(axis=1, level="state").sum()
eeagg3 = eeagg3.loc[:, [s for s in state_order if s in eeagg3.columns]].T

agg2 = agg.copy()
agg2.columns = new_cols(agg.columns, years=False)
agg3 = agg2.groupby(axis=1, level="state").sum()
agg3 = agg3.loc[:, [s for s in state_order if s in agg3.columns]]

A = agg3.T.copy()
A.columns = new_cols(A.columns, years=True)
A = A.groupby(axis=1, level=["year", "state"]).sum()
years = A.columns.get_level_values("year").unique()
ncs = [(y, s) for y in years for s in state_order if (y, s) in A.columns]
A = A.loc[:, ncs].T

agg_srow2 = agg_srow.copy()
agg_srow2.columns = new_cols(agg_srow2.columns, years = False)
agg_srow3 = agg_srow2.groupby(axis=1, level="state").sum()
agg_srow3 = agg_srow3.loc[:, [s for s in state_order if s in agg_srow3.columns]]

agg_scol2 = agg_scol.T.copy()
agg_scol2.columns = new_cols(agg_scol2.columns, years = True)
agg_scol3 = agg_scol2.groupby(axis=1, level=["year", "state"]).sum()
ncs = [(y, s) for y in years for s in state_order if (y, s) in agg_scol3.columns]
agg_scol3 = agg_scol3.loc[:, ncs].T

all = []
ns = []
for y in years:
    couples = A.loc[y].values
    singlef = agg_srow3.loc[y].values
    singlem = agg_scol3.loc[y].values
    ns.append(singlem.sum() + singlef.sum() + 2 * couples.sum())
    couples_m = np.concat([couples, singlem], axis = 1)
    single_f_ext = np.concat([singlef, np.array([0])], axis = 0).reshape(1, -1)
    couples_mf = np.concat([couples_m, single_f_ext], axis = 0)
    all.append(couples_mf)

final = torch.tensor(all)

# think about nhat. is it really 2*couples + singlef + singlem. yes, for margins!!
muhat = final/torch.tensor(ns).unsqueeze(-1).unsqueeze(-1)
muhat.sum(dim=2)[:,0:3]
muhat.sum(dim=1)[:,0:3]

# transitions

def bodge_trans(p):
    spi = p.index.get_level_values('COMB') // 1000000000 == 3
    p0 = p.iloc[spi, :]
    pcpl = p.iloc[~spi, :]
    # cannot be done in one go, otherwise singles will be aggregated with native/imm
    pagg = pcpl.groupby(level=['child_p', 'mar_p', 'whyo_p', 'child', 'mar', 'whyo'], axis=0).sum()
    agg_s = p0.groupby(level=['child_p', 'mar_p', 'whyo_p', 'child', 'mar', 'whyo'], axis=0).sum()
    def reassign(pagg1):
        pagg2 = pagg1.T.reset_index(level = [0], drop=True).T.copy()
        pagg2.columns = new_cols(pagg2.columns, years=False)
        pagg3 = pagg2.groupby(axis=1, level="state").sum()
        pagg3 = pagg3.loc[:, [s for s in state_order if s in agg3.columns]]
        pA = pagg3.T.copy()
        pA.columns = new_cols(pA.columns, years=False, trans = True)
        pA = pA.groupby(axis=1, level=["state_p", "state"]).sum()
        ncs = [(sp, s) for sp in state_order for s in state_order 
               if (sp, s) in pA.columns]
        pA = pA.loc[:, ncs].T
        return pA
    nP0 = np.concat([reassign(pagg).values, reassign(agg_s).values], axis = 0)
    tP0 = torch.tensor(nP0).view(4, 3, 3)
    return (tP0 / tP0.sum(dim=2, keepdim=True)).nan_to_num(0.0).to(dtype=torch.float32)

p_pre  = bodge_trans(trans("men", 1)).transpose(0,1).numpy()
p      = bodge_trans(trans("men", 2)).transpose(0,1).numpy()
p_post = bodge_trans(trans("men", 3)).transpose(0,1).numpy()
q_pre  = bodge_trans(trans("women", 1)).numpy()
q      = bodge_trans(trans("women", 2)).numpy()
q_post = bodge_trans(trans("women", 3)).numpy()

eehat = (torch.tensor(eeagg3.values) / torch.tensor(ns).mean()).to(dtype=torch.float32).numpy()

MSnew = Dataset.from_dict({'p': [np.stack([p_pre, p, p_post])],
                           'q': [np.stack([q_pre, q, q_post])],
                           'couplings': [muhat.to(dtype=torch.float32).numpy()],
                           'entryexit': [eehat]})

# Combine datasets into a DatasetDict
# they are really all just functions of muhat with the J_m, J_f projections
dataset_dict = DatasetDict({
    #'K': data_old(reduce_to_kids, reduce_trans_to_kids, 2),
    #'Mold': data_old(reduce_to_marriage_old, reduce_trans_to_marriage_old, 3),
    #'KMold': data_old(reduce_to_kids_marriage, reduce_trans_to_kids_marriage, 6),
    'M': data(reduce_to_marriage, reduce_trans_to_marriage, reduce_ee_to_marriage, 2),
    'KM': data(reduce_to_kids_marriage, reduce_trans_to_kids_marriage, reduce_ee_to_kids_marriage, 4),
    'MS': MSnew,
    'KMS': data(reduce_to_kids_marriage_spec, reduce_trans_to_kids_marriage_spec, reduce_ee_to_kids_marriage_spec, 8),
})


from dotenv import load_dotenv
from huggingface_hub import login
import os

load_dotenv()
hf_rwtoken = os.getenv('HF_TOKEN')
from huggingface_hub import login
login(token=hf_rwtoken)
dataset_dict.push_to_hub("StefanHubner/DivorceData", token = hf_rwtoken)


