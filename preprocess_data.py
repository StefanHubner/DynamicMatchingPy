import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
import json
import torch

file_path = './revised_data.xls'

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

def reduce_to(d1, so, ntypes):
    n = d1.shape[1]
    dims = [2, 2, 2, 3]
    t1 = torch.tensor(d1[:-1, :-1]).flatten().view(dims + dims)
    scol = torch.tensor(d1[:np.prod(dims), -1]).view(dims)
    srow = torch.tensor(d1[-1, :np.prod(dims)]).view(dims)
    cs = torch.sum(t1, dim=tuple([so, len(dims) + so])).view([ntypes, ntypes])
    sf = torch.sum(scol, dim=so).view([ntypes, 1])
    sm = torch.sum(srow, dim=so).view([1, ntypes])
    byT = torch.cat([
        torch.cat([cs, sf], dim=1),
        torch.cat([sm, torch.tensor(0).view(1, 1)], dim=1)
    ], dim=0)
    return byT.detach().numpy()

def reduce_trans_to(ts, so, ntypes):
    def offset(lst, offs):
        return [(x + offs) for x in lst]
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

def data(red, tsred, ntypes):
    block = couplings()
    muhat = torch.tensor(block.values).reshape(25, 25, 25)
    p_pre, q_pre   = tt(1, ntypes, tsred)
    p, q           = tt(2, ntypes, tsred)
    p_post, q_post = tt(3, ntypes, tsred)
    return Dataset.from_dict({'p': [np.stack([p_pre, p, p_post])],
                              'q': [np.stack([q_pre, q, q_post])],
                              'couplings': [muhat]})

# Combine datasets into a DatasetDict
# they are really all just functions of muhat with the J_m, J_f projections
dataset_dict = DatasetDict({
    'K': data(reduce_to_kids, reduce_trans_to_kids, 2),
    'M': data(reduce_to_marriage, reduce_trans_to_marriage, 3),
    'KM': data(reduce_to_kids_marriage, reduce_trans_to_kids_marriage, 6)
})

hf_rwtoken = "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"
dataset_dict.push_to_hub("StefanHubner/DivorceData", token = hf_rwtoken)
