import torch

torch.set_printoptions(precision=4)
avg = tMuHat.mean(axis=0)
avg_m0 = avg[:-1,-1]
avg_0f = avg[-1,:-1]
torch.log(avg[:-1,:-1] / torch.outer(avg_m0, avg_0f))
