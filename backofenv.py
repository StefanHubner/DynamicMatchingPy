import torch

torch.set_printoptions(precision=4)
avg = tMuHat.mean(axis=0)
avg_m0 = avg[:-1,-1]
avg_0f = avg[-1,:-1]
t = avg[:-1,:-1] / torch.outer(avg_m0, avg_0f)
torch.outer(avg_m0, avg_0f)
avg_m0
avg_0f
torch.log(t)

# do this for pre, during, post (6:13 is treat)
torch.set_printoptions(precision=4)
avg = tMuHat[14:,:,:].mean(axis=0)
avg_m0 = avg[:-1,-1]
avg_0f = avg[-1,:-1]
torch.log(torch.square(avg[:-1,:-1]) / torch.outer(avg_m0, avg_0f))


