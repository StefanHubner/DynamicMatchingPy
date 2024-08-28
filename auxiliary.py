

## one by one (non-recursive) matching of moments

    # (iota^T kronecker J_m) vec mu = M
    # (J_f kronecker iota) vec mu = F


    # mu* (optimal QE control/decision)
    mus = dr(ss, xi)



# simulates from a gumbel distribution
import numpy as np
import matplotlib.pyplot as plt

def gumbel_simulate(n, mu, beta):
    # Simulate from Gumbel distribution
    u = np.random.uniform(0, 1, n)
    x = mu - beta * np.log(-np.log(u))
    return x

mu = 0
beta = 1
x = np.linspace(-10, 10, 1000)
y = np.exp(-(x - mu) / beta) * np.exp(-np.exp(-(x - mu) / beta)) / beta
plt.plot(x, y)
plt.show()

x = gumbel_simulate(n, mu, beta)
plt.hist(x, bins=100, density=True)
plt.show()


# plots an entropy function

import numpy as np
import matplotlib.pyplot as plt

def entropy(p):
    return -(p * np.log(p)) - (1 - p) * np.log(1 - p)


p = np.linspace(0.01, 0.99, 100)
H = entropy(p)
plt.plot(p, H)
plt.show()





import torch


def transform_couplings(mm_slack, row_margs, col_margs):
    def normalise(matrix, row_margs, col_margs):
        row_sums = torch.sum(matrix, dim=1, keepdim=True)
        col_sums = torch.sum(matrix, dim=0, keepdim=True)
        # Normalize rows
        matrix = matrix * (row_margs / row_sums)
        # Normalize columns
        col_sums = torch.sum(matrix, dim=0, keepdim=True)
        matrix = matrix * (col_margs / col_sums)
        return matrix

    # Step 1: Affine Transformation
    shift_constant = torch.abs(torch.min(mm_slack)) + 1
    transformed = mm_slack + shift_constant

    # Step 2: Normalise
    return normalise(transformed.clone(), row_margs, col_margs)

# Example usage
coupling_matrix = torch.tensor([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0]])
row_margs = [3.0, 9.0, 12.0]
col_margs = [6.0, 7.0, 11.0]

positive_coupling_matrix = transform_and_normalize(coupling_matrix, row_margs, col_margs)
print(positive_coupling_matrix)

ds = torch.tensor([[7/12,0,5/12],[1/6,1/2,1/3],[1/4,1/2,1/4]])

gen = torch.distributions.dirichlet.Dirichlet(torch.tensor(4*[1.0]))
matrix = gen.sample()
M = torch.tensor((0.1238, 0.0039))
F = torch.tensor((0.6484, 0.2239))
mu = torch.tensor([[0.2,0.4],[0.3,0.1]])
mum0 = M - mu.sum(dim=1)
mu0f = F - mu.sum(dim=0)
#bind mum0 to mu on the right and muf0 to mu on the bottom
mum = torch.cat((mu,mum0.view(2,1)),dim=1)
muf_row = torch.cat((mu0f, torch.tensor(0).view(1))).view(1,3)

col_margs = torch.cat((F, mum0.sum().view(1))).view(1,-1)
row_margs = torch.cat((M, mu0f.sum().view(1))).view(-1,1)

mm_slack = torch.cat((mum, muf_row))


dsm = transform_couplings(mm_slack, row_margs, col_margs)

print(dsm)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrix

# Simulate data
np.random.seed(234)
n = 150
grid = np.linspace(0, 4 * np.pi, n)
data = pd.DataFrame({
         'x': grid,
         'y': np.sin(grid) + np.sqrt(grid) * np.random.normal(0, 0.5, n)
       })

# Split into training and test data
train = data.sample(n=n//2)
test = data.drop(train.index)

# Fit splines and compute MSE
k_vals = [1, 2, 3, 4, 8, 16, 32]
r = np.arange(len(k_vals))

# Define basis functions
def px(k, x):
    match k:
        case 1: return dmatrix("x")
        case 2: return dmatrix("x + x2", { "x2": x**2 })
        case _: return dmatrix("bs(x, df={})".format(k))

# Fit splines and compute MSE
mse_train, mse_test, yhats = [], [], []
for k in k_vals:
    fit = np.linalg.lstsq(px(k, train.x), train.y, rcond=None)[0]
    yhat = np.dot(px(k, data.x), fit)
    mse_train.append(np.mean((train.y - yhat[train.index]) ** 2))
    mse_test.append(np.mean((test.y - yhat[test.index]) ** 2))
    yhats.append(yhat)

preds = pd.DataFrame(np.column_stack(yhats), columns=k_vals)
preds['x'] = data.x
preds['y'] = data.y
preds['test'] = preds.index.isin(test.index)

plt.figure(figsize=(10, 6))
plt.scatter(preds['x'], preds['y'], c=preds['test'])
for col in preds.columns.difference(['x', 'y', 'test']):
    plt.plot(preds['x'], preds[col], label=f'k={col}')
plt.title('Predictions with different Model Complexities')
plt.legend()
plt.show()


# Create the plot
plt.figure(figsize=(10, 6))
plt.bar(r - 0.2, mse_train, width = 0.4, color = '#1f77b4', label='Train')
plt.bar(r + 0.2, mse_test, width = 0.4, color = '#ff7f0e', label='Test')
plt.xlabel('Model Compexlity')
plt.ylabel('Train and Test MSE (Mean Squared Error)')
plt.xticks(r, k_vals)
plt.ylim(1,2)
plt.legend()
plt.show()
