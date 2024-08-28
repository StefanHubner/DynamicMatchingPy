import torch
import torch.nn as nn
import math

class Perceptron(nn.Module):
    def __init__(self, hidden_sizes, ntypes, constrained = False, llb = 0.0):
        super(Perceptron, self).__init__()
        layers = []
        input_size = 2 * ntypes - 1 # adding up
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        # mu, nu, V (dim mu = dim nu, if constrained)
        nout = (1+int(constrained)) * (ntypes ** 2) + 1
        layers.append(nn.Linear(input_size, nout))
        self.layers = nn.Sequential(*layers)
        self.nout = nout
        self.ntypes = ntypes
        self.constrained = constrained
        self.lastlayerbias = llb
        self._initialize_weights()
    # this makes sure that startup mus are feasiable
    # in case of nan's adjust this
    def _initialize_weights(self):
        gain = nn.init.calculate_gain('relu')
        for idx, m in enumerate(self.layers):
            if isinstance(m, nn.Linear):
                bound = gain * math.sqrt(6 / m.weight.size(1))
                nn.init.uniform_(m.weight, a=-bound, b=+bound)
                if m.bias is not None:
                    #bias for last layer to be inside the feasible region
                    b = self.lastlayerbias * int(idx == len(self.layers) - 1)
                    nn.init.constant_(m.bias, b)
    def forward(self, x):
        y = self.layers(x)
        if self.constrained: 
            mu = y[:, :self.ntypes**2]  # mu's (singles are implied)
            nu = y[:, self.ntypes**2:2*self.ntypes**2]  # nu's (LM) 
            V = y[:, 2*self.ntypes**2]  # cont. value
        else:
            mu = torch.sigmoid(y[:, :self.nout-1])
            nu = ()
            V = torch.exp(y[:, self.nout-1])
        return (mu, nu, V)


