import torch
import torch.nn as nn
from torch.autograd import grad as autograd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from abc import ABC, abstractmethod

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def numpy_to_tensor(x):
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float).to(device).reshape(n_samples, -1)

class Network(nn.Module):
    def dynamic_fit(self, X, y, lr=1e-3, epochs=1000, output=[], testing=None):
        if not output:
            output = [int(i) for i in np.linspace(0, epochs, 11)]
        Xt = numpy_to_tensor(X)
        yt = numpy_to_tensor(y)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.forward(Xt)
            loss = sum([a*b(Xt, out, yt, self) for a, b in self.loss])
            loss.backward()
            optimizer.step()
            if epoch in output or epoch == epochs-1:
                print(f"Epoch {epoch+1}/{epochs} loss: {loss.item():.2g}")
                if testing is not None:
                    test = self.predict(testing)
                    self.train()
                else:
                    test = None
                yield loss.item, test

    def fit(self, X, y, lr=1e-3, epochs=1000):
        return [i for i, j in self.dynamic_fit(X, y, lr, epochs)]

    def predict(self, x):
        self.eval()
        out = self.forward(numpy_to_tensor(x))
        return out.detach().cpu().numpy()

class PINN(Network):
    def __init__(self, in_dim, out_dim, layer_size, layer_count, loss):
        super().__init__()
        self.loss = loss
        layers = [nn.Linear(in_dim, layer_size), nn.ReLU()]
        layers += [nn.ReLU() if i%2 else nn.Linear(layer_size, layer_size) for i in range(2*layer_count)]
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(layer_size, out_dim)

    def forward(self, x):
        h = self.layers(x)
        return self.out(h)

class DGM_LSTM(nn.Module):
    def __init__(self, in_dim, out_dim):
        '''
        args:
            in_dim: dimension of input data
            out_dim: number of LSTM layer outputs
            
        returns: custom layer object
        '''
        super().__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim

        self.activate1 = nn.Tanh()
        self.activate2 = nn.Tanh()

        self.uz = nn.Parameter(torch.Tensor(in_dim, out_dim).random_())
        self.ug = nn.Parameter(torch.Tensor(in_dim, out_dim).random_())
        self.ur = nn.Parameter(torch.Tensor(in_dim, out_dim).random_())
        self.uh = nn.Parameter(torch.Tensor(in_dim, out_dim).random_())

        self.wz = nn.Parameter(torch.Tensor(out_dim, out_dim).random_())
        self.wg = nn.Parameter(torch.Tensor(out_dim, out_dim).random_())
        self.wr = nn.Parameter(torch.Tensor(out_dim, out_dim).random_())
        self.wh = nn.Parameter(torch.Tensor(out_dim, out_dim).random_())

        self.bz = nn.Parameter(torch.ones(1, out_dim))
        self.bg = nn.Parameter(torch.ones(1, out_dim))
        self.br = nn.Parameter(torch.ones(1, out_dim))
        self.bh = nn.Parameter(torch.ones(1, out_dim))
    
    def forward(self, S, X):
        Z = self.activate1(torch.add(torch.add(torch.matmul(X, self.uz), torch.matmul(S, self.wz)), self.bz))
        G = self.activate1(torch.add(torch.add(torch.matmul(X, self.ug), torch.matmul(S, self.wg)), self.bg))
        R = self.activate1(torch.add(torch.add(torch.matmul(X, self.ur), torch.matmul(S, self.wr)), self.br))

        H = self.activate2(torch.add(torch.add(torch.matmul(X, self.uh), torch.matmul(torch.mul(S, R), self.wh)), self.bh))

        S_new = torch.add(torch.mul(torch.sub(torch.ones_like(G), G), H), torch.mul(Z, S))

        return S_new

class DGM(Network):
    def __init__(self, in_dim, out_dim, layer_size, layer_count, loss):
        super().__init__()
        self.loss = loss
        self.first = nn.Sequential(nn.Linear(in_dim, layer_size), nn.ReLU())
        self.layers = [
            DGM_LSTM(in_dim, layer_size) for _ in range(layer_count)
        ]
        self.out = nn.Linear(layer_size, out_dim)

    def forward(self, x):
        x_copy = x
        temp = self.first(x)
        for layer in self.layers:
            temp = layer.forward(temp, x_copy)
        return self.out(temp)

class LOSS(ABC):
    @abstractmethod
    def __call__(self, input, result, target, model) -> torch.Tensor:
        return torch.Tensor([0])
