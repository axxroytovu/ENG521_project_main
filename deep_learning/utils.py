import torch
import torch.nn as nn
from torch.autograd import grad as autograd
import numpy as np

class bounds():
    def __init__(self, **kwargs):
        self.bounds = {}
        for k, v in kwargs.items():
            dim, edge = k.split("_")
            if dim not in self.bounds:
                self.bounds[dim] = {}
            self.bounds[dim][edge] = v
    
    def train_bounds(self, **kwargs):
        for k, v in kwargs.items():
            dim, edge = k.split("_")
            self.bounds[dim][edge+'_train'] = v
    
    def get(self, dim, dir, train=False):
        if dir == 'center':
            return (self.get(dim, 'low', train) + self.get(dim, 'high', train))/2
        if train:
            return self.bounds[dim].get(dir+'_train', self.bounds[dim][dir])
        return self.bounds[dim][dir]

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += autograd(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div

def grad(outputs, inputs):
    return autograd(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True
    )

# general laplacian isn't currently working
def laplacian(y, x):
    lap = torch.zeros_like(y)
    for xi in x:
        dy_dxi = autograd(y, xi, torch.ones_like(y), create_graph=True)[0]
        lap += autograd(dy_dxi, xi, torch.ones_like(y), create_graph=True)[0]
    return lap

def laplacian_1d(y, x):
    dy_dx = autograd(y, x, torch.ones_like(y), create_graph=True)[0]
    return autograd(dy_dx, x, torch.ones_like(dy_dx), create_graph=True)[0]

def laplacian_2d(z, x, y):
    lap = torch.zeros_like(z)
    dz_dx = autograd(z, x, torch.ones_like(z), create_graph=True)[0]
    lap += autograd(dz_dx, x, torch.ones_like(dz_dx), create_graph=True)[0]
    dz_dy = autograd(z, y, torch.ones_like(z), create_graph=True)[0]
    lap += autograd(dz_dy, y, torch.ones_like(dz_dy), create_graph=True)[0]
    return lap

def dy_dt(y, t):
    return autograd(y, t, torch.ones_like(y), create_graph=True)[0]
