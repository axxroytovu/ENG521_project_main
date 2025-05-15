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
