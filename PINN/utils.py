import torch
import torch.nn as nn
from torch.autograd import grad as autograd
import numpy as np

class bounds():
    def __init__(self, xl, xh, yl, yh):
        self.bounds = {'x': {'low': xl, 'high': xh}, 
                       'y': {'low': yl, 'high': yh}}
    
    def train_bounds(self, **kwargs):
        if 'x_low' in kwargs:
            self.bounds['x']['low_train'] = kwargs['x_low']
        if 'x_high' in kwargs:
            self.bounds['x']['high_train'] = kwargs['x_high']
        if 'y_low' in kwargs:
            self.bounds['y']['low_train'] = kwargs['y_low']
        if 'y_high' in kwargs:
            self.bounds['y']['high_train'] = kwargs['y_high']
    
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

def build_data(base_metal, bound, size, seed=1234, train=False, interp=False):
    x = np.linspace(bound.get('x', 'low', train), bound.get('x', 'high', train), size[0])
    y = np.linspace(bound.get('y', 'low', train), bound.get('y', 'high', train), size[1])
    X, Y = np.meshgrid(x, y)
    U, V = base_metal(X, Y)
    inpt = np.column_stack((X.ravel(), Y.ravel()))
    oupt = np.column_stack((U.ravel(), V.ravel()))
    if interp:
        return (x,y), (U, V), X.shape
    return inpt, oupt, X.shape