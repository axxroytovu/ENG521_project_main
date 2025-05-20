from deep_learning import deep_network_core as core, utils
import torch
import torch.nn as nn
from torch.autograd import grad as autograd
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MSE_Loss(core.LOSS):
    def __init__(self):
        self.loss = nn.MSELoss()
        
    def __call__(self, input, result, target, model):
        return self.loss(target, result)

class PHYSICS_Loss(core.LOSS):
    def __init__(self, bounds):
        self.bounds = bounds
    
    def __call__(self, input, result, target, model):
        x = torch.empty((100, 1)).uniform_(self.bounds.get('x', 'low'), self.bounds.get('x', 'high')).requires_grad_(True).to(device)
        y = torch.empty((100, 1)).uniform_(self.bounds.get('y', 'low'), self.bounds.get('y', 'high')).requires_grad_(True).to(device)
        inp = torch.cat((x, y), axis=1)
        zs = model(inp)
        pde = utils.divergence(zs, inp)
        return torch.mean(pde**2)

class SYMMETRY_Loss(core.LOSS):
    def __init__(self, bounds):
        self.bounds = bounds
    
    def __call__(self, input, result, target, model):
        x = torch.empty((100, 1)).uniform_(self.bounds.get('x', 'low'), self.bounds.get('x', 'center')).requires_grad_(True).to(device)
        y = torch.empty((100, 1)).uniform_(self.bounds.get('y', 'low'), self.bounds.get('y', 'high')).requires_grad_(True).to(device)
        center = torch.tensor([self.bounds.get('x', 'center'), self.bounds.get('y', 'center')])
        inp = torch.cat((x, y), axis=1)
        t2 = (-1 * (inp - center)) + center
        sym = model(inp) + model(t2)
        return torch.mean(sym**2)

def taylor_green(x, y):
    u = np.sin(2*np.pi*x) * np.cos(2*np.pi*y)
    v = -np.cos(2*np.pi*x) * np.sin(2*np.pi*y)
    return u, v

quad_bound = utils.bounds(x_low=0, x_high=1, y_low=0, y_high=1)

torch.manual_seed(1234)
np.random.seed(1234)
x_train = np.linspace(quad_bound.get('x', 'low', True), quad_bound.get('x', 'high', True), 10)
y_train = np.linspace(quad_bound.get('y', 'low', True), quad_bound.get('y', 'high', True), 10)
Xr, Yr = np.meshgrid(x_train, y_train)
Ur, Vr = taylor_green(Xr, Yr)
inpt = np.column_stack((Xr.ravel(), Yr.ravel()))
oupt = np.column_stack((Ur.ravel(), Vr.ravel()))

network_pinn = core.PINN(2, 2, 32, 2, [(1, MSE_Loss()), (1, PHYSICS_Loss(quad_bound)), (1, SYMMETRY_Loss(quad_bound))])
network_dgm = core.DGM(2, 2, 64, 3, [(1, MSE_Loss()), (1, PHYSICS_Loss(quad_bound)), (1, SYMMETRY_Loss(quad_bound))])

print("Training PINN")
network_pinn.fit(inpt, oupt, lr=1e-4, epochs=10000)

print("Training DGM")
network_dgm.fit(inpt, oupt, lr=1e-4, epochs=10000)

x_test = np.linspace(quad_bound.get('x', 'low', False), quad_bound.get('x', 'high', False), 100)
y_test = np.linspace(quad_bound.get('y', 'low', False), quad_bound.get('y', 'high', False), 100)
Xt, Yt = np.meshgrid(x_test, y_test)
Ut, Vt = taylor_green(Xt, Yt)
in_tst = np.column_stack((Xt.ravel(), Yt.ravel()))
ou_tst = np.column_stack((Ut.ravel(), Vt.ravel()))

pinn_pred = network_pinn.predict(in_tst)
dgm_pred = network_dgm.predict(in_tst)

plt.figure()
plt.streamplot(Xt, Yt, Ut, Vt)
plt.title("Truth")

plt.figure()
plt.streamplot(Xt, Yt, pinn_pred[:,0].reshape((100, 100)), pinn_pred[:,1].reshape((100, 100)))
plt.title("PINN")

plt.figure()
plt.streamplot(Xt, Yt, dgm_pred[:,0].reshape((100, 100)), dgm_pred[:,1].reshape((100, 100)))
plt.title("DGM")
plt.show()
