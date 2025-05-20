from deep_learning import deep_network_core as core, utils
import torch
import torch.nn as nn
from torch.autograd import grad as autograd
import numpy as np
import matplotlib.pyplot as plt
import pickle

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MSE_loss(core.LOSS):
    def __init__(self):
        self.loss = nn.MSELoss()
        
    def __call__(self, input, result, target, model):
        return self.loss(target, result)

class initial_condition_loss(core.LOSS):
    def __init__(self, bounds):
        self.bounds = bounds
        self.mse = nn.MSELoss()

    def __call__(self, input, result, target, model):
        x = torch.empty((100, 1)).uniform_(self.bounds.get('x', 'low'), self.bounds.get('x', 'high'))
        ydot = torch.empty((100, 1)).uniform_(self.bounds.get('ydot', 'low'), self.bounds.get('ydot', 'high'))
        t = torch.zeros((100, 1))
        inp = torch.cat((x, ydot, t), axis=1)
        outpt = model(inp)
        tgt = torch.cat((x, t, t, ydot), axis=1)
        return self.mse(tgt, outpt)

class first_order_loss(core.LOSS):
    def __init__(self, bounds):
        self.bounds = bounds
    
    def __call__(self, input, result, target, model):
        x0 = torch.empty((100, 1)).uniform_(self.bounds.get('x', 'low'), self.bounds.get('x', 'high'))
        ydot0 = torch.empty((100, 1)).uniform_(self.bounds.get('ydot', 'low'), self.bounds.get('ydot', 'high'))
        t = torch.empty((100, 1)).uniform_(self.bounds.get('t', 'low'), self.bounds.get('t', 'high')).requires_grad_(True).to(device)
        inp = torch.cat((x0, ydot0, t), axis=1)
        output = model(inp)
        xdot_pred = autograd(output[:, 0], t, grad_outputs=torch.ones((100)), create_graph=True)[0]
        xdot_calc = output[:, 2]
        ydot_pred = autograd(output[:, 1], t, grad_outputs=torch.ones((100)), create_graph=True)[0]
        ydot_calc = output[:, 3]
        return torch.mean((xdot_calc-xdot_pred)**2) + torch.mean((ydot_calc-ydot_pred)**2)

class second_order_loss(core.LOSS):
    def __init__(self, bounds, mu=1/87.21):
        self.bounds = bounds
        self.mu = mu
    
    def __call__(self, input, result, target, model):
        x1 = -self.mu
        x2 = 1 - self.mu
        x0 = torch.empty((100, 1)).uniform_(self.bounds.get('x', 'low'), self.bounds.get('x', 'high'))
        ydot0 = torch.empty((100, 1)).uniform_(self.bounds.get('ydot', 'low'), self.bounds.get('ydot', 'high'))
        t = torch.empty((100, 1)).uniform_(self.bounds.get('t', 'low'), self.bounds.get('t', 'high')).requires_grad_(True).to(device)
        inp = torch.cat((x0, ydot0, t), axis=1)
        output = model(inp)
        xxdot_pred = autograd(output[:, 2], t, grad_outputs=torch.ones((100)), create_graph=True)[0]
        yydot_pred = autograd(output[:, 3], t, grad_outputs=torch.ones((100)), create_graph=True)[0]
        r13 = -torch.threshold(-torch.pow(torch.pow(output[:, 0]-x1, 2) + torch.pow(output[:, 1], 2), 3/2), -0.0001, -0.0001)
        r23 = -torch.threshold(-torch.pow(torch.pow(output[:, 0]-x2, 2) + torch.pow(output[:, 1], 2), 3/2), -0.0001, -0.0001)
        xxdot_calc = output[:, 0] + 2*output[:, 3] + (1-self.mu)*(x1 - output[:, 0])/r13 + self.mu*(x2 - output[:, 0])/r23
        yydot_calc = output[:, 1] - 2*output[:, 2] + (self.mu-1)*output[:, 1]/r13 - self.mu*output[:, 1]/r23
        return torch.mean((xxdot_calc - xxdot_pred)**2) + torch.mean((yydot_calc-yydot_pred)**2)


data = []
with open("3bodygrid2.csv", 'r') as datafile:
    for line in datafile.readlines():
        raw = [float(x) for x in line.split(',')]
        data.append((raw[:3], raw[3:]))

bound = utils.bounds(x_low=0.68, x_high=0.7, ydot_low=0.64, ydot_high=0.66, t_low=0, t_high=10)

torch.manual_seed(123)
random = np.random.default_rng(123)

random.shuffle(data)

network_pinn = core.PINN(3, 4, 64, 3, [(1, MSE_loss()), (1, initial_condition_loss(bound)), (1, first_order_loss(bound))])
network_dgm = core.DGM(3, 4, 64, 3, [(1, MSE_loss()), (1, initial_condition_loss(bound)), (1, first_order_loss(bound))])

pinn_pred = []
pinn_loss = []
dgm_pred = []
dgm_loss = []

x0test = 0.690681027
ydottest = .659003204

x_test = np.stack((np.ones(100)*x0test, np.ones(100)*ydottest, np.linspace(0, 10, 100)), axis=1)

pinn_pred.append(network_pinn.predict(x_test))
dgm_pred.append(network_dgm.predict(x_test))

train_in = np.array([x[0] for x in data])
train_out = np.array([x[1] for x in data])

try:
    print(f"Training PINN")
    for loss, test in network_pinn.dynamic_fit(train_in, train_out, lr=1e-4, epochs=1000, testing=x_test):
        pinn_loss.append(loss)
        pinn_pred.append(test)

    print(f"Training DGM")
    for loss, test in network_dgm.dynamic_fit(train_in, train_out, lr=1e-4, epochs=1000, testing=x_test):
        dgm_loss.append(loss)
        dgm_pred.append(test)
except KeyboardInterrupt:
    pass

plt.figure()
for i, pred in enumerate(pinn_pred):
    plt.plot(pred[:,0], pred[:,1], label=i*100)
plt.legend()
plt.title("PINN")

plt.figure()
for i, pred in enumerate(dgm_pred):
    plt.plot(pred[:,0], pred[:,1], label=i*100)
plt.legend()
plt.title("DGM")
plt.show()

with open("dgm.pkl", 'wb') as dmod:
    pickle.dump(network_dgm, dmod)
with open("pinn.pkl", 'wb') as pmod:
    pickle.dump(network_pinn, pmod)
with open("ddat.pkl", 'wb') as ddat:
    pickle.dump(dgm_pred, ddat)
with open("pdat.pkl", 'wb') as pdat:
    pickle.dump(pinn_pred, pdat)
