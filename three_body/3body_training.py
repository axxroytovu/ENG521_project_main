from deep_learning import deep_network_core as core, utils
import torch
import torch.nn as nn
from torch.autograd import grad as autograd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.integrate import solve_ivp

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MSE_loss(core.LOSS):
    def __init__(self):
        self.loss = nn.MSELoss()
        
    def __call__(self, input, result, target, model):
        return self.loss(target, result[:, 0:2])

# Removed because this no longer applies
#class initial_condition_loss(core.LOSS):
#    def __init__(self, bounds):
#        self.bounds = bounds
#        self.mse = nn.MSELoss()
#
#    def __call__(self, input, result, target, model):
#        t = torch.zeros((100, 1))
#        inp = t
#        outpt = model(inp)
#        tgt = torch.cat((x, t, t, ydot), axis=1)
#        return self.mse(tgt, outpt)

class first_order_loss(core.LOSS):
    def __init__(self, bounds):
        self.bounds = bounds
    
    def __call__(self, input, result, target, model):
        t = torch.empty((100, 1)).uniform_(self.bounds.get('t', 'low'), self.bounds.get('t', 'high')).requires_grad_(True).to(device)
        inp = t
        output = model(inp)
        xdot_pred = torch.reshape(autograd(output[:, 0], t, grad_outputs=torch.ones((100)), create_graph=True)[0], (-1,))
        xdot_calc = output[:, 2]
        ydot_pred = torch.reshape(autograd(output[:, 1], t, grad_outputs=torch.ones((100)), create_graph=True)[0], (-1,))
        ydot_calc = output[:, 3]
        return torch.mean((xdot_calc-xdot_pred)**2) + torch.mean((ydot_calc-ydot_pred)**2)

class second_order_loss(core.LOSS):
    def __init__(self, bounds, mu=1/87.21):
        self.bounds = bounds
        self.mu = mu
    
    def __call__(self, input, result, target, model):
        x1 = -self.mu
        x2 = 1 - self.mu
        t = torch.empty((100, 1)).uniform_(self.bounds.get('t', 'low'), self.bounds.get('t', 'high')).requires_grad_(True).to(device)
        inp = t
        output = model(inp)
        r13 = torch.pow(torch.pow(output[:, 0]-x1, 2) + torch.pow(output[:, 1], 2), 3/2)
        r23 = torch.pow(torch.pow(output[:, 0]-x2, 2) + torch.pow(output[:, 1], 2), 3/2)
        xxdot_pred = torch.reshape(autograd(output[:, 2], t, grad_outputs=torch.ones((100)), create_graph=True)[0], (-1,)) * r13 * r23
        yydot_pred = torch.reshape(autograd(output[:, 3], t, grad_outputs=torch.ones((100)), create_graph=True)[0], (-1,)) * r13 * r23
        xxdot_calc = output[:, 0] * r13 * r23 + 2*output[:, 3] * r13 * r23 + (1-self.mu)*(x1 - output[:, 0])*r23 + self.mu*(x2 - output[:, 0])*r13
        yydot_calc = output[:, 1] * r13 * r23 - 2*output[:, 2] * r13 * r23 + r23 * (self.mu-1)*output[:, 1] - r13 * self.mu*output[:, 1]
        return torch.mean((xxdot_calc - xxdot_pred)**2) + torch.mean((yydot_calc-yydot_pred)**2)

class second_order_loss2(core.LOSS):
    def __init__(self, bounds, mu=1/87.21):
        self.bounds = bounds
        self.mu = mu
    
    def __call__(self, input, result, target, model):
        x1 = -self.mu
        x2 = 1 - self.mu
        t = torch.empty((100, 1)).uniform_(self.bounds.get('t', 'low'), self.bounds.get('t', 'high')).requires_grad_(True).to(device)
        inp = t
        output = model(inp)
        r13 = torch.pow(torch.pow(output[:, 0]-x1, 2) + torch.pow(output[:, 1], 2), 3/2)
        r23 = torch.pow(torch.pow(output[:, 0]-x2, 2) + torch.pow(output[:, 1], 2), 3/2)
        xxdot_pred = torch.reshape(autograd(output[:, 2], t, grad_outputs=torch.ones((100)), create_graph=True)[0], (-1,))
        yydot_pred = torch.reshape(autograd(output[:, 3], t, grad_outputs=torch.ones((100)), create_graph=True)[0], (-1,))
        xxdot_calc = output[:, 0] + 2*output[:, 3] + (1-self.mu)*(x1 - output[:, 0])/r13 + self.mu*(x2 - output[:, 0])/r23
        yydot_calc = output[:, 1] - 2*output[:, 2] + (self.mu-1)*output[:, 1]/r13 - self.mu*output[:, 1]/r23
        return torch.mean((xxdot_calc - xxdot_pred)**2) + torch.mean((yydot_calc-yydot_pred)**2)

mu = 1/81.27 # Mass ratio of the moon to the earth

def remadedynamics(t, x):
    x1 = -mu
    x2 = 1-mu
    r13 = np.pow(np.pow(x[0]-x1, 2) + np.pow(x[1], 2), 3/2)
    r23 = np.pow(np.pow(x[0]-x2, 2) + np.pow(x[1], 2), 3/2)
    dxdt = x[2]
    dydt = x[3]
    ddxdt = x[0] + 2*x[3] + (1-mu)*(x1 - x[0])/r13 + mu*(x2 - x[0])/r23
    ddydt = x[1] - 2*x[2] + (mu-1)*x[1]/r13 - mu*x[1]/r23
    return np.array([dxdt, dydt, ddxdt, ddydt])

bound = utils.bounds(x_low=0.68, x_high=0.7, ydot_low=0.64, ydot_high=0.66, t_low=0, t_high=8)

torch.manual_seed(123)
random = np.random.default_rng(123)

x0 = 0.690681027
ydot = .659003204
t_span = (0, 8)
v = np.array([x0, 0, 0, ydot])

ivp = solve_ivp(remadedynamics, t_span, v, method='Radau', dense_output=True, rtol=1e-9, atol=1e-9)

plotting = np.array([ivp.sol(t) for t in np.linspace(0, 8, 1001)])
data = [(t, ivp.sol(t)[:2]) for t in np.linspace(0, 8, 20)]

loss1 = [(1, MSE_loss())]
loss2 = [(1, MSE_loss()), (1, first_order_loss(bound)), (1e-4, second_order_loss2(bound))]
loss_temp = [(1, MSE_loss()), (1, first_order_loss(bound))]

network_pinn1 = core.PINN(1, 2, 64, 3, loss1)
network_pinn2 = core.PINN(1, 4, 64, 3, loss2)
network_dgm1 = core.DGM(1, 2, 64, 3, loss1)
network_dgm2 = core.DGM(1, 4, 64, 3, loss2)

pinn1_pred = []
pinn1_loss = []
pinn2_pred = []
pinn2_loss = []
dgm1_pred = []
dgm1_loss = []
dgm2_pred = []
dgm2_loss = []

x_test = np.linspace(0, 8, 101).reshape((101, 1))
out_test = np.array([ivp.sol(t) for t in x_test[:, 0]]).reshape((101, 4))

pinn1_pred.append(np.concatenate((network_pinn1.predict(x_test), network_pinn1.derivs(x_test)), axis=1))
pinn2_pred.append(network_pinn2.predict(x_test))
dgm1_pred.append(np.concatenate((network_dgm1.predict(x_test), network_dgm1.derivs(x_test)), axis=1))
dgm2_pred.append(network_dgm2.predict(x_test))
bulkloss = lambda x, y: np.mean(((x-y)**2).ravel())

train_in = np.array([x[0] for x in data])
train_out = np.array([x[1] for x in data])

scale = 10000
try:
    print(f"Training PINN1")
    for loss, test in network_pinn1.dynamic_fit(train_in, train_out, lr=1e-4, epochs=10*scale, testing=x_test, output=1000):
        deriv = network_pinn1.derivs(x_test)
        full_out = np.concatenate((test, deriv), axis=1)
        pinn1_loss.append(bulkloss(full_out, out_test))
        pinn1_pred.append(full_out)
    print(f"Training PINN2")
    for loss, test in network_pinn2.dynamic_fit(train_in, train_out, lr=1e-4, epochs=10*scale, testing=x_test, output=1000):
        pinn2_loss.append(bulkloss(test, out_test))
        pinn2_pred.append(test)
    print(f"Training DGM1")
    for loss, test in network_dgm1.dynamic_fit(train_in, train_out, lr=1e-4, epochs=10*scale, testing=x_test, output=1000):
        deriv = network_pinn1.derivs(x_test)
        full_out = np.concatenate((test, deriv), axis=1)
        dgm1_loss.append(bulkloss(full_out, out_test))
        dgm1_pred.append(full_out)
    print(f"Training DGM2")
    for loss, test in network_dgm2.dynamic_fit(train_in, train_out, lr=1e-4, epochs=10*scale, testing=x_test, output=1000):
        dgm2_loss.append(bulkloss(test, out_test))
        dgm2_pred.append(test)
except KeyboardInterrupt:
    pass

fig, (a1, a2) = plt.subplots(1, 2)

#plt.plot(plotting[:, 0], plotting[:, 1])
a1.scatter([d[1][0] for d in data], [d[1][1] for d in data], marker='o', c='blue', label="Training")
a1.plot(plotting[:, 0], plotting[:, 1], c='blue')
a2.plot(plotting[:, 2], plotting[:, 3], c='blue', label="Truth")
a1.plot(pinn1_pred[-1][:,0], pinn1_pred[-1][:,1], label="FF No Physics", c='red')
a2.plot(pinn1_pred[-1][:,2], pinn1_pred[-1][:,3], label="FF No Physics", c='red')
a1.plot(pinn2_pred[-1][:,0], pinn2_pred[-1][:,1], label="FF Physics", c='orange')
a2.plot(pinn2_pred[-1][:,2], pinn2_pred[-1][:,3], label="FF Physics", c='orange')
a1.plot(dgm1_pred[-1][:,0], dgm1_pred[-1][:,1], label="DGM No Physics", c='purple')
a2.plot(dgm1_pred[-1][:,2], dgm1_pred[-1][:,3], label="DGM No Physics", c='purple')
a1.plot(dgm2_pred[-1][:,0], dgm2_pred[-1][:,1], label="DGM Physics", c='green')
a2.plot(dgm2_pred[-1][:,2], dgm2_pred[-1][:,3], label="DGM Physics", c='green')
#plt.plot(pinn3_pred[-1][:,0], pinn3_pred[-1][:,1], label="2nd order")
#plt.quiver(pinn3_pred[-1][:,0], pinn3_pred[-1][:,1], pinn3_pred[-1][:,2], pinn3_pred[-1][:,3])
plt.legend()
a1.set_title("Position plot")
a2.set_title("Velocity plot")
a1.set_xlabel("x")
a1.set_ylabel("y")
a2.set_xlabel("x'")
a2.set_ylabel("y'")

torch.save(network_pinn1, "pinn_no_physics.pt")
torch.save(network_pinn2, "pinn_physics.pt")
torch.save(network_dgm1, "dgm_no_physics.pt")
torch.save(network_dgm2, "dgm_physics.pt")

plt.figure()
plt.title("Training Rate")
plt.semilogy(np.linspace(0, 10*scale, len(pinn1_loss)), pinn1_loss, label="FF No Physics", c='red')
plt.semilogy(np.linspace(0, 10*scale, len(pinn2_loss)), pinn2_loss, label="FF Physics", c='orange')
plt.semilogy(np.linspace(0, 10*scale, len(dgm1_loss)), dgm1_loss, label="DGM No Physics", c='purple')
plt.semilogy(np.linspace(0, 10*scale, len(dgm2_loss)), dgm2_loss, label="DGM Physics", c='green')
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("total loss")

plt.show()