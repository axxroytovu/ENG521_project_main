import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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


x0test = 0.690681027
ydottest = .659003204
t_span = (0, 10)
v = np.array([x0test, 0, 0, ydottest])
ivp = solve_ivp(remadedynamics, t_span, v, method='Radau', dense_output=True)
truth = np.array([ivp.sol(t) for t in np.linspace(0, 10, 1001)])

with open("pdat.pkl", 'rb') as pfile:
    pdata = pickle.load(pfile)
with open("ddat.pkl", 'rb') as dfile:
    ddata = pickle.load(dfile)

plt.figure()
plt.plot(truth[:, 0], truth[:, 1], label="Truth")
for i, x in enumerate(pdata):
    plt.plot(x[:, 0], x[:, 1], label=f"Epoch {i*1000}")
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.title("PINN")
plt.legend()

plt.figure()
plt.plot(truth[:, 0], truth[:, 1], label="Truth")
for i, x in enumerate(ddata):
    plt.plot(x[:, 0], x[:, 1], label=f"Epoch {i*1000}")
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.title("DGM")
plt.legend()
plt.show()