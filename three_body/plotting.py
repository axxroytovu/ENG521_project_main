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
ivp = solve_ivp(remadedynamics, t_span, v, method='Radau', dense_output=True, rtol=1e-9, atol=1e-9)
truth = np.array([ivp.sol(t) for t in np.linspace(0, 10, 1001)])

valid=[
    (0.6906122448979591, 0.6587755102040816),
    (0.6906122448979591, 0.6591836734693878),
    (0.6910204081632653, 0.6587755102040816),
    (0.6910204081632653, 0.6591836734693878)
    ]

d = {v: [] for v in valid}

with open("3bodygrid3.csv", 'r') as dfile:
    for l in dfile.readlines():
        data = [float(x) for x in l.split(',')]
        if (data[0], data[1]) in valid:
            d[(data[0], data[1])].append((data[2], data[3], data[4]))

for v in valid:
    d[v] = sorted(d[v])

with open("pdat1.pkl", 'rb') as pfile:
    pdata = pickle.load(pfile)
with open("ddat1.pkl", 'rb') as dfile:
    ddata = pickle.load(dfile)

plt.figure()
plt.plot(truth[:, 0], truth[:, 1], label="Truth")
for i, x in enumerate(pdata):
    plt.plot(x[:, 0], x[:, 1], label=f"Epoch {i*1000}")
for v in d.values():
    plt.plot([x[1] for x in v], [x[2] for x in v], label="Data")
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