import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import solve_ivp

mu = 1/81.27 # Mass ratio of the moon to the earth

def dynamics(x):
    '''
    x is a numpy array of:
    x[:, 0] = x
    x[:, 1] = y
    x[:, 2] = dx/dt
    x[:, 3] = dy/dt
    note: x and y are in sidereal coordinates
    '''
    if len(x.shape) == 1 or x.shape[1] == 1:
        x = x.reshape(1, -1)
    x1 = -mu
    x2 = 1 - mu
    r13 = np.pow(np.pow(x[:,0]-x1, 2) + np.pow(x[:,1], 2), 3/2)
    r23 = np.pow(np.pow(x[:,0]-x2, 2) + np.pow(x[:,1], 2), 3/2)
    full_lib = np.concatenate((
        x, 
        (x[:, 0]/r13).reshape(-1, 1), 
        (x[:, 0]/r23).reshape(-1, 1), 
        (x[:, 1]/r13).reshape(-1, 1), 
        (x[:, 1]/r23).reshape(-1, 1),
        (1/r13).reshape(-1, 1),
        (1/r23).reshape(-1, 1)
    ), axis=1)
    timestepmat = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 2, -(1-mu), -mu, 0, 0, (1-mu)*x1, mu*x2],
        [0, 1, -2, 0, 0, 0, -(1-mu), -mu, 0, 0]
    ])
    return full_lib @ timestepmat.T

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

def rk4(func, starting, delta=0.0001, steps=100000):
    tstep = [starting]
    with tqdm(range(steps)) as tq:
        for t in tq:
            k1 = delta * func(tstep[-1])
            k2 = delta * func(tstep[-1] + k1*0.5)
            k3 = delta * func(tstep[-1] + k2*0.5)
            k4 = delta * func(tstep[-1] + k3)
            tstep.append(tstep[-1] + (k1 + 2*k2 + 2*k3 + k4)/6)
    return np.array(tstep)

rando = np.random.default_rng()

xmin = 0.68
xmax = 0.7
xdelt = xmax - xmin
ydotmin = 0.64
ydotmax = 0.66
ydotdelt = ydotmax-ydotmin

x = np.linspace(xmin, xmax, 50)
y = np.linspace(ydotmin, ydotmax, 50)
X, Y = np.meshgrid(x, y)
x_ = X.ravel()
y_ = Y.ravel()

initial = np.stack((x_, np.zeros(x_.shape), np.zeros(x_.shape), y_), axis=-1)
print(initial.shape)
t_span = (0, 10)

#full = rk4(dynamics, periods)
#test = rk4(lambda x: remadedynamics(0, x), periods[0])

#for i in range(3):
#    plt.plot(full[:,i,0], full[:,i,1], label=str(i))
#plt.plot(test[:, 0], test[:, 1], label='rk4')
threebdfile = "3bodygrid3.csv"
with tqdm(list(enumerate(initial))) as tq: 
    for i, v in tq:
        try:
            ivp = solve_ivp(remadedynamics, t_span, v, method='Radau', dense_output=True, rtol=1e-9, atol=1e-9)
            with open(threebdfile, 'a') as tf:
                for t in np.linspace(0, 10, 101):
                    data = ivp.sol(t)
                    tf.write(f"{v[0]},{v[3]},{t},{data[0]},{data[1]},{data[2]},{data[3]}\n")
        except KeyboardInterrupt:
            pass
