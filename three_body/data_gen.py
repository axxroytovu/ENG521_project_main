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

data = '''4 1.420066759 -.750089226 .991171812
55 1.436012194 -.773041206 .990641562
56 1.482472688 -.839913548 .989488452
57 1.482479885 -.839923904 .989488310
58 1.482480749 -.839925147 .989488293
59 1.482480839 -.839925277 .989488291
60 1.484291896 -.842531112 .989452980
61 1.500000000 -.865127084 .989171608
62 1.527747691 -.905008902 .988770468
63 1.571709588 -.968071397 .988331206
64 1.599999994 -1.008550650 .988144472
65 1.693663090 -1.141870270 .987865285
66 1.693667486 -1.141876501 .987865280
67 1.693668589 -1.141878069 .987865273
68 1.695875689 -1.145005836 .987863040
69 1.711218700 -1.166731114 .987851332
70 1.719999999 -1.179151093 .987847426
71 1.759999990 -1.235596959 .987850654
72 1.779999986 -1.263741936 .987862566
73 1.799999997 -1.291836326 .987879650
74 1.800024092 -1.291870143 .987879673
75 1.820071399 -1.319981429 .987900960
76 1.840141490 -1.348076531 .987925568
77 1.860233486 -1.376155712 .987952773
78 1.880346999 -1.404220071 .987981963
79 1.899999991 -1.431600306 .988011882
80 1.900480792 -1.432269651 .988012628
81 1.920634091 -1.460305256 .988044339
82 1.940805793 -1.488327336 .988076740
83 1.960995197 -1.516336987 .988109537
84 1.981200993 -1.544334542 .988142488
85 2.000000000 -1.570353544 .988173087
86 2.001422375 -1.572321102 .988175396
87 2.021658182 -1.600297347 .988208099
88 2.041907191 -1.628263930 .988240472
89 2.062168181 -1.656221567 .988272412
90 2.082440287 -1.684171519 .988303844
91 2.099999994 -1.708365128 .988330601
92 2.102722079 -1.712114324 .988334708
93 2.123012185 -1.740050652 .988364964'''
rando = np.random.default_rng()
c = [a.split(' ') for a in data.split('\n')]

initial = np.hstack((rando.random((1000,1)), np.zeros((1000,1)), np.zeros((1000, 1)), rando.random((1000,1))))
t_span = (0, 10)

#full = rk4(dynamics, periods)
#test = rk4(lambda x: remadedynamics(0, x), periods[0])

#for i in range(3):
#    plt.plot(full[:,i,0], full[:,i,1], label=str(i))
#plt.plot(test[:, 0], test[:, 1], label='rk4')
with tqdm(enumerate(initial)) as tq:
    for i, v in tq:
        ivp = solve_ivp(remadedynamics, t_span, v, method='Radau')
        plt.plot(ivp.y[0], ivp.y[1], label=i)
plt.legend()
plt.show()