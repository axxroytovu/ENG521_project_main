# Plate

# Following this video: https://www.youtube.com/watch?v=CXOrkQs4WYo

import numpy as np
import matplotlib.pyplot as plt

alpha = 110
length = 50 # length of the plate, 50 mm
time = 2 # total time, 10 s
nodes = 40 # number of nodes - 1

dx = length / nodes # distance between nodes
dy = length / nodes # distance between nodes
# time step, must be less or equal to than min of dx^2 / (4 * alpha) and dy^2 / (4 * alpha)
dt = np.min([0.25 * dx**2 / alpha, 0.25 * dy**2 / alpha]) 

u = np.zeros((nodes, nodes)) + 20 # middle of the plate is 20 degrees
# for i in range(len(u[0, :])): # top ranges from 20 to 78 degrees linearly
#     u[0, i] = 20 + 3*i
u[0, :] = 100 # bottom side of the plate is 100 degrees
# u[-1, :] = 100 # top side of the plate is 100 degrees
# u[:, 0] = 100 # left side of the plate is 100 degrees
# u[:, -1] = 100 # right side of the plate is 100 degrees

# fig, axis = plt.subplots()
# pcm = axis.pcolormesh(u, cmap='gist_rainbow', vmin=0, vmax=100)
# plt.colorbar(pcm, ax=axis)

heat_data = np.zeros((nodes, nodes, int(time / dt) + 2))

counter = 0
while counter * dt < time:
    w = u.copy()

    for i in range(1, nodes - 1):
        for j in range(1, nodes - 1):
            dd_ux = (w[i - 1, j] - 2 * w[i, j] + w[i + 1, j]) / dx**2
            dd_uy = (w[i, j - 1] - 2 * w[i, j] + w[i, j + 1]) / dy**2

            u[i, j] = dt * alpha * (dd_ux + dd_uy) + w[i, j]

    counter += 1
    heat_data[:, :, counter] = u.copy()

    # pcm.set_array(u)
    # plt.pause(0.01)
    # axis.set_title(f't: {counter:.3f} s')
    print(f't: {counter * dt:.3f} s, Ave temp: {np.mean(u):.2f} C')

# plt.show()
print(heat_data.shape)