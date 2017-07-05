#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_prop_cycle(plt.cycler('color', ['c', 'm', 'y', 'k']))

x = np.linspace(-1.0, 1.0, 50)
for f in [1.0, 2.0, 3.0, 4.0]:
    ax.plot(x, np.sin(x * f))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_prop_cycle(plt.cycler('color', ['c', 'm', 'y', 'k']))

x = np.linspace(-1.0, 1.0, 50)
y = np.linspace(-1.0, 1.0, 50)
z = np.zeros((len(x), len(y)))


def get_z(x, y, f):
    for i, u in enumerate(x):
        for j, v in enumerate(y):
            z[i, j] = (f * u) ** 2 + (f * v) ** 2
    return z


for f in [1.0, 2.0, 3.0, 4.0]:
    ax.contour(x, y, get_z(x, y, f), levels=[1], linewidth=2)

plt.show()
