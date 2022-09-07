import numpy as np
import matplotlib.pyplot as plt
import math

x = np.arange(0, 10, 0.1)
y = np.sin(x)
x_m, y_m = np.meshgrid(x, y)
z = x_m + 5 * y_m
print(z)
print(z.shape)

x = np.linspace(-10, 10, 21)
y = x.copy().T # transpose
z = np.ones((21, 21))

with open('../dumpZ.txt', 'rt') as f:
    for i in range(21):
        line = f.readline().split()
        for j in range(21):
            # z[i, j] = math.log10(float(line[j]))
            z[i, j] = float(line[j])
print(z)
print(z.shape)

fig = plt.figure()
ax = plt.axes(projection='3d')

x, y = np.meshgrid(x, y)

ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
ax.set_title('Loss with delta weight')

plt.show()