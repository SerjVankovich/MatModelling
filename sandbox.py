import numpy as np
import matplotlib.pyplot as plt
from task8 import fun

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

x, y = np.meshgrid(np.linspace(-6, 6, 100), np.linspace(-10, 27, 100))

z = fun(x, y)

ax.plot_surface(x, y, z)

plt.show()
