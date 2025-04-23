import numpy as np
from matplotlib import pyplot as plt
from diffKDE import diffKDE



xmin = 0
xmax = 1

data = np.array([0.1, 0.2, 0.3, 0.33, 0.34, 0.35, 0.36, 0.37, 0.5, 0.55, 0.7, 0.8])

result = diffKDE.KDE(data, xmin = xmin, xmax = xmax)


# color map
cm = plt.get_cmap('tab20')

fig = plt.figure()
ax = fig.add_subplot()

plt.plot(result[1], result[0], label = 'diffKDE') # plot against the discretization area omega
plt.legend()
plt.grid()
plt.xlabel('data value')
plt.ylabel('estimated density')
plt.title('Diffusion-based density estimate of a data sample')
plt.tight_layout()
plt.show()
#plt.savefig("Testbild.png")

exit()
