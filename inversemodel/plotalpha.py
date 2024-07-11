import numpy as np
import matplotlib.pyplot as plt
p = np.load("gellpoints.npy")
vi = np.loadtxt("cell_vertices_initial.txt")
alpha = np.load("alphafinal.npy")
print("alpha",alpha)
d = np.zeros((len(p),1))
# a = np.zeros((len(p),1))
for i in range(len(p)):
    x0 = p[i]
    rs = np.min(np.linalg.norm(x0-vi,axis=1)) # distance to cell surface
    d[i]=rs
plt.scatter(d,alpha)
plt.minorticks_on()
# plt.tick_params(which='minor', length=10, color='green')
plt.grid()
plt.xlabel("distance from cell surface")
plt.ylabel("alpha values")
plt.savefig("alphavsdist.png")