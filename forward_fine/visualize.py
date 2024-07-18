import matplotlib.pyplot as plt
import numpy as np

data0 = np.loadtxt("cell_vertices_final.txt") # cell_vertices_initial
x = data0[:,0]
y = data0[:,1]
z = data0[:,2]


# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x,y,z, c='red', marker='.', s=10)

# data = np.loadtxt("cell_vertices_initial.txt") # cell_vertices_initial
# x = data[:,0]
# y = data[:,1]
# z = data[:,2]
import meshio
mesh = meshio.read('data/vtk/celltest.vtu')
data = mesh.points
x = data[:,0]
y = data[:,1]
z = data[:,2]

ax.scatter(x,y,z, c='blue', marker='.', s=1, alpha = .3)

plt.show()

for point in data:
    for actpoint in data0:
        if np.linalg.norm(point-actpoint)<10**-1:
            print(1)