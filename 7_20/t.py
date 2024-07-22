
import numpy as np
import meshio
mesh = meshio.read("reference_domain.xdmf")
print(mesh.points.shape)
for i,p in enumerate(mesh.points):
    if np.linalg.norm(p-np.array([8.262040710449218750e+01, 8.185975646972656250e+01, 1.208771514892578125e+01]))<10**-2:
        print(i)

print(mesh.points[-1])