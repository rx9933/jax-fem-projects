import meshio
import numpy as np
m = meshio.read("reference_domain.xdmf")
for p in m.points:
    if np.linalg.norm(p-np.array([67.1986,227.034,-25.635]))<10**-1:
        print(p)
            