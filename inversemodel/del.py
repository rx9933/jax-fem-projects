import meshio 
meshio = meshio.read("reference_gell.msh")
print(meshio.points.shape) # 11718

import numpy as np
print(np.load("gellpoints.npy").shape)
