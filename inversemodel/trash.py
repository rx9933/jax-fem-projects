import time
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.append('/workspace')

from math import sqrt
import jax
import jax.numpy as np
import numpy as onp
import basix
from jax_fem import logger
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh, cells_out, read_in_mesh
from jax_fem.basis import get_elements
from jax import jit


# ele_type = 'TET4'
# cell_type = get_meshio_cell_type(ele_type)     
# meshio_mesh = read_in_mesh("cell_vertices_initial.txt", cell_type)
# points = onp.asarray(meshio_mesh.points    )
# points = onp.loadtxt("cell_vertices_initial.txt")
# alpha = np.load('alpha.npy')
# print("alpha", alpha)
# x0 = onp.array([8.262040710449218750e+01, 8.185975646972656250e+01, 1.208771514892578125e+01])
# # ind = np.where(np.isclose(x0, points, atol=1e-5))
# tol = .01
# print(points[0])
# print(x0)
# # ind = np.where(np.linalg.norm(x0-points) < tol)
# print(points.shape)
# print(x0.shape)
# ind = onp.where(points==x0)
# # print(ind)
# ind = np.where(np.absolute(points-x0) < tol, 1, 0)
# i = np.nonzero(ind,size=1)#(ind==np.array([1,1,1]))
# print(i[0])
# print(alpha[i[0]])

import meshio
mesh = meshio.read('reference_gell.msh') # or something else?
pdata = mesh.points
# alpha_0 = np.ravel(np.ones((19634,4))) # 19634
alpha_0 = np.ones((len(pdata),))
np.save("alpha.npy", alpha_0)
print(np.load("alpha.npy"))
