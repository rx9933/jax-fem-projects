import os
import sys
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as np
import numpy as onp
# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

# Import the custom solution functions.
from u_current import return_sol as get_ucurr
from u_real import return_sol as get_ureal

ureal = get_ureal()
ucurrent = get_ucurr()

# Calculate the difference field.
b = 2 * (np.array(ucurrent) - np.array(ureal))
print("B shape:", b.shape)

# Specify mesh-related information.
ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 1., 1.
meshio_mesh = rectangle_mesh(Nx=32, Ny=32, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
p = np.array(meshio_mesh.points)

# Define the Poisson problem class.
class Poisson(Problem):
    def get_tensor_map(self):
        return lambda x: x

    def get_mass_map(self):
        def mass_map(u, x):
            
            # Reshape points and act for broadcasting
            points_reshaped = x.reshape(-1, 2)
            act_reshaped = p.reshape(1, -1, 2)

            # Use broadcasting to find matching indices
            matches = np.all(points_reshaped[:, None, :] == act_reshaped, axis=-1)

            # Get the index positions of the matches
            index_positions = np.argmax(matches, axis=1)

            # Reshape to the original shape of points
            index_matrix = index_positions.reshape(x.shape)#[:-1])
            print(x.shape)
            jax.debug.print("i:{}",index_matrix)
            val = -np.array([10*np.exp(-(np.power(x[0] - 0.5, 2) + np.power(x[1] - 0.5, 2)) / 0.02)])
            return val
        return mass_map

# Define boundary locations.
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)

def bottom(point):
    return np.isclose(point[1], 0., atol=1e-5)

def top(point):
    return np.isclose(point[1], Ly, atol=1e-5)

# Define Dirichlet boundary values.
def dirichlet_val(point):
    return 0.

location_fns = [left, right, bottom, top]
value_fns = [dirichlet_val] * len(location_fns)
vecs = [0] * len(location_fns)
dirichlet_bc_info = [location_fns, vecs, value_fns]

# Create an instance of the Poisson class.
problem = Poisson(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)

# Solve the problem.
sol = solver(problem, linear=True, use_petsc=True)

# Save the solution to a local folder for visualization with ParaView.
data_dir = os.path.join(os.path.dirname(__file__), 'data')
vtk_path = os.path.join(data_dir, f'vtk/adjoint.vtu')
save_sol(problem.fes[0], sol[0], vtk_path)
