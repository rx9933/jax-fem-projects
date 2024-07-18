# Import some useful modules.
import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import matplotlib.pyplot as plt


# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

# Define constitutive relationship.
class Poisson(Problem):
    def get_tensor_map(self):
        return lambda x: x
    
    def get_mass_map(self):
        def mass_map(u, x):
            # val = -np.array([10*np.exp(-(np.power(x[0] - 0.5, 2) + np.power(x[1] - 0.5, 2)) / 0.02)])
            val = self.bv
            return val
        return mass_map

    def set_params(self, params):
        global bv
        bv = params
        self.bv = bv
        # self.fe.dirichlet_bc_info[-1][-1] = get_dirichlet_bottom(scale_d)
        # self.fe.update_Dirichlet_boundary_conditions(self.fe.dirichlet_bc_info)

# Specify mesh-related information. 
ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly = 1., 1.
meshio_mesh = rectangle_mesh(Nx=32, Ny=32, domain_x=Lx, domain_y=Ly)

mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

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

# Create an instance of the Class 'Poisson'. 
problem = Poisson(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)

# Solve the problem.
sol_0 = solver(problem, linear=True, use_petsc=True) # ensure to use ORiginal B value

# Define parameters.
b = 0.5*np.ones((problem.fe.num_cells, problem.fe.num_quads))
params = [b]


# Implicit differentiation wrapper
fwd_pred = ad_wrapper(problem) 
sol_list = fwd_pred(params)

vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
save_sol(problem.fe, sol_list[0], vtk_path)

# change to objective, J
def test_fn(sol_list):
    u_0quad = problem.fes[0].convert_from_dof_to_quad(sol_0[0])[:,:,0]
    cells_JxW = problem.JxW[:,0,:]
    u_cquad = problem.fes[0].convert_from_dof_to_quad(sol_list[0])[:,:,0] # modify to (sol[0] - u_0)**2 for J?

    obj = np.sum((u_cquad-u_0quad)*cells_JxW)

    return obj

def composed_fn(params):
    return test_fn(fwd_pred(params))

val = test_fn(sol_list)

h = 1e-3 # small perturbation


# Forward difference
b_plus = (1 + h)*b
params_b = [b]
dJdb_fd = (composed_fn(params_b) - val)/(h*b)

# Derivative obtained by automatic differentiation
dJdb = jax.grad(composed_fn)(params)


# Comparison
# print(f"\nDerivative comparison between automatic differentiation (AD) and finite difference (FD)")
# print(f"\ndE = {dE}, dE_fd = {dE_fd}, WRONG results! Please avoid gradients w.r.t self.E")
# print(f"This is due to the use of glob variable self.E, inside a jax jitted function.")
# print(f"\ndrho[0, 0] = {drho[0, 0]}, drho_fd_00 = {drho_fd_00}")
print(f"\dJdb = {dJdb}, dJdb_fd = {dJdb_fd}")


