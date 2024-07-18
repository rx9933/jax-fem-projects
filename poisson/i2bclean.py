import os
import sys
import numpy as onp
import jax
import jax.numpy as np
import meshio
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from matplotlib import pyplot as plt

# Set random seed for reproducibility
rng = onp.random.default_rng(seed=42)

# Define Poisson Problem class
class Poisson(Problem):
    def custom_init(self):
        self.fe = self.fes[0]

    def get_tensor_map(self):
        return lambda x: x
    
    def get_mass_map(self):
        def mass_map(u, x):
            b_quad = problem.fes[0].convert_from_dof_to_quad(b)[:, :, 0]
            x_points = problem.fe.get_physical_quad_points()
            matching_indices = np.where((np.array(x_points) == x).all(axis=-1), size=1)
            val = b_quad[matching_indices]
            
            if flag_1:
                val = -np.array([10 * np.exp(-((x[0] - 0.5)**2 + (x[1] - 0.5)**2) / 0.02)])
            return val
        return mass_map

    def set_params(self, params):
        global b
        b = np.array(params[0])

ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly = 1., 1.

meshio_mesh = rectangle_mesh(Nx=10, Ny=10, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

# Define boundary conditions
def left(point): return np.isclose(point[0], 0., atol=1e-5)
def right(point): return np.isclose(point[0], Lx, atol=1e-5)
def bottom(point): return np.isclose(point[1], 0., atol=1e-5)
def top(point): return np.isclose(point[1], Ly, atol=1e-5)
def dirichlet_val(point): return 0.

location_fns = [left, right, bottom, top]
value_fns = [dirichlet_val] * len(location_fns)
vecs = [0] * len(location_fns)
dirichlet_bc_info = [location_fns, vecs, value_fns]

# Create Poisson problem instance
problem = Poisson(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)

# Define real/ideal objective
global flag_1
flag_1 = True
b_ex = 2 * np.ones((problem.fe.num_cells, 1))
params = [b_ex]
problem.set_params(params)

# Implicit differentiation wrapper
fwd_pred = ad_wrapper(problem)
sol_0 = solver(problem, linear=True, use_petsc=True)

def test_fn(sol_list):
    u_0quad = problem.fes[0].convert_from_dof_to_quad(sol_0[0])[:, :, 0]
    cells_JxW = problem.JxW[:, 0, :]
    u_cquad = problem.fes[0].convert_from_dof_to_quad(sol_list[0])[:, :, 0]
    obj = np.sum((u_cquad - u_0quad)**2 * cells_JxW)
    print("Objective:", obj)
    return obj

def composed_fn(params):
    return test_fn(fwd_pred(params))

print("0 objective:", composed_fn(params))

flag_1 = False
b_fd_0 = np.array(rng.random((problem.fe.num_cells, 1)))
params = [b_fd_0]
problem.set_params(params)
Jb0 = composed_fn(params)
print("Current objective:", Jb0)
dJdb = jax.grad(composed_fn)(params)[0]

val = test_fn(fwd_pred(params))
h = 1e-6
db = h * b_fd_0
b_fd_1 = b_fd_0 + db
params_f = [b_fd_1]
problem.set_params(params_f)
dJdb_fd = (composed_fn(params_f) - val) / h
print("FD perturbation objective:", composed_fn([b_fd_0]))

g = np.einsum('ao,ao->o', dJdb, b_fd_0)[0]
f = dJdb_fd
print(f"AD gradient: {g:.11f}")
print(f"FD gradient: {f:.11f}")

x = []
y = []
for k in range(1, 8):
    e = 10**-k
    db = e * b_fd_0
    b_fd_1 = b_fd_0 + db
    params_f = [b_fd_1]
    problem.set_params(params_f)
    Jbp = composed_fn(params_f)
    Q = np.log(np.abs(Jb0 + e * g - Jbp))
    x.append(np.log(e))
    y.append(Q)

plt.plot(x, y, "*")
plt.title("Taylor Test for Convergence of dJ/db for Poisson Equation")
plt.xlabel("Log of Epsilon Value")
plt.ylabel("Log of Error (Q)")
plt.savefig("taylortest.png")

for i in range(len(x) - 1):
    dx = x[i + 1] - x[i]
    dy = y[i + 1] - y[i]
    print(dy / dx)
