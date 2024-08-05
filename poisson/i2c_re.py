import os
import sys
os.environ['JAX_PLATFORMS']='cpu'
import numpy as onp
import jax
import jax.numpy as np
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from matplotlib import pyplot as plt

rng = onp.random.default_rng(seed=42)
# Define constitutive relationship.
class Poisson(Problem):
    def custom_init(self):
        self.fe = self.fes[0]
    def get_tensor_map(self):
        return lambda x: x
    
    def get_mass_map(self):
        def mass_map(u, x):
            # if self.b is None:
            #     raise ValueError("Parameter 'b' has not been set.")
            b_quad = problem.fes[0].convert_from_dof_to_quad(b)[:, :, 0]
            x_points = problem.fe.get_physical_quad_points()
            matching_indices = np.where((np.array(x_points) == x).all(axis=-1), size =1)
            val = b_quad[matching_indices]
                        
            # let b be the cellsx1
            # calc b at quad points
            # calc which indicies of quad points x is 
            # index b at quad points with x
            # return this as val
            # print(val.shape)
            if flag_1:
                val = -np.array([10 * np.exp(-((x[0] - 0.5)**2 + (x[1] - 0.5)**2) / 0.02)])
            return val  # Use self.b here
        return mass_map

    def set_params(self, params):
        global b
        b = np.array(params[0])

# Specify mesh-related information.
ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)

# data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly = 1., 1.

# meshio_mesh = rectangle_mesh(Nx=10, Ny=10, domain_x=Lx, domain_y=Ly)
# mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
import meshio
mesh = meshio.read('data/msh/poisson.msh')
pdata = mesh.points
print(mesh.cells_dict.keys())
mesh = Mesh(pdata, mesh.cells_dict[cell_type])

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

problem = Poisson(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)

###### Define real/ideal, objective ~ 0?
global flag_1
flag_1 = True # ideal, use pre-specified b val
b_ex = 2 * np.ones((problem.fe.num_cells,1)) 
params = [b_ex]

# Set parameters in the problem instance
problem.set_params(params)

# Implicit differentiation wrapper
fwd_pred = ad_wrapper(problem) 
sol_list = fwd_pred(params)
# Solve the problem.
sol_0 = solver(problem, linear=True, use_petsc=True)  # Ensure to use the original b value

######### 


# Define objective function J
def test_fn(sol_list):
    u_0quad = problem.fes[0].convert_from_dof_to_quad(sol_0[0])[:, :, 0]
    cells_JxW = problem.JxW[:, 0, :]
    u_cquad = problem.fes[0].convert_from_dof_to_quad(sol_list[0])[:, :, 0]
    obj = np.sum((u_cquad - u_0quad)**2 * cells_JxW)
    print("OBJE",obj)
    return obj

def composed_fn(params):
    # print("Fwd prd", len(fwd_pred(params)[0]))
    return test_fn(fwd_pred(params))

# Derivative obtained by automatic differentiation

print("0 objective", composed_fn(params))

flag_1 = False

b_fd_0 = np.array(rng.random((problem.fe.num_cells,1)))
params = [b_fd_0]
#@@@@
problem = Poisson(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
problem.set_params(params)
#@@@@

# Implicit differentiation wrapper
fwd_pred = ad_wrapper(problem) 
sol_list = fwd_pred(params)
Jb0 = composed_fn(params)
print("Curr objective", Jb0) 
dJdb = jax.grad(composed_fn)(params)[0]


val = test_fn(sol_list)
# Small perturbation
h = 10**-6
db = h*b_fd_0

# Forward difference for finite difference approximation
b_fd_1 = b_fd_0 + db
params_f = [b_fd_1]

#@@@@
problem = Poisson(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
problem.set_params(params_f)
#@@@@

# Implicit differentiation wrapper
fwd_pred = ad_wrapper(problem) 
sol_list = fwd_pred(params_f)

dJdb_fd = (composed_fn(params_f) - val)/h 
print("FD pertuberation objective",composed_fn([b_fd_0]))


g = np.einsum('ao,ao->o',dJdb, b_fd_0)[0]
f = dJdb_fd


print(f"{g:.11f}")
print(f"{f:11f}")

x = []
y = []
for k in range(1,8):
    e = 10**-k

    h = e

    db = h*b_fd_0

    # Forward difference for finite difference approximation
    b_fd_1 = b_fd_0 + db
    params_f = [b_fd_1]

    #@@@@
    problem = Poisson(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
    problem.set_params(params_f)
    #@@@@

    # Implicit differentiation wrapper
    fwd_pred = ad_wrapper(problem) 
    sol_list = fwd_pred(params_f)

    Jbp = composed_fn(params_f) 

    Q = np.log(np.abs(Jb0+e*g-Jbp))
    
    x.append(np.log(e))
    y.append(Q)

plt.plot(x,y,"*")
plt.title("Taylor Test for Convergence of dJ/db for Poisson Equation")
plt.xlabel("Log of Episilon Value")
plt.ylabel("Log of Error (Q)")
plt.savefig("taylortest.png")

for i in range(len(x)-1):
    dx = x[i+1]-x[i]
    dy = y[i+1]-y[i]
    print(dy/dx)
