import time
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.append('/workspace')
os.environ['JAX_PLATFORMS'] = 'cpu'
from math import sqrt
import jax
import jax.numpy as np
import numpy as onp
import basix
from jax_fem import logger
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh, cells_out, read_in_mesh
from jax_fem.basis import get_elements
from jax import jit
from scipy.optimize import minimize
class UnitSystem:
    def __init__(self, simlen_to_meters):
        self.simlen_to_meters = simlen_to_meters
        self.meters_to_simlen = 1. / simlen_to_meters
        self.um = 1e-6 * self.simlen_to_meters
        self.GPa = 1e6 * self.meters_to_simlen

units = UnitSystem(300e-6)
CENTER = np.array([0, 0, 0])
radius_cell = 30 #* units.um
gel_section_width = 300 #* units.um
SQRT2 = np.sqrt(2)
beta = np.log(0.5) / (np.log(radius_cell) - np.log(0.5 * gel_section_width * SQRT2 - radius_cell))
logger.info(f"Found beta={beta}")



def localize(orig_mat):
    cells, points = cells_out()
    # Number of points
    num_points = points.shape[0]

    # Flatten cells and orig_mat
    flattened_cells = cells.flatten()
    flattened_orig_mat = orig_mat.flatten()

    # Create indices for repeated points
    repeated_indices = np.repeat(np.arange(cells.shape[0]), cells.shape[1])
    point_indices = flattened_cells

    # Create an array to match points to their original indices
    updates_local_mat = np.zeros(num_points).at[point_indices].add(flattened_orig_mat)
    updates_num_repeat = np.zeros(num_points).at[point_indices].add(1)

    # Normalize local_mat by num_repeat
    local_mat = np.where(updates_num_repeat == 0, 0, updates_local_mat / updates_num_repeat)

    return local_mat

def get_alpha(x0):
    vi = onp.loadtxt('cell_vertices_initial.txt') 
    rff = 60 # characteristic distance "ff" for farfield
    vi = np.array(vi) # mesh vertices on cell surface

    rs = np.min(np.linalg.norm(x0-vi,axis=1)) # distance to cell surface
    rsc = np.minimum(rs, rff) # clipped distance to cell surface
    a0 = 2.5
    rcrit = rff * np.sqrt(2*a0-1) / (np.sqrt(2*a0-1) + 1) # characteristic distance for most degraded gel portion
    aideal = 1/2 * (((rsc-rcrit)/(rff-rcrit))**2 + 1)
    return aideal

    
class HyperElasticity(Problem):
    def custom_init(self):
        self.fe = self.fes[0]
    
    def get_universal_kernel(self):
        tensor_map = self.get_tensor_map_spatial_var()
        # JAX FEM call signature for laplace kernel; cannot change w/o modifying problem.py
        # pass a_quad through cell_internal_vars
        def laplace_kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars): 
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_shape_grads = cell_shape_grads[:, :self.fes[0].num_nodes, :]
            cell_sol = cell_sol_list[0]
            cell_v_grads_JxW = cell_v_grads_JxW[:, :self.fes[0].num_nodes, :, :]
            vec = self.fes[0].vec
            u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :]
            u_grads = np.sum(u_grads, axis=1)
            u_grads_reshape = u_grads.reshape(-1, vec, self.dim)
        
            u_physics = jax.vmap(tensor_map, (0,0), 0)(u_grads_reshape, *cell_internal_vars).reshape(u_grads.shape) ##ADDED PHYSICAL QUAD POINTS
            val = np.sum(u_physics[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1))
            val = jax.flatten_util.ravel_pytree(val)[0]
            return val

        return laplace_kernel
    
    def get_tensor_map_spatial_var(self):

        def psi(F, a):
            alpha = a
            C1 = 50 
            D1 = 10000 * C1 
            J = np.linalg.det(F)
            I1 = np.trace(F.T @ F)
            energy = alpha * C1 * (I1 - 3.)  - 2 *C1* np.log(J) + D1 *(np.log(J))**2
            
            if energy.shape!=():
                return energy[0]
                
            return energy
        
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, a_quad): # alpha becomes traced
            I = np.eye(self.dim)# num_cells ( is used in vmap), num_quads_per_cell, 3, 3; identity on position to position matrix xyz->xyz
            F = u_grad + I
            P = P_fn(F,a_quad)
            return P

        return first_PK_stress
    
    def set_params(self, params):
        a_quad = params
        self.internal_vars = [a_quad]
        
        

ele_type = 'TET4'
cell_type = get_meshio_cell_type(ele_type)     
meshio_mesh = read_in_mesh("reference_domain.xdmf", cell_type)
points = meshio_mesh.points    

def get_j(F):
    return np.linalg.det(F)

def get_f(u_grad):
    I = np.identity(u_grad.shape[0])
    F = u_grad + I
    return F

def get_c(f_vals):
    C = f_vals.swapaxes(2, 3) @ f_vals
    return C

def get_shape_grads_ref(ele_type):
    element_family, basix_ele, _, _, degree, re_order = get_elements(ele_type)
    element = basix.create_element(element_family, basix_ele, degree)
    node_points = element.points
    vals_and_grads = element.tabulate(1, node_points)[:, :, re_order, :]
    shape_grads_ref = onp.transpose(vals_and_grads[1:, :, :, 0], axes=(1, 2, 0))
    logger.debug(f"ele_type = {ele_type}, node_points.shape = (num_nodes, dim) = {node_points.shape}")
    return shape_grads_ref
 
def get_shape_grads_physical(problem):
    shape_grads_ref = get_shape_grads_ref(problem.fes[0].ele_type)
    physical_coos = problem.fes[0].points[problem.fes[0].cells]
    jacobian_dx_deta = onp.sum(physical_coos[:, None, :, :, None] * shape_grads_ref[None, :, :, None, :], axis=2, keepdims=True)
    jacobian_deta_dx = onp.linalg.inv(jacobian_dx_deta)
    shape_grads_physical = (shape_grads_ref[None, :, :, None, :] @ jacobian_deta_dx)[:, :, :, 0, :]
    return shape_grads_physical

init_pos = np.asarray(onp.loadtxt("cell_vertices_initial.txt"))
disp = np.asarray(onp.loadtxt("cell_vertices_final.txt")) - init_pos
tol = 10**-9

def zero_dirichlet_val(point, load_factor=1):
    return 0.


def xcell_displacement(point, load_factor=1):
    ind = np.where(np.absolute(init_pos-point) < tol, 1, 0)
    i = np.nonzero(ind,size=1)
    return disp[i,:][0][0][0]*load_factor
    
def ycell_displacement(point, load_factor=1):
    ind = np.where(np.absolute(init_pos-point) < tol, 1, 0)
    i = np.nonzero(ind,size=1)
    return disp[i,:][0][0][1]*load_factor
    
def zcell_displacement(point, load_factor=1):
    ind = np.where(np.absolute(init_pos-point) < tol, 1, 0)
    i = np.nonzero(ind,size=1)
    return disp[i,:][0][0][2]*load_factor
    
    
def get_c(f_vals):
    C = f_vals.swapaxes(2, 3) @ f_vals
    return C


box_length = 300
distol = 10**-3

centroid = np.array([77.30223623, 77.03447408, 66.74390624])
bounds = np.stack((centroid-box_length/2, centroid+box_length/2))


def gel_surface(point):
    px, py, pz = point[0], point[1], point[2]
    nx, mx = bounds[:,0][0], bounds[:,0][1]
    ny, my = bounds[:,1][0], bounds[:,1][1]
    nz, mz = bounds[:,2][0], bounds[:,2][1]

    left = np.isclose(point[0], nx, atol=1e-5)
    right = np.isclose(point[0], mx, atol=1e-5)
    front = np.isclose(point[1], ny, atol=1e-5)
    back = np.isclose(point[1], my, atol=1e-5)
    top = np.isclose(point[2], nz, atol=1e-5)
    bottom = np.isclose(point[2], mz, atol=1e-5)
    return left | right | front | back | top | bottom

pdata = onp.loadtxt('cell_vertices_initial.txt')
def cell_surface(point):
    return np.any(np.all(np.isclose(point, np.array(pdata),atol =10**-5), axis =1))


def save_sol_all(sol,problem):
    shape_grads_physical = get_shape_grads_physical(problem)
    cell_sols = sol[0][np.array(problem.fes[0].cells)]
    u_grads = cell_sols[:, None, :, :, None] * shape_grads_physical[:, :, :, None, :]
    u_grads = np.sum(u_grads, axis=2)
    vectorized_get_f = np.vectorize(get_f, signature='(n,m)->(n,m)')
    f_vals = vectorized_get_f(u_grads)

    C = get_c(f_vals)
    return C 

def main():

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    dirichlet_bc_info = [[gel_surface] * 3 + [cell_surface] * 3, 
                        [0, 1, 2] * 2, 
                        [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val] + 
                        [xcell_displacement, ycell_displacement, zcell_displacement]]
    
    ref_problem = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)

    vectorized_get_alpha = np.vectorize(get_alpha, signature='(n)->()')
    alpha_mat = vectorized_get_alpha(ref_problem.fes[0].points) # 3906,
    a_quad = ref_problem.fes[0].convert_from_dof_to_quad_a(alpha_mat.reshape(3906,1))[:, :, 0]

    ref_fwd_pred = ad_wrapper(ref_problem)
    sol_0 = ref_fwd_pred(a_quad)

    C_0 = save_sol_all(sol_0,ref_problem)
    C_0quad = ref_problem.fes[0].convert_from_dof_to_quad_C(C_0)[:, :, 0,0] # 4 points per tetra
    cells_JxW = ref_problem.JxW[:, 0, :] # THIS SAME FOR ALL PROBLEMS



    #NEW PROBLEM
    a_0 = np.ones((ref_problem.fe.num_cells,ref_problem.fe.num_quads))

    curr_problem = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
    curr_fwd_pred = ad_wrapper(curr_problem)

    def objective(params):
        curr_sol = curr_fwd_pred(params)
        # C_c = save_sol_all(curr_sol, problem)
        # obj = np.sum((C_0quad - C_cquad)**2 * cells_JxW)
        # obj = np.sum(params[0]**2) 
        # obj = np.sum((C_c-C_0)**2) 
        # obj = np.sum((params)**2)
        obj = np.sum(curr_sol[0]) ####
        return obj

    obj_and_intergrad = jax.value_and_grad(objective)
    
    def obj_and_grad(alpha):
        J,dJda = obj_and_intergrad(alpha)
        # assert np.all(np.isclose(dJda, 0))
        print(dJda)
        print("djda",np.linalg.norm(dJda))
        
        # Jprime = np.einsum('ao,ao->o',dJda, alpha)
        return (J, dJda)
    
    out = obj_and_grad(a_0)

    print(out[0])
    # print(out[1])
    #scipy.minimize
    #result = minimize(obj_and_grad, a_0, method='L-BFGS-B', jac=True, bounds = [(.5, 10)]*a_0.size, tol = 10**-8,options = {"maxiter":1,"gtol":10**-8,"disp": True}, )
    #print(result.x)
    

if __name__ == "__main__":
    main()