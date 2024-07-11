import time
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.append('/workspace')
# os.environ['JAX_PLATFORMS'] = 'cpu'
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


class HyperElasticity(Problem):
    def get_universal_kernel(self):
        tensor_map = self.get_tensor_map_spatial_var()
        def laplace_kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_shape_grads = cell_shape_grads[:, :self.fes[0].num_nodes, :]
            cell_sol = cell_sol_list[0]
            cell_v_grads_JxW = cell_v_grads_JxW[:, :self.fes[0].num_nodes, :, :]
            vec = self.fes[0].vec
            u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :]
            u_grads = np.sum(u_grads, axis=1)
            u_grads_reshape = u_grads.reshape(-1, vec, self.dim)
            u_physics = jax.vmap(tensor_map)(u_grads_reshape, physical_quad_points, *cell_internal_vars).reshape(u_grads.shape)
            val = np.sum(u_physics[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1))
            val = jax.flatten_util.ravel_pytree(val)[0]
            return val

        return laplace_kernel
    def get_tensor_map_spatial_var(self):
        def psi(F, X):

            # alpha = (radius_cell / np.linalg.norm(X - CENTER)) ** beta
            a = get_alpha(X)
            # alpha = 1
            C1 = 50 
            D1 = 10000 * C1 
            J = np.linalg.det(F)
            I1 = np.trace(F.T @ F)
            energy = a * C1 * (I1 - 3.)  - 2 * np.log(J) + D1 *(np.log(J))**2
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, X):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F, X)
            return P

        return first_PK_stress
    
ele_type = 'TET4'
cell_type = get_meshio_cell_type(ele_type)     
meshio_mesh = read_in_mesh("reference_domain.xdmf", cell_type)
points = meshio_mesh.points    
print("81",points.shape)
@jax.jit
def get_alpha(x0):
    ind = np.where(np.linalg.norm(points-x0) < tol, 1, 0)
    print("ALPHAF", alpha_f)
    return alpha_f[ind]

def get_j(F):
    return np.linalg.det(F)

def get_f(u_grad):
    I = np.identity(u_grad.shape[0])
    F = u_grad + I
    return F


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
    i = np.nonzero(ind,size=1)#(ind==np.array([1,1,1]))
    return disp[i,:][0][0][0]*load_factor
def ycell_displacement(point, load_factor=1):
    ind = np.where(np.absolute(init_pos-point) < tol, 1, 0)
    i = np.nonzero(ind,size=1)#(ind==np.array([1,1,1]))
    return disp[i,:][0][0][1]*load_factor
def zcell_displacement(point, load_factor=1):
    ind = np.where(np.absolute(init_pos-point) < tol, 1, 0)
    i = np.nonzero(ind,size=1)#(ind==np.array([1,1,1]))
    return disp[i,:][0][0][2]*load_factor
def get_c(f_vals):
    print(f_vals.shape)
    C = f_vals.swapaxes(3, 4) @ f_vals
    return C


box_length = 300
distol = .2
r = 30
centroid = np.array([77.30223623, 77.03447408, 66.74390624])
bounds = np.stack((centroid-box_length/2, centroid+box_length/2))

def gel_surface(point):
    px, py, pz = point[0], point[1], point[2]
    nx, mx = bounds[:,0][0], bounds[:,0][1]
    ny, my = bounds[:,1][0], bounds[:,1][1]
    nz, mz = bounds[:,2][0], bounds[:,2][1]

    return np.logical_or(
        np.isclose(np.abs(px - nx), 0, atol=distol),
        np.logical_or(
            np.isclose(np.abs(px - mx), 0, atol=distol),
            np.logical_or(
                np.isclose(np.abs(py - ny), 0, atol=distol),
                np.logical_or(
                    np.isclose(np.abs(py - my), 0, atol=distol),
                    np.logical_or(
                        np.isclose(np.abs(pz - nz), 0, atol=distol),
                        np.isclose(np.abs(pz - mz), 0, atol=distol)
                    )
                )
            )
        )
    )
pdata = onp.loadtxt('cell_vertices_initial.txt')

def cell_surface(point):
    return np.any(np.isclose(point, np.array(pdata)))

def problem(s=False):
    
    def apply_load_steps(problem, num_steps = 2):
        load_factor = 1 / num_steps
        for step in np.arange(0, 1 + 1/num_steps, 1 / num_steps ):
            logger.info(f"STEP {step}")
            print("STEP",step)
            load_factor = step
            print("LF =",load_factor)
            problem.dirichlet_bc_info[0][2][3:] = [
                lambda point, load_factor=load_factor: xcell_displacement(point, load_factor),
                lambda point, load_factor=load_factor: ycell_displacement(point, load_factor),
                lambda point, load_factor=load_factor: zcell_displacement(point, load_factor)
            ]
            
            problem = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
            if step ==0 :
                sol = solver(problem, use_petsc=True)
            else:
                sol = solver(problem, use_petsc=True, initial_guess=sol)
        return sol
        
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    dirichlet_bc_info = [[gel_surface] * 3 + [cell_surface] * 3, 
                        [0, 1, 2] * 2, 
                        [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val] + 
                        [xcell_displacement, ycell_displacement, zcell_displacement]]
    
    problem = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)

    sol = apply_load_steps(problem, 2)
    print("DONE SOLVING")

    cells, points = cells_out()

    def localize(orig_mat):
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
    
    def save_sol_par():
      

        shape_grads_physical = get_shape_grads_physical(problem)
        cell_sols = sol[0][np.array(problem.fes[0].cells)]
        u_grads = cell_sols[:, None, :, :, None] * shape_grads_physical[:, :, :, None, :]
        u_grads = np.sum(u_grads, axis=2)

        # Initialize J matrix, and alpha matrix
        ug_s = u_grads.shape
        j_mat = np.zeros(ug_s[:2])
        print("JMAT",cells.shape)
        alpha_mat = np.zeros(ug_s[:2])

        # # Get global point indices
        global_point_inds = cells

        # # Get point values
        point_vals = points[global_point_inds]

        # Vectorize the operations for j_mat, and alpha_mat
        vectorized_get_j = np.vectorize(get_j, signature='(n,m)->()')
        vectorized_get_f = np.vectorize(get_f, signature='(n,m)->(n,m)')
        vectorized_get_alpha = np.vectorize(get_alpha, signature='(n)->()')
        
        f_vals = vectorized_get_f(u_grads)

        j_mat = vectorized_get_j(f_vals)

        alpha_mat = vectorized_get_alpha(points)
        # print("amat shape", alpha_f)
        local_j = localize(j_mat)
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        vtk_path = os.path.join(data_dir, f'vtk/varalpha.vtu')
        
        save_sol(problem.fes[0], sol[0], vtk_path, point_infos = [{"j":local_j}, {"alpha":alpha_f}])
        return f_vals
    if s:
        f_vals = save_sol_par()
    if not(s):     # get required values to return c
        shape_grads_physical = get_shape_grads_physical(problem)
        cell_sols = sol[0][np.array(problem.fes[0].cells)]
        u_grads = cell_sols[:, None, :, :, None] * shape_grads_physical[:, :, :, None, :] 

        vectorized_get_f = np.vectorize(get_f, signature='(n,m)->(n,m)')
        print("VG",vectorized_get_f)
        print("ugrad",u_grads.shape)
        f_vals = vectorized_get_f(u_grads)

        print("FVV",f_vals.shape)
        return get_c(f_vals) # C sim 

def main(alpha,s):
    global alpha_f 
    alpha_f = np.array(alpha)
   
    l=1
    t = np.zeros(l)
    for i in range(l):
        start_time = time.time()
        Csim = problem(s)
        t = t.at[i].set(time.time() - start_time)
    # print(t)
    print("avg t",np.average(t), "std", np.std(t))

    if not(s):
        return Csim
    
# if __name__ == "__main__":
#     global alpha
#     alpha = np.load('alpha.npy')
#     s = True # default value when running just the forward model, save the alpha/j values
#     main(alpha,s)