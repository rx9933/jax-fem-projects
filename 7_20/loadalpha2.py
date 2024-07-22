import time
import os
import sys
os.environ['JAX_PLATFORMS'] = 'cpu'
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
            E = 0.01 * units.GPa
            nu = 0.49
            # alpha = (radius_cell / np.linalg.norm(X - CENTER)) ** beta
            alpha = get_alpha(X)
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J ** (-2. / 3.)
            I1 = np.trace(F.T @ F)
            energy = alpha * (mu / 2.) * (Jinv * I1 - 3.) + (0.5 * kappa) * (J * (J - 1) - np.log(J))
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, X):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F, X)
            return P

        return first_PK_stress
    
def get_alpha(X):
    alpha = (radius_cell / np.linalg.norm(X - CENTER))**beta
    return alpha
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


stretch_factor = 1.1# 3.87 gives NaNs, for 10 steps
a = stretch_factor - 1
b = 1 / sqrt(stretch_factor) - 1
c = 1 / sqrt(stretch_factor) - 1


def surface_x(point,load_factor=1):
    return point[0] * a *load_factor

def surface_y(point,load_factor=1):
    return point[1] * b *load_factor

def surface_z(point,load_factor=1):
    return point[2] * c *load_factor

# def apply_load_steps(problem, initial_sol, num_steps):
#     sol = initial_sol
#     load_factor = 1 / num_steps
#     for step in np.arange(0, 1+ 1/ num_steps, 1 / num_steps):
#         logger.info(f"STEP {step}")
#         print("STEP",step)
#         load_factor = step
#         problem.dirichlet_bc_info[0][2][3:] = [
#             lambda point, load_factor=load_factor: surface_x(point, load_factor),
#             lambda point, load_factor=load_factor: surface_y(point, load_factor),
#             lambda point, load_factor=load_factor: surface_z(point, load_factor)
#         ]
        
#         if step ==0 :
#             print("on 0")
#             print(problem.dirichlet_bc_info[0][2][3](np.array([1,1,1])))
#             sol = solver(problem, use_petsc=True)
#         else:
#             sol = solver(problem, use_petsc=True, initial_guess=sol)
#     return sol


def problem():
    ele_type = 'TET4'
    cell_type = get_meshio_cell_type(ele_type)
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    # Lx, Ly, Lz = 1., 1., 1.
    meshio_mesh = read_in_mesh("largesphere.msh", cell_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    box_length = 300
    tol = .01
    r = 30

    def cube(point):
        return np.logical_or(
            np.isclose(np.abs(point[0]), box_length / 2., atol=1e-5),
            np.logical_or(
                np.isclose(np.abs(point[1]), box_length / 2., atol=1e-5),
                np.isclose(np.abs(point[2]), box_length / 2., atol=1e-5)
            )
        )

    def sphere_surface(point):
        return np.isclose(np.linalg.norm(point), r, atol=tol)

    def zero_dirichlet_val(point):
        return 0.
    
    dirichlet_bc_info = [[cube] * 3 + [sphere_surface] * 3,
                         [0, 1, 2] * 2,
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val] +
                         [surface_x, surface_y, surface_z]]
    problem = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
    num_steps = 1
    load_factor = 1 / num_steps
    for step in np.arange(0, 1 + 1/num_steps, 1 / num_steps ):
        logger.info(f"STEP {step}")
        print("STEP",step)
        # load_factor = (step + 1) / num_steps
        load_factor = step
        print("LF =",load_factor)
        problem.dirichlet_bc_info[0][2][3:] = [
            lambda point, load_factor=load_factor: surface_x(point, load_factor),
            lambda point, load_factor=load_factor: surface_y(point, load_factor),
            lambda point, load_factor=load_factor: surface_z(point, load_factor)
        ]
        
        problem = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
        if step ==0 :
            print("on 0")
            print(problem.dirichlet_bc_info[0][2][3](np.array([1,1,1])))
            sol = solver(problem, use_petsc=True)
            initial_sol = sol
        else:
            sol = solver(problem, use_petsc=True, initial_guess=initial_sol)
            initial_sol = sol

        
        
        
    
    # sol = apply_load_steps(problem, initial_sol=None, num_steps=num_steps)
    
    print("DONE SOLVING")
    vtk_path = os.path.join(data_dir, f'vtk/alphagellmodj.vtu')
    save_sol(problem.fes[0], sol[0], vtk_path)#, point_infos = [{"j":local_j}, {"alpha":local_alpha}])
    """
    first_PK_stress = problem.get_tensor_map_spatial_var()
    cells, points = cells_out()
    shape_grads_physical = get_shape_grads_physical(problem)
    cell_sols = sol[0][np.array(problem.fes[0].cells)]
    u_grads = cell_sols[:, None, :, :, None] * shape_grads_physical[:, :, :, None, :]
    u_grads = np.sum(u_grads, axis=2)


    # Initialize stress tensor, J matrix, and alpha matrix
    ug_s = u_grads.shape
    stress_tensor = np.zeros(ug_s)
    j_mat = np.zeros(ug_s[:2])
    alpha_mat = np.zeros(ug_s[:2])

    # Get global point indices
    global_point_inds = cells

    # Get point values
    point_vals = points[global_point_inds]

    # Vectorize the operations for stress tensor, j_mat, and alpha_mat
    vectorized_first_PK_stress = np.vectorize(first_PK_stress, signature='(n,m),(o)->(n,m)')
    vectorized_get_j = np.vectorize(get_j, signature='(n,m)->()')
    vectorized_get_f = np.vectorize(get_f, signature='(n,m)->(n,m)')
    vectorized_get_alpha = np.vectorize(get_alpha, signature='(n)->()')

    # Compute stress tensor
    stress_tensor = vectorized_first_PK_stress(u_grads, point_vals)

    # Compute deformation gradient and J matrix
    f_vals = vectorized_get_f(u_grads)
    j_mat = vectorized_get_j(f_vals)

    # Compute alpha matrix
    alpha_mat = vectorized_get_alpha(point_vals)

    # Print progress for every 100th cell
                # num_cells = ug_s[0]
                # for i in range(0, num_cells, 100):
                #     print("i:", i)

    # Localize function remains unchanged
    # def localize(orig_mat):
    #     local_mat = np.zeros(len(points))
    #     num_repeat = np.zeros(len(points))
    #     flat_cells = cells.flatten()
    #     for i, point in enumerate(flat_cells):
    #         ind = np.where(np.all(points == points[point], axis=1))[0]
    #         local_mat.at[ind].add(orig_mat.flatten()[i])
    #         num_repeat.at[ind].add(1)
    #     local_mat /= num_repeat
    #     return local_mat
    def localize(orig_mat):
        local_mat = np.zeros(len(points))
        num_repeat = np.zeros(len(points))
        for r in range(cells.shape[0]):
            for c in range(cells.shape[1]):
                point = points[cells[r,c]]
                ind = np.where(np.all(points == point, axis = 1))
                local_mat = local_mat.at[ind].add(orig_mat[r,c])
                num_repeat = num_repeat.at[ind].add(1)
            if r%100==0:
                print("row num", r) 
        local_mat = local_mat/num_repeat
        return local_mat

    # Localize J and alpha matrices
    local_j = localize(j_mat)
    local_alpha = localize(alpha_mat)

    print(j_mat)
    vtk_path = os.path.join(data_dir, f'vtk/alphagellmodj.vtu')
    save_sol(problem.fes[0], sol[0], vtk_path, point_infos = [{"j":local_j}, {"alpha":local_alpha}])
    """
if __name__ == "__main__":
    # start_time = time.time()
    # problem()
    # print("--- %s seconds ---" % (time.time() - start_time))
    l=1
    t = np.zeros(l)
    for i in range(l):
        start_time = time.time()
        problem()
        t = t.at[i].set(time.time() - start_time)
    print(t) #avg t 3.945726823806763 std 0.5014158602174434
    # print("--- %s seconds ---" % (time.time() - start_time))
    print("avg t",np.average(t), "std", np.std(t))
    # w/o jit 5.1428914070129395