import time
import os
import sys
# os.environ['JAX_PLATFORMS'] = 'cpu'
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
            # alpha = 1
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
    
     
def get_alpha(x0):
    vi = onp.loadtxt('cell_vertices_initial.txt') 
    rff = 60 # characteristic distance "ff" for farfield
    vi = np.array(vi) # mesh vertices on cell surface

    rs = np.min(np.linalg.norm(x0-vi,axis=1)) # distance to cell surface
    rsc = np.minimum(rs, rff) # clipped distance to cell surface
    a0 = 2.5
    rcrit = rff*np.sqrt(2*a0-1)/(np.sqrt(2*a0-1)+1) #characterestic distance for most degraded gel portion
    aideal = 1/2*(((rsc-rcrit)/(rff-rcrit))**2 + 1)
    return aideal

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
    print("ind",ind)
    i = np.nonzero(ind[:,0],size=1)#(ind==np.array([1,1,1]))
    return disp[i,:][0][0][0]*load_factor

def ycell_displacement(point, load_factor=1):
    ind = np.where(np.absolute(init_pos-point) < tol, 1, 0)
    i = np.nonzero(ind[:,0],size=1)#(ind==np.array([1,1,1]))
    # print(disp[i,:][0][0][1])
    return disp[i,:][0][0][1]*load_factor
def zcell_displacement(point, load_factor=1):
    ind = np.where(np.absolute(init_pos-point) < tol, 1, 0)
    i = np.nonzero(ind[:,0],size=1)#(ind==np.array([1,1,1]))
    return disp[i,:][0][0][2]*load_factor

def problem():
    ele_type = 'TET4'
    cell_type = get_meshio_cell_type(ele_type)
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    # Lx, Ly, Lz = 1., 1., 1.
    meshio_mesh = read_in_mesh("reference_domain.xdmf", cell_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    box_length = 300
    distol = 10**-4
    r = 30
    centroid = np.array([77.30223623, 77.03447408, 66.74390624])
    bounds = np.stack((centroid-box_length/2, centroid+box_length/2))


    def gel_surface(point):
        px, py, pz = point[0], point[1], point[2]
        nx, mx = bounds[:,0][0], bounds[:,0][1]
        ny, my = bounds[:,1][0], bounds[:,1][1]
        nz, mz = bounds[:,2][0], bounds[:,2][1]
        # print("m",nx, mx)

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
    
    #TESTS
    # print(cell_surface(np.array([8.254631805419921875e+01, 8.463780975341796875e+01, 2.081762695312500000e+01])))
    # print(gel_surface(np.array([77.30223623+150, 2, 2])))
    # act = onp.loadtxt('cell_vertices_final.txt')
    c = 0
    # for point in pdata:
    #     # point = onp.array([8.254631805419921875e+01, 8.463780975341796875e+01, 2.081762695312500000e+01])
        
    #     x = onp.asarray(xcell_displacement(point))
    #     y = onp.asarray(ycell_displacement(point))
    #     z = onp.asarray(zcell_displacement(point))
    #     disp = onp.stack((x,y,z))
    #     # print("y val",y)

    #     pred = disp + point#onp.array([8.254631805419921875e+01, 8.463780975341796875e+01, 2.081762695312500000e+01])
    #     for r in act:
    #         if onp.linalg.norm(r-pred)< 10**-5:
    #             c +=1
    # assert c == pdata.shape[0]
    # x = onp.asarray(xcell_displacement(np.array([8.262040710449218750e+01, 8.185975646972656250e+01, 1.208771514892578125e+01])))
    # 


    # dirichlet_bc_info = [[cell_surface] * 3,
    #                      [0, 1, 2] * 1,
    #                      [xcell_displacement, ycell_displacement,zcell_displacement]]
    # a = gel_surface(np.array([77.30223623+150, 2, 2]))
    # print("a", gel_surface(np.array([77.30223623+150, 2, 2])))
    
    dirichlet_bc_info = [[gel_surface] * 3 + [cell_surface] * 3, 
                        [0, 1, 2] * 2, 
                        [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val] + 
                        [xcell_displacement, ycell_displacement, zcell_displacement]]
    
    problem = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
    num_steps = 2
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
            initial_sol = sol
        else:
            sol = solver(problem, use_petsc=True, initial_guess=initial_sol)
            initial_sol = sol
    print("DONE SOLVING")
    vtk_path = os.path.join(data_dir, f'vtk/speed_ind.vtu')
    save_sol(problem.fes[0], sol[0], vtk_path)#, point_infos = [{"j":local_j}, {"alpha":local_alpha}])
    
if __name__ == "__main__":
    l=1
    t = np.zeros(l)
    for i in range(l):
        start_time = time.time()
        problem()
        t = t.at[i].set(time.time() - start_time)
    print(t) #avg t 3.945726823806763 std 0.5014158602174434
    print("avg t",np.average(t[1:]), "std", np.std(t[1:]))