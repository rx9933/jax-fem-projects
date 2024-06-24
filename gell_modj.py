import os
# for computer (no gpu)
os.environ['JAX_PLATFORMS'] = 'cpu'

# Import some useful modules.
import jax
import jax.numpy as np
import numpy as onp
import os
import basix

# Import JAX-FEM specific modules.
from jax_fem import logger
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh, cells_out, read_in_mesh
from jax_fem.basis import get_elements


class HyperElasticity(Problem):
    def get_tensor_map(self):
        def psi(F):

            # E = 10.    # Young's modulus
            # nu = 0.3   # Poisson's ratio
            # mu = E / (2. * (1. + nu))
            # kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            # Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F) 
            C1 = 50
            D1 = 1000*C1
            energy = C1 * (I1 - 3. - 2*np.log(J) ) + D1 * (np.log(J))**2.
            return energy

        
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            F = get_f(u_grad)
            P = P_fn(F)
            return P

        return first_PK_stress
        
def get_j(F):
    return np.linalg.det(F)
def get_f(u_grad):
    I = np.identity(u_grad.shape[0])
    F = u_grad + I
    return F
def get_shape_grads_ref(ele_type):
    """Get shape function gradients in reference domain
    The gradient is w.r.t reference coordinates.

    Returns
    -------
    shape_grads_ref: onp.ndarray
        e.g., for a HEX8 element we have (8, 8, 3) = (num_points, num_nodes, dim)
        num_points indicates the number of points being evaluated for shape gradients.
        num_nodes indicates the number of shape functions.
        (num_points and num_nodes really are the same.)
    """
    element_family, basix_ele, _, _, degree, re_order = get_elements(ele_type)
    element = basix.create_element(element_family, basix_ele, degree)
    node_points = element.points
    vals_and_grads = element.tabulate(1, node_points)[:, :, re_order, :]
    shape_grads_ref = onp.transpose(vals_and_grads[1:, :, :, 0], axes=(1, 2, 0))
    logger.debug(f"ele_type = {ele_type}, node_points.shape = (num_nodes, dim) = {node_points.shape}")
    return shape_grads_ref


def get_shape_grads_physical(problem):
    """Get shape function gradients in physical domain
    The gradient is w.r.t physical coordinates.

    See Hughes, Thomas JR. The finite element method: linear static and dynamic finite element analysis. Courier Corporation, 2012.
    Page 147, Eq. (3.9.3)

    Returns
    -------
    shape_grads_physical : onp.ndarray
        (num_cells, num_points, num_nodes, dim)
    """
    shape_grads_ref = get_shape_grads_ref(problem.fes[0].ele_type) # (num_points, num_nodes, dim)
    physical_coos = problem.fes[0].points[problem.fes[0].cells] # (num_cells, num_nodes, dim)

    # (num_cells, 1, num_nodes, dim, 1) * (1, num_points, num_nodes, 1, dim) ->
    # (num_cells, num_points, num_nodes, dim, dim) -> (num_cells, num_points, 1, dim, dim)
    jacobian_dx_deta = onp.sum(physical_coos[:, None, :, :, None] * shape_grads_ref[None, :, :, None, :], axis=2, keepdims=True)

    jacobian_deta_dx = onp.linalg.inv(jacobian_dx_deta) # (num_cells, num_points, 1, dim, dim)

    # (1, num_points, num_nodes, 1, dim) @ (num_cells, num_points, 1, dim, dim) ->
    # (num_cells, num_points, num_nodes, 1, dim) -> (num_cells, num_points, num_nodes, dim)
    shape_grads_physical = (shape_grads_ref[None, :, :, None, :] @ jacobian_deta_dx)[:, :, :, 0, :]

    return shape_grads_physical


def problem():
    # Specify mesh-related information (first-order hexahedron element).
    # Specify mesh-related information (first-order hexahedron element).
    ele_type = 'TET4'
    cell_type = get_meshio_cell_type(ele_type)
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    Lx, Ly, Lz = 1., 1., 1.

    # Generate the mesh
    # meshio_mesh = box_mesh(Nx=20,
    #                        Ny=20,
    #                        Nz=20,
    #                        Lx=Lx,
    #                        Ly=Ly,
    #                        Lz=Lz,
    #                        data_dir=data_dir,
    #                        ele_type=ele_type)
    print("celltype", cell_type)
    meshio_mesh = read_in_mesh("sphere.msh", cell_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    box_length = 300
    tol = .01

    r = 30
    # 4, .5, .5
    a = 4
    b = -.5
    c = -.5

    # Define boundary locations.
    def cube(point):
        return np.logical_or(
            np.isclose(np.abs(point[0]), box_length / 2., atol=1e-5),
            np.logical_or(
                np.isclose(np.abs(point[1]), box_length / 2., atol=1e-5),
                np.isclose(np.abs(point[2]), box_length / 2., atol=1e-5)
            )
        )

    def sphere_surface(point):
        return np.isclose(np.linalg.norm(point) , r, atol=tol)
    #########

    # Define Dirichlet boundary values.
    def zero_dirichlet_val(point):
        return 0.

    def surface_x(point):
        # assert np.linalg.norm(point) == r
        return point[0]*a/r

    def surface_y(point):
        # assert np.linalg.norm(point) == r
        return point[1]*b/r

    def surface_z(point):
        # assert np.linalg.norm(point) == r
        return point[2]*c/r

    # Apply zero Dirichlet boundary values on the left side and pulling on the right side along the x-axis.

    dirichlet_bc_info = [[cube] * 3 + [sphere_surface] * 3, 
                        [0, 1, 2] * 2, 
                        [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val] + 
                        [surface_x, surface_y, surface_z]]


    # Create an instance of the problem.
    problem = HyperElasticity(mesh,
                            vec=3,
                            dim=3,
                            ele_type=ele_type,
                            dirichlet_bc_info=dirichlet_bc_info)
    # Solve the defined problem.
    sol = solver(problem, use_petsc=True)
    
    shape_grads_physical = get_shape_grads_physical(problem)
    # print(type(problem.fes[0].cells)) # 30974,4
    cell_sols = sol[0][np.array(problem.fes[0].cells)]

    # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_points, num_nodes, 1, dim) -> (num_cells, num_points, num_nodes, vec, dim)
    u_grads = cell_sols[:, None, :, :, None] * shape_grads_physical[:, :, :, None, :]
    u_grads = np.sum(u_grads, axis=2)  # (num_cells, num_points, vec, dim)

    print(f"\nu_grads.shape = {u_grads.shape}, expected (num_cells, num_points, vec, dim)") 
    
    first_PK_stress = problem.get_tensor_map()
    
    ####
    ug_s = u_grads.shape
    stress_tensor = np.zeros(ug_s)
    j_mat = np.zeros(ug_s[:2])
    num_cells = ug_s[0] # 30974
    # print("NC",num_cells)
    num_points = ug_s[1]
    for i in range(num_cells):
        for j in range(num_points):
            stress_tensor = stress_tensor.at[i,j,:,:].set(first_PK_stress(u_grads[i,j]))
            j_mat = j_mat.at[i,j].set(get_j(get_f(u_grads[i,j])))

        if i%100 == 0:
            print("i:",i)

    cells, points = cells_out()   
    num_repeat = np.zeros(len(points))
    local_j = np.zeros(len(points))

    for r in range(cells.shape[0]):
        for c in range(cells.shape[1]):
            point = points[cells[r,c]]
            ind = np.where(np.all(points == point, axis = 1))
            local_j = local_j.at[ind].add(j_mat[r,c])
            num_repeat = num_repeat.at[ind].add(1)
    local_j = local_j/num_repeat
    print("lj shape", local_j.shape) # 9261,
    # print("Dshape", sol.shape) # 3
    # Store the solution to local file.
    vtk_path = os.path.join(data_dir, f'vtk/gell_modj.vtu')
    save_sol(problem.fes[0], sol[0], vtk_path, point_infos = [{"j":local_j}])

if __name__ == "__main__":
    problem()
