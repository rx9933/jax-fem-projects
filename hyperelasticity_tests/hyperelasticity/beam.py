""" 06/10/24
extension of hyperelasticity demo
10x1x1 (x,y,z) beam with 5 cells per unit. E=1, nu =.3.
beam is pulled in x dimension with 20 stretch ratios (lamda = l/L, l = final length, L = initial length) between 0 and 0.3.
produces graph with analytical solution (based of 1st Piola Kirchoff tensor) and of the calculated solutions.

//// DATA ////
lamdas = [0.01    0.02526 0.04053 0.05579 0.07105 0.08632 0.10158 0.11684 0.13211 0.14737 0.16263 0.17789 0.19316 0.20842 0.22368 0.23895 0.25421 0.26947 0.28474 0.3    ]
experimental forces = [Array(0.01049, dtype=float64), Array(0.02607, dtype=float64), Array(0.04116, dtype=float64), Array(0.05579, dtype=float64), Array(0.06999, dtype=float64), Array(0.08379, dtype=float64), Array(0.09719, dtype=float64), Array(0.11024, dtype=float64), Array(0.12294, dtype=float64), Array(0.13531, dtype=float64), Array(0.14737, dtype=float64), Array(0.15914, dtype=float64), Array(0.17064, dtype=float64), Array(0.18187, dtype=float64), Array(0.19284, dtype=float64), Array(0.20358, dtype=float64), Array(0.2141, dtype=float64), Array(0.22439, dtype=float64), Array(0.23448, dtype=float64), Array(0.24438, dtype=float64)]
analytical forces =[0.01142 0.02844 0.04496 0.06103 0.07667 0.09189 0.10673 0.12121 0.13533 0.14914 0.16263 0.17582 0.18874 0.20139 0.21379 0.22595 0.23788 0.2496  0.26111 0.27242]
RMSE error =  0.016688569097990363
"""
import os
import sys
os.environ['JAX_PLATFORMS'] = 'cpu'
# Modify sys.path to include necessary directories
sys.path.insert(0, os.path.abspath('..'))  # Include parent directory
sys.path.append('/workspace')             # Include workspace directory

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
from jax_fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh, cells_out
from jax_fem.basis import get_elements
# from matplotlib import pyplot as plt

class HyperElasticity(Problem):
    def get_tensor_map(self):
        def psi(F):
            E = 1.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = get_j(F)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
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


def problem(lamda): # lamda = .05 # 10 %  stretch ratio
    # Specify mesh-related information (first-order hexahedron element).
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    Lx, Ly, Lz = 10., 1., 1.

    Nx,Ny,Nz = Lx*1,Ly*1,Lz*1
    meshio_mesh = box_mesh(Nx,
                           Ny,
                           Nz,
                           Lx=Lx,
                           Ly=Ly,
                           Lz=Lz,
                           data_dir=data_dir,
                           ele_type=ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    # Define boundary locations.
    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def pull_dirichlet_val_x(point):
        return lamda*point[0] 

    # Apply zero Dirichlet boundary values on the left side and pulling on the right side along the x-axis.
    dirichlet_bc_info = [[left] * 3 + [right] * 3, 
                        [0, 1, 2] * 2, 
                        [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val] + 
                        [pull_dirichlet_val_x, zero_dirichlet_val, zero_dirichlet_val]]

    # Create an instance of the problem.
    problem = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)

    # Solve the defined problem.

    sol_list = solver(problem, use_petsc=True)
    sol = sol_list[0]
    # print("sol shape", sol.shape, "\n\n")
    
    shape_grads_physical = get_shape_grads_physical(problem)
    cell_sols = sol[problem.fes[0].cells]

    # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_points, num_nodes, 1, dim) -> (num_cells, num_points, num_nodes, vec, dim)
    u_grads = cell_sols[:, None, :, :, None] * shape_grads_physical[:, :, :, None, :]
    u_grads = np.sum(u_grads, axis=2)  # (num_cells, num_points, vec, dim)

    print(f"\nu_grads.shape = {u_grads.shape}, expected (num_cells, num_points, vec, dim)") 
    
    first_PK_stress = problem.get_tensor_map()
    
    ####
    ug_s = u_grads.shape
    P_mat = np.zeros(ug_s) # 8000,8,3,3
    j_mat = np.zeros(ug_s[:2])
    num_cells = ug_s[0] # 8000
    num_points = ug_s[1]
    for i in range(num_cells):
        for j in range(num_points):
            P_mat = P_mat.at[i,j,:,:].set(first_PK_stress(u_grads[i,j]))
            j_mat = j_mat.at[i,j].set(get_j(get_f(u_grads[i,j])))

        if i%100 == 0:
            print("i:",i)

    cells, points = cells_out()   
    num_repeat_j = np.zeros(len(points))
    num_repeat_p = np.zeros(len(points))
    local_j = np.zeros(len(points))
    local_p = np.zeros((len(points),3,3)) # num points (9261) x num vectors x num dim
    ps = points[cells]
    # print("PS",ps)
    for r in range(cells.shape[0]):
        for c in range(cells.shape[1]):
            point = points[cells[r,c]]
            ind = np.where(np.all(points == point, axis = 1))
            local_j = local_j.at[ind].add(j_mat[r,c])
            local_p = local_p.at[ind].add(P_mat[r,c,:,:])
            num_repeat_j = num_repeat_j.at[ind].add(1)
            num_repeat_p = num_repeat_p.at[ind].add(1)
    local_j = local_j/num_repeat_j
    local_p = local_p/num_repeat_p[:, np.newaxis, np.newaxis]

    # face_points = cells[num_cells -1,:]
    # Face points on x axis/end

    end_ind = np.argwhere(points[:,0] == Lx)
    all_p_face = local_p[end_ind,:,0] 
    all_p_facex = all_p_face[:,0]
    xyz = np.mean(all_p_facex, axis = 0)
    print(xyz)
    F_mag = np.linalg.norm(xyz)

    # Store the solution to local file.


    vtk_path = os.path.join(data_dir, f'vtk/pull10j.vtu')
    print("j shape", local_j.shape)
    print("points", len(points))
    # print("sol", len(sol[0]))
    save_sol(problem.fes[0], sol, vtk_path, point_infos = [{"j":local_j}])
    return F_mag

def graph(num_iter):
    lamda_max = .3
    lamdas = np.linspace(0.01,lamda_max, num_iter)
    # lamdas = [.3]
    y = [0]*len(lamdas)
    for l in range(len(lamdas)):
        print("iter",l)
        print(problem(lamdas[l]))
        y[l] = problem(lamdas[l])
    print("lamdas", lamdas)
    print("exp forces", y)
    # plt.style.use(["ggplot",
    # "figures.mplstyle"])
    # plt.style.use(["ggplot", "/oden/asadam/CARDIAX/jax_fem/figures.mplstyle"])

    # plt.plot(lamdas, y, marker = "*", label = "Experimental Solution")

    Ly,Lz=1,1
    A = Ly*Lz
    E = 1.
    nu = 0.3
    mu = E / (2. * (1. + nu))
    C = mu/2
    s = 1+ lamdas
    analytical_Soln = 2*C*A*(s-1/s**2)
    print("ana forces", analytical_Soln)
    rmse = np.sqrt(np.mean((np.array(y)-analytical_Soln)**2))
    # plt.plot(lamdas, analytical_Soln, marker = "o", label = "Analytical Solution")
    # plt.title(f"Force $||F||_2$ vs Stretch Ratio $\\lambda$")
    # plt.xlabel("$\\lambda$")
    # plt.ylabel("$||F||_2$")
    # plt.legend()
    # plt.savefig("beam2plot.png")
    # plt.show()
    print("RMSE error = ", rmse)
if __name__ == "__main__":
    graph(1)
