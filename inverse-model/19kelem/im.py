# fixing displacement application
import time
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.append('/workspace')
#os.environ['JAX_PLATFORMS'] = 'cpu'
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
@jax.jit
def zero_dirichlet_val(point, load_factor=1):
    return 0.

@jax.jit
def xcell_displacement(point, load_factor=1):
    ind = np.where(np.absolute(init_pos-point) < tol, 1, 0)
    i = np.nonzero(ind,size=1)
    return disp[i,:][0][0][0]*load_factor
@jax.jit 
def ycell_displacement(point, load_factor=1):
    ind = np.where(np.absolute(init_pos-point) < tol, 1, 0)
    i = np.nonzero(ind,size=1)
    return disp[i,:][0][0][1]*load_factor
@jax.jit
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


def Cauchy(sol,problem):
    shape_grads_physical = get_shape_grads_physical(problem)
    cell_sols = sol[0][np.array(problem.fes[0].cells)]
    u_grads = cell_sols[:, None, :, :, None] * shape_grads_physical[:, :, :, None, :]
    u_grads = np.sum(u_grads, axis=2)
    vectorized_get_f = np.vectorize(get_f, signature='(n,m)->(n,m)')
    f_vals = vectorized_get_f(u_grads)

    C = get_c(f_vals)
    return C 
"""
def calc_elem_adj(cells):
    # Function to find shared nodes between two elements
    def shared_nodes(element1, element2):
        return len(set(element1).intersection(element2))

    adjacency_list = []

    # Check for adjacency by shared nodes
    for i in range(len(cells)):
        print(i/len(cells))
        for j in range(i + 1, len(cells)):
            if shared_nodes(cells[i], cells[j]) >= 3:  # At least 3 common points to have a face in contact
                adjacency_list.append([i, j])

    # Convert the adjacency list to a numpy array
    adjacency_matrix = np.array(adjacency_list)

    return adjacency_matrix
"""
def calc_elem_adj(cells):
    # Convert cells to a NumPy array
    cells = onp.array(cells)
    
    # Initialize adjacency list
    adjacency_list = []

    # Create a set representation for fast intersection
    cell_sets = [set(cell) for cell in cells]

    # Check for adjacency by shared nodes
    for i in range(len(cells)):
        print(i / len(cells))
        for j in range(i + 1, len(cells)):
            if len(cell_sets[i].intersection(cell_sets[j])) >= 3:  # At least 3 common points to have a face in contact
                adjacency_list.append([i, j])

    # Convert the adjacency list to a numpy array
    adjacency_matrix = np.array(adjacency_list)

    return adjacency_matrix

def cross_area(common_p, points):
    pval = points[common_p[:,0]]
    vec1 = points[common_p[:,1]] - pval
    vec2 = points[common_p[:,2]] - pval

    c_p = np.cross(vec1, vec2)
    areas = 0.5 * np.linalg.norm(c_p, axis = 1)

    return areas 

def all_area(elem_adjacency):
    print("a")
    num_int_facets = elem_adjacency.shape[0]

    print("b")
    pc1 = cells[elem_adjacency[:,0]]
    pc2 = cells[elem_adjacency[:,1]]
    print("c")
    
    common_p = np.zeros((num_int_facets, 3))
    for i in range(num_int_facets):
        r = np.intersect1d(pc1[i], pc2[i])
        common_p = common_p.at[i,:].set(r)
    common_p = common_p.astype('i')
    print("d")
    A = cross_area(common_p, points)
    return A
cells, points = cells_out()



def regularization(a, elem_adjacency, A):
    a_diff = np.abs(a[elem_adjacency[:,0]]-a[elem_adjacency[:,1]])
    gamma = 10**(-9)
    # assert not(np.all(np.isclose(A,0)))
    # assert not(np.all(np.isclose(a_diff,0))) # will equal 0 on first iteration when inputting alpha as np.ones
    tv_reg = gamma * np.einsum('fo,f->fo', a_diff, A)
    print("TV REG", )
    print("%.7f", np.sum(np.linalg.norm(tv_reg))); 
    # assert not(np.all(np.isclose(tv_reg,0)))
    
    return np.sum(tv_reg)

def main():
    
    elem_adjacency = calc_elem_adj(cells)  ####
    # elem_adjacency = np.array(onp.loadtxt("adjacency_matrix.txt")).astype(np.int64)
    A = all_area(elem_adjacency)
    
    start_time = time.time()
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    vtk_path = os.path.join(data_dir, f'vtk/test.vtu')

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
    
    C_0 = Cauchy(sol_0,ref_problem)
    C_0quad = ref_problem.fes[0].convert_from_dof_to_quad_C(C_0)[:, :, 0,0] # 4 points per tetra
    cells_JxW = ref_problem.JxW[:, 0, :] # THIS IS CONSTANT FOR ALL PROBLEMS
    
    def objective(params):
        curr_problem = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
        curr_fwd_pred = ad_wrapper(curr_problem)
        curr_sol = curr_fwd_pred(params)
        C_c = Cauchy(curr_sol, curr_problem)
        C_cquad = ref_problem.fes[0].convert_from_dof_to_quad_C(C_c)[:, :, 0,0] # 4 points per tetra
        obj = np.sum((C_0quad - C_cquad)**2 * cells_JxW) ## RESOURCE EXHAUSTED MEMORY ERROR ON GPU
        
        tik = regularization(params, elem_adjacency, A)
        # tik = 0
        print("obj + tik",obj + tik)
        return obj + tik

    obj_and_intergrad = jax.value_and_grad(objective)
    
    def obj_and_grad(alpha):
        alpha = np.reshape(alpha,(ref_problem.fe.num_cells,ref_problem.fe.num_quads)) ####
        J,dJda = obj_and_intergrad(alpha)
        return (J, dJda)
    
    #scipy.minimize
    
    fin_problem = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
    a_0 = np.ones((fin_problem.fe.num_cells,fin_problem.fe.num_quads))
    a_in = np.reshape(a_0,(a_0.size,))
    
    # 5000 iterations in FEniCS inverse model
    result = minimize(obj_and_grad, a_in, method='L-BFGS-B', jac=True, bounds = [(.5, 10)]*a_0.size, tol = 10**-8,options = {"maxiter":5000,"gtol":10**-8,"disp": True,"iprint":100}, )

    a_f = np.reshape(result.x,(ref_problem.fe.num_cells,ref_problem.fe.num_quads)) 
    # a_f = a_0

    fin_fwd_pred = ad_wrapper(fin_problem)
    fin_sol = fin_fwd_pred(a_f)

    # to save sol, alpha should be of shape (num_nodes,); need to convert from quad to dof for nodal solution
    # currently assigning 1 quad point value (per cell) as cell val
    
    save_sol(fin_problem.fes[0], fin_sol[0], vtk_path, cell_infos = [{"alpha":np.ravel(a_f)}])   
    print("TIME:",time.time() - start_time)


if __name__ == "__main__":
    main()
    # regularization()
