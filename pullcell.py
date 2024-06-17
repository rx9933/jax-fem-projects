import os
import sys

# Modify sys.path to include necessary directories
sys.path.insert(0, os.path.abspath('..'))  # Include parent directory
sys.path.append('/workspace')             # Include workspace directory

# Import necessary modules
import jax
import jax.numpy as np
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh, read_in_mesh

# Define a problem class for hyperelasticity
class HyperElasticity(Problem):
    """
    Represents a hyperelasticity problem.

    Args:
        mesh (Mesh): The computational mesh for the problem.
        vec (int): Vector size.
        dim (int): Dimension of the problem (2 or 3).
        ele_type (str): Element type (e.g., 'HEX8').
        dirichlet_bc_info (list): Information about Dirichlet boundary conditions.
        mesh: Mesh                    
    """

    def get_tensor_map(self):
        """
        Computes the first Piola-Kirchhoff stress tensor based on the strain energy density function 'psi'.
        The material properties are defined by Young's modulus (E) and Poisson's ratio (nu) using an Isochoric Neohookeen model. 
        Automatic differentiation provided by JAX computes the gradient of 'psi' with respect to the deformation gradient 'F', resulting in the function 'P_fn'.

        Returns:
            first_PK_stress: A function that computes the first Piola-Kirchhoff stress tensor.
        """
        def psi(F):
            """
            Computes the strain energy density.

            Args:
                F (array): Deformation gradient tensor.

            Returns:
                float: Strain energy density.
            """
            E = 10.    # Young's modulus
            nu = 0.3   # Poisson's ratio
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            """
            Computes the first Piola-Kirchhoff stress tensor.

            Args:
                u_grad (array): Gradient of the displacement field.

            Returns:
                array: First Piola-Kirchhoff stress tensor.
            """
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P
        return first_PK_stress


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
meshio_mesh = read_in_mesh("pullcellmesh.msh", cell_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


# Define boundary locations.
lcount = 0
rcount = 0
def left(point):
    act = np.array([.4, .5, .5])
    print(np.isclose(np.linalg.norm(point-act), 0.0, atol=.015))
    return np.isclose(np.linalg.norm(point-act), 0.0, atol=.015)


def right(point):
    act = np.array([.6, .5, .5])
    return np.isclose(np.linalg.norm(point-act) , 0.0, atol=.015)

def zero_dirichlet_val(point):
    return 0.

def pull_dirichlet_val_x(point):
    return .5*point[0] 

# Apply zero Dirichlet boundary values on the left side and pulling on the right side along the x-axis.
dirichlet_bc_info = [[left] * 3 + [right] * 3, 
                    [0, 1, 2] * 2, 
                    [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val] + 
                    [zero_dirichlet_val, zero_dirichlet_val, pull_dirichlet_val_x]]


# Create an instance of the problem.
problem = HyperElasticity(mesh,
                          vec=3,
                          dim=3,
                          ele_type=ele_type,
                          dirichlet_bc_info=dirichlet_bc_info)
# Solve the defined problem.
sol = solver(problem, use_petsc=True)
# Store the solution to local file.
vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
save_sol(problem.fes[0], sol[0], vtk_path)
# print(lcount)
# print(rcount)