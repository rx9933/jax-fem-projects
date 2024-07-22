import numpy as np
import meshio

# Load initial positions and displacements
init_pos = np.asarray(np.loadtxt("cell_vertices_initial.txt"))
disp = np.asarray(np.loadtxt("cell_vertices_final.txt")) - init_pos
tol = 10**-3

def zero_dirichlet_val(point, load_factor=1):
    return 0.0

def xcell_displacement(point, load_factor=1):
    ind = np.argwhere(np.abs(init_pos[:,0] - point[0]) < tol)
    if ind.size > 0:
        return disp[ind[0,0]] * load_factor
    return np.zeros_like(point)

def ycell_displacement(point, load_factor=1):
    ind = np.argwhere(np.absolute(init_pos[:,0]-point[0]) < tol,size=1)
    return disp[ind[0,0]][1]*load_factor

def zcell_displacement(point, load_factor=1):
    ind = np.argwhere(np.absolute(init_pos[:,0]-point[0]) < tol,size=1)
    return disp[ind[0,0]][2]*load_factor
# Read the input mesh
input_mesh_file = "reference_domain.xdmf"
mesh = meshio.read(input_mesh_file)

# Apply displacements to the mesh points
new_points = mesh.points.copy()
for i, point in enumerate(mesh.points):
    displacement = xcell_displacement(point)
    new_points[i] += displacement

# Create the final mesh with updated points
final_mesh = meshio.Mesh(points=new_points, cells=mesh.cells)

# Write the final mesh to a file
output_mesh_file = "output_mesh.vtu"
meshio.write(output_mesh_file, final_mesh)

print(f"Converted {input_mesh_file} to {output_mesh_file} with applied displacements.")
