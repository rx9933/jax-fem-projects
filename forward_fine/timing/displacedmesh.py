import numpy as np
import meshio

# Load initial positions and displacements
init_pos = np.asarray(np.loadtxt("cell_vertices_initial.txt"))
fin_pos = np.asarray(np.loadtxt("cell_vertices_final.txt")) 
disp = fin_pos- init_pos
tol = 10**-3

def zero_dirichlet_val(point, load_factor=1):
    return 0.0

def xcell_displacement(point, load_factor=1):
    # ind = np.argwhere(np.linalg.norm(init_pos[:,:]-point[:],axis=1) < tol)
    ind = np.where(np.linalg.norm(init_pos-point,axis=1) < 10**-2)[0]
    # print(ind)
    # print("v",init_pos[2472])

    if len(ind)!=0:
        # if np.linalg.norm(init_pos[2472]-point)<10**-2:
        #     print("DD",disp[ind[0,0]])
        return disp[ind[0]] * load_factor
    else:
        return np.zeros(3)

input_mesh_file = "reference_domain.xdmf"
mesh = meshio.read(input_mesh_file)

# Apply displacements to the mesh points
new_points = mesh.points.copy()
# print(new_points[2722])
for i, point in enumerate(mesh.points):
    
    displacement = xcell_displacement(point)

    if np.linalg.norm(new_points[i]+displacement-np.array([77.2221,65.865,62.3789]))<10**-3:
        print(new_points[i])
        print(xcell_displacement(point))
        print(new_points[i]+displacement)
        print(i)
    
    new_points[i] += displacement
    # assert np.linalg.norm(new_points[i]-np.array([76.0919,74.906,59.0598]))<10**-3

    # if (xcell_displacement(point)==np.zeros(3)).all:
        # assert(np.array([76.0919,74.906,59.0598]) in fin_pos)
    # assert (new_points[i]==fin_pos[i]).all

# Create the final mesh with updated points

final_mesh = meshio.Mesh(points=new_points, cells=mesh.cells)

# Write the final mesh to a file
output_mesh_file = "output_mesh.vtu"
meshio.write(output_mesh_file, final_mesh)

print(f"Converted {input_mesh_file} to {output_mesh_file} with applied displacements.")
