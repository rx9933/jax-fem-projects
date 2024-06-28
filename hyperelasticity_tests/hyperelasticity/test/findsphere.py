import numpy as np
import meshio

def read_msh(filename):
    mesh = meshio.read(filename)
    points = mesh.points
    cells = mesh.cells_dict["tetra"]
    return points, cells

def transform_sphere_to_ellipsoid(sphere_points, a, b, c):
    # Scale factors for the ellipsoid
    # print(sphere_points)
    scale_factors = np.array([a, b, c])

    # Normalize points to lie on the unit sphere
    norms = np.linalg.norm(sphere_points, axis=1, keepdims=True)
    print(norms)
    unit_sphere_points = sphere_points / norms

    # Scale unit sphere points to the ellipsoid
    ellipsoid_points = unit_sphere_points * scale_factors

    # Apply transformation only to the sphere surface points
    # ellipsoid_points[sphere_indices] = ellipsoid_points[sphere_indices] * scale_factors

    return ellipsoid_points

def write_points_to_msh(points, cells, filename):
    mesh = meshio.Mesh(points=points, cells={"tetra": cells})
    meshio.write(filename, mesh, binary=False)


# Read the input .msh file
input_filename = "sphere.msh"
points, cells = read_msh(input_filename)

# Assuming the sphere is the last set of cells
# sphere_indices = cells[-1000:]  # Adjust this based on your mesh structure
sphere_indicies = np.where(np.linalg.norm(points, axis = 1) <= 50)
# Ellipsoid semi-axes
box_length = 300
a = 4 * box_length
b = .5 * box_length
c = .5 * box_length

# Transform sphere points to ellipsoid
ellipsoid_points = transform_sphere_to_ellipsoid(points[sphere_indicies], a, b, c)

# Write the ellipsoid mesh to a new .msh file
output_filename = "ellipsoid.msh"
write_points_to_msh(ellipsoid_points, cells, output_filename)

print("Ellipsoid mesh generated and saved as", output_filename)