import numpy as np
from scipy.spatial import Delaunay
import meshio

# Generate points inside a sphere
def generate_points_in_sphere(radius, num_points):
    points = []
    while len(points) < num_points:
        point = np.random.uniform(-radius, radius, 3)
        if np.linalg.norm(point) <= radius:
            points.append(point)
    return np.array(points)

# Generate tetrahedral mesh
def generate_tetrahedral_mesh(points):
    delaunay = Delaunay(points)
    return delaunay.points, delaunay.simplices

# Save the mesh to an STL file
def save_mesh_to_stl(points, simplices, filename):
    cells = [("tetra", simplices)]
    mesh = meshio.Mesh(points, cells)
    mesh.write(filename)

# Parameters
radius = 1.0
num_points = 1000
filename = "sphere_filled_with_tetrahedrons.stl"

# Generate points and tetrahedral mesh
points = generate_points_in_sphere(radius, num_points)
mesh_points, mesh_simplices = generate_tetrahedral_mesh(points)

# Save the mesh to an STL file
save_mesh_to_stl(mesh_points, mesh_simplices, filename)

print(f"STL file saved as {filename}")