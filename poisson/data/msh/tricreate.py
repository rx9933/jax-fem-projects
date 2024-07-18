import numpy as np
import meshio

# Define the number of divisions along each axis
Nx = 10
Ny = 10

# Define the domain dimensions
Lx = 1.0
Ly = 1.0

# Generate points
x = np.linspace(0, Lx, Nx + 1)
y = np.linspace(0, Ly, Ny + 1)
points = np.array([[xi, yi, 0.0] for yi in y for xi in x])

# Generate cells (triangles)
triangles = []
for i in range(Ny):
    for j in range(Nx):
        # Define the four corners of the current cell
        bottom_left = i * (Nx + 1) + j
        bottom_right = bottom_left + 1
        top_left = bottom_left + (Nx + 1)
        top_right = top_left + 1
        
        # Create two triangles for each square cell
        triangles.append([bottom_left, bottom_right, top_right])
        triangles.append([bottom_left, top_right, top_left])

triangles = np.array(triangles)

# Create mesh object
mesh = meshio.Mesh(
    points=points,
    cells=[("triangle", triangles)]
)

# Save mesh to file
mesh_file = "tri3_mesh.vtk"
meshio.write(mesh_file, mesh)

print(f"Mesh saved to {mesh_file}")
