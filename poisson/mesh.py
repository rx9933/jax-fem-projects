import numpy as np
import meshio

# Define the number of divisions along each axis
Nx = 8
Ny = 8

# Define the domain dimensions
Lx = 1.0
Ly = 1.0

# Generate points
x = np.linspace(0, Lx, 2*Nx + 1)  # Mid-side nodes included
y = np.linspace(0, Ly, 2*Ny + 1)  # Mid-side nodes included
points = np.array([[xi, yi, 0.0] for yi in y for xi in x])

# Generate cells
quads = []
for i in range(Ny):
    for j in range(Nx):
        # Define corner and mid-side nodes for quad8
        bottom_left = (2 * i) * (2 * Nx + 1) + 2 * j
        bottom_right = bottom_left + 2
        top_right = bottom_left + 2 * (2 * Nx + 1) + 2
        top_left = bottom_left + 2 * (2 * Nx + 1)
        
        mid_bottom = bottom_left + 1
        mid_right = bottom_left + (2 * Nx + 1) + 1
        mid_top = top_left + 1
        mid_left = bottom_left + (2 * Nx + 1)
        
        quads.append([
            bottom_left, bottom_right, top_right, top_left, 
            mid_bottom, mid_right, mid_top, mid_left
        ])

quads = np.array(quads)

# Create mesh object
mesh = meshio.Mesh(
    points=points,
    cells=[("quad8", quads)]
)

# Save mesh to file
meshio.write("quad8_mesh.vtk", mesh)

# Optionally, print mesh information
print(mesh)
