import numpy as np
import meshio

def create_sphere(radius, center, n_theta, n_phi):
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta, phi = np.meshgrid(theta, phi)

    x = radius * np.sin(theta) * np.cos(phi) + center[0]
    y = radius * np.sin(theta) * np.sin(phi) + center[1]
    z = radius * np.cos(theta) + center[2]

    points = np.array([x.flatten(), y.flatten(), z.flatten()]).T

    faces = []
    for i in range(n_phi - 1):
        for j in range(n_theta - 1):
            p1 = i * n_theta + j
            p2 = p1 + n_theta
            p3 = p1 + 1
            p4 = p2 + 1
            faces.append([p1, p2, p3])
            faces.append([p3, p2, p4])
    return points, np.array(faces)

# Parameters for the sphere
radius = 30
center = [0, 0, 0]
n_theta = 20 # Number of divisions along the theta direction
n_phi = 40 # Number of divisions along the phi direction

# Generate sphere points and faces
points, faces = create_sphere(radius, center, n_theta, n_phi)

# Create the mesh
mesh = meshio.Mesh(points=points, cells=[("triangle", faces)])

# Write the mesh to an STL file
meshio.write("CytoD_downsampled.stl", mesh)