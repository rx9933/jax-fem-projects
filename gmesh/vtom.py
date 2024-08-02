import pymesh

# Path to your VTK file
vtk_file = 'p.vtk'

# Load the VTK file
mesh = pymesh.load_mesh(vtk_file)

# Specify output MSH file path
msh_file = 'p.msh'

# Save the mesh to MSH format
pymesh.save_mesh(msh_file, mesh)

print(f"Converted {vtk_file} to {msh_file}")
