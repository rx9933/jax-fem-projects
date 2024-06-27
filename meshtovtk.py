import meshio

def convert_msh_to_vtk(input_filename, output_filename):
    """
    Convert a .msh file to a .vtk file.
    
    Parameters:
    - input_filename: The path to the input .msh file.
    - output_filename: The path to the output .vtk file.
    """
    # Read the .msh file
    mesh = meshio.read(input_filename)
    
    # Write the mesh to a .vtk file
    meshio.write(output_filename, mesh)
    
    print(f"Converted {input_filename} to {output_filename}")

# Example usage
input_msh_file = "largesphere.msh"
output_vtk_file = "largesphere.vtk"

convert_msh_to_vtk(input_msh_file, output_vtk_file)
