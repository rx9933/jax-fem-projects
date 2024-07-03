import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # Include parent directory
sys.path.append('/workspace')             # Include workspace director

import meshio

def convert_xdmf_to_msh(xdmf_filename, msh_filename):
    # Read the .xdmf file
    mesh = meshio.read(xdmf_filename)
   
    # Write to .msh file
    meshio.write(msh_filename, mesh)

if __name__ == "__main__":
    # Replace 'input.xdmf' with your .xdmf file path
    xdmf_filename = "reference_domain.xdmf"
   
    # Replace 'output.msh' with your desired .msh file path
    msh_filename = "output.msh"
   
    convert_xdmf_to_msh(xdmf_filename, msh_filename)
    print(f"Conversion complete: {xdmf_filename} -> {msh_filename}")