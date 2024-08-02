import meshio
vtkf = "p.vtk"
mf = "output.msh"
mesh = meshio.read(vtkf)
meshio.write(mf,mesh)
print("converted")
