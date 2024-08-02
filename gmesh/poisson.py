import gmsh

# Initialize Gmsh
gmsh.initialize()

# Create a model
gmsh.model.add("quad8_mesh")
#elementType = 16
# Define the corner points of the square
gmsh.model.geo.addPoint(0, 0, 0, 1.0, 1)
gmsh.model.geo.addPoint(1, 0, 0, 1.0, 2)
gmsh.model.geo.addPoint(1, 1, 0, 1.0, 3)
gmsh.model.geo.addPoint(0, 1, 0, 1.0, 4)

# Define the edges of the square
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

# Define the curve loop and the surface
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addPlaneSurface([1], 1)

# Synchronize to process the geometric entities
gmsh.model.geo.synchronize()

# Transfinite definition for structured mesh
gmsh.model.mesh.setTransfiniteCurve(1, 32)
gmsh.model.mesh.setTransfiniteCurve(3, 32)
gmsh.model.mesh.setTransfiniteCurve(2, 32)
gmsh.model.mesh.setTransfiniteCurve(4, 32)
gmsh.model.mesh.setTransfiniteSurface(1)

# Recombine the mesh into quadrangles
gmsh.model.mesh.setRecombine(2, 1)

# Set the mesh order to 2 for quadratic elements
gmsh.model.mesh.setOrder(2)

# Get the element type for quadratic quadrangle elements (quad8)
elementType = gmsh.model.mesh.getElementType("quadrangle",2)
# elementType = 16
#print("Ele",elementType)

# Generate the mesh
gmsh.model.mesh.generate(3)

# Save the mesh
gmsh.write("quad8_32x32.msh")

# Finalize Gmsh
gmsh.finalize()
