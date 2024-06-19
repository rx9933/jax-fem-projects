// -----------------------------------------------------------------------------
//
//  Gmsh GEO tutorial 18
//
//  Periodic meshes
//
// -----------------------------------------------------------------------------

// Periodic meshing constraints can be imposed on surfaces and curves.

// Let's use the OpenCASCADE geometry kernel to build two geometries.

SetFactory("OpenCASCADE");

// We start with a cube and a sphere:
Box(1) = {0, 0, 0, 1, 1, 1};
Sphere(2) = {0.5, 0.5, 0.5, 0.35}; // Placing the sphere in the center of the cube

// Subtract the sphere from the cube to create the remaining volume
v() = BooleanDifference { Volume{1}; Delete; } { Volume{2}; Delete; };

// Set a non-uniform mesh size constraint (for visualization) for the surface of the sphere
MeshSize { PointsOf{ Volume{2}; }} = 0.0005; // smaller mesh size

// Set a non-uniform mesh size constraint (for visualization) for the rest of the volume
MeshSize { PointsOf{ Volume{v()}; }} = 0.1;

// Generate tetrahedral mesh inside the remaining volume
t() = Tetrahedron;
v() {t()} = Volume{v()};

Mesh 3;
Mesh.MshFileVersion = 2.2;
Save "sphere_filled_with_tetrahedrons.msh";
