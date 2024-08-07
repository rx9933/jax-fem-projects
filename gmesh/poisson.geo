

// Define the corner points of the square
Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {1, 1, 0, 1.0};
Point(4) = {0, 1, 0, 1.0};

// Define the edges of the square
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Define the curve loop and the surface
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Transfinite definition for structured mesh
Transfinite Line {1, 3} = 32 Using Progression 1;
Transfinite Line {2, 4} = 32 Using Progression 1;
Transfinite Surface {1};

// Recombine the mesh into quadrangles
Recombine Surface {1};

// Mesh the surface with quadratic elements
Mesh.getElementType= ("quadrangle", 2)

Mesh.ElementOrder = 2;
Mesh 2;

// Save the mesh
Save "quad8_32x32.msh";
