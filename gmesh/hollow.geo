// Merge the STL file containing the cell surface
Merge "CytoD_downsampled.stl";

// Lengths for the outer boundary box
l = 300.;
length = l;
height = l;
depth = l;
lcar1 = 0.5;

// Centroid of the cell
// centroidx = 77.30223623;
// centroidy = 77.03447408;
// centroidz = 66.74390624;
centroidx = 0;
centroidy = 0;
centrodz = 0;

// Outer Boundary Points
Point(1) = {centroidx+length/2, centroidy+height/2, centroidz+depth/2, lcar1}; 
Point(2) = {centroidx+length/2, centroidy+height/2, centroidz-depth/2, lcar1}; 
Point(3) = {centroidx-length/2, centroidy+height/2, centroidz+depth/2, lcar1}; 
Point(4) = {centroidx-length/2, centroidy-height/2, centroidz+depth/2, lcar1}; 
Point(5) = {centroidx+length/2, centroidy-height/2, centroidz+depth/2, lcar1}; 
Point(6) = {centroidx+length/2, centroidy-height/2, centroidz-depth/2, lcar1}; 
Point(7) = {centroidx-length/2, centroidy+height/2, centroidz-depth/2, lcar1}; 
Point(8) = {centroidx-length/2, centroidy-height/2, centroidz-depth/2, lcar1}; 

// Outer Box
Line(1) = {3, 1};
Line(2) = {3, 7};
Line(3) = {7, 2};
Line(4) = {2, 1};
Line(5) = {1, 5};
Line(6) = {5, 4};
Line(7) = {4, 8};
Line(8) = {8, 6};
Line(9) = {6, 5};
Line(10) = {6, 2};
Line(11) = {3, 4};
Line(12) = {8, 7};
Line Loop(13) = {-6, -5, -1, 11};
Plane Surface(14) = {13};
Line Loop(15) = {4, 5, -9, 10};
Plane Surface(16) = {15};
Line Loop(17) = {-3, -12, 8, 10};
Plane Surface(18) = {17};
Line Loop(19) = {7, 12, -2, 11};
Plane Surface(20) = {19};
Line Loop(21) = {-4, -3, -2, 1};
Plane Surface(22) = {21};
Line Loop(23) = {8, 9, 6, 7};
Plane Surface(24) = {23};

// Define the outer gel volume
Surface Loop(25) = {14, 16, 18, 20, 22, 24};
Volume(26) = {25};

// Cell surface from STL
Physical Surface(202) = {Surface{1}}; // Assuming the cell surface is surface 1 from the STL

// Create volume for the cell (hollow part)
Surface Loop(27) = {Surface{1}}; // Cell surface loop
Volume(28) = {27};

// Subtract cell volume from gel volume
BooleanDifference{ Volume{26}; Delete; }{ Volume{28}; Delete; }

// Physical entities
Physical Surface(201) = {14, 16, 18, 20, 22, 24}; // outer box
Physical Volume(301) = {26}; // gel

// Mesh settings
Mesh.CharacteristicLengthFactor = 1.2;
Characteristic Length{PointsOf{Physical Surface{201};}} = 20;

// Generate the mesh
Mesh 3;
Mesh.MshFileVersion = 2.2;
Save "CytoD.msh";