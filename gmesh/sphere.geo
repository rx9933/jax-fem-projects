SetFactory("OpenCASCADE");

// Parameters
l = 300; // Side length of box
major = 60; // Major axis length ( diameter)
minor = 30;

// Geometry
Box(1) = {-l/2, -l/2, -l/2, l, l, l};
Sphere(2) = {0, 0, 0, major/2};
//Dilate {{0,0,0}, {minor/major,minor/major, 1.}} { Volume{2}; }
BooleanDifference{Volume{1}; Delete;}{Volume{2}; Delete; }

// Physical groups
Physical Volume(301) = {1};
Physical Surface(201) = {8, 9, 10, 11, 12, 13};
Physical Surface(202) = {7};

Mesh.CharacteristicLengthFactor = 1.2;
Characteristic Length{PointsOf{Physical Surface{202};}} = 1; 
Characteristic Length{PointsOf{Physical Surface{201};}} = 30; 

// Generate Mesh
Mesh 3;
Mesh.MshFileVersion = 2.2;
Save "standardsphere.msh";
