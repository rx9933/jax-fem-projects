import meshio

filename = "simulation_output_f.xdmf"
meshname = "fenics_f"

with meshio.xdmf.TimeSeriesReader(filename) as reader:
    points, cells = reader.read_points_cells()
    t, point_data, cell_data = reader.read_data(0)

mesh = meshio.Mesh(points, cells, point_data=point_data, cell_data=cell_data)
mesh.write(meshname + ".vtu", file_format="vtu")
