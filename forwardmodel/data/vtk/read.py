import meshio
import numpy as np
mesh = meshio.read("celltest.vtu")
mpoints = mesh.points
print("CELLS",mesh.cells)
with open ('meshpointsout.txt','w+') as f: 
    for i in mpoints:
        f.write(str(i)+"\n")
print(mpoints.size)
cellfinal = np.loadtxt('../../cell_vertices_final.txt')
print(cellfinal.size)
# cellfinal = cellfinal[0]
# print(cellfinal)
# diff = []
# for m in mpoints:
#     diff.append(np.linalg.norm(m-cellfinal))
# print(np.argmin(np.array(diff)))
# print(mpoints[158])
c = 0
# for point in cellfinal:
# point = cellfinal[0]

point = np.loadtxt('../../cell_vertices_initial.txt')[0]
print("p",point)
for m in mpoints:
    if np.linalg.norm(point-m)<2.6: # 80.1682663  79.7334137  17.97245026], 2.6
        print("m",m)
        c+=1
            # break
print(c)