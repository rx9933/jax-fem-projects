import numpy as np
import meshio
mesh = meshio.read('reference_gell.msh')
pdata = mesh.points
np.save('gellpoints.npy', pdata)
 
# alpha_0 = np.ones((len(pdata),))

def get_alpha_0(x0):
    vi = np.loadtxt('cell_vertices_initial.txt') 
    rff = 60 # characteristic distance "ff" for farfield 
    vi = np.array(vi) # mesh vertices on cell surface

    rs = np.min(np.linalg.norm(x0-vi,axis=1)) # distance to cell surface
    rsc = np.minimum(rs, rff) # clipped distance to cell surface
    a0 = 2.5
    rcrit = rff*np.sqrt(2*a0-1)/(np.sqrt(2*a0-1)+1) #characterestic distance for most degraded gel portion
    aideal = 1/2*(((rsc-rcrit)/(rff-rcrit))**2 + 1)
    return aideal
a_target = np.array([get_alpha_0(x0) for x0 in pdata])

a_sim = np.load("alphafinal.npy")

rmse = 1/len(pdata) * (np.linalg.norm(a_target-a_sim))**2
print("RMSE", rmse)