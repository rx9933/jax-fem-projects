import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.append('/workspace')

import numpy as np
import jax 
import jax.numpy as jnp
from scipy.optimize import minimize

import meshio
mesh = meshio.read('reference_gell.msh')
pdata = mesh.points
np.save('gellpoints.npy', pdata)

# alpha_0 = np.ones((len(pdata),))
def get_alpha_0(x0):
    vi = np.loadtxt('cell_vertices_initial.txt') 
    rff = 100 # characteristic distance "ff" for farfield #CHANGE TO 60
    vi = np.array(vi) # mesh vertices on cell surface

    rs = np.min(np.linalg.norm(x0-vi,axis=1)) # distance to cell surface
    rsc = np.minimum(rs, rff) # clipped distance to cell surface
    a0 = 2.5
    rcrit = rff*np.sqrt(2*a0-1)/(np.sqrt(2*a0-1)+1) #characterestic distance for most degraded gel portion
    aideal = 1/2*(((rsc-rcrit)/(rff-rcrit))**2 + 1)
    return aideal
alpha_0 = np.array([get_alpha_0(x0) for x0 in pdata])

import f
s = False
Ctarget = f.main(alpha_0, s)

def objective(alpha): 
    Csim = f.main(alpha,s)
    disp_matching = jnp.asarray(np.linalg.norm(Ctarget-Csim)**2)
    print("DP", disp_matching)

    regularization=10**3
    tik = jnp.sum(jnp.gradient(alpha)**2)
 
    tikhanov = tik*regularization
    print("TK",tikhanov)
    
    obj = (disp_matching+tikhanov)
    print("OBJECTIVE", obj)
    return obj


dphi = jax.grad(objective)

# global iter
# iter = 0
result = minimize(objective, alpha_0, method='L-BFGS-B', jac=dphi, bounds = [(.5, 10)]*alpha_0.size)
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
# Compute solution with final alpha and save alpha/j to vtk
print("SOLUTION",solution)
f.main(solution,True) 

# print('Solution: f(%s) = %.5f' % (solution, evaluation))




