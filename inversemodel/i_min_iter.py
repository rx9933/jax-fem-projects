import os
import sys
import time
start_time = time.time()

sys.path.insert(0, os.path.abspath('..'))
sys.path.append('/workspace')

import numpy as np
import jax 
import jax.numpy as jnp
from scipy.optimize import minimize

import meshio
mesh = meshio.read('reference_gell.msh')
pdata = mesh.points
print("PDATA",pdata.shape)
np.save('gellpoints.npy', pdata)

import f
s = False

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
alpha_0 = np.array([get_alpha_0(x0) for x0 in pdata])
Ctarget = f.main(alpha_0, s)




def objective(alpha): 
    Csim = f.main(alpha,s)
    print("CT-CS",np.linalg.norm(Ctarget-Csim))
    # print("CT",Ctarget)
    # disp_matching = jnp.asarray(np.linalg.norm(Ctarget-Csim)**2)
    disp_matching = jnp.sum((Ctarget-Csim)**2)
    # assert jnp.sum((Ctarget-Csim)**2) == jnp.asarray(np.linalg.norm(Ctarget-Csim)**2)

    print("DP", disp_matching)

    regularization=10**-8
    tik = jnp.sum(jnp.gradient(alpha)**2)
 
    tikhanov = tik*regularization
    print("TK",tikhanov)
    
    obj = (disp_matching+tikhanov)
    print("OBJECTIVE", obj)
    return obj


dphi = jax.grad(objective)

# global iter
# iter = 0
alpha_1 = np.ones((len(pdata),))

result = minimize(objective, alpha_1, method='L-BFGS-B', jac=dphi, bounds = [(.5, 10)]*alpha_1.size, tol = 10**-8,options = {"maxiter":1,"gtol":10**-8,"disp": True}, )
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
# Compute solution with final alpha and save alpha/j to vtk
print("SOLUTION",solution)
np.save("alphafinal.npy",solution)
f.main(solution,True) 

dt = time.time() - start_time
print("total time for convergence",dt)
# print('Solution: f(%s) = %.5f' % (solution, evaluation))




