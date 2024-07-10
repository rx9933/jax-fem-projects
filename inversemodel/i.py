import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.append('/workspace')

import numpy as np
import jax 
import jax.numpy as jnp
from scipy.optimize import minimize

import meshio
mesh = meshio.read('reference_gell.msh') # or something else?
pdata = mesh.points
np.save('gellpoints.npy', pdata)
# alpha_0 = np.ravel(np.ones((19634,4))) # 19634
alpha_0 = np.ones((len(pdata),))
np.save('alpha.npy',alpha_0)

import f

s = False
def objective(alpha): 
    print(alpha)
    Ctarget = np.load('C.npy')
    global iter
    print("ITER", iter,'\n')
    iter +=1
    f.main(alpha,s) # pass in new alpha, as scipy optimizes phi(alpha)
    
    Csim = np.load('Csim.npy')
    disp_matching = jnp.asarray(np.linalg.norm(Ctarget-Csim)**2)
    
    regularization=10**-3
    tik = jnp.sum(jnp.gradient(alpha)**2)
    
    tikhanov = tik*regularization
    print("OBJECTIVE", disp_matching+tikhanov)
    return disp_matching+tikhanov

dphi = jax.grad(objective)

global iter
iter = 0
result = minimize(objective, alpha_0, method='L-BFGS-B', jac=dphi, bounds = [(.5, 10)]*alpha_0.size)
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
# Compute solution with final alpha and save alpha/j
f.main(solution,True)

# print('Solution: f(%s) = %.5f' % (solution, evaluation))




