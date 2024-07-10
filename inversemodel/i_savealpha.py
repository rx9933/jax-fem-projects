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
alpha_0 = np.ravel(np.ones((19634,4))) # 19634
np.save('alpha.npy',alpha_0)

import f


def objective(alpha): # phi
    Ctarget = np.load('C.npy')
    global iter
    print("ITER", iter,'\n')
    iter +=1
    f.main() # uses prev value of alpha (every f.main call resets alpha at end of computations)
    print("ALPHa", alpha[0])
    np.save('alpha.npy', alpha)
    print("SAVED alpha")
    Csim = np.load('Csim.npy')

    disp_matching = jnp.asarray(np.linalg.norm(Ctarget-Csim)**2)
    
    tik = 0
    regularization=10**-2
    # grad_alpha = jax.grad(v.get_alpha) # REPLACE with alpha (function)
    # grad_alpha = jax.grad(alpha)
    # grad_alpha = jnp.gradient(alpha) # not jax.grad gives the gradient of a callable function while jnp.grad gives grad of ndarray
    tik = jnp.sum(jnp.gradient(alpha)**2)
    # for point in pdata:
    #     if not(v.gel_surface(point)) and not(v.cell_surface(point)):
    #         # print(grad_alpha(point))
    #         grad_alpha = jnp.gradient(alpha)
    #         tik+= jnp.sum(float(grad_alpha(point)**2))
    
    tikhanov = tik*regularization
    print("OBJECTIVE", disp_matching+tikhanov)
    return disp_matching+tikhanov

dphi = jax.grad(objective)
# define range for input

# def alpha_0(x0): # alpha_0 = jnp.ones((19634,4))
#     return 1

global iter
iter = 0
# perform the bfgs algorithm search
result = minimize(objective, alpha_0, method='L-BFGS-B', jac=dphi, bounds = [(2.5, 10)]*alpha_0.size)
# summarize the result

print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
# print('Solution: f(%s) = %.5f' % (solution, evaluation))



