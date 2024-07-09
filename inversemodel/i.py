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
alpha_0 = np.ravel(np.ones((19,4))) # 19634
np.save('alpha.npy',alpha_0)

import f


def objective(alpha): # phi
    print("AAA",jnp.asarray(alpha))
    Ctarget = np.load('C.npy')
    ## need to call v. main and use alpha in this call to reset value of csim
    f.main() # call f main but don't need to use alpha, since previous alpha value has already been saved in a.npy

    np.save('alpha.npy',np.asarray(alpha))
    Csim = np.load('Csim.npy')
    disp_matching = np.linalg.norm(Ctarget-Csim)**2
    
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
    # print(disp_matching+tikhanov)
    return disp_matching+tikhanov # hanov


dphi = jax.grad(objective)
# define range for input

# def alpha_0(x0): # alpha_0 = jnp.ones((19634,4))
#     return 1


# perform the bfgs algorithm search
result = minimize(objective, alpha_0, method='BFGS', jac=dphi)
# summarize the result

# print('Status : %s' % result['message'])
# print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
# print('Solution: f(%s) = %.5f' % (solution, evaluation))



