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

# def get_alpha(x0):
#     vi = np.loadtxt('cell_vertices_initial.txt') 
#     rff = 100 # characteristic distance "ff" for farfield
#     vi = np.array(vi) # mesh vertices on cell surface

#     rs = np.min(np.linalg.norm(x0-vi,axis=1)) # distance to cell surface
#     rsc = np.minimum(rs, rff) # clipped distance to cell surface
#     a0 = 2.5
#     rcrit = rff*np.sqrt(2*a0-1)/(np.sqrt(2*a0-1)+1) #characterestic distance for most degraded gel portion
#     aideal = 1/2*(((rsc-rcrit)/(rff-rcrit))**2 + 1)
#     return aideal
# alpha_0 = np.array([get_alpha(point) for point in pdata])



alpha_0 = np.ones((len(pdata),))


np.save('alpha.npy',alpha_0)

import f

def objective(alpha): 
    # print(alpha)
    Ctarget = np.load('C.npy')
    # global iter
    # print("ITER", iter,'\n')
    # iter +=1
    # f.main(alpha,s) # pass in new alpha, as scipy optimizes phi(alpha)

    Csim = np.load('Csim.npy')
    disp_matching = jnp.asarray(np.linalg.norm(Ctarget-Csim)**2)
    
    regularization=10
    tik = jnp.sum(jnp.gradient(alpha)**2)
 
    tikhanov = tik*regularization
    print("OBJECTIVE", disp_matching+tikhanov)
    return disp_matching+tikhanov
    
s = False
alpha = alpha_0
for i in range(5):
    r = .01
    gradobj = jax.grad(objective)
    print("OBJ GRAD",gradobj(alpha))
    alpha = alpha-r*gradobj(alpha)
    f.main(alpha,s)
print("Final alpha",alpha)
f.main(alpha,True)
# dphi = jax.grad(objective)

# global iter
# iter = 0
# result = minimize(objective, alpha_0, method='L-BFGS-B', jac=dphi, bounds = [(.5, 10)]*alpha_0.size)
# print('Status : %s' % result['message'])
# print('Total Evaluations: %d' % result['nfev'])
# # evaluate solution
# solution = result['x']
# evaluation = objective(solution)
# # Compute solution with final alpha and save alpha/j
# f.main(solution,True)

# print('Solution: f(%s) = %.5f' % (solution, evaluation))




