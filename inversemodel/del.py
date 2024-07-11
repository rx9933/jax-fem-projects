
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

def objective(alpha): 
  
    regularization=10**-3
    # tik = jnp.sum(jnp.gradient(alpha)**2)
    tik = jnp.linalg.norm(alpha)
    tikhanov = tik*regularization
    return tikhanov
    
s = False
alpha = alpha_0
for i in range(3):
    r = 10
    gradobj = jax.grad(objective)
    print("OBJ GRAD",gradobj(alpha))
    alpha = alpha-r*gradobj(alpha)
    print("CURR ALPHA", alpha)
    f.main(alpha,s)
print("Final alpha",alpha)





