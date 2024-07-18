import os
import sys
import time
start_time = time.time()

sys.path.insert(0, os.path.abspath('..'))
sys.path.append('/workspace')

import numpy as onp
import jax
import jax.numpy as np
from scipy.optimize import minimize
import f


problem=f.solve_problem.get_problem()

# get u_0 (send in b = 10 e**)
u_0 = f.solve_problem(1,True) # b val doesn't matter, uses orig as defined in f.py
u_0quad = problem.fes[0].convert_from_dof_to_quad(u_0)[:,:,0]
cells_JxW = problem.JxW[:,0,:]

def objective(b): 
    # solve f with b
    # get u_c, sol[0]

    u_cquad = problem.fes[0].convert_from_dof_to_quad(u_c)[:,:,0] # modify to (sol[0] - u_0)**2 for J?

    obj = np.sum((u_cquad-u_0quad)*cells_JxW)

    return obj

dJdb = jax.grad(objective) #ADJOINT STUFF
b_1 = 1 
numnodes = u_0quad.shape[0]
# actual bounds are .16
result = minimize(objective, b_1, method='L-BFGS-B', jac=dJdb, bounds = [(0, .2)]*numnodes, tol = 10**-8,options = {"maxiter":20,"gtol":10**-8,"disp": True}, )
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
# Compute solution with final alpha and save alpha/j to vtk
print("SOLUTION",solution)

dt = time.time() - start_time
print("total time for convergence",dt)




