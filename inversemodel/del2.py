# While loop conditioned on x and n with a jitted body.
import time
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.append('/workspace')
# os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
pos = jnp.array([[.1,.1,.1],[.2,.2,.2],[.3,.3,.3]]) 
a = jnp.array([[1,2],[3,4]])
@jax.jit
def ind(x):
    # i = jnp.where(pos==x)
    tol = .01
    i = jnp.where(jnp.linalg.norm(pos-x) < tol, 1, 0)
    print(x)
    print(i)
    print("a",type(a))
    return a[i]

def mat():
    x = jnp.array([.1,.1,.1])
    val = ind(x)
    # print(a)
    return val
# print(mat())
a =1
a = a + 1
print(a)