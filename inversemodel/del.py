# While loop conditioned on x and n with a jitted body.
import time
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.append('/workspace')
# os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
@jax.jit
def loop_body(prev_i):
  print(prev_i)
  return prev_i + 1

def g_inner_jitted(x, n):
  i = 0
  while i < n:
    i = loop_body(i)
  return x + i

print(g_inner_jitted(10, 20))

