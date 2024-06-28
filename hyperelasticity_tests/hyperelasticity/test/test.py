import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.append('/workspace')
import jax
import jax.numpy as np
a =np.array([[1,2,3],[2,3,4]])
# a= [1,2]
print(type(a))