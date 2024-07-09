import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.append('/workspace')

import jax
gp = np.load('gellpoints.npy')
print(jax.numpy.gradient(jax.numpy.asarray(gp)))