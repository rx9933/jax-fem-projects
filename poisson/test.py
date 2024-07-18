import jax.numpy as np
twoxfour = np.array([[.1,.1],[.2,.2],[.3,.3],[.4,.4]])
points = np.array([twoxfour, np.multiply(twoxfour,np.array(2)),np.multiply(twoxfour,np.array(3))])
sequence = np.arange(.1,1.3,.1)
act = np.tile(sequence,(2,1)).T




