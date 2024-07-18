import os
import sys
os.environ['JAX_PLATFORMS']='cpu'
import jax.numpy as np

# Define the initial arrays
twoxfour = np.array([[.1, .1], [.2, .2], [.3, .3], [.4, .4]])
x = np.array([twoxfour, np.multiply(twoxfour, np.array(2)), np.multiply(twoxfour, np.array(3))]) # X array
sequence = np.arange(.1, 1.3, .1)
points = np.tile(sequence, (2, 1)).T # POINTS ARRAY
print(points.shape)
# Reshape points and act for broadcasting
points_reshaped = x.reshape(-1, 2)
act_reshaped = points.reshape(1, -1, 2)

# Use broadcasting to find matching indices
matches = np.all(points_reshaped[:, None, :] == act_reshaped, axis=-1)

# Get the index positions of the matches
index_positions = np.argmax(matches, axis=1)

# Reshape to the original shape of points
index_matrix = index_positions.reshape(x.shape[:-1])

print(points[index_matrix]) #use b instead


