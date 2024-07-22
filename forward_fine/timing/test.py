import jax.numpy as np
init_pos=np.array([[.2,.2,.3],[.1,.1,.1],[.3,.3,.1]])
point = np.array([.1,.1,.1])
print(point.shape)
print(init_pos.shape)
# print(np.linalg.norm(init_pos-point,axis=1))
# print(np.where(np.linalg.norm(init_pos-point,axis=1) < 10**-2,size=1)[0][0])
print(np.all(np.isclose(point, init_pos)))