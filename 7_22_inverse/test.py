import jax.numpy as np
num_cells = 10

# cell_infos = [["Cell info",np.ones((num_cells,))]]
# for cell_info in cell_infos:
#     print(cell_info)
#     name, data = cell_info
#     assert data.shape == (num_cells,), f"cell data wrong shape, get {data.shape}, while num_cells = {num_cells}"
       
alpha = np.array([1,2,3,4,7,8,9,10])
regularization=1
tik = np.sum(np.gradient(alpha)**2)*regularization
# print(tik)

cells = np.array([[1,2,3],[2,3,4],[4,5,6]])
ind = np.array([0,1,2,1,2])
print(cells[ind])

num_int_facets = 2
pc1 = np.array([[1,2,3,4],[6,7,8,9]])
pc2 = np.array([[2,3,4,6],[7,8,6,30]])

common_points = np.zeros((num_int_facets, 3))
for i in range(num_int_facets):
    r = np.intersect1d(pc1[i], pc2[i])
    common_points = common_points.at[i,:].set(r)

common_points.astype('i')