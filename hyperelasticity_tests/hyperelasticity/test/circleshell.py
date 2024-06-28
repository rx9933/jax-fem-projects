# $MeshFormat
# 2.2 0 8
# $EndMeshFormat
# $Nodes
# 2204
# $MeshFormat
# 2.2 0 8
# $EndMeshFormat
# $Nodes
# 98111


# $MeshFormat
# 2.2 0 8
# $EndMeshFormat
# $Nodes
# 177901


import numpy as np
done = False
t = 10
curr = [1]
with open('hollowdata.txt', 'r') as file:
    for line in file:
        if curr[0] == 2204:
            break
        values = line.split()
        curr = np.array([float(value) for value in values])
        if np.linalg.norm(curr[1:])-30<=t:
            print(curr)
        if np.linalg.norm(curr[1:])-20<=t:
            print(curr[1:])
            print("in trouble, vol within sphere is filled")
            # print(curr[1:])



# # Print the result
# print(f'.4: {min1}, .6: {max1}')
# print(300*np.array([.6,.5,.5]))