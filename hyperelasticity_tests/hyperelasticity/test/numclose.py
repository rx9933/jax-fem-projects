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
t = 1
while not(done):
    # t+=.1
    curr = [0]
    min1 = 0
    max1 = 0
    with open('hollowdata.txt', 'r') as file:
        for line in file:
            if curr[0] == 177901:
                break
            values = line.split()
            curr = np.array([float(value) for value in values])
            if np.linalg.norm(curr[1:] - 300*np.array([.4,.5,.5])) <= t:
                min1 += 1
                # print(curr[1:])
            if np.linalg.norm(curr[1:] - 300*np.array([.6,.5,.5])) <= t:
                max1 += 1
                # print(curr[1:])
    if min1 == max1 and min1!=0:
        print("t=", t)
        done = True


# Print the result
print(f'.4: {min1}, .6: {max1}')
print(300*np.array([.6,.5,.5]))