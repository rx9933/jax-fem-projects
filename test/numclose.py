
import numpy as np
t = 0.015
count_close_to_half = 0

with open('data.txt', 'r') as file:
    for line in file:
        values = line.split()
        curr = np.array([float(value) for value in values])
        if np.linalg.norm(curr[1:] - np.array([.4,.5,.5])) <= t:
            count_close_to_half += 1

# Print the result
print(f'Numbers close to 0.5 within a tolerance of {t}: {count_close_to_half}')