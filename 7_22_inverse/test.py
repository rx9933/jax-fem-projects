import numpy as np

# Given temperature array
T = np.array([283.0022, 275.50772, 272.43988, 270.96865, 270.91143, 269.17876, 
              266.53206, 273.07951, 270.83309, 266.93154, 279.04068, 269.70494, 269.81309])

# Calculate the average (mean)
mean_T = np.mean(T)

# Calculate the standard deviation
std_dev_T = np.std(T)

print(f"Average (Mean): {mean_T}")
print(f"Standard Deviation: {std_dev_T}")
