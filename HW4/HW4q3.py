import numpy as np

# Input matrix
input_matrix = np.array([
    [12, 14, 14, 12],
    [0, 10, 10, 0],
    [-5, 5, 5, -5]
])

# Function to calculate mean and standard deviation
def calculate_mean_std(matrix):
    mean = np.mean(matrix, axis=1)
    std = np.std(matrix, axis=1)
    return mean, std

# Function to normalize the input matrix
def normalize(matrix, mean, std):
    return (matrix - mean[:, np.newaxis]) / std[:, np.newaxis]

# Function to apply scaling and shifting
def scale_shift(normalized_matrix, gamma, beta):
    # Reshape gamma and beta to match the dimensions of normalized_matrix
    return (gamma[:, np.newaxis] * normalized_matrix) + beta[:, np.newaxis]

# Calculate mean and standard deviation
mean, std = calculate_mean_std(input_matrix)

# Normalize the input matrix
normalized_matrix = normalize(input_matrix, mean, std)

# Scaling (gamma) and shifting (beta) parameters
gamma = np.array([1, 1, 1])  # For each row
beta = np.array([0, -10, 10])  # For each row

# Apply scaling and shifting
scaled_shifted_matrix = scale_shift(normalized_matrix, gamma, beta)

# Output results
print("Input Matrix:")
print(input_matrix)
print("\nMean:")
print(mean)
print("\nStandard Deviation:")
print(std)
print("\nNormalized Matrix (ẑᵢ):")
print(normalized_matrix)
print("\nScaled and Shifted Matrix (z̃ᵢ):")
print(scaled_shifted_matrix)