import numpy as np


def batch_normalization_analysis():
    """
    Analyze batch normalization for problem 3 of HW4.

    Problem: We have a mini-batch of 4 examples with 3 features each.
    The input matrix is:
    [12  0  -5]
    [14 10   5]
    [14 10   5]
    [12  0  -5]
    """
    print("Batch Normalization Analysis - Problem 3\n")

    # Given input matrix (4 examples x 3 features)
    input_matrix = np.array([
        [12, 0, -5],
        [14, 10, 5],
        [14, 10, 5],
        [12, 0, -5]
    ])

    # Part (a): Calculate z_hat
    # First, calculate the mean for each feature across the batch
    mean = np.mean(input_matrix, axis=0)
    print(f"Feature means: {mean}")

    # Then calculate the variance for each feature
    variance = np.mean((input_matrix - mean) ** 2, axis=0)
    print(f"Feature variances: {variance}")

    # Normalize each feature (epsilon = 0 as stated in the problem)
    z_hat = (input_matrix - mean) / np.sqrt(variance)

    # Transpose for the requested 3×4 matrix format (features x samples)
    z_hat_transposed = z_hat.T
    print("\nPart (a): z_hat matrix (3x4):")
    print(np.round(z_hat_transposed, 4))

    # Part (b): Calculate z_tilde using gamma and beta
    gamma = np.array([1, 1, 1])
    beta = np.array([0, -10, 10])

    # Apply scaling and shifting
    z_tilde = np.zeros_like(input_matrix)
    for i in range(3):  # For each feature
        z_tilde[:, i] = gamma[i] * z_hat[:, i] + beta[i]

    # Transpose for the requested 3×4 matrix format
    z_tilde_transposed = z_tilde.T
    print("\nPart (b): z_tilde matrix (3x4):")
    print(np.round(z_tilde_transposed, 4))

    # Part (c): Explain differences in batch normalization during training vs testing
    print("\nPart (c): Differences between batch normalization during training and testing:")
    print("1. During training:")
    print("   - Batch normalization uses statistics (mean and variance) computed from the current mini-batch")
    print("   - These batch statistics are used for normalization in the forward pass")
    print("   - Running averages of means and variances are updated for later use in testing")
    print("   - This introduces noise that acts as regularization and helps with training")

    print("\n2. During testing:")
    print("   - Instead of batch statistics, the stored population statistics (running mean and variance)")
    print("     accumulated during training are used")
    print("   - This ensures consistent normalization regardless of test batch size")
    print("   - No further updates to the running statistics occur")
    print("   - The model behaves deterministically, without the regularizing noise of batch statistics")

    # Part (d): Effect of batch size during testing
    print("\nPart (d): How batch size during testing affects testing results:")
    print("1. When using population statistics (standard practice):")
    print("   - The batch size during testing should have no effect on results")
    print("   - Each sample is normalized using the same fixed statistics regardless of batch size")
    print("   - This allows for consistent inference even with a single sample")

    print("\n2. If batch statistics were used during testing (not standard practice):")
    print("   - Smaller batch sizes would lead to less reliable statistics")
    print("   - Results would be inconsistent and depend on which samples are batched together")
    print("   - Performance would likely degrade as batch size decreases")
    print("   - Single-sample inference would be impossible as variance cannot be calculated")
    print("   - Different inferences on the same input would be possible depending on the batch")


if __name__ == "__main__":
    batch_normalization_analysis()