import numpy as np
import matplotlib.pyplot as plt


def demonstrate_symmetric_svd():
    """
    This script demonstrates the solution to Question 2 from the homework.
    It shows how to find the SVD of a symmetric matrix when its eigendecomposition is known.
    """

    print("Question 2 Solution Demonstration")
    print("=================================")

    # Define a symmetric matrix A with its eigendecomposition
    # We'll use a simple example to demonstrate the concept
    # For this demonstration, we'll create a small matrix with known eigenvalues [3, -2, 1]

    # Create a random orthogonal matrix U
    # In practice, these would be the eigenvectors given in the problem
    np.random.seed(421)  # For reproducibility
    U_random = np.random.randn(3, 3)
    U, _, _ = np.linalg.svd(U_random)  # This gives us an orthogonal matrix

    # Define the diagonal matrix of eigenvalues
    Lambda = np.diag([3, -2, 1])

    # Compute the symmetric matrix A using eigendecomposition
    A = U @ Lambda @ U.T

    print("\nOriginal Matrix A (from eigendecomposition):")
    print(A)

    print("\nEigenvalues of A:")
    eigenvalues = np.linalg.eigvals(A)
    print(eigenvalues)

    print("\nSolution Steps:")
    print("1. For a symmetric matrix, the singular values are the absolute values of the eigenvalues.")

    # For SVD, we need the absolute values of eigenvalues
    singular_values = np.abs([3, -2, 1])
    print("\nSingular values:", singular_values)

    print("\n2. We need to ensure the singular values are in descending order.")
    # Sort singular values in descending order
    sorted_indices = np.argsort(singular_values)[::-1]
    sorted_singular_values = singular_values[sorted_indices]
    print("Sorted singular values:", sorted_singular_values)

    print("\n3. For the right singular vectors, we need to adjust for negative eigenvalues.")
    print("   - For positive eigenvalues, the corresponding eigenvector stays the same")
    print("   - For negative eigenvalues, the corresponding eigenvector gets a sign flip")

    # Create matrices for SVD
    W = np.zeros_like(U)
    V = np.zeros_like(U)

    # Eigenvalues in original order
    eigenvalues_original = np.array([3, -2, 1])

    # Adjust columns based on eigenvalue signs and sorting
    for i, idx in enumerate(sorted_indices):
        # If the eigenvalue is negative, flip the sign in V
        if eigenvalues_original[idx] < 0:
            W[:, i] = U[:, idx]
            V[:, i] = -U[:, idx]  # Sign flip for the right singular vector
        else:
            W[:, i] = U[:, idx]
            V[:, i] = U[:, idx]

    # SVD form: A = W Σ V^T
    Sigma = np.diag(sorted_singular_values)

    print("\nResulting SVD components:")
    print("W (left singular vectors):")
    print(W)
    print("\nΣ (diagonal matrix of singular values):")
    print(Sigma)
    print("\nV (right singular vectors):")
    print(V)

    # Verify the SVD decomposition
    A_reconstructed = W @ Sigma @ V.T
    print("\nReconstructed Matrix A (from SVD):")
    print(A_reconstructed)

    print("\nVerification - Error between original and reconstructed matrix:")
    error = np.linalg.norm(A - A_reconstructed)
    print(f"Frobenius norm of difference: {error:.10f}")

    print("\nConclusion:")
    print("For a symmetric matrix with eigendecomposition A = U Λ U^T:")
    print("1. The singular values are the absolute values of the eigenvalues")
    print("2. The singular vectors need to be adjusted for the sign of the eigenvalues")
    print("3. Both singular values and vectors need to be reordered to maintain descending order of singular values")


if __name__ == "__main__":
    demonstrate_symmetric_svd()
