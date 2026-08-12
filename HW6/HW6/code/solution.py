import numpy as np
'''
Homework5: Principal Component Analysis

Helper functions
----------------
In this assignment, you may want to use following helper functions:
- np.linalg.svd(): compute the singular value decomposition on a given matrix. 
- np.dot(): matrices multiplication operation.
- np.mean(): compute the mean value on a given matrix.
- np.ones(): generate a all '1' matrix with a given shape.
- np.transpose(): matrix transpose operation.
- np.linalg.norm(): compute the norm value of a matrix.

'''

class PCA():

	def __init__(self, X, n_components):
		'''
		Args:
			X: The data matrix of shape [n_samples, n_features].
			n_components: The number of principal components. A scaler number.
		'''

		self.n_components = n_components
		self.X = X
		self.Up, self.Xp = self._do_pca()

	
	def _do_pca(self):
		'''
		To do PCA decomposition.
		Returns:
			Up: Principal components (transform matrix) of shape [n_features, n_components].
			Xp: The reduced data matrix after PCA of shape [n_samples, n_components].
		'''
		### YOUR CODE HERE
		# Calculate the mean of each feature
		mean_vector = np.mean(self.X, axis=0)

		# Center the data by subtracting the mean
		X_centered = self.X - mean_vector

		# Compute SVD of the centered data matrix
		U, sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)

		# We need to use V, not U for the principal components
		# The principal components are the right singular vectors (V)
		# Select the first n_components principal components
		Up = Vt.T[:, :self.n_components]

		# Project the centered data onto the principal components
		Xp = X_centered @ Up

		return Up, Xp
		### END YOUR CODE

	def get_reduced(self, X=None):
		'''
		To return the reduced data matrix.
		Args:
			X: The data matrix with shape [n_any, n_features] or None. 
			   If None, return reduced training X.
		Returns:
			Xp: The reduced data matrix of shape [n_any, n_components].
		'''
		if X is None:
			return self.Xp
		else:
			return X@self.Up

	def reconstruction(self, Xp):
		'''
		To reconstruct reduced data given principal components Up.

		Args:
		Xp: The reduced data matrix after PCA of shape [n_samples, n_components].

		Return:
		X_re: The reconstructed matrix of shape [n_samples, n_features].
		'''
		### YOUR CODE HERE
		# Reconstruct the data by multiplying the reduced data with the transpose
		# of the principal components
		X_re = Xp @ self.Up.T

		# Add the mean back to get the original scale
		X_re = X_re + np.mean(self.X, axis=0)

		return X_re
		### END YOUR CODE


def reconstruct_error(A, B):
	'''
	To compute the reconstruction error.

	Args: 
	A & B: Two matrices needed to be compared with. Should be of same shape.

	Return: 
	error: the Frobenius norm's square of the matrix A-B. A scaler number.
	'''
	### YOUR CODE HERE

	# Compute the Frobenius norm squared of the difference matrix
	error = np.linalg.norm(A - B, 'fro') ** 2
	return error

	### END YOUR CODE