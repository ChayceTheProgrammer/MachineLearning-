import numpy as np
import sys

"""
This script implements a two-class logistic regression model.
"""

class logistic_regression(object):

    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

        ### YOUR CODE HERE

        self.W = np.zeros(n_features)  # init weights

        for _ in range(self.max_iter):
            grad = np.zeros(n_features)
            for i in range(n_samples):
                grad += self._gradient(X[i], y[i])

            #Avg Gradient
            self.W -= self.learning_rate * (grad / n_samples)

        ### END YOUR CODE
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)

        for _ in range(self.max_iter):

            # Shuffle the data
            indices = np.random.permutation(n_samples)

            # Mini-batch updates
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:min(i + batch_size, n_samples)]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # Compute average gradient for the mini-batch
                grad = np.zeros(n_features)
                for j in range(len(batch_indices)):
                    grad += self._gradient(X_batch[j], y_batch[j])

                # Update weights using average gradient
                self.W -= self.learning_rate * (grad / len(batch_indices))

        ### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE

        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)

        for _ in range(self.max_iter):
            # Shuffle the data
            indices = np.random.permutation(n_samples)

            # Update for each sample
            for i in indices:
                grad = self._gradient(X[i], y[i])
                self.W -= self.learning_rate * grad


        ### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
        ### YOUR CODE HERE

        z = np.dot(self.W, _x)
        grad = (-_y / (1 + np.exp(_y * z))) * _x
        return grad

        ### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
        ### YOUR CODE HERE

        scores = np.dot(X, self.W)
        proba_sensitive = 1 / (1 + np.exp(-scores))
        return np.column_stack((1 - proba_sensitive, proba_sensitive))

        ### END YOUR CODE

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
        ### YOUR CODE HERE

        scores = np.dot(X, self.W)
        return np.where(scores >= 0, 1, -1)

        ### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
        ### YOUR CODE HERE

        predictions = self.predict(X)
        return np.mean(predictions == y)

        ### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self

