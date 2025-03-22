#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author: Unknown or CSCE 421 Professor
@edited by: Chayce Leonard for HW2
"""

import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):

    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        
    def fit_miniBGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch GD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,...,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

        ### YOUR CODE HERE
        n_samples, n_features = X.shape

        # Convert labels to integers and then to one-hot vectors
        labels = labels.astype(int)  # Convert to integers first
        one_hot = np.zeros((n_samples, self.k))
        one_hot[np.arange(n_samples), labels] = 1

        # Initialize weights
        self.W = np.random.randn(n_features, self.k) * 0.01

        for epoch in range(self.max_iter):
            indices = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                X_batch = X[batch_indices]
                y_batch = one_hot[batch_indices]

                grad = np.zeros_like(self.W)
                for j in range(X_batch.shape[0]):
                    grad += self._gradient(X_batch[j], y_batch[j])

                self.W -= self.learning_rate * (grad / X_batch.shape[0])

        return self
        ### END YOUR CODE

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
        ### YOUR CODE HERE

        scores = np.dot(self.W.T, _x)
        probs = self.softmax(scores)
        error = probs - _y  #_y is one-hot encoded
        grad = np.outer(_x, error)
        return grad

        ### END YOUR CODE
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

        ### YOUR CODE HERE

        e_x = np.exp(x - np.max(x)) #This is A numerically stable representation of e^x
        return e_x / e_x.sum()

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

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
        ### YOUR CODE HERE
        # Compute scores for each class
        scores = np.dot(X, self.W)  # [n_samples, k]

        # Get predictions by taking argmax over classes
        predictions = np.argmax(scores, axis=1)

        return predictions
        ### END YOUR CODE

    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
        ### YOUR CODE HERE

        # Get predictions
        predictions = self.predict(X)

        # Calculate accuracy
        accuracy = np.mean(predictions == labels)

        return accuracy

        ### END YOUR CODE

