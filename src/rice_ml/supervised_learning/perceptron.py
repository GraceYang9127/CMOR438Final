"""
perceptron.py

Binary Perceptron classifier. Labels must be -1 or +1.

The perceptron finds a linear decision boundary w · x + b = 0 by iterating
over misclassified training samples and updating the weight vector.

Classes
-------
Perceptron
    Single-layer binary classifier with step activation.
"""

import numpy as np


class Perceptron(object):
    """
    Binary Perceptron classifier.

    Learns a linear decision boundary for -1/+1 labeled data.
    Converges when all training samples are correctly classified.

    Parameters
    ----------
    eta: float, default=0.5
        Learning rate (controls update step size).
    epochs: int, default=50
        Maximum number of passes over the training data.

    Attributes
    ----------
    w_: ndarray of shape (n_features + 1,)
        Weight vector after fitting. w_[:-1] are the feature weights.
        w_[-1] is the bias term.
    errors_: list of int
        Number of misclassifications per epoch. Empty if convergence
        is reached before max epochs.

    Notes
    -----
    Labels must be -1 or +1. Passing 0/1 labels will produce incorrect
    weight updates because the update rule computes (y_hat - y), which
    only works correctly on the -1/+1 scale.
    """

    def __init__(self, eta=0.5, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def net_input(self, X):
        """
        Compute the linear combination z = w · x + b.

        Parameters
        ----------
        X: array-like of shape (n_features,) or (n_samples, n_features)

        Returns
        -------
        float or ndarray
        """
        return np.dot(X, self.w_[:-1]) + self.w_[-1]

    def predict(self, X):
        """
        Predict class labels for X using the step activation.

        Returns +1 if net_input >= 0, else -1.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred: ndarray of shape (n_samples,), values in {-1, +1}
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def fit(self, X, y):
        """
        Fit the perceptron to training data.

        Iterates over all samples each epoch and applies the update rule
        w -= eta * (y_hat - y) * x whenever a sample is misclassified.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,), values in {-1, +1}

        Returns
        -------
        self
        """
        X, y = np.array(X, dtype=float), np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        #Initialize weights randomly; last entry is the bias
        self.w_ = np.random.rand(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                #update = eta * (y_hat - y); zero when prediction is correct
                update = self.eta * (self.predict(xi) - target)
                self.w_[:-1] -= update * xi
                self.w_[-1] -= update
                errors += int(update != 0)
            if errors == 0:
                return self
            self.errors_.append(errors)
        return self
