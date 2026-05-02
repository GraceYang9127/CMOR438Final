"""
logistic_regression.py

Binary logistic regression classifier.

Unlike the perceptron, logistic regression outputs a probability in (0, 1)
using the sigmoid activation, which handles non-linearly separable data.

Classes
-------
LogisticRegression: Binary classifier trained with stochastic gradient descent on cross-entropy loss.
"""

import numpy as np


def _sigmoid(z):
    """
    Sigmoid activation function.

    sigma(z) = 1 / (1 + e^{-z})

    Clipped to prevent overflow in exp for very large |z|.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegression(object):
    """
    Binary logistic regression via stochastic gradient descent (Lecture 5.1).

    Models the probability that a sample belongs to the positive class:
        p = sigma(w · x + b) = 1 / (1 + e^{-(w · x + b)})

    Loss function (binary cross-entropy):
        L = -y * log(p) - (1 - y) * log(1 - p)

    SGD update rule:
        w -= alpha * (p - y) * x
        b -= alpha * (p - y)

    Labels must be 0 or 1.

    Parameters
    ----------
    alpha: float, default=0.01
        Learning rate.
    epochs: int, default=1000
        Number of passes over the training data.

    Attributes
    ----------
    w_: ndarray of shape (n_features + 1,)
        Fitted weights. w_[:-1] are the feature weights
        w_[-1] is the bias term.
    errors_: list of float
        Mean cross-entropy loss per epoch.
    """

    def __init__(self, alpha=0.01, epochs=1000):
        self.alpha = alpha
        self.epochs = epochs

    def net_input(self, X):
        """
        Compute linear combination z = w · x + b.

        Parameters
        ----------
        X : array-like of shape (n_features,) or (n_samples, n_features)

        Returns
        -------
        float or ndarray
        """
        return np.dot(X, self.w_[:-1]) + self.w_[-1]

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        p = sigma(w · x + b)

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        proba: ndarray of shape (n_samples,), values in (0, 1)
        """
        X = np.array(X, dtype=float)
        return _sigmoid(self.net_input(X))

    def predict(self, X):
        """Predict class labels for X.

        Applies a 0.5 decision threshold to predict_proba.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred: ndarray of shape (n_samples,), values in {0, 1}
        """
        return (self.predict_proba(X) >= 0.5).astype(int)

    def fit(self, X, y):
        """Fit logistic regression using stochastic gradient descent.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,), values in {0, 1}

        Returns
        -------
        self
        """
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        n_samples, n_features = X.shape
        self.w_ = np.random.rand(n_features + 1)
        self.errors_ = []

        for _ in range(self.epochs):
            total_loss = 0.0
            for xi, yi in zip(X, y):
                p = _sigmoid(np.dot(xi, self.w_[:-1]) + self.w_[-1])
                # Gradient of cross-entropy w.r.t. w and b: (p - y)
                error = p - yi
                self.w_[:-1] -= self.alpha * error * xi
                self.w_[-1] -= self.alpha * error
                # Binary cross-entropy loss for this sample
                total_loss += -yi * np.log(p + 1e-10) - (1 - yi) * np.log(1 - p + 1e-10)
            self.errors_.append(total_loss / n_samples)
        return self
