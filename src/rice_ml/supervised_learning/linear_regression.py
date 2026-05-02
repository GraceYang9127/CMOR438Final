"""
linear_regression.py

Ordinary least-squares linear regression.

Two fitting methods:
  'normal': closed-form normal equation (exact solution)
  'sgd': stochastic gradient descent (one sample at a time)

Classes
-------
LinearRegression: Fits a linear model y_hat = w · x + b by minimizing MSE.
"""

import numpy as np

class LinearRegression(object):
    """
    Ordinary least-squares linear regression

    Fits a model of the form: y_hat = w · x + b   (linear activation - identity function)

    Cost function (MSE): C(w, b) = (1 / 2N) * sum( (y_hat_i - y_i)^2 )

    SGD update rule:
        w -= alpha * (y_hat - y) * x
        b -= alpha * (y_hat - y)

    Parameters
    ----------
    alpha: float, default=0.01
        Learning rate (used only when method='sgd').
    epochs: int, default=1000
        Number of SGD passes over the data (used only when method='sgd').
    method: {'normal', 'sgd'}, default='normal'
        'normal' solves the normal equation exactly using least squares.
        'sgd' uses stochastic gradient descent.

    Attributes
    ----------
    w_: ndarray of shape (n_features + 1,)
        Fitted weight vector. w_[:-1] are the feature weights;
        w_[-1] is the bias term.
    errors_: list of float
        MSE cost per epoch (populated only when method='sgd').
    """

    def __init__(self, alpha=0.01, epochs=1000, method="normal"):
        self.alpha = alpha
        self.epochs = epochs
        self.method = method

    def net_input(self, X):
        """
        Compute linear combination z = w · x + b.

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
        Predict target values for X.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred: ndarray of shape (n_samples)
        """
        X = np.array(X, dtype=float)
        return self.net_input(X)

    def fit(self, X, y):
        """
        Fit the linear regression model.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples)

        Returns
        -------
        self
        """
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        n_samples, n_features = X.shape

        if self.method == "normal":
            # Normal equation: w = (X^T X)^{-1} X^T y
            # least squares on the bias-augmented matrix [X | 1]
            X_aug = np.c_[X, np.ones(n_samples)]
            self.w_ = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        else:
            # Stochastic gradient descent 
            self.w_ = np.random.rand(n_features + 1)
            self.errors_ = []
            for _ in range(self.epochs):
                total_error = 0.0
                for xi, yi in zip(X, y):
                    error = self.predict(xi) - yi
                    self.w_[:-1] -= self.alpha * error * xi
                    self.w_[-1] -= self.alpha * error
                    total_error += 0.5 * error ** 2
                self.errors_.append(total_error / n_samples)
        return self
