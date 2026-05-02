"""
knn.py

K-Nearest Neighbors classifier and regressor.

For each query point x:
  1. Compute Euclidean distance to every training point.
  2. Select the k nearest neighbors.
  3. Classify by majority vote (KNN) or average target (KNNRegressor).

Classes
-------
KNN: K-Nearest Neighbors classifier.
KNNRegressor: K-Nearest Neighbors regressor.
"""

import numpy as np

class KNN(object):
    """
    K-Nearest Neighbors classifier.

    A non-parametric, instance-based classifier with no explicit training phase.
    At prediction time, distances to all stored training points are computed.

    Parameters
    ----------
    k: int, default=3
        Number of nearest neighbors to use for majority vote.

    Attributes
    ----------
    X_train_: ndarray of shape (n_samples, n_features)
        Training feature matrix stored during fit.
    y_train_: ndarray of shape (n_samples,)
        Training labels stored during fit.
    """
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """
        Store the training data (no computation performed).

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        self.X_train_ = np.array(X, dtype=float)
        self.y_train_ = np.array(y)
        return self

    def predict(self, X):
        """
        Predict class labels for X by majority vote of k nearest neighbors.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred: ndarray of shape (n_samples,)
        """
        X = np.array(X, dtype=float)
        predictions = []
        for x in X:
            # Euclidean distances from x to every training point
            distances = np.linalg.norm(self.X_train_ - x, axis=1)
            k_indices = np.argsort(distances)[: self.k]
            k_labels = self.y_train_[k_indices]
            # Return the most common label among the k neighbors
            values, counts = np.unique(k_labels, return_counts=True)
            predictions.append(values[np.argmax(counts)])
        return np.array(predictions)

    def score(self, X, y):
        """
        Return classification accuracy on (X, y).

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)

        Returns
        -------
        float in [0, 1]
        """
        return float(np.mean(self.predict(X) == np.array(y)))

class KNNRegressor(object):
    """
    K-Nearest Neighbors regressor.

    Predicts the target of a query point as the mean of its k nearest
    neighbors' targets.

    Parameters
    ----------
    k: int, default=3
        Number of nearest neighbors to average.

    Attributes
    ----------
    X_train_: ndarray of shape (n_samples, n_features)
    y_train_: ndarray of shape (n_samples,)
    """
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """
        Store the training data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        self.X_train_ = np.array(X, dtype=float)
        self.y_train_ = np.array(y, dtype=float)
        return self

    def predict(self, X):
        """
        Predict target values by averaging k nearest neighbors.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred: ndarray of shape (n_samples,)
        """
        X = np.array(X, dtype=float)
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train_ - x, axis=1)
            k_indices = np.argsort(distances)[: self.k]
            predictions.append(self.y_train_[k_indices].mean())
        return np.array(predictions)
