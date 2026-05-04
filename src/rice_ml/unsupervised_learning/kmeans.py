"""
kmeans.py

K-Means clustering algorithm.

Algorithm:
  1. Initialize k random centroids (randomly chosen data points).
  2. Assign each point to its nearest centroid: label(x) = argmin_j || x - centroid_j ||
  3. Recompute each centroid as the mean of its assigned points: centroid_j = (1 / |C_j|) * sum_{x in C_j} x
  4. Repeat steps 2-3 until centroids shift less than tol or max_iters is reached.

Inertia (within-cluster sum of squared distances): sum_j sum_{x in C_j} || x - centroid_j ||^2

Classes
-------
KMeans: Lloyd's K-Means algorithm.
"""

import numpy as np


def _assign_clusters(X, centroids):
    """
    Assign each sample in X to its nearest centroid.

    Parameters
    ----------
    X: ndarray of shape (n_samples, n_features)
    centroids: ndarray of shape (k, n_features)

    Returns
    -------
    labels: ndarray of shape (n_samples,)
        Cluster index (0 to k-1) for each sample.
    """
    # Broadcasting: compute distance from every point to every centroid
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


class KMeans(object):
    """
    Lloyd's K-Means clustering algorithm.

    Parameters
    ----------
    k: int, default=3
        Number of clusters.
    max_iters: int, default=100
        Maximum number of assign-update iterations.
    tol: float, default=1e-4
        Convergence threshold on centroid shift (L2 norm).

    Attributes
    ----------
    centroids_: ndarray of shape (k, n_features)
        Final centroid positions after fitting.
    labels_: ndarray of shape (n_samples,)
        Cluster assignment for each training sample.
    inertia_: float
        Total within-cluster sum of squared distances.
    """

    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        """
        Fit K-Means to X using Lloyd's algorithm.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.array(X, dtype=float)
        # Initialize centroids by randomly selecting k data points
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids_ = X[idx].copy()

        for _ in range(self.max_iters):
            labels = _assign_clusters(X, self.centroids_)
            # Recompute each centroid. Keep old centroid if cluster is empty
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if (labels == j).any() else self.centroids_[j]
                for j in range(self.k)
            ])
            # Stop if centroids have converged
            if np.linalg.norm(new_centroids - self.centroids_) < self.tol:
                break
            self.centroids_ = new_centroids

        self.labels_ = _assign_clusters(X, self.centroids_)
        # Inertia = sum of squared distances to the nearest centroid
        self.inertia_ = float(sum(
            np.linalg.norm(X[self.labels_ == j] - self.centroids_[j]) ** 2
            for j in range(self.k) if (self.labels_ == j).any()
        ))
        return self

    def predict(self, X):
        """
        Assign new samples to the nearest fitted centroid.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        labels: ndarray of shape (n_samples,)

        Raises
        ------
        AttributeError: If called before fit().
        """
        if self.centroids_ is None:
            raise AttributeError("Model not fitted yet. Call fit() first.")
        X = np.array(X, dtype=float)
        return _assign_clusters(X, self.centroids_)

    def fit_predict(self, X):
        """Fit to X and return cluster labels.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        labels: ndarray of shape (n_samples,)
        """
        return self.fit(X).labels_
