"""
dbscan.py

DBSCAN: Density-Based Spatial Clustering of Applications with Noise.

Point types:
  Core point — has >= min_samples neighbors within radius eps.
  Border point — within eps of a core point, but not a core point itself.
  Noise point — not reachable from any core point; labeled -1.

Classes
-------
DBSCAN: Density-based clustering with automatic noise detection.
"""

import numpy as np
from collections import deque

class DBSCAN(object):
    """
    Density-Based Spatial Clustering of Applications with Noise.

    Parameters
    ----------
    eps: float, default=0.5
        Maximum distance between two samples for one to be considered in the neighborhood of the other.
    min_samples: int, default=5
        Minimum number of samples within eps to classify a point as a core point.

    Attributes
    ----------
    labels_: ndarray of shape (n_samples,)
        Cluster label for each sample. Noise points are labeled -1.
        Clusters are labeled 0, 1, 2, ...
    """
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def _region_query(self, X, idx):
        """
        Return indices of all points within eps of X[idx].

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
        idx: int
            Index of the query point.

        Returns
        -------
        neighbors: ndarray of int
        """
        distances = np.linalg.norm(X - X[idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def fit(self, X):
        """
        Run DBSCAN on X.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.array(X, dtype=float)
        n = len(X)
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._region_query(X, i)

            if len(neighbors) < self.min_samples:
                # Point has too few neighbors — mark as noise
                labels[i] = -1
            else:
                # Start a new cluster and expand via BFS
                labels[i] = cluster_id
                queue = deque(neighbors.tolist())
                while queue:
                    j = queue.popleft()
                    if not visited[j]:
                        visited[j] = True
                        j_neighbors = self._region_query(X, j)
                        # If j is a core point, add its neighbors to the queue
                        if len(j_neighbors) >= self.min_samples:
                            queue.extend(j_neighbors.tolist())
                    # Absorb border points into the current cluster
                    if labels[j] == -1:
                        labels[j] = cluster_id
                cluster_id += 1

        self.labels_ = labels
        return self

    def fit_predict(self, X):
        """
        Fit to X and return cluster labels.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        labels: ndarray of shape (n_samples,)
            -1 for noise, >= 0 for cluster membership.
        """
        return self.fit(X).labels_
