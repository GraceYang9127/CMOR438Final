"""
label_propagation.py

Label Propagation for semi-supervised classification.

Algorithm:
  1. Build affinity matrix W using the RBF kernel: W_ij = exp( -gamma * ||x_i - x_j||^2 )
  2. Row-normalize to get transition matrix: T = D^{-1} W,  where D = diag(row sums of W)
  3. Iterate:
       F = T @ F
       Clamp labeled rows back to their true one-hot labels after each step.
  4. Assign each unlabeled point to argmax class in F.

Labeled rows are "clamped": their label distribution never changes.
Unlabeled rows gradually absorb label information from their neighbors.

Classes
-------
LabelPropagation: Semi-supervised classifier via graph-based label spreading.
"""

import numpy as np

class LabelPropagation(object):
    """
    Semi-supervised label propagation on a fully-connected graph (RBF kernel).

    Parameters
    ----------
    gamma: float, default=20
        RBF kernel bandwidth: W_ij = exp(-gamma * ||x_i - x_j||^2).
        Larger gamma gives a tighter neighborhood (fewer connections).
    n_iterations: int, default=1000
        Number of propagation steps before convergence is assumed.

    Attributes
    ----------
    labels_: ndarray of shape (n_samples)
        Predicted label for every sample (labeled and unlabeled).
    classes_: ndarray
        Unique class labels seen in y (excluding -1).
    """

    def __init__(self, gamma=20, n_iterations=1000):
        self.gamma = gamma
        self.n_iterations = n_iterations
        self.labels_ = None
        self.classes_ = None

    def _build_affinity(self, X):
        """
        Build the RBF affinity matrix.

        W_ij = exp( -gamma * ||x_i - x_j||^2 )

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)

        Returns
        -------
        W: ndarray of shape (n_samples, n_samples)
        """
        dists = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
        return np.exp(-self.gamma * dists)

    def fit(self, X, y):
        """
        Propagate labels from labeled to unlabeled samples.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)
            Known labels for labeled samples; -1 for unlabeled samples.

        Returns
        -------
        self
        """
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.classes_ = np.unique(y[y != -1])
        n_classes = len(self.classes_)
        n_samples = len(X)
        labeled_mask = y != -1

        label_to_idx = {c: i for i, c in enumerate(self.classes_)}

        # Initialize label matrix F: one-hot for labeled, zeros for unlabeled
        F = np.zeros((n_samples, n_classes))
        for i, yi in enumerate(y):
            if yi != -1:
                F[i, label_to_idx[yi]] = 1.0

        F_clamped = F.copy()

        # Row-normalized affinity: T = D^{-1} W
        W = self._build_affinity(X)
        row_sums = W.sum(axis=1, keepdims=True)
        T = W / (row_sums + 1e-10)

        # Propagate labels, resetting labeled rows each iteration
        for _ in range(self.n_iterations):
            F = T @ F
            # Clamping: labeled points keep their true labels
            F[labeled_mask] = F_clamped[labeled_mask]

        # Assign each point to the class with the highest score
        self.labels_ = np.array([
            y[i] if labeled_mask[i] else self.classes_[np.argmax(F[i])]
            for i in range(n_samples)
        ])
        return self

    def predict(self, X=None):
        """
        Return the propagated labels for all samples.

        Parameters
        ----------
        X: ignored
            Present for API consistency. Label propagation does not generalize
            to unseen points; labels are computed during fit() for all samples.

        Returns
        -------
        labels_: ndarray of shape (n_samples,)
        """
        return self.labels_
