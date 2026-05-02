"""
pca.py

Principal Component Analysis (PCA).

Steps:
  1. Center the data: X_c = X - mean(X)
  2. Covariance matrix: S = (1/(n-1)) * X_c^T * X_c
  3. Eigendecomposition: S * v_i = lambda_i * v_i
  4. Sort descending by eigenvalue.
  5. Project: Z = X_c @ V_k  (top-k eigenvectors as columns)

Explained variance ratio for component i:
    r_i = lambda_i / sum(lambda)

Classes
-------
PCA: Dimensionality reduction via eigendecomposition of the covariance matrix.
"""

import numpy as np

class PCA(object):
    """
    Principal Component Analysis.

    Parameters
    ----------
    n_components: int or None, default=None
        Number of principal components to keep.
        None keeps all components.

    Attributes
    ----------
    components_: ndarray of shape (n_components, n_features)
        Top-k eigenvectors (rows = principal components).
    explained_variance_: ndarray of shape (n_components,)
        Eigenvalues corresponding to each component.
    explained_variance_ratio_: ndarray of shape (n_components,)
        Fraction of total variance explained by each component.
    mean_: ndarray of shape (n_features,)
        Per-feature mean of the training data.
    """
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        """
        Compute principal components from training data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.array(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Covariance matrix: S = (1/(n-1)) * X_c^T * X_c
        cov = np.cov(X_centered, rowvar=False)

        #eigh guarantees real eigenvalues and sorts ascending
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        #Sort descending so the first component has maximum variance
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        k = self.n_components or X.shape[1]
        # Rows of components_ are the principal component directions
        self.components_ = eigenvectors[:, :k].T
        self.explained_variance_ = eigenvalues[:k]
        # Explained variance ratio: lambda_i / sum(all lambda)
        self.explained_variance_ratio_ = eigenvalues[:k] / eigenvalues.sum()
        return self

    def transform(self, X):
        """
        Project X onto the principal components.

        Z = (X - mean) @ V_k^T

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        Z: ndarray of shape (n_samples, n_components)
        """
        X = np.array(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        """
        Fit to X then project X onto the principal components.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        Z: ndarray of shape (n_samples, n_components)
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Reconstruct data from the reduced representation.

        X_approx = Z @ V_k + mean

        Parameters
        ----------
        X: array-like of shape (n_samples, n_components)

        Returns
        -------
        X_approx: ndarray of shape (n_samples, n_features)
        """
        X = np.array(X, dtype=float)
        return X @ self.components_ + self.mean_
