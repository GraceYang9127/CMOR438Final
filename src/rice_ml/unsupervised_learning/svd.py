"""
svd.py

Truncated Singular Value Decomposition (SVD).

Full SVD factorization: A = U * Sigma * V^T

Where:
    U: left singular vectors  (n_samples x n_samples)
    Sigma:diagonal of singular values sigma_1 >= sigma_2 >= ... >= 0
    V^T: right singular vectors (n_features x n_features)

TruncatedSVD keeps only the top n_components singular values/vectors,
giving the best rank-k approximation of A (Eckart-Young theorem):
    A_k = U_k * Sigma_k * V_k^T

Dimensionality reduction: X_reduced = X @ V_k

Explained variance ratio: r_i = sigma_i^2 / sum(sigma^2)


Classes
-------
TruncatedSVD: Low-rank SVD-based dimensionality reduction.
"""

import numpy as np

class TruncatedSVD(object):
    """
    Dimensionality reduction via Truncated SVD.

    Parameters
    ----------
    n_components: int, default=2
        Number of singular value components to keep.

    Attributes
    ----------
    components_: ndarray of shape (n_components, n_features)
        Top-k right singular vectors (rows of V^T).
    singular_values_: ndarray of shape (n_components,)
        Top-k singular values in descending order.
    explained_variance_ratio_: ndarray of shape (n_components,)
        Fraction of total variance explained by each component:
        sigma_i^2 / sum(sigma^2).
    """

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        """
        Compute the top n_components singular vectors of X.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.array(X, dtype=float)
        # Full SVD: A = U * Sigma * V^T
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        # Keep only the top n_components right singular vectors
        self.components_ = Vt[: self.n_components]
        self.singular_values_ = s[: self.n_components]
        # Explained variance ratio: sigma_i^2 / sum(sigma^2)
        self.explained_variance_ratio_ = (s[: self.n_components] ** 2) / (s ** 2).sum()
        return self

    def transform(self, X):
        """
        Project X onto the top singular vectors.

        X_reduced = X @ V_k^T

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        X_reduced: ndarray of shape (n_samples, n_components)
        """
        X = np.array(X, dtype=float)
        return X @ self.components_.T

    def fit_transform(self, X):
        """Fit to X and return the reduced representation.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        X_reduced: ndarray of shape (n_samples, n_components)
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Reconstruct the approximate original data from the reduced representation.

        X_approx = X_reduced @ V_k

        Parameters
        ----------
        X: array-like of shape (n_samples, n_components)

        Returns
        -------
        X_approx: ndarray of shape (n_samples, n_features)
        """
        X = np.array(X, dtype=float)
        return X @ self.components_
