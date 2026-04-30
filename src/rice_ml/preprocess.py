"""
preprocess.py

Preprocessing utilities for the rice_ml package.

Classes:
- StandardScaler: Standardize features to zero mean and unit variance: z = (x - mu) / sigma
- MinMaxScaler: Scale features to a given range [lo, hi]: x_scaled = (x - x_min) / (x_max - x_min)
- OrdinalEncoder: Encode categorical columns as integer codes (alphabetical order).

Functions:
- train_test_split(*arrays, test_size, random_state): Split arrays or DataFrames into random train and test subsets.

"""

import numpy as np


class StandardScaler(object):

    def __init__(self):
        # Learned mean and standard deviation, set during fit
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """Compute column-wise mean and standard deviation from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.array(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # Avoid division by zero for constant features
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        """Standardize X using fitted mean and std: z = (x - mu) / sigma.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
        """
        X = np.array(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """Fit to X then return the standardized array.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Reverse standardization: x = z * sigma + mu.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
        """
        X = np.array(X, dtype=float)
        return X * self.std_ + self.mean_


class MinMaxScaler(object):

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        # Learned min and max, set during fit
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        """Compute column-wise min and max from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.array(X, dtype=float)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X):
        """Scale X to feature_range: x_scaled = (x - x_min) / (x_max - x_min) * (hi - lo) + lo.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
        """
        X = np.array(X, dtype=float)
        scale = self.max_ - self.min_
        scale[scale == 0] = 1.0
        lo, hi = self.feature_range
        return lo + (X - self.min_) / scale * (hi - lo)

    def fit_transform(self, X):
        """Fit to X then return the scaled array.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Reverse scaling: x = (x_scaled - lo) / (hi - lo) * (x_max - x_min) + x_min.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
        """
        X = np.array(X, dtype=float)
        scale = self.max_ - self.min_
        scale[scale == 0] = 1.0
        lo, hi = self.feature_range
        return (X - lo) / (hi - lo) * scale + self.min_


class OrdinalEncoder(object):

    def __init__(self):
        # One array of sorted unique categories per column
        self.categories_ = None

    def fit(self, X):
        """Learn the sorted unique categories for each column.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        """Encode categories as integer codes in alphabetical order.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)

        Returns
        -------
        X_encoded : ndarray, same shape as X, dtype float
        """
        X = np.array(X)
        scalar = X.ndim == 1
        if scalar:
            X = X.reshape(-1, 1)
        out = np.zeros_like(X, dtype=float)
        for i, cats in enumerate(self.categories_):
            lookup = {v: j for j, v in enumerate(cats)}
            out[:, i] = np.array([lookup[v] for v in X[:, i]], dtype=float)
        return out.ravel() if scalar else out

    def fit_transform(self, X):
        """Fit to X then return the encoded array.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)

        Returns
        -------
        X_encoded : ndarray, same shape as X, dtype float
        """
        return self.fit(X).transform(X)


def train_test_split(*arrays, test_size=0.2, random_state=None):
    """Split arrays or DataFrames into random train and test subsets.

    Parameters
    ----------
    *arrays : array-likes with the same first-dimension length
        Any number of arrays, DataFrames, or Series to split together.
    test_size : float, default=0.2
        Fraction of samples to use for the test set.
    random_state : int or None, default=None
        Seed for reproducibility.

    Returns
    -------
    list
        Alternating train/test pairs for each input array:
        [X_train, X_test, y_train, y_test, ...]
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(arrays[0])
    indices = np.random.permutation(n)
    n_test = max(1, int(np.floor(test_size * n)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    result = []
    for arr in arrays:
        try:
            import pandas as pd
            if isinstance(arr, (pd.DataFrame, pd.Series)):
                result.append(arr.iloc[train_idx])
                result.append(arr.iloc[test_idx])
                continue
        except ImportError:
            pass
        arr = np.array(arr)
        result.append(arr[train_idx])
        result.append(arr[test_idx])
    return result
