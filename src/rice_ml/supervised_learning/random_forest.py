"""
random_forest.py

Random Forest classifier and regressor.

For each tree:
  1. Draw a bootstrap sample.
  2. At each split, consider only sqrt(n_features) randomly chosen features.
  3. Grow a full decision tree on the bootstrap sample.

Predictions:
  - Classifier: majority vote across all trees.
  - Regressor: average prediction across all trees.

Classes
-------
RandomForestClassifier
RandomForestRegressor
"""

import numpy as np

from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestClassifier(object):
    """
    Random Forest classifier.

    Parameters
    ----------
    n_estimators: int, default=100
        Number of trees in the forest.
    max_depth: int or None, default=None
        Maximum depth of each tree.
    min_samples_split: int, default=2
        Minimum samples to split a node.
    max_features: int or None, default=None
        Features to consider per split. Default is sqrt(n_features).

    Attributes
    ----------
    trees_: list of DecisionTreeClassifier
        Fitted trees after calling fit().
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees_ = []

    def fit(self, X, y):
        """
        Train n_estimators trees on bootstrap samples.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = np.array(X, dtype=float), np.array(y)
        n_samples, n_features = X.shape
        # Default: sqrt(n_features) features considered at each split
        max_features = self.max_features or int(np.sqrt(n_features))
        self.trees_ = []
        for _ in range(self.n_estimators):
            # Bootstrap sample: draw n_samples indices with replacement
            idx = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=max_features,
            )
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
        return self

    def predict(self, X):
        """
        Predict class labels by majority vote across all trees.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred: ndarray of shape (n_samples,)
        """
        X = np.array(X, dtype=float)
        # Collect predictions from all trees: shape (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.trees_])
        result = []
        for col in all_preds.T:
            values, counts = np.unique(col, return_counts=True)
            result.append(values[np.argmax(counts)])
        return np.array(result)


class RandomForestRegressor(object):
    """
    Random Forest regressor.

    Parameters
    ----------
    n_estimators: int, default=100
    max_depth: int or None, default=None
    min_samples_split: int, default=2
    max_features: int or None, default=None

    Attributes
    ----------
    trees_: list of DecisionTreeRegressor
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees_ = []

    def fit(self, X, y):
        """
        Train n_estimators regression trees on bootstrap samples.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        n_samples, n_features = X.shape
        max_features = self.max_features or int(np.sqrt(n_features))
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=max_features,
            )
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
        return self

    def predict(self, X):
        """
        Predict target values by averaging predictions across all trees.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred: ndarray of shape (n_samples,)
        """
        X = np.array(X, dtype=float)
        all_preds = np.array([tree.predict(X) for tree in self.trees_])
        return all_preds.mean(axis=0)
