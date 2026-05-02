"""
decision_tree.py

Decision tree classifier and regressor using recursive binary splitting (CART).

Classifier uses Gini impurity to select splits.
Regressor uses MSE (variance) to select splits.

Each internal node stores a (feature index, threshold) pair.
Each leaf node stores a predicted value:
  - Classifier: majority class of samples reaching that leaf.
  - Regressor:  mean target value of samples reaching that leaf.

Classes
-------
DecisionTreeClassifier: Binary splitting on Gini impurity.
DecisionTreeRegressor: Binary splitting on MSE criterion.
"""

import numpy as np

class _Node(object):
    """
    A single node in the decision tree.

    Attributes
    ----------
    feature: int or None
        Index of the feature used to split (None for leaf nodes).
    threshold: float or None
        Split threshold: go left if x[feature] <= threshold.
    left: _Node or None
        Left subtree (samples where x[feature] <= threshold).
    right: _Node or None
        Right subtree (samples where x[feature] > threshold).
    value: scalar or None
        Prediction value; set only for leaf nodes.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def _gini(y):
    """
    Compute Gini impurity of label array y.

    Gini = 1 - sum(p_k^2)  where p_k = fraction of class k in y.

    Parameters
    ----------
    y: ndarray

    Returns
    -------
    float in [0, 0.5]
    """
    n = len(y)
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return 1.0 - float(np.sum(probs ** 2))


def _mse_impurity(y):
    """
    Compute MSE impurity (variance) of target array y.

    MSE impurity = Var(y)

    Parameters
    ----------
    y: ndarray

    Returns
    -------
    float
    """
    return float(np.var(y)) if len(y) > 0 else 0.0


def _best_split(X, y, criterion_fn, n_features):
    """
    Find the feature and threshold that maximize information gain.

    Searches over n_features randomly selected features and all unique
    threshold values for each feature.

    Parameters
    ----------
    X: ndarray of shape (n_samples, n_total_features)
    y: ndarray of shape (n_samples,)
    criterion_fn: callable
        Impurity function — _gini or _mse_impurity.
    n_features: int
        Number of features to consider at this split.

    Returns
    -------
    best_feature: int or None
    best_threshold: float or None
    """
    n_samples, n_total = X.shape
    parent_impurity = criterion_fn(y)
    best_gain = -np.inf
    best_feature, best_threshold = None, None

    # Randomly sample n_features features to consider (random forest support)
    feat_indices = np.random.choice(n_total, n_features, replace=False)
    for feature in feat_indices:
        for threshold in np.unique(X[:, feature]):
            left = X[:, feature] <= threshold
            right = ~left
            if left.sum() == 0 or right.sum() == 0:
                continue
            # Information gain = parent impurity - weighted child impurity
            gain = parent_impurity - (
                left.sum() * criterion_fn(y[left]) + right.sum() * criterion_fn(y[right])
            ) / n_samples
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold


class DecisionTreeClassifier(object):
    """
    Decision tree classifier using Gini impurity (CART algorithm).

    Parameters
    ----------
    max_depth: int or None, default=None
        Maximum depth of the tree. None means nodes expand until pure or min_samples_split is reached.
    min_samples_split: int, default=2
        Minimum samples required to split an internal node.
    n_features: int or None, default=None
        Number of features to consider per split. None means all features are used (set to sqrt(n_features) for random forests).

    Attributes
    ----------
    root_: _Node
        Root node of the fitted tree.
    """

    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root_ = None

    def fit(self, X, y):
        """
        Build the decision tree from training data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = np.array(X, dtype=float), np.array(y)
        n_features = self.n_features or X.shape[1]
        self.root_ = self._grow(X, y, depth=0, n_features=n_features)
        return self

    def _grow(self, X, y, depth, n_features):
        """
        Recursively grow the tree by finding the best split.

        Parameters
        ----------
        X: ndarray
        y: ndarray
        depth: int
            Current depth in the tree.
        n_features: int
            Features to consider at this node.

        Returns
        -------
        _Node
        """
        classes, counts = np.unique(y, return_counts=True)
        # Leaf prediction: majority class
        leaf_value = classes[np.argmax(counts)]

        # Stop conditions: max depth reached, too few samples, or pure node
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(y) < self.min_samples_split
            or len(classes) == 1
        ):
            return _Node(value=leaf_value)

        feature, threshold = _best_split(X, y, _gini, n_features)
        if feature is None:
            return _Node(value=leaf_value)

        left_mask = X[:, feature] <= threshold
        left = self._grow(X[left_mask], y[left_mask], depth + 1, n_features)
        right = self._grow(X[~left_mask], y[~left_mask], depth + 1, n_features)
        return _Node(feature=feature, threshold=threshold, left=left, right=right)

    def _traverse(self, x, node):
        """
        Traverse the tree for a single sample x.

        Parameters
        ----------
        x: ndarray of shape (n_features,)
        node: _Node

        Returns
        -------
        prediction: scalar
        """
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred: ndarray of shape (n_samples,)
        """
        X = np.array(X, dtype=float)
        return np.array([self._traverse(x, self.root_) for x in X])


class DecisionTreeRegressor(object):
    """
    Decision tree regressor using MSE criterion (CART algorithm).

    Parameters
    ----------
    max_depth: int or None, default=None
    min_samples_split: int, default=2
    n_features: int or None, default=None

    Attributes
    ----------
    root_: _Node
    """

    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root_ = None

    def fit(self, X, y):
        """
        Build the regression tree from training data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        n_features = self.n_features or X.shape[1]
        self.root_ = self._grow(X, y, depth=0, n_features=n_features)
        return self

    def _grow(self, X, y, depth, n_features):
        """Recursively grow the regression tree.

        Parameters
        ----------
        X: ndarray
        y: ndarray
        depth: int
        n_features: int

        Returns
        -------
        _Node
        """
        # Leaf prediction: mean of target values in this region
        leaf_value = float(np.mean(y))

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(y) < self.min_samples_split
        ):
            return _Node(value=leaf_value)

        feature, threshold = _best_split(X, y, _mse_impurity, n_features)
        if feature is None:
            return _Node(value=leaf_value)

        left_mask = X[:, feature] <= threshold
        if left_mask.sum() == 0 or (~left_mask).sum() == 0:
            return _Node(value=leaf_value)

        left = self._grow(X[left_mask], y[left_mask], depth + 1, n_features)
        right = self._grow(X[~left_mask], y[~left_mask], depth + 1, n_features)
        return _Node(feature=feature, threshold=threshold, left=left, right=right)

    def _traverse(self, x, node):
        """
        Traverse the tree for a single sample x.
        """
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        """
        Predict target values for X.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred: ndarray of shape (n_samples,)
        """
        X = np.array(X, dtype=float)
        return np.array([self._traverse(x, self.root_) for x in X])
