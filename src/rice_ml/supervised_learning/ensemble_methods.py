"""
ensemble_methods.py

Ensemble methods: Bagging and Gradient Boosting.

Classes
-------
BaggingClassifier: Bootstrap aggregating with any base estimator.
GradientBoostingRegressor:Sequential residual-fitting regressor.
"""

import copy
import numpy as np

from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

class BaggingClassifier(object):
    """
    Bootstrap Aggregating (Bagging) classifier.

    Trains n_estimators independent copies of a base estimator on random
    bootstrap samples, then aggregates predictions by majority vote.

    Parameters
    ----------
    base_estimator: estimator or None, default=None
        Base classifier with fit/predict API.
        Defaults to DecisionTreeClassifier(max_depth=None).
    n_estimators: int, default=10
        Number of base estimators to train.
    max_samples: float, default=1.0
        Fraction of training samples to draw for each estimator.

    Attributes
    ----------
    estimators_: list
        Fitted estimator copies.
    """

    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.estimators_ = []

    def fit(self, X, y):
        """
        Train n_estimators copies on bootstrap samples.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = np.array(X, dtype=float), np.array(y)
        n = len(y)
        n_draw = int(self.max_samples * n)
        base = self.base_estimator or DecisionTreeClassifier(max_depth=None)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = np.random.choice(n, n_draw, replace=True)
            estimator = copy.deepcopy(base)
            estimator.fit(X[idx], y[idx])
            self.estimators_.append(estimator)
        return self

    def predict(self, X):
        """
        Predict class labels by majority vote across all estimators.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred: ndarray of shape (n_samples,)
        """
        X = np.array(X, dtype=float)
        all_preds = np.array([est.predict(X) for est in self.estimators_])
        result = []
        for col in all_preds.T:
            values, counts = np.unique(col, return_counts=True)
            result.append(values[np.argmax(counts)])
        return np.array(result)


class GradientBoostingRegressor(object):
    """
    Gradient Boosting regressor via sequential residual fitting.

    Each new tree is trained on the residuals (errors) of the current ensemble,
    progressively correcting mistakes made by earlier trees.

    Round 1: fit tree_1 on (X, y)
    Round 2: fit tree_2 on (X, y - tree_1.predict(X))
    Round t: fit tree_t on (X, residuals from round t-1)

    Final prediction: y_0 + learning_rate * sum(tree_t.predict(X)) where y_0 = mean(y) is the initial baseline prediction.

    Parameters
    ----------
    n_estimators: int, default=100
        Number of boosting stages (trees) to fit.
    learning_rate: float, default=0.1
        Shrinks the contribution of each tree. Lower values require more trees
        but often yield better generalization.
    max_depth: int, default=3
        Maximum depth of each regression tree (shallow trees work best).

    Attributes
    ----------
    estimators_: list of DecisionTreeRegressor
        Fitted trees, one per boosting round.
    y0_: float
        Baseline prediction (mean of training targets).
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.estimators_ = []
        self.y0_ = None

    def fit(self, X, y):
        """
        Fit the gradient boosting model by sequentially fitting residuals.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        # Baseline: predict the mean of y
        self.y0_ = float(y.mean())
        residuals = y - self.y0_
        self.estimators_ = []

        for _ in range(self.n_estimators):
            # Fit a shallow tree to the current residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.estimators_.append(tree)
            # Update residuals: subtract this tree's contribution
            residuals = residuals - self.learning_rate * tree.predict(X)

        return self

    def predict(self, X):
        """
        Predict target values for X.

        y_hat = y_0 + learning_rate * sum(tree_t.predict(X))

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred: ndarray of shape (n_samples,)
        """
        X = np.array(X, dtype=float)
        y_pred = np.full(len(X), self.y0_)
        for tree in self.estimators_:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred
