import numpy as np
import pytest

from rice_ml.supervised_learning.random_forest import RandomForestClassifier, RandomForestRegressor
from rice_ml.supervised_learning.ensemble_methods import BaggingClassifier, GradientBoostingRegressor

@pytest.fixture
def classification_data():
    # Well-separated binary classes
    rng = np.random.default_rng(8)
    X0 = rng.normal([-3, -3], 0.5, (30, 2))
    X1 = rng.normal([3, 3], 0.5, (30, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 30 + [1] * 30)
    return X, y

@pytest.fixture
def regression_data():
    # y = x — identity relationship
    rng = np.random.default_rng(9)
    X = rng.uniform(0, 10, (40, 1))
    y = X.ravel()
    return X, y


# RandomForestClassifier
def test_rfc_fit_returns_self(classification_data):
    """
    Test that fit() returns the fitted forest.

    Checks
    ------
    - fit() returns self.
    """
    X, y = classification_data
    clf = RandomForestClassifier(n_estimators=10, max_depth=3)
    assert clf.fit(X, y) is clf

def test_rfc_trees_populated(classification_data):
    """
    Test that trees_ contains n_estimators trees after fitting.

    Checks
    ------
    - len(trees_) == n_estimators.
    """
    X, y = classification_data
    clf = RandomForestClassifier(n_estimators=10, max_depth=3).fit(X, y)
    assert len(clf.trees_) == 10

def test_rfc_predict_shape(classification_data):
    """
    Test predict returns an array of the correct length.

    Checks
    ------
    - predict(X).shape == (n_samples,).
    """
    X, y = classification_data
    clf = RandomForestClassifier(n_estimators=10, max_depth=3).fit(X, y)
    assert clf.predict(X).shape == (len(X),)

def test_rfc_high_accuracy(classification_data):
    """
    Test RandomForest achieves high accuracy on separable data.

    Checks
    ------
    - Accuracy >= 0.90 on training data.
    """
    X, y = classification_data
    clf = RandomForestClassifier(n_estimators=20, max_depth=5).fit(X, y)
    accuracy = np.mean(clf.predict(X) == y)
    assert accuracy >= 0.90


# RandomForestRegressor
def test_rfr_predict_shape(regression_data):
    """
    Test RandomForestRegressor predict returns the correct shape.

    Checks
    ------
    - predict(X).shape == (n_samples,).
    """
    X, y = regression_data
    reg = RandomForestRegressor(n_estimators=10, max_depth=3).fit(X, y)
    assert reg.predict(X).shape == (len(X),)

def test_rfr_low_training_error(regression_data):
    """
    Test that the forest fits training data reasonably well.

    Checks
    ------
    - Mean absolute error on training data is less than 2.0.
    """
    X, y = regression_data
    reg = RandomForestRegressor(n_estimators=20, max_depth=None).fit(X, y)
    mae = np.mean(np.abs(reg.predict(X) - y))
    assert mae < 2.0

# BaggingClassifier
def test_bagging_fit_returns_self(classification_data):
    """
    Test that fit() returns the fitted Bagging model.

    Checks
    ------
    - fit() returns self.
    """
    X, y = classification_data
    clf = BaggingClassifier(n_estimators=10)
    assert clf.fit(X, y) is clf


def test_bagging_predict_shape(classification_data):
    """
    Test BaggingClassifier predict returns the correct shape.

    Checks
    ------
    - predict(X).shape == (n_samples,).
    """
    X, y = classification_data
    clf = BaggingClassifier(n_estimators=10).fit(X, y)
    assert clf.predict(X).shape == (len(X),)


# GradientBoostingRegressor
def test_gbr_fit_returns_self(regression_data):
    """
    Test that fit() returns the fitted GradientBoostingRegressor.

    Checks
    ------
    - fit() returns self.
    """
    X, y = regression_data
    reg = GradientBoostingRegressor(n_estimators=20, learning_rate=0.1, max_depth=2)
    assert reg.fit(X, y) is reg

def test_gbr_estimators_populated(regression_data):
    """
    Test that estimators_ is populated after fitting.

    Checks
    ------
    - len(estimators_) == n_estimators.
    """
    X, y = regression_data
    reg = GradientBoostingRegressor(n_estimators=20).fit(X, y)
    assert len(reg.estimators_) == 20

def test_gbr_predict_shape(regression_data):
    """
    Test GradientBoostingRegressor predict returns the correct shape.

    Checks
    ------
    - predict(X).shape == (n_samples,).
    """
    X, y = regression_data
    reg = GradientBoostingRegressor(n_estimators=20).fit(X, y)
    assert reg.predict(X).shape == (len(X),)

def test_gbr_low_training_error(regression_data):
    """
    Test GradientBoostingRegressor fits training data closely.

    Checks
    ------
    - Mean absolute error on training data is less than 1.0.
    """
    X, y = regression_data
    reg = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3).fit(X, y)
    mae = np.mean(np.abs(reg.predict(X) - y))
    assert mae < 1.0
