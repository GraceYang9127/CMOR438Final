import numpy as np
import pytest

from rice_ml.supervised_learning.decision_tree import DecisionTreeRegressor


@pytest.fixture
def regression_data():
    # y = 2x + 3 — simple linear signal
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 10, (40, 1))
    y = 2 * X.ravel() + 3
    return X, y

# Fit
def test_dtr_fit_returns_self(regression_data):
    """
    Test that fit() returns the fitted regressor.

    Checks
    ------
    - fit() returns self.
    """
    X, y = regression_data
    reg = DecisionTreeRegressor(max_depth=3)
    assert reg.fit(X, y) is reg

def test_dtr_root_is_set(regression_data):
    """
    Test that root_ is populated after fitting.

    Checks
    ------
    - root_ is not None after fit.
    """
    X, y = regression_data
    reg = DecisionTreeRegressor(max_depth=3).fit(X, y)
    assert reg.root_ is not None


# Predict
def test_dtr_predict_shape(regression_data):
    """
    Test predict returns a 1D array of the correct length.

    Checks
    ------
    - predict(X).shape == (n_samples,).
    """
    X, y = regression_data
    reg = DecisionTreeRegressor(max_depth=3).fit(X, y)
    assert reg.predict(X).shape == (len(X),)

def test_dtr_deep_tree_low_training_error(regression_data):
    """
    Test that an unconstrained tree fits training data closely.

    Checks
    ------
    - Mean absolute error on training data is small with max_depth=None.
    """
    X, y = regression_data
    reg = DecisionTreeRegressor(max_depth=None).fit(X, y)
    mae = np.mean(np.abs(reg.predict(X) - y))
    assert mae < 1.0


def test_dtr_stump_has_higher_error_than_deep_tree(regression_data):
    """
    Test that a shallow tree has higher training error than a deep tree.

    Checks
    ------
    - MSE of depth-1 stump >= MSE of unconstrained tree.
    """
    X, y = regression_data
    stump = DecisionTreeRegressor(max_depth=1).fit(X, y)
    deep = DecisionTreeRegressor(max_depth=None).fit(X, y)
    mse_stump = np.mean((stump.predict(X) - y) ** 2)
    mse_deep = np.mean((deep.predict(X) - y) ** 2)
    assert mse_stump >= mse_deep


def test_dtr_predict_in_target_range(regression_data):
    """
    Test that predictions stay within the range of training targets.

    Checks
    ------
    - All predictions are between y.min() and y.max().
    """
    X, y = regression_data
    reg = DecisionTreeRegressor(max_depth=4).fit(X, y)
    preds = reg.predict(X)
    assert np.all(preds >= y.min()) and np.all(preds <= y.max())
