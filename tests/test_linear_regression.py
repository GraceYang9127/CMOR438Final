import numpy as np
import pytest

from rice_ml.supervised_learning.linear_regression import LinearRegression


@pytest.fixture
def linear_data():
    # y = 2x + 1 — exact linear relationship, no noise
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = 2 * X.ravel() + 1.0
    return X, y



# Normal equation
def test_normal_equation_fit_returns_self(linear_data):
    """
    Test that fit() returns the fitted model instance.

    Checks
    ------
    - fit() returns self.
    """
    X, y = linear_data
    model = LinearRegression(method="normal")
    assert model.fit(X, y) is model


def test_normal_equation_weights_shape(linear_data):
    """
    Test that w_ has shape (n_features + 1,) after fitting.

    Checks
    ------
    - w_ stores [w1, ..., wn, bias].
    """
    X, y = linear_data
    model = LinearRegression(method="normal").fit(X, y)
    assert model.w_.shape == (X.shape[1] + 1,)


def test_normal_equation_exact_fit(linear_data):
    """
    Test that the normal equation recovers exact coefficients for noiseless data.

    Checks
    ------
    - w_[-1] (bias) is approximately 1.0.
    - w_[0] (slope) is approximately 2.0.
    """
    X, y = linear_data
    model = LinearRegression(method="normal").fit(X, y)
    assert np.isclose(model.w_[-1], 1.0, atol=1e-6)
    assert np.isclose(model.w_[0], 2.0, atol=1e-6)


def test_normal_equation_predict_shape(linear_data):
    """
    Test predict returns a 1D array with one value per sample.

    Checks
    ------
    - predict(X).shape == (n_samples,).
    """
    X, y = linear_data
    model = LinearRegression(method="normal").fit(X, y)
    assert model.predict(X).shape == (len(X),)


def test_normal_equation_perfect_predictions(linear_data):
    """
    Test that predictions match targets exactly for noiseless linear data.

    Checks
    ------
    - predict(X) is approximately equal to y.
    """
    X, y = linear_data
    model = LinearRegression(method="normal").fit(X, y)
    assert np.allclose(model.predict(X), y, atol=1e-6)



# Gradient descent
def test_sgd_fit_populates_errors(linear_data):
    """
    Test that SGD training populates the errors_ list.

    Checks
    ------
    - errors_ is a non-empty list after training.
    """
    X, y = linear_data
    model = LinearRegression(method="sgd", alpha=0.01, epochs=500).fit(X, y)
    assert isinstance(model.errors_, list) and len(model.errors_) > 0


def test_sgd_error_decreases(linear_data):
    """
    Test that SGD training error decreases over epochs.

    Checks
    ------
    - Final error is less than initial error.
    """
    X, y = linear_data
    model = LinearRegression(method="sgd", alpha=0.01, epochs=500).fit(X, y)
    assert model.errors_[-1] < model.errors_[0]


def test_sgd_approximate_fit(linear_data):
    """
    Test that SGD reaches a reasonable prediction on simple linear data.

    Checks
    ------
    - Predictions are within 1.0 of true values after sufficient training.
    """
    X, y = linear_data
    model = LinearRegression(method="sgd", alpha=0.01, epochs=2000).fit(X, y)
    assert np.allclose(model.predict(X), y, atol=1.0)
