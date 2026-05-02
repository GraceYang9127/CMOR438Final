import numpy as np
import pytest

from rice_ml.supervised_learning.perceptron import Perceptron

@pytest.fixture
def separable_data():
    # Two tight clusters well separated — perceptron always converges
    rng = np.random.default_rng(0)
    X_neg = rng.normal([-3, -3], 0.3, (20, 2))
    X_pos = rng.normal([3, 3], 0.3, (20, 2))
    X = np.vstack([X_neg, X_pos])
    y = np.array([-1] * 20 + [1] * 20)
    return X, y

# Initialization and fit
def test_perceptron_fit_returns_self(separable_data):
    """
    Test that fit() returns the fitted Perceptron instance.

    Checks
    ------
    - fit() returns self, enabling method chaining.
    """
    X, y = separable_data
    clf = Perceptron(eta=0.5, epochs=200)
    assert clf.fit(X, y) is clf


def test_perceptron_weights_shape(separable_data):
    """
    Test that w_ has length n_features + 1 after fitting.

    Checks
    ------
    - w_ stores [w1, ..., wn, bias], so shape is (n_features + 1,).
    """
    X, y = separable_data
    clf = Perceptron(eta=0.5, epochs=200).fit(X, y)
    assert clf.w_.shape == (X.shape[1] + 1,)


def test_perceptron_errors_tracked(separable_data):
    """
    Test that errors_ is a list populated during training.

    Checks
    ------
    - errors_ is a list after fit.
    """
    X, y = separable_data
    clf = Perceptron(eta=0.5, epochs=200).fit(X, y)
    assert isinstance(clf.errors_, list)



# Predict
def test_perceptron_predict_only_valid_labels(separable_data):
    """
    Test that predict returns only -1 or +1.

    Checks
    ------
    - All predicted values belong to {-1, 1}.
    """
    X, y = separable_data
    clf = Perceptron(eta=0.5, epochs=200).fit(X, y)
    preds = clf.predict(X)
    assert set(preds).issubset({-1, 1})


def test_perceptron_predict_shape(separable_data):
    """
    Test that predict returns an array of the correct length.

    Checks
    ------
    - Output shape matches number of input samples.
    """
    X, y = separable_data
    clf = Perceptron(eta=0.5, epochs=200).fit(X, y)
    assert clf.predict(X).shape == (len(X),)


def test_perceptron_converges_on_separable_data(separable_data):
    """
    Test that the perceptron achieves high accuracy on linearly separable data.

    Checks
    ------
    - Accuracy >= 0.95 on training data after sufficient epochs.
    """
    X, y = separable_data
    clf = Perceptron(eta=0.5, epochs=500).fit(X, y)
    accuracy = np.mean(clf.predict(X) == y)
    assert accuracy >= 0.95

# net_input
def test_perceptron_net_input_shape(separable_data):
    """
    Test that net_input returns one value per sample.

    Checks
    ------
    - net_input(X) shape equals (n_samples,).
    """
    X, y = separable_data
    clf = Perceptron(eta=0.5, epochs=200).fit(X, y)
    assert clf.net_input(X).shape == (len(X),)
