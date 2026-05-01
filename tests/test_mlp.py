import numpy as np
import pytest

from rice_ml.supervised_learning.multilayer_perceptron import DenseNetwork

@pytest.fixture
def binary_data():
    # Two well-separated clusters — binary classification
    rng = np.random.default_rng(2)
    X0 = rng.normal([-2, -2], 0.4, (20, 2))
    X1 = rng.normal([2, 2], 0.4, (20, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 20 + [1] * 20)
    return X, y

# Initialization
def test_dense_network_weights_initialized():
    """
    Test that W and B are initialized with the correct number of layers.

    Checks
    ------
    - len(W) == len(layers) (index 0 is placeholder).
    - len(B) == len(layers).
    """
    net = DenseNetwork(layers=[2, 4, 1])
    assert len(net.W) == 3
    assert len(net.B) == 3

def test_dense_network_weight_shapes():
    """
    Test that each weight matrix has the correct shape.

    Checks
    ------
    - W[1] shape is (hidden_size, input_size).
    - W[2] shape is (output_size, hidden_size).
    """
    net = DenseNetwork(layers=[2, 4, 1])
    assert net.W[1].shape == (4, 2)
    assert net.W[2].shape == (1, 4)

# Fit
def test_dense_network_fit_returns_self(binary_data):
    """
    Test that fit() returns the network instance.

    Checks
    ------
    - fit() returns self.
    """
    X, y = binary_data
    net = DenseNetwork(layers=[2, 4, 1])
    assert net.fit(X, y, alpha=0.01, epochs=10) is net

def test_dense_network_errors_tracked(binary_data):
    """
    Test that errors_ is populated during training.

    Checks
    ------
    - errors_ is a list with one entry per epoch.
    """
    X, y = binary_data
    net = DenseNetwork(layers=[2, 4, 1])
    net.fit(X, y, alpha=0.01, epochs=20)
    assert isinstance(net.errors_, list) and len(net.errors_) == 20

def test_dense_network_error_decreases(binary_data):
    """
    Test that training loss decreases over epochs.

    Checks
    ------
    - Final error is less than initial error.
    """
    X, y = binary_data
    net = DenseNetwork(layers=[2, 8, 1])
    net.fit(X, y, alpha=0.05, epochs=200)
    assert net.errors_[-1] < net.errors_[0]

# Predict
def test_dense_network_predict_shape(binary_data):
    """
    Test predict returns an array of the correct length.

    Checks
    ------
    - predict(X).shape == (n_samples,).
    """
    X, y = binary_data
    net = DenseNetwork(layers=[2, 4, 1])
    net.fit(X, y, alpha=0.01, epochs=10)
    assert net.predict(X).shape == (len(X),)

def test_dense_network_predict_binary_labels(binary_data):
    """
    Test predict returns only 0 or 1 for binary output layer.

    Checks
    ------
    - All predicted labels belong to {0, 1}.
    """
    X, y = binary_data
    net = DenseNetwork(layers=[2, 4, 1])
    net.fit(X, y, alpha=0.01, epochs=10)
    preds = net.predict(X)
    assert set(preds).issubset({0, 1})


def test_dense_network_multiclass_predict(binary_data):
    """
    Test predict works correctly for a 3-class output layer.

    Checks
    ------
    - predict returns integer class indices.
    """
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (30, 2))
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    net = DenseNetwork(layers=[2, 8, 3])
    net.fit(X, y, alpha=0.01, epochs=10)
    preds = net.predict(X)
    assert set(preds).issubset({0, 1, 2})
