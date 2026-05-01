import numpy as np
import pytest

from rice_ml.supervised_learning.logistic_regression import LogisticRegression


@pytest.fixture
def binary_data():
    # Two well-separated clusters with labels 0 and 1
    rng = np.random.default_rng(1)
    X0 = rng.normal([-3, -3], 0.3, (30, 2))
    X1 = rng.normal([3, 3], 0.3, (30, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 30 + [1] * 30)
    return X, y

# Fit
def test_logistic_fit_returns_self(binary_data):
    """
    Test that fit() returns the fitted model instance.

    Checks
    ------
    - fit() returns self.
    """
    X, y = binary_data
    model = LogisticRegression(alpha=0.01, epochs=200)
    assert model.fit(X, y) is model

def test_logistic_weights_shape(binary_data):
    """
    Test that w_ has shape (n_features + 1,) after fitting.

    Checks
    ------
    - w_ stores [w1, ..., wn, bias].
    """
    X, y = binary_data
    model = LogisticRegression().fit(X, y)
    assert model.w_.shape == (X.shape[1] + 1,)

def test_logistic_errors_decrease(binary_data):
    """
    Test that cross-entropy loss decreases during training.

    Checks
    ------
    - Final loss is less than initial loss.
    """
    X, y = binary_data
    model = LogisticRegression(alpha=0.1, epochs=500).fit(X, y)
    assert model.errors_[-1] < model.errors_[0]

# predict_probability
def test_logistic_proba_range(binary_data):
    """
    Test that predict_proba returns values in (0, 1).

    Checks
    ------
    - All probabilities are >= 0 and <= 1.
    """
    X, y = binary_data
    model = LogisticRegression(alpha=0.1, epochs=500).fit(X, y)
    proba = model.predict_proba(X)
    assert np.all(proba >= 0) and np.all(proba <= 1)

# predict
def test_logistic_predict_only_binary_labels(binary_data):
    """
    Test that predict returns only 0 or 1.

    Checks
    ------
    - All predicted labels belong to {0, 1}.
    """
    X, y = binary_data
    model = LogisticRegression(alpha=0.1, epochs=500).fit(X, y)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})

def test_logistic_predict_shape(binary_data):
    """
    Test predict returns a 1D array of the correct length.

    Checks
    ------
    - predict(X).shape == (n_samples,).
    """
    X, y = binary_data
    model = LogisticRegression(alpha=0.1, epochs=500).fit(X, y)
    assert model.predict(X).shape == (len(X),)

def test_logistic_high_accuracy(binary_data):
    """
    Test that logistic regression achieves high accuracy on separable data.

    Checks
    ------
    - Accuracy >= 0.95 on training data after sufficient epochs.
    """
    X, y = binary_data
    model = LogisticRegression(alpha=0.1, epochs=1000).fit(X, y)
    accuracy = np.mean(model.predict(X) == y)
    assert accuracy >= 0.95
