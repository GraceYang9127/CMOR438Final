import numpy as np
import pytest

from rice_ml.supervised_learning.knn import KNN, KNNRegressor

@pytest.fixture
def classification_data():
    # Two tight clusters — class 0 near origin, class 1 far away
    rng = np.random.default_rng(4)
    X0 = rng.normal([0, 0], 0.2, (20, 2))
    X1 = rng.normal([5, 5], 0.2, (20, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 20 + [1] * 20)
    return X, y


@pytest.fixture
def regression_data():
    # y = 3x simple linear relationship
    rng = np.random.default_rng(5)
    X = rng.uniform(0, 10, (30, 1))
    y = 3 * X.ravel()
    return X, y



# KNN Classifier
def test_knn_fit_returns_self(classification_data):
    """
    Test that fit() stores training data and returns self.

    Checks
    ------
    - fit() returns the KNN instance.
    - X_train_ and y_train_ are set after fit.
    """
    X, y = classification_data
    clf = KNN(k=3)
    result = clf.fit(X, y)
    assert result is clf
    assert clf.X_train_ is not None
    assert clf.y_train_ is not None

def test_knn_predict_shape(classification_data):
    """
    Test predict returns an array of the correct length.

    Checks
    ------
    - predict(X).shape == (n_samples,).
    """
    X, y = classification_data
    clf = KNN(k=3).fit(X, y)
    assert clf.predict(X).shape == (len(X),)

def test_knn_predict_valid_labels(classification_data):
    """
    Test that predict returns only labels seen during training.

    Checks
    ------
    - All predicted labels belong to {0, 1}.
    """
    X, y = classification_data
    clf = KNN(k=3).fit(X, y)
    preds = clf.predict(X)
    assert set(preds).issubset({0, 1})

def test_knn_high_accuracy_separable(classification_data):
    """
    Test KNN achieves high accuracy on well-separated data.

    Checks
    ------
    - Accuracy >= 0.95 on training data.
    """
    X, y = classification_data
    clf = KNN(k=3).fit(X, y)
    assert clf.score(X, y) >= 0.95

def test_knn_k1_perfect_on_training(classification_data):
    """
    Test that k=1 gives perfect accuracy on the training set.

    Checks
    ------
    - With k=1, the nearest neighbor is the point itself, accuracy == 1.0.
    """
    X, y = classification_data
    clf = KNN(k=1).fit(X, y)
    assert clf.score(X, y) == 1.0



# KNN Regressor
def test_knn_regressor_predict_shape(regression_data):
    """
    Test KNNRegressor predict returns the correct shape.

    Checks
    ------
    - predict(X).shape == (n_samples,).
    """
    X, y = regression_data
    reg = KNNRegressor(k=3).fit(X, y)
    assert reg.predict(X).shape == (len(X),)

def test_knn_regressor_k1_exact_on_training(regression_data):
    """
    Test that k=1 reproduces training targets exactly.

    Checks
    ------
    - With k=1, predict returns the exact training target for each point.
    """
    X, y = regression_data
    reg = KNNRegressor(k=1).fit(X, y)
    assert np.allclose(reg.predict(X), y, atol=1e-10)
