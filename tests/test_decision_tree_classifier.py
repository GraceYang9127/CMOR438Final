import numpy as np
import pytest

from rice_ml.supervised_learning.decision_tree import DecisionTreeClassifier


@pytest.fixture
def separable_data():
    # Two tight clusters: class 0 near origin, class 1 in upper quadrant
    rng = np.random.default_rng(6)
    X0 = rng.normal([0, 0], 0.3, (25, 2))
    X1 = rng.normal([4, 4], 0.3, (25, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 25 + [1] * 25)
    return X, y

# Fit
def test_dtc_fit_returns_self(separable_data):
    """
    Test that fit() returns the fitted classifier.

    Checks
    ------
    - fit() returns self.
    """
    X, y = separable_data
    clf = DecisionTreeClassifier(max_depth=3)
    assert clf.fit(X, y) is clf

def test_dtc_root_is_set(separable_data):
    """
    Test that root_ is populated after fitting.

    Checks
    ------
    - root_ is not None after fit.
    """
    X, y = separable_data
    clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert clf.root_ is not None


# Predict
def test_dtc_predict_shape(separable_data):
    """
    Test predict returns the correct shape.

    Checks
    ------
    - predict(X).shape == (n_samples,).
    """
    X, y = separable_data
    clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert clf.predict(X).shape == (len(X),)


def test_dtc_predict_valid_labels(separable_data):
    """
    Test that predict returns only labels seen in training.

    Checks
    ------
    - All predicted labels belong to {0, 1}.
    """
    X, y = separable_data
    clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
    preds = clf.predict(X)
    assert set(preds).issubset({0, 1})

def test_dtc_high_accuracy(separable_data):
    """
    Test that an unconstrained tree achieves high accuracy on separable data.

    Checks
    ------
    - Accuracy >= 0.95 on training data with max_depth=None.
    """
    X, y = separable_data
    clf = DecisionTreeClassifier(max_depth=None).fit(X, y)
    accuracy = np.mean(clf.predict(X) == y)
    assert accuracy >= 0.95


def test_dtc_max_depth_one_is_stump(separable_data):
    """
    Test that max_depth=1 still produces valid predictions.

    Checks
    ------
    - A decision stump predicts only valid labels.
    - Output shape is correct.
    """
    X, y = separable_data
    clf = DecisionTreeClassifier(max_depth=1).fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset({0, 1})
