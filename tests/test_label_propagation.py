import numpy as np
import pytest

from rice_ml.unsupervised_learning.label_propagation import LabelPropagation

@pytest.fixture
def semi_labeled_data():
    # Two tight clusters. Only first and last point are labeled
    rng = np.random.default_rng(4)
    X0 = rng.normal([0, 0], 0.3, (10, 2))
    X1 = rng.normal([5, 5], 0.3, (10, 2))
    X = np.vstack([X0, X1])
    y = np.full(20, -1)
    # Label one point from each cluster
    y[0] = 0
    y[19] = 1
    return X, y

# Fit
def test_lp_fit_returns_self(semi_labeled_data):
    """
    Test that fit() returns the fitted LabelPropagation instance.

    Checks
    ------
    - fit() returns self.
    """
    X, y = semi_labeled_data
    lp = LabelPropagation(gamma=1.0, n_iterations=10)
    assert lp.fit(X, y) is lp

def test_lp_labels_shape(semi_labeled_data):
    """
    Test that labels_ has one entry per sample.

    Checks
    ------
    - labels_.shape == (n_samples,).
    """
    X, y = semi_labeled_data
    lp = LabelPropagation(gamma=1.0, n_iterations=50).fit(X, y)
    assert lp.labels_.shape == (len(X),)


def test_lp_labeled_points_preserved(semi_labeled_data):
    """
    Test that labeled points retain their original labels after propagation.

    Checks
    ------
    - Labeled inputs (y != -1) have the same label in labels_ as in y.
    """
    X, y = semi_labeled_data
    lp = LabelPropagation(gamma=1.0, n_iterations=50).fit(X, y)
    labeled_mask = y != -1
    assert np.all(lp.labels_[labeled_mask] == y[labeled_mask])


def test_lp_no_unlabeled_remain(semi_labeled_data):
    """
    Test that no points remain unlabeled (-1) after propagation.

    Checks
    ------
    - None of the output labels are -1.
    """
    X, y = semi_labeled_data
    lp = LabelPropagation(gamma=1.0, n_iterations=100).fit(X, y)
    assert np.all(lp.labels_ != -1)


def test_lp_labels_valid_classes(semi_labeled_data):
    """
    Test that all output labels belong to the known classes {0, 1}.

    Checks
    ------
    - All labels are in {0, 1}.
    """
    X, y = semi_labeled_data
    lp = LabelPropagation(gamma=1.0, n_iterations=100).fit(X, y)
    assert set(lp.labels_).issubset({0, 1})


def test_lp_propagates_correctly(semi_labeled_data):
    """
    Test that label propagation correctly assigns well-separated clusters.

    Checks
    ------
    - Cluster 0 (first 10 points) predicted mostly as label 0.
    - Cluster 1 (last 10 points) predicted mostly as label 1.
    """
    X, y = semi_labeled_data
    lp = LabelPropagation(gamma=5.0, n_iterations=200).fit(X, y)
    # First 10 points should be labeled 0
    assert np.mean(lp.labels_[:10] == 0) >= 0.8
    # Last 10 points should be labeled 1
    assert np.mean(lp.labels_[10:] == 1) >= 0.8

# Predict
def test_lp_predict_returns_labels(semi_labeled_data):
    """
    Test that predict() returns the same array as labels_.

    Checks
    ------
    - predict() output equals labels_.
    """
    X, y = semi_labeled_data
    lp = LabelPropagation(gamma=1.0, n_iterations=50).fit(X, y)
    np.testing.assert_array_equal(lp.predict(), lp.labels_)
