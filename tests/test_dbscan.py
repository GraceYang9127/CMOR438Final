import numpy as np
import pytest

from rice_ml.unsupervised_learning.dbscan import DBSCAN

@pytest.fixture
def two_cluster_data():
    # Two tight blobs with a gap — DBSCAN should find 2 clusters, 0 noise
    rng = np.random.default_rng(1)
    X0 = rng.normal([0, 0], 0.2, (15, 2))
    X1 = rng.normal([5, 5], 0.2, (15, 2))
    return np.vstack([X0, X1])


@pytest.fixture
def noise_data():
    # Dense cluster plus one outlier far away
    rng = np.random.default_rng(2)
    X_cluster = rng.normal([0, 0], 0.2, (20, 2))
    outlier = np.array([[50.0, 50.0]])
    return np.vstack([X_cluster, outlier])



# Fit
def test_dbscan_fit_returns_self(two_cluster_data):
    """
    Test that fit() returns the DBSCAN instance.

    Checks
    ------
    - fit() returns self.
    """
    db = DBSCAN(eps=0.5, min_samples=3)
    assert db.fit(two_cluster_data) is db

def test_dbscan_labels_shape(two_cluster_data):
    """
    Test that labels_ has one entry per sample.

    Checks
    ------
    - labels_.shape == (n_samples,)
    """
    db = DBSCAN(eps=0.5, min_samples=3).fit(two_cluster_data)
    assert db.labels_.shape == (len(two_cluster_data),)



# Clustering correctness
def test_dbscan_finds_two_clusters(two_cluster_data):
    """
    Test that DBSCAN discovers exactly 2 clusters on two-blob data.

    Checks
    ------
    - Number of unique non-noise labels == 2.
    """
    db = DBSCAN(eps=0.5, min_samples=3).fit(two_cluster_data)
    cluster_labels = db.labels_[db.labels_ != -1]
    assert len(np.unique(cluster_labels)) == 2

def test_dbscan_no_noise_on_dense_blobs(two_cluster_data):
    """
    Test that tight clusters produce very few noise points.

    Checks
    ------
    - Fewer than 10% of points are labeled noise on dense, well-separated blobs.
    """
    db = DBSCAN(eps=0.8, min_samples=3).fit(two_cluster_data)
    noise_fraction = np.mean(db.labels_ == -1)
    assert noise_fraction < 0.10

def test_dbscan_detects_outlier(noise_data):
    """
    Test that a far-away point is labeled as noise (-1).

    Checks
    ------
    - The outlier at index -1 has label == -1.
    """
    db = DBSCAN(eps=0.5, min_samples=3).fit(noise_data)
    assert db.labels_[-1] == -1


def test_dbscan_all_noise_when_eps_tiny():
    """
    Test that every point is noise when eps is extremely small.

    Checks
    ------
    - All labels == -1 with eps near 0 and min_samples > 1.
    """
    X = np.random.rand(10, 2)
    db = DBSCAN(eps=1e-10, min_samples=2).fit(X)
    assert np.all(db.labels_ == -1)



# fit_predict
def test_dbscan_fit_predict_matches_labels(two_cluster_data):
    """
    Test that fit_predict returns the same labels as fit then labels_.

    Checks
    ------
    - fit_predict(X) == db.labels_ after fit(X).
    """
    db = DBSCAN(eps=0.5, min_samples=3)
    preds = db.fit_predict(two_cluster_data)
    np.testing.assert_array_equal(preds, db.labels_)
