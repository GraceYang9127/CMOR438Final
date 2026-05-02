import numpy as np
import pytest

from rice_ml.unsupervised_learning.kmeans import KMeans

@pytest.fixture
def blob_data():
    # Two well-separated blobs. k=2 should find them cleanly
    rng = np.random.default_rng(0)
    X0 = rng.normal([0, 0], 0.3, (20, 2))
    X1 = rng.normal([8, 8], 0.3, (20, 2))
    return np.vstack([X0, X1])

# Fit
def test_kmeans_fit_returns_self(blob_data):
    """
    Test that fit() returns the fitted KMeans instance.

    Checks
    ------
    - fit() returns self.
    """
    km = KMeans(k=2)
    assert km.fit(blob_data) is km

def test_kmeans_centroids_shape(blob_data):
    """
    Test that centroids_ has shape (k, n_features) after fitting.

    Checks
    ------
    - centroids_.shape == (k, n_features).
    """
    km = KMeans(k=2).fit(blob_data)
    assert km.centroids_.shape == (2, 2)

def test_kmeans_labels_shape(blob_data):
    """
    Test that labels_ has one entry per sample after fitting.

    Checks
    ------
    - labels_.shape == (n_samples,)
    """
    km = KMeans(k=2).fit(blob_data)
    assert km.labels_.shape == (len(blob_data),)

def test_kmeans_labels_valid(blob_data):
    """
    Test that all labels are valid cluster indices in {0, ..., k-1}.

    Checks
    ------
    - All labels belong to {0, 1}.
    """
    km = KMeans(k=2).fit(blob_data)
    assert set(km.labels_).issubset({0, 1})

def test_kmeans_inertia_positive(blob_data):
    """
    Test that inertia_ is a positive float after fitting.

    Checks
    ------
    - inertia_ > 0.
    """
    km = KMeans(k=2).fit(blob_data)
    assert km.inertia_ > 0

# Predict
def test_kmeans_predict_shape(blob_data):
    """
    Test predict returns an array of the correct length.

    Checks
    ------
    - predict(X).shape == (n_samples,).
    """
    km = KMeans(k=2).fit(blob_data)
    assert km.predict(blob_data).shape == (len(blob_data),)

def test_kmeans_predict_before_fit_raises():
    """
    Test that predict raises AttributeError before fit.

    Checks
    ------
    - Calling predict() before fit() raises AttributeError.
    """
    km = KMeans(k=2)
    with pytest.raises(AttributeError):
        km.predict(np.random.rand(4, 2))

def test_kmeans_finds_two_clusters(blob_data):
    """
    Test that KMeans discovers exactly 2 clusters on two-blob data.

    Checks
    ------
    - len(unique labels) == 2.
    """
    km = KMeans(k=2).fit(blob_data)
    assert len(np.unique(km.labels_)) == 2

def test_kmeans_fit_predict_matches_labels(blob_data):
    """
    Test that fit_predict returns the same labels as fit then labels_.

    Checks
    ------
    - fit_predict(X) == km.labels_ after fit(X).
    """
    km = KMeans(k=2)
    preds = km.fit_predict(blob_data)
    np.testing.assert_array_equal(preds, km.labels_)