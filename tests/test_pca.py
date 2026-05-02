import numpy as np
import pytest

from rice_ml.unsupervised_learning.pca import PCA


@pytest.fixture
def simple_X():
    rng = np.random.default_rng(3)
    return rng.normal(0, 1, (50, 4))



# Fit
def test_pca_fit_returns_self(simple_X):
    """
    Test that fit() returns the fitted PCA instance.

    Checks
    ------
    - fit() returns self.
    """
    pca = PCA(n_components=2)
    assert pca.fit(simple_X) is pca

def test_pca_components_shape(simple_X):
    """
    Test that components_ has shape (n_components, n_features).

    Checks
    ------
    - components_.shape == (n_components, n_features).
    """
    pca = PCA(n_components=2).fit(simple_X)
    assert pca.components_.shape == (2, simple_X.shape[1])


def test_pca_mean_stored(simple_X):
    """
    Test that mean_ is set to the column-wise mean of the training data.

    Checks
    ------
    - mean_ matches np.mean(X, axis=0).
    """
    pca = PCA(n_components=2).fit(simple_X)
    assert np.allclose(pca.mean_, simple_X.mean(axis=0))


def test_pca_explained_variance_ratio_sums_to_at_most_one(simple_X):
    """
    Test that explained_variance_ratio_ values sum to <= 1.

    Checks
    ------
    - sum(explained_variance_ratio_) <= 1.0 + small tolerance.
    """
    pca = PCA(n_components=2).fit(simple_X)
    assert pca.explained_variance_ratio_.sum() <= 1.0 + 1e-9


def test_pca_full_components_ratio_sums_to_one(simple_X):
    """
    Test that all components explain 100% of the variance.

    Checks
    ------
    - With n_components == n_features, ratio sums to approximately 1.0.
    """
    pca = PCA(n_components=simple_X.shape[1]).fit(simple_X)
    assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0, atol=1e-6)


def test_pca_variance_descending(simple_X):
    """
    Test that explained variances are in descending order.

    Checks
    ------
    - Each variance >= the next variance.
    """
    pca = PCA(n_components=3).fit(simple_X)
    variances = pca.explained_variance_
    assert np.all(variances[:-1] >= variances[1:])



# Transform
def test_pca_transform_shape(simple_X):
    """
    Test that transform reduces dimensionality to n_components.

    Checks
    ------
    - transform(X).shape == (n_samples, n_components).
    """
    pca = PCA(n_components=2).fit(simple_X)
    assert pca.transform(simple_X).shape == (len(simple_X), 2)


def test_pca_fit_transform_matches(simple_X):
    """
    Test that fit_transform gives the same result as fit then transform.

    Checks
    ------
    - fit_transform(X) equals fit(X).transform(X) element-wise.
    """
    pca1 = PCA(n_components=2).fit(simple_X)
    result1 = pca1.transform(simple_X)
    result2 = PCA(n_components=2).fit_transform(simple_X)
    assert np.allclose(np.abs(result1), np.abs(result2), atol=1e-8)

# Inverse transform
def test_pca_inverse_transform_shape(simple_X):
    """
    Test that inverse_transform returns the original number of features.

    Checks
    ------
    - inverse_transform(transform(X)).shape == X.shape.
    """
    pca = PCA(n_components=2).fit(simple_X)
    X_back = pca.inverse_transform(pca.transform(simple_X))
    assert X_back.shape == simple_X.shape
