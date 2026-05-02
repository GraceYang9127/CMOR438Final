import numpy as np
import pytest

from rice_ml.unsupervised_learning.svd import TruncatedSVD

@pytest.fixture
def simple_X():
    rng = np.random.default_rng(7)
    return rng.normal(0, 1, (50, 6))



# Fit
def test_svd_fit_returns_self(simple_X):
    """
    Test that fit() returns the fitted TruncatedSVD instance.

    Checks
    ------
    - fit() returns self.
    """
    svd = TruncatedSVD(n_components=2)
    assert svd.fit(simple_X) is svd

def test_svd_components_shape(simple_X):
    """
    Test that components_ has shape (n_components, n_features).

    Checks
    ------
    - components_.shape == (n_components, n_features).
    """
    svd = TruncatedSVD(n_components=3).fit(simple_X)
    assert svd.components_.shape == (3, simple_X.shape[1])

def test_svd_singular_values_shape(simple_X):
    """
    Test that singular_values_ has length n_components.

    Checks
    ------
    - singular_values_.shape == (n_components,).
    """
    svd = TruncatedSVD(n_components=3).fit(simple_X)
    assert svd.singular_values_.shape == (3,)


def test_svd_singular_values_descending(simple_X):
    """
    Test that singular values are stored in descending order.

    Checks
    ------
    - Each singular value >= the next one.
    """
    svd = TruncatedSVD(n_components=4).fit(simple_X)
    s = svd.singular_values_
    assert np.all(s[:-1] >= s[1:])


def test_svd_explained_variance_ratio_shape(simple_X):
    """
    Test that explained_variance_ratio_ has length n_components.

    Checks
    ------
    - explained_variance_ratio_.shape == (n_components,).
    """
    svd = TruncatedSVD(n_components=2).fit(simple_X)
    assert svd.explained_variance_ratio_.shape == (2,)


def test_svd_explained_variance_ratio_sums_to_at_most_one(simple_X):
    """
    Test that explained_variance_ratio_ values sum to <= 1.

    Checks
    ------
    - sum(explained_variance_ratio_) <= 1.0 + small tolerance.
    """
    svd = TruncatedSVD(n_components=2).fit(simple_X)
    assert svd.explained_variance_ratio_.sum() <= 1.0 + 1e-9


def test_svd_full_components_ratio_sums_to_one(simple_X):
    """
    Test that keeping all components explains 100% of variance.

    Checks
    ------
    - With n_components == min(n_samples, n_features), ratio sums to ~1.0.
    """
    k = min(simple_X.shape)
    svd = TruncatedSVD(n_components=k).fit(simple_X)
    assert np.isclose(svd.explained_variance_ratio_.sum(), 1.0, atol=1e-6)


# Transform
def test_svd_transform_shape(simple_X):
    """
    Test that transform reduces dimensionality to n_components.

    Checks
    ------
    - transform(X).shape == (n_samples, n_components).
    """
    svd = TruncatedSVD(n_components=2).fit(simple_X)
    assert svd.transform(simple_X).shape == (len(simple_X), 2)

def test_svd_fit_transform_matches(simple_X):
    """
    Test that fit_transform gives the same result as fit then transform.

    Checks
    ------
    - fit_transform(X) equals fit(X).transform(X) element-wise.
    """
    result1 = TruncatedSVD(n_components=2).fit(simple_X).transform(simple_X)
    result2 = TruncatedSVD(n_components=2).fit_transform(simple_X)
    assert np.allclose(result1, result2, atol=1e-10)


# Inverse transform
def test_svd_inverse_transform_shape(simple_X):
    """
    Test that inverse_transform returns the original number of features.

    Checks
    ------
    - inverse_transform(transform(X)).shape == X.shape.
    """
    svd = TruncatedSVD(n_components=2).fit(simple_X)
    X_back = svd.inverse_transform(svd.transform(simple_X))
    assert X_back.shape == simple_X.shape


def test_svd_full_rank_reconstruction(simple_X):
    """
    Test that keeping all components reconstructs X approximately.

    Checks
    ------
    - inverse_transform(transform(X)) ≈ X when n_components == rank.
    """
    k = min(simple_X.shape)
    svd = TruncatedSVD(n_components=k).fit(simple_X)
    X_back = svd.inverse_transform(svd.transform(simple_X))
    assert np.allclose(X_back, simple_X, atol=1e-8)
