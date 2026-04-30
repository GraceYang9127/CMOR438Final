import numpy as np
import pytest
from rice_ml.preprocess import StandardScaler, MinMaxScaler, OrdinalEncoder, train_test_split


@pytest.fixture
def simple_X():
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


# StandardScaler
def test_standard_scaler_zero_mean(simple_X):
    """
    Test StandardScaler produces zero-mean features after transform.

    Checks
    ------
    - Column means of the transformed array are all approximately 0.
    """
    Xt = StandardScaler().fit_transform(simple_X)
    assert np.allclose(Xt.mean(axis=0), 0.0, atol=1e-10)


def test_standard_scaler_unit_std(simple_X):
    """
    Test StandardScaler produces unit-variance features after transform.

    Checks
    ------
    - Column standard deviations of the transformed array are all approximately 1.
    """
    Xt = StandardScaler().fit_transform(simple_X)
    assert np.allclose(Xt.std(axis=0), 1.0, atol=1e-10)


def test_standard_scaler_fit_stores_params(simple_X):
    """
    Test that fit stores mean_ and std_ correctly.

    Checks
    ------
    - mean_ matches the column-wise mean of the training data.
    - std_ matches the column-wise standard deviation.
    """
    sc = StandardScaler().fit(simple_X)
    assert np.allclose(sc.mean_, simple_X.mean(axis=0))
    assert np.allclose(sc.std_, simple_X.std(axis=0))


def test_standard_scaler_inverse_roundtrip(simple_X):
    """
    Test that inverse_transform recovers the original data.

    Checks
    ------
    - inverse_transform(transform(X)) equals X up to floating-point precision.
    """
    sc = StandardScaler().fit(simple_X)
    assert np.allclose(sc.inverse_transform(sc.transform(simple_X)), simple_X, atol=1e-10)


def test_standard_scaler_constant_column():
    """
    Test StandardScaler does not produce NaN for constant features.

    Checks
    ------
    - A column with zero variance is scaled without producing NaN values.
    """
    Xc = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    Xt = StandardScaler().fit_transform(Xc)
    assert not np.any(np.isnan(Xt))


# -------------------------------------------------------------------
# MinMaxScaler
# -------------------------------------------------------------------

def test_minmax_scaler_zero_one_range(simple_X):
    """
    Test MinMaxScaler scales features to [0, 1].

    Checks
    ------
    - Minimum of each scaled column is 0.
    - Maximum of each scaled column is 1.
    """
    Xt = MinMaxScaler().fit_transform(simple_X)
    assert np.allclose(Xt.min(axis=0), 0.0, atol=1e-10)
    assert np.allclose(Xt.max(axis=0), 1.0, atol=1e-10)


def test_minmax_scaler_custom_range(simple_X):
    """
    Test MinMaxScaler respects a custom feature_range.

    Checks
    ------
    - With feature_range=(-1, 1), column min is -1 and max is 1.
    """
    Xt = MinMaxScaler(feature_range=(-1, 1)).fit_transform(simple_X)
    assert np.allclose(Xt.min(axis=0), -1.0, atol=1e-10)
    assert np.allclose(Xt.max(axis=0), 1.0, atol=1e-10)


def test_minmax_scaler_inverse_roundtrip(simple_X):
    """
    Test that inverse_transform recovers the original data.

    Checks
    ------
    - inverse_transform(transform(X)) equals X up to floating-point precision.
    """
    sc = MinMaxScaler().fit(simple_X)
    assert np.allclose(sc.inverse_transform(sc.transform(simple_X)), simple_X, atol=1e-10)


def test_minmax_scaler_constant_column():
    """
    Test MinMaxScaler does not produce NaN for constant features.

    Checks
    ------
    - A column with identical values is scaled without producing NaN values.
    """
    Xc = np.array([[3.0, 0.0], [3.0, 0.0]])
    Xt = MinMaxScaler().fit_transform(Xc)
    assert not np.any(np.isnan(Xt))


# OrdinalEncoder
def test_ordinal_encoder_1d():
    """
    Test OrdinalEncoder on a 1D array of strings.

    Checks
    ------
    - Categories are assigned integer codes in alphabetical order.
    - bird=0, cat=1, dog=2.
    """
    enc = OrdinalEncoder()
    y = np.array(["cat", "dog", "cat", "bird"])
    out = enc.fit_transform(y)
    assert list(out) == [1.0, 2.0, 1.0, 0.0]


def test_ordinal_encoder_2d():
    """
    Test OrdinalEncoder on a 2D array of strings.

    Checks
    ------
    - Output shape matches input shape.
    - First column is encoded independently from the second.
    """
    X2 = np.array([["a", "x"], ["b", "y"], ["a", "x"]])
    out = OrdinalEncoder().fit_transform(X2)
    assert out.shape == (3, 2)
    assert out[0, 0] == 0.0 and out[1, 0] == 1.0


def test_ordinal_encoder_fit_transform_consistent():
    """
    Test that fit then transform gives the same result as fit_transform.

    Checks
    ------
    - fit().transform(X) equals fit_transform(X) element-wise.
    """
    y = np.array(["red", "blue", "green", "red"])
    enc = OrdinalEncoder()
    enc.fit(y)
    assert np.array_equal(enc.transform(y), enc.fit_transform(y))


# train_test_split
def test_train_test_split_shapes():
    """
    Test train_test_split produces arrays of the correct size.

    Checks
    ------
    - With test_size=0.2 and 100 samples, train has 80 and test has 20.
    """
    y = np.arange(100)
    X_train, X_test = train_test_split(y, test_size=0.2, random_state=0)
    assert len(X_train) == 80 and len(X_test) == 20


def test_train_test_split_no_overlap():
    """
    Test that train and test sets share no samples.

    Checks
    ------
    - The intersection of train and test index sets is empty.
    """
    y = np.arange(50)
    train, test = train_test_split(y, test_size=0.3, random_state=42)
    assert len(set(train) & set(test)) == 0


def test_train_test_split_multiple_arrays():
    """
    Test train_test_split with multiple arrays applies the same split.

    Checks
    ------
    - Both arrays have matching train lengths and matching test lengths.
    """
    Xa = np.arange(60)
    Xb = np.arange(60) * 2
    Xa_tr, Xa_te, Xb_tr, Xb_te = train_test_split(Xa, Xb, test_size=0.2, random_state=7)
    assert len(Xa_tr) == len(Xb_tr)
    assert len(Xa_te) == len(Xb_te)


def test_train_test_split_reproducible():
    """
    Test that the same random_state produces the same split.

    Checks
    ------
    - Two calls with the same random_state return identical arrays.
    """
    y = np.arange(80)
    tr1, te1 = train_test_split(y, test_size=0.25, random_state=1)
    tr2, te2 = train_test_split(y, test_size=0.25, random_state=1)
    assert np.array_equal(tr1, tr2)
    assert np.array_equal(te1, te2)
