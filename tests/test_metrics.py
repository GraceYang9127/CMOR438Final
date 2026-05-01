import numpy as np
import pytest

from rice_ml.metrics import mse, rmse, r2_score, accuracy_score, confusion_matrix, classification_report


# -------------------------------------------------------------------
# MSE
# -------------------------------------------------------------------

def test_mse_perfect_predictions():
    """
    Test MSE is zero when predictions exactly match true values.

    Checks
    ------
    - MSE([1, 2, 3], [1, 2, 3]) == 0.0
    """
    assert mse([1, 2, 3], [1, 2, 3]) == 0.0


def test_mse_known_value():
    """
    Test MSE on a known input-output pair.

    Checks
    ------
    - MSE([0, 0], [2, 2]) == 4.0
    """
    assert mse([0, 0], [2, 2]) == 4.0


# -------------------------------------------------------------------
# RMSE
# -------------------------------------------------------------------

def test_rmse_perfect_predictions():
    """
    Test RMSE is zero when predictions exactly match true values.

    Checks
    ------
    - RMSE([5, 5], [5, 5]) == 0.0
    """
    assert rmse([5, 5], [5, 5]) == 0.0


def test_rmse_known_value():
    """
    Test RMSE on a known input-output pair.

    Checks
    ------
    - RMSE([0, 0], [3, 4]) == sqrt(12.5)
    """
    assert np.isclose(rmse([0, 0], [3, 4]), np.sqrt(12.5))


# -------------------------------------------------------------------
# R²
# -------------------------------------------------------------------

def test_r2_perfect_fit():
    """
    Test R² is 1.0 for a perfect prediction.

    Checks
    ------
    - R²([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    """
    assert np.isclose(r2_score([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)


def test_r2_known_value():
    """
    Test R² against a manually computed expected value.

    Checks
    ------
    - R² matches 1 - SS_res / SS_tot for a known dataset.
    """
    y_true = np.array([3, -0.5, 2, 7], dtype=float)
    y_pred = np.array([2.5, 0.0, 2, 8], dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    expected = 1 - ss_res / ss_tot
    assert np.isclose(r2_score(y_true, y_pred), expected)


def test_r2_constant_prediction():
    """
    Test R² is less than 1 when prediction is a constant.

    Checks
    ------
    - Predicting the mean of y gives R² < 1 for non-trivial y.
    """
    y = [1, 2, 3, 4]
    score = r2_score(y, [2.5] * 4)
    assert score < 1.0


# -------------------------------------------------------------------
# Accuracy
# -------------------------------------------------------------------

def test_accuracy_perfect():
    """
    Test accuracy_score is 1.0 for perfect predictions.

    Checks
    ------
    - All labels match → accuracy == 1.0
    """
    assert accuracy_score([0, 1, 2], [0, 1, 2]) == 1.0


def test_accuracy_zero():
    """
    Test accuracy_score is 0.0 when no prediction is correct.

    Checks
    ------
    - No labels match → accuracy == 0.0
    """
    assert accuracy_score([0, 1, 2], [1, 2, 0]) == 0.0


def test_accuracy_partial():
    """
    Test accuracy_score on partially correct predictions.

    Checks
    ------
    - 3 of 4 labels match → accuracy == 0.75
    """
    assert accuracy_score([0, 0, 1, 1], [0, 1, 1, 1]) == 0.75


def test_accuracy_mismatched_lengths():
    """
    Test accuracy_score raises ValueError for inputs of different lengths.

    Checks
    ------
    - Passing arrays of different lengths raises ValueError.
    """
    with pytest.raises(ValueError):
        accuracy_score([0, 1, 2], [0, 1])


# -------------------------------------------------------------------
# Confusion matrix
# -------------------------------------------------------------------

def test_confusion_matrix_binary_perfect():
    """
    Test confusion matrix for a perfect binary classifier.

    Checks
    ------
    - Diagonal entries equal the class counts; off-diagonals are 0.
    """
    cm = confusion_matrix([0, 1, 0, 1], [0, 1, 0, 1])
    assert np.array_equal(cm, [[2, 0], [0, 2]])


def test_confusion_matrix_all_wrong():
    """
    Test confusion matrix when all predictions are wrong.

    Checks
    ------
    - All errors appear in off-diagonal entries.
    """
    cm = confusion_matrix([0, 0, 1, 1], [1, 1, 0, 0])
    assert np.array_equal(cm, [[0, 2], [2, 0]])


def test_confusion_matrix_shape():
    """
    Test confusion matrix shape for a 3-class problem.

    Checks
    ------
    - Shape is (n_classes, n_classes).
    """
    cm = confusion_matrix([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 2])
    assert cm.shape == (3, 3)


def test_confusion_matrix_sum():
    """
    Test that confusion matrix entries sum to the total number of samples.

    Checks
    ------
    - sum of all entries == len(y_true)
    """
    y = [0, 1, 1, 0, 2]
    cm = confusion_matrix(y, y)
    assert cm.sum() == len(y)


# -------------------------------------------------------------------
# Classification report
# -------------------------------------------------------------------

def test_classification_report_keys_present():
    """
    Test that classification_report returns macro and weighted averages.

    Checks
    ------
    - 'macro avg' key is present in the report.
    - 'weighted avg' key is present in the report.
    """
    report = classification_report([0, 1, 0, 1], [0, 1, 1, 0])
    assert "macro avg" in report
    assert "weighted avg" in report


def test_classification_report_perfect_scores():
    """
    Test classification_report returns 1.0 for all metrics on perfect predictions.

    Checks
    ------
    - precision, recall, and f1-score are all 1.0 for every class.
    """
    y = [0, 1, 2, 0, 1, 2]
    report = classification_report(y, y)
    for cls in ["0", "1", "2"]:
        assert report[cls]["precision"] == 1.0
        assert report[cls]["recall"] == 1.0
        assert report[cls]["f1-score"] == 1.0


def test_classification_report_support_sums_to_n():
    """
    Test that per-class support values sum to the total number of samples.

    Checks
    ------
    - Sum of support across all classes equals len(y_true).
    """
    y = [0, 0, 1, 1, 2]
    report = classification_report(y, y)
    total = sum(report[k]["support"] for k in ["0", "1", "2"])
    assert total == len(y)
