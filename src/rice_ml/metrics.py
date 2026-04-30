"""
metrics.py

A minimal implementation of fundamental classification and regression metrics, including:

mse, 
rmse, 
r2_score, 
accuracy_score, 
confusion_matrix, 
classification_report   


"""

import numpy as np

def mse(y_true, y_pred):
    # MSE = (1/n) * sum((y_i - y_hat_i)^2)
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    # RMSE = sqrt(MSE)
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true, y_pred):
    # R^2 = 1 - SS_res / SS_tot
    # SS_res = sum((y_i - y_hat_i)^2), SS_tot = sum((y_i - y_bar)^2)
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


def accuracy_score(y_true, y_pred):
    # accuracy = (number of correct predictions) / n
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None):
    # Entry [i, j] = count of samples with true label i predicted as label j
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


def classification_report(y_true, y_pred, labels=None):
    # Per-class precision, recall, f1, support + macro and weighted averages
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = {}
    for i, lbl in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        report[str(lbl)] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1-score": round(f1, 4),
            "support": support,
        }

    total = len(y_true)
    macro_p = np.mean([v["precision"] for v in report.values()])
    macro_r = np.mean([v["recall"] for v in report.values()])
    macro_f = np.mean([v["f1-score"] for v in report.values()])
    weighted_p = sum(v["precision"] * v["support"] for v in report.values()) / total
    weighted_r = sum(v["recall"] * v["support"] for v in report.values()) / total
    weighted_f = sum(v["f1-score"] * v["support"] for v in report.values()) / total

    report["macro avg"] = {
        "precision": round(float(macro_p), 4),
        "recall": round(float(macro_r), 4),
        "f1-score": round(float(macro_f), 4),
        "support": total,
    }
    report["weighted avg"] = {
        "precision": round(float(weighted_p), 4),
        "recall": round(float(weighted_r), 4),
        "f1-score": round(float(weighted_f), 4),
        "support": total,
    }
    return report
