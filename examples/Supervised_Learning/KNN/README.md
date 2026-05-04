# K-Nearest Neighbors (KNN)

KNN is a non-parametric, instance-based classifier. There is no explicit training phase — the model memorizes the training set and classifies new points by majority vote among their k nearest neighbors.

---

## Algorithm

For a query point $\mathbf{x}$:

1. Compute Euclidean distance to every training point: $d(\mathbf{x}, \mathbf{x}^{(i)}) = \|\mathbf{x} - \mathbf{x}^{(i)}\|_2$
2. Find the $k$ nearest neighbors
3. Predict by majority vote (classification) or mean (regression)

A small $k$ causes high variance (overfitting); a large $k$ causes high bias (underfitting).

---

## Dataset

**Heart Disease UCI** (`heart.csv`) - 1025 patients, 13 clinical features. Binary classification: predict heart disease.

---

## Notebook Covers

- KNN concept diagram (query point with k=3 neighbor circle and distance lines)
- EDA: age vs max heart rate scatter; boxplots of age, thalach, chol by class
- Accuracy vs k plot showing bias-variance tradeoff
- Best k selection and confusion matrix
- Feature scaling discussion (KNN is sensitive to scale)
