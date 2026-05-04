# Principal Component Analysis (PCA) and PCA + K-Means Pipeline

Two notebooks are in this folder:

- `pca.ipynb` — standalone PCA for dimensionality reduction and visualization
- `pca_kmeans_combo.ipynb` — PCA as a preprocessing step before K-Means clustering

---

## Algorithm

1. Center the data: $\tilde{X} = X - \bar{X}$
2. Compute the covariance matrix: $C = \frac{1}{n} \tilde{X}^\top \tilde{X}$
3. Eigendecomposition: $C = V \Lambda V^\top$
4. Sort eigenvectors by eigenvalue (descending)
5. Project: $X_{\text{reduced}} = \tilde{X} \cdot V_k$

**Explained variance ratio:** $r_i = \lambda_i / \sum_j \lambda_j$

---

## Why PCA Before K-Means?

In high dimensions, distances become meaningless (curse of dimensionality). PCA denoises the data and compresses it to the directions of maximum variance before clustering, often producing cleaner and more separable clusters.

---

## Dataset

**Gym Members Exercise Tracking** (`gym_members_exercise_tracking.csv`) - 973 gym members, 15 features (age, weight, BMI, calories burned, session duration, etc.). Falls back to sklearn iris if CSV not found.

---

## Notebooks Cover

**pca.ipynb:**
- Geometric PCA diagram (PC1/PC2 arrows on correlated 2D data)
- Correlation heatmap motivating PCA
- Scree plot and cumulative variance plot
- 2D projection visualization
- Reconstruction quality vs number of components

**pca_kmeans_combo.ipynb:**
- Pipeline overview and curse of dimensionality explanation
- Elbow method comparison: raw features vs PCA-reduced
- Cluster visualization in 2D PCA space
- Inertia and silhouette score comparison table (raw vs PCA)
