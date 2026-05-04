# Truncated Singular Value Decomposition (SVD)

SVD factorizes any matrix into three components. The truncated version keeps only the top $k$ singular values, giving the best rank-$k$ approximation of the original matrix (Eckart-Young theorem).

---

## Algorithm

**Full SVD:**

$$A = U \Sigma V^\top$$

- $U \in \mathbb{R}^{n \times n}$: left singular vectors
- $\Sigma \in \mathbb{R}^{n \times p}$: diagonal matrix of singular values $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$
- $V^\top \in \mathbb{R}^{p \times p}$: right singular vectors

**Truncated SVD (keep top $k$):**

$$A_k = U_k \Sigma_k V_k^\top$$

**Dimensionality reduction:** $X_{\text{reduced}} = X \cdot V_k^\top$

**Explained variance ratio:** $r_i = \sigma_i^2 / \sum_j \sigma_j^2$

**Key difference from PCA:** SVD does not require centering the data first. PCA is equivalent to SVD applied to the centered data matrix.

---

## Dataset

**Food Nutrition** (`FOOD-DATA-GROUP1.csv`) - 551 food items, 37 nutritional features. The 551x37 matrix is decomposed to find the most important directions of nutritional variation. Falls back to sklearn digits if CSV not found.

---

## Notebook Covers

- Matrix factorization box diagram (A = U_k x Sigma_k x V_k^T with labeled shapes)
- EDA: top-10 highest-protein foods; submatrix heatmap of raw data
- Singular value decay plot and cumulative explained variance
- 2D SVD projection visualization
- Reconstruction error vs number of components
