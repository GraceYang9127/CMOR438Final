# K-Means Clustering

K-Means partitions $n$ samples into $k$ clusters by iteratively assigning each point to its nearest centroid and recomputing centroids. It minimizes total inertia (within-cluster sum of squared distances).

---

## Algorithm (Lloyd's)

1. Initialize $k$ centroids (randomly or with k-means++)
2. Assign each point to its nearest centroid: $c_i = \arg\min_k \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$
3. Update centroids: $\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{i \in C_k} \mathbf{x}_i$
4. Repeat steps 2-3 until centroids stop moving

**Inertia:** $\sum_{i=1}^{N} \|\mathbf{x}_i - \boldsymbol{\mu}_{c_i}\|^2$

---

## Dataset

**Food Nutrition** (`FOOD-DATA-GROUP1.csv`) - 551 food items, 37 nutritional features (calories, fat, protein, carbohydrates, vitamins, minerals). Clusters correspond to natural food groups.

---

## Notebook Covers

- Lloyd's algorithm 3-panel diagram (init, assign, update)
- EDA: top-10 most caloric foods bar chart; fat vs protein scatter colored by caloric value
- Elbow method (inertia vs k) and silhouette score vs k
- Cluster assignments visualized in 2D PCA space
- Cluster size and composition summary
