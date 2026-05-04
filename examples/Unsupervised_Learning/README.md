# Unsupervised Learning

Unsupervised learning algorithms find structure in data without labels. All models are implemented from scratch using NumPy and NetworkX.

---

## What Is Unsupervised Learning?

Given only input features $X$ with no labels $y$, the goal is to discover hidden structure: natural clusters, low-dimensional representations, or community structure in graphs.

Three main tasks:

- **Clustering**: group similar samples together (K-Means, DBSCAN)
- **Dimensionality reduction**: find compact representations (PCA, SVD)
- **Graph analysis**: detect communities in network data (Community Detection, Label Propagation)

---

## Algorithms Covered

| Algorithm | Task | Dataset | Notebook |
|---|---|---|---|
| K-Means | Clustering | `FOOD-DATA-GROUP1.csv` | `K-means/kmeans.ipynb` |
| DBSCAN | Clustering | `FOOD-DATA-GROUP1.csv` | `DBSCAN/dbscan.ipynb` |
| PCA | Dimensionality Reduction | `gym_members_exercise_tracking.csv` | `PCA/pca.ipynb` |
| PCA + K-Means | Pipeline | `gym_members_exercise_tracking.csv` | `PCA/pca_kmeans_combo.ipynb` |
| Truncated SVD | Dimensionality Reduction | `FOOD-DATA-GROUP1.csv` | `SVD/svd.ipynb` |
| Community Detection | Graph Analysis | `FOOD-DATA-GROUP1.csv` | `Community_detection/community_detection.ipynb` |
| Label Propagation | Semi-Supervised | Synthetic blobs | `Label_Propagation/label_propagation.ipynb` |

---

## Notebook Structure

Each notebook follows the same structure:

1. Intuition and math overview
2. Dataset description
3. Exploratory data analysis (EDA)
4. Illustrative diagram
5. Data loading and preprocessing
6. Model fitting with `rice_ml` classes
7. Evaluation and visualizations
8. Hyperparameter sensitivity analysis
9. Interpretation

---

## Key Concept

No single unsupervised technique is universally optimal. K-Means assumes spherical clusters; DBSCAN handles arbitrary shapes; PCA assumes linear structure. Understanding these assumptions is as important as running the algorithm.
