# CMOR 438 / INDE 577 — Data Science & Machine Learning

A from-scratch Python machine learning package built for Rice University's CMOR 438 / INDE 577 course. 


---

## Overview

`rice_ml` is a educational machine learning framework covering supervised learning, unsupervised learning, preprocessing, and evaluation. It is designed for transparency: every algorithm exposes its math through clean, readable implementations.

Key features:

- Implementations of 16 algorithms built from scratch using NumPy only.
- Built-in preprocessing tools: scalers, encoders, and train/test split with pandas support.
- Evaluation metrics for regression and classification: MSE, RMSE, R², accuracy, confusion matrix, classification report.
- Example Jupyter notebooks for every algorithm with EDA, visualizations, and interpretation.
- 150 unit tests covering algorithms, preprocessing, and metrics.
- CI via GitHub Actions on every push and pull request.

---

## Project Structure

```
CMOR438/
├── examples/
│   ├── Supervised_Learning/
│   │   ├── Decision Trees/
│   │   ├── Ensembles/
│   │   ├── KNN/
│   │   ├── Linear Regression/
│   │   ├── Logistic Regression/
│   │   ├── Neural Networks/
│   │   ├── Perceptron/
│   │   └── Regression Trees/
│   └── Unsupervised_Learning/
│       ├── Community_detection/
│       ├── DBSCAN/
│       ├── K-means/
│       ├── Label_Propagation/
│       ├── PCA/
│       └── SVD/
├── src/
│   └── rice_ml/
│       ├── __init__.py
│       ├── metrics.py
│       ├── preprocess.py
│       ├── supervised_learning/
│       └── unsupervised_learning/
├── tests/
│   ├── test_community_detection.py
│   ├── test_dbscan.py
│   ├── test_decision_tree_classifier.py
│   ├── test_decision_tree_regressor.py
│   ├── test_ensemble.py
│   ├── test_kmeans.py
│   ├── test_knn.py
│   ├── test_label_propagation.py
│   ├── test_linear_regression.py
│   ├── test_logistic_regression.py
│   ├── test_metrics.py
│   ├── test_mlp.py
│   ├── test_pca.py
│   ├── test_perceptron.py
│   ├── test_preprocess.py
│   └── test_svd.py
├── pyproject.toml
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Algorithms

### Supervised Learning

| Algorithm | Class | Description |
|---|---|---|
| Perceptron | `Perceptron` | Binary classifier with step activation. Labels must be -1 / +1. Converges on linearly separable data. |
| Linear Regression | `LinearRegression` | Fits $\hat{y} = \mathbf{w} \cdot \mathbf{x} + b$ via normal equation or SGD. Minimizes MSE. |
| Logistic Regression | `LogisticRegression` | Sigmoid output, cross-entropy loss, SGD updates. Binary classification. |
| K-Nearest Neighbors | `KNN` | Non-parametric classifier. Majority vote among $k$ nearest Euclidean neighbors. |
| Decision Tree | `DecisionTreeClassifier` | Recursive Gini-impurity splits. Leaf returns majority class. |
| Regression Tree | `DecisionTreeRegressor` | Recursive variance-reduction splits. Leaf returns mean of samples. |
| Random Forest | `RandomForestClassifier`, `RandomForestRegressor` | Bootstrap samples + random feature subsets. Majority vote or mean. |
| Bagging | `BaggingClassifier` | Bootstrap aggregating over any base estimator. Reduces variance. |
| Gradient Boosting | `GradientBoostingRegressor` | Sequential trees fit to residuals: $F_m = F_{m-1} + \eta h_m$. |
| Neural Network | `DenseNetwork`, `MLP` | Fully connected MLP. He initialization, sigmoid activations, backpropagation. |

### Unsupervised Learning

| Algorithm | Class | Description |
|---|---|---|
| K-Means | `KMeans` | Lloyd's algorithm. Minimizes inertia. Includes elbow method and silhouette score. |
| DBSCAN | `DBSCAN` | Density-based clustering. Finds arbitrary-shaped clusters. Labels noise as -1. |
| PCA | `PCA` | Eigendecomposition of covariance matrix. Returns components, explained variance, inverse transform. |
| Truncated SVD | `TruncatedSVD` | Low-rank matrix approximation via SVD. Explained variance ratio included. |
| Label Propagation | `LabelPropagation` | Semi-supervised. RBF affinity, row-normalized transition matrix, iterative label spreading with clamping. |
| Community Detection | `CommunityDetector` | Greedy modularity maximization (Clauset-Newman-Moore) and Girvan-Newman. Wraps NetworkX. |

---

## Utilities

### Preprocessing (`rice_ml.preprocess`)

| Class / Function | Description |
|---|---|
| `StandardScaler` | Standardize features to zero mean and unit variance: $z = (x - \mu) / \sigma$ |
| `MinMaxScaler` | Scale features to a target range (default [0, 1]) |
| `OrdinalEncoder` | Convert categorical text columns to integer codes (alphabetical ordering) |
| `train_test_split` | Randomly split arrays or DataFrames into train and test sets. Supports `random_state`. |

### Metrics (`rice_ml.metrics`)

| Function | Description |
|---|---|
| `mse` | Mean Squared Error: $(1/n)\sum(\hat{y}_i - y_i)^2$ |
| `rmse` | Root Mean Squared Error: $\sqrt{\text{MSE}}$ |
| `r2_score` | Coefficient of determination: $1 - SS_{res}/SS_{tot}$ |
| `accuracy_score` | Fraction of correctly classified samples |
| `confusion_matrix` | Count matrix: entry $[i,j]$ = samples with true label $i$ predicted as $j$ |
| `classification_report` | Per-class precision, recall, F1, support with macro and weighted averages |

---

## Datasets

Place CSV files in the `data/` directory (not committed to git). Each notebook falls back to a sklearn built-in if the CSV is not found.

| File | Description | Used In |
|---|---|---|
| `heart.csv` | 1025 patients, 13 clinical features, binary heart disease label | Perceptron, KNN, Ensembles |
| `diabetes.csv` | 768 patients, 8 medical features (glucose, BMI, insulin), diabetes outcome | Logistic Regression |
| `Obesity_levels.csv` | 2111 samples, 16 lifestyle features, 7 obesity categories | Neural Network, Decision Tree |
| `gym_members_exercise_tracking.csv` | 973 gym members, 15 features, predict calories burned or session duration | Linear Regression, Regression Tree, PCA |
| `bodyfat.csv` | 252 samples, 14 body measurements, predict body fat percentage | Ensembles |
| `FOOD-DATA-GROUP1.csv` | 551 food items, 37 nutritional features (calories, fat, protein, vitamins) | K-Means, DBSCAN, SVD, Community Detection |

---

## Installation

```bash
git clone https://github.com/GraceYang9127/CMOR438.git
cd CMOR438
pip install -e .
```

---

## Getting Started

```python
from rice_ml.supervised_learning import LinearRegression
from rice_ml.preprocess import StandardScaler, train_test_split
from rice_ml.metrics import mse, r2_score

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = LinearRegression(method='normal')
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(f'R²:   {r2_score(y_test, y_pred):.4f}')
print(f'MSE:  {mse(y_test, y_pred):.4f}')
```

```python
from rice_ml.unsupervised_learning import KMeans, PCA
from rice_ml.preprocess import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

km = KMeans(k=4, max_iters=200)
km.fit(X_reduced)
print(f'Inertia: {km.inertia_:.2f}')
print(f'Cluster sizes: {[int((km.labels_ == i).sum()) for i in range(4)]}')
```

---

## Testing

```bash
pytest -v
```

150 tests covering all algorithms, preprocessing utilities, and evaluation metrics. Tests include correctness checks, edge cases, and shape validation.

---

## Running Notebooks

Download the datasets from Kaggle and place them in `data/`. Then:

```bash
find examples -name "*.ipynb" -exec jupyter nbconvert --to notebook --execute --inplace {} \;
```

Or open any notebook in VS Code and click Run All.

---

## Author

Grace Yang - Rice University, CMOR 438 / INDE 577
