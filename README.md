# CMOR 438 / INDE 577 — Data Science & Machine Learning

A from-scratch Python machine learning package built for Rice University's CMOR 438 / INDE 577 course. 

![CI](https://github.com/GraceYang9127/CMOR438/actions/workflows/test.yml/badge.svg)

---

## Package: `rice_ml`

Install locally with:

```bash
pip install -e .
```

### Supervised Learning

| Algorithm | Class | Description |
|---|---|---|
| Perceptron | `Perceptron` | Binary classifier with step activation, -1/+1 labels |
| Linear Regression | `LinearRegression` | Normal equation and SGD, predicts continuous values |
| Logistic Regression | `LogisticRegression` | Sigmoid output, cross-entropy loss, binary classification |
| K-Nearest Neighbors | `KNN` | Non-parametric, majority vote or mean of k neighbors |
| Decision Tree | `DecisionTreeClassifier` | Gini impurity splits, recursive partitioning |
| Regression Tree | `DecisionTreeRegressor` | Variance reduction splits, leaf predicts mean |
| Random Forest | `RandomForestClassifier`, `RandomForestRegressor` | Bootstrap + random feature subsets |
| Bagging | `BaggingClassifier` | Bootstrap aggregating over any base estimator |
| Gradient Boosting | `GradientBoostingRegressor` | Sequential residual fitting |
| Neural Network | `DenseNetwork` | Fully connected MLP with backpropagation, He initialization |

### Unsupervised Learning

| Algorithm | Class | Description |
|---|---|---|
| K-Means | `KMeans` | Lloyd's algorithm, minimizes inertia |
| DBSCAN | `DBSCAN` | Density-based clustering, automatic noise detection |
| PCA | `PCA` | Eigendecomposition of covariance matrix |
| Truncated SVD | `TruncatedSVD` | Low-rank matrix approximation |
| Label Propagation | `LabelPropagation` | Semi-supervised, RBF affinity graph |
| Community Detection | `CommunityDetector` | Greedy modularity and Girvan-Newman |

### Preprocessing and Metrics

| Module | Functions / Classes |
|---|---|
| `rice_ml.preprocess` | `StandardScaler`, `MinMaxScaler`, `OrdinalEncoder`, `train_test_split` |
| `rice_ml.metrics` | `mse`, `rmse`, `r2_score`, `accuracy_score`, `confusion_matrix`, `classification_report` |

---

## Repository Structure

```
CMOR438/
├── src/rice_ml/          - Package source code
├── tests/                - Pytest test suite
├── examples/
│   ├── Supervised_Learning/    - Notebooks for supervised algorithms
│   └── Unsupervised_Learning/  - Notebooks for unsupervised algorithms
├── data/                 - CSV datasets (not committed)
├── requirements.txt
└── pyproject.toml
```

---

## Datasets

All datasets are downloaded separately and placed in `data/`. Each notebook falls back to a sklearn built-in if the CSV is not found.

| File | Source | Used In |
|---|---|---|
| `heart.csv` | Kaggle - Heart Disease UCI | Perceptron, KNN, Ensembles |
| `diabetes.csv` | Kaggle - Pima Indians Diabetes | Logistic Regression |
| `Obesity_levels.csv` | Kaggle - Obesity Levels | Neural Network, Decision Trees |
| `gym_members_exercise_tracking.csv` | Kaggle - Gym Members | Linear Regression, Regression Tree, PCA |
| `bodyfat.csv` | Kaggle - Body Fat Prediction | Ensembles |
| `FOOD-DATA-GROUP1.csv` | Kaggle - Nutrition Facts | K-Means, DBSCAN, SVD, Community Detection |

---

## Running Tests

```bash
pytest -v
```

## Running Notebooks

```bash
find examples -name "*.ipynb" -exec jupyter nbconvert --to notebook --execute --inplace {} \;
```

---

## Author

Grace Yang - Rice University, CMOR 438 / INDE 577
