# Supervised Learning

Supervised learning algorithms learn a mapping from input features to output labels using labeled training data. Every model here is implemented from scratch using only NumPy.

---

## What Is Supervised Learning?

Given a dataset of input-output pairs $(X, y)$, the goal is to learn a function $f$ such that $f(x) \approx y$ for unseen inputs. Two main tasks:

- **Classification**: predict a discrete class label (e.g. heart disease: yes/no)
- **Regression**: predict a continuous value (e.g. calories burned)

---

## Algorithms Covered

| Algorithm | Type | Dataset | Notebook |
|---|---|---|---|
| Perceptron | Classification | `heart.csv` | `Perceptron/perceptron.ipynb` |
| Linear Regression | Regression | `gym_members_exercise_tracking.csv` | `Linear Regression/linear_regression.ipynb` |
| Logistic Regression | Classification | `diabetes.csv` | `Logistic Regression/logistic_regression.ipynb` |
| K-Nearest Neighbors | Classification | `heart.csv` | `KNN/knn.ipynb` |
| Decision Tree | Classification | `Obesity_levels.csv` | `Decision Trees/decision_tree.ipynb` |
| Regression Tree | Regression | `gym_members_exercise_tracking.csv` | `Regression Trees/regression_tree.ipynb` |
| Ensemble Methods | Classification + Regression | `heart.csv`, `bodyfat.csv` | `Ensembles/ensemble.ipynb` |
| Neural Network | Classification | `Obesity_levels.csv` | `Neural Networks/neural_network.ipynb` |

---

## Notebook Structure

Each notebook follows the same structure:

1. Intuition and math overview
2. Dataset description
3. Exploratory data analysis (EDA)
4. Illustrative diagram
5. Preprocessing with `rice_ml.preprocess`
6. Model training with `rice_ml` classes
7. Evaluation with `rice_ml.metrics`
8. Visualizations (confusion matrix, cost curves, decision boundaries)
9. Interpretation

---

## Key Principle

All models are implemented from scratch. No scikit-learn classifiers or regressors are used for training or prediction. scikit-learn is only used for built-in fallback datasets when a CSV is not found.
