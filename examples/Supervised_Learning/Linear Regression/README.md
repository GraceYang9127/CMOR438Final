# Linear Regression

Linear regression fits a model $\hat{y} = \mathbf{w} \cdot \mathbf{x} + b$ by minimizing mean squared error. Two methods are provided: a closed-form normal equation and stochastic gradient descent.

---

## Algorithm

**Normal Equation** (exact solution):

$$\mathbf{w} = (X^\top X)^{-1} X^\top \mathbf{y}$$

**SGD** (iterative):

$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \cdot (\hat{y} - y) \cdot \mathbf{x}$$

Cost function (MSE):

$$C = \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

---

## Dataset

**Gym Members Exercise Tracking** (`gym_members_exercise_tracking.csv`) - 973 gym members, 15 features. Target: calories burned per session.

---

## Notebook Covers

- Regression line concept diagram with residuals
- Correlation heatmap to identify predictive features
- Normal equation vs SGD comparison
- Actual vs predicted scatter plot
- Residuals analysis (scatter + distribution)
- MSE, RMSE, and R² evaluation
