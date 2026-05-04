# Regression Tree

A regression tree applies the same recursive binary splitting as a decision tree classifier, but uses variance reduction as the splitting criterion and predicts the mean of training samples at each leaf.

---

## Algorithm

**Variance reduction (splitting criterion):**

$$\text{Gain}(j, t) = \text{Var}(\text{parent}) - \frac{N_L}{N}\,\text{Var}(\text{left}) - \frac{N_R}{N}\,\text{Var}(\text{right})$$

**Leaf prediction:**

$$\hat{y} = \frac{1}{|S|} \sum_{i \in S} y_i$$

This produces piecewise constant predictions — each region of the feature space maps to one constant value.

---

## Dataset

**Gym Members Exercise Tracking** (`gym_members_exercise_tracking.csv`) - 973 gym members. Target: session duration (hours), predicted from age, weight, calories burned, workout frequency, and body fat percentage.

---

## Notebook Covers

- Piecewise constant prediction diagram (step function vs smooth data)
- EDA: calories vs session duration; weight vs session duration
- Depth vs R² plot (train and test)
- Actual vs predicted scatter with perfect-fit line
- Residuals analysis (scatter + distribution)
