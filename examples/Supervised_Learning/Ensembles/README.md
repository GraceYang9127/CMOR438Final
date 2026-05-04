# Ensemble Methods

Ensemble methods combine multiple base learners to produce more accurate and robust models than any single learner. Two strategies are covered: bagging and gradient boosting.

---

## Algorithms

**Bagging (Bootstrap Aggregating):**

Train $T$ trees on independent bootstrap samples. Aggregate by majority vote:

$$\hat{y} = \text{mode}\{h_1(\mathbf{x}), h_2(\mathbf{x}), \dots, h_T(\mathbf{x})\}$$

**Random Forest** extends bagging by sampling $\sqrt{p}$ random features at each split, further reducing tree correlation.

**Gradient Boosting:**

Fit trees sequentially to the residuals of the current ensemble:

$$F_0(x) = \bar{y}, \quad F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

where $h_m$ fits the residuals $r_i = y_i - F_{m-1}(x_i)$.

---

## Datasets

- **Classification:** `heart.csv` (1025 patients, predict heart disease)
- **Regression:** `bodyfat.csv` (252 samples, predict body fat percentage)

---

## Notebook Covers

- Bagging diagram (data to bootstrap samples to trees to vote)
- Gradient boosting diagram (4-panel sequential residual fitting)
- Dataset description for both heart and bodyfat
- EDA: heart disease feature boxplots; body fat feature scatter
- Classification comparison: single tree vs bagging vs random forest
- Regression comparison: single tree vs random forest vs gradient boosting
- Actual vs predicted and residuals for gradient boosting
