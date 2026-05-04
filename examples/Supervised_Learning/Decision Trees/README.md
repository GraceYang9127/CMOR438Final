# Decision Tree Classifier

A decision tree recursively partitions the feature space using binary splits. At each node it selects the feature and threshold that maximizes Gini gain, producing an interpretable tree of if-else rules.

---

## Algorithm

**Gini impurity:**

$$\text{Gini}(S) = 1 - \sum_{k} p_k^2$$

**Gini gain for a split:**

$$\text{Gain}(j, t) = \text{Gini}(\text{parent}) - \frac{N_L}{N}\,\text{Gini}(\text{left}) - \frac{N_R}{N}\,\text{Gini}(\text{right})$$

Splitting continues until max depth is reached or a node has fewer than `min_samples_split` samples. Leaf predicts the majority class.

---

## Dataset

**Obesity Levels** (`Obesity_levels.csv`) - 2111 samples, 16 health and lifestyle features (age, height, weight, diet, activity). Target: obesity category (7 classes).

---

## Notebook Covers

- Full intuition, algorithm steps, advantages/disadvantages, summary
- Dataset description table for all 16 features
- Boxplots of weight, height, age, physical activity by obesity class
- Scatter of age vs weight colored by class
- PCA projection colored by class (EDA)
- Depth vs accuracy plot (train and test) for bias-variance tradeoff
- Confusion matrix for best depth
- Decision boundary visualization in 2D PCA space
