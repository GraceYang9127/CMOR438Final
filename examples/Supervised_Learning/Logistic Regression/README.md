# Logistic Regression

Logistic regression is a probabilistic binary classifier. It applies a sigmoid function to a linear combination of features to output a probability, then thresholds at 0.5.

---

## Algorithm

Sigmoid function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \mathbf{w} \cdot \mathbf{x} + b$$

Cross-entropy loss:

$$L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

SGD update rule:

$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \cdot (\sigma(z) - y) \cdot \mathbf{x}$$

---

## Dataset

**Pima Indians Diabetes** (`diabetes.csv`) - 768 patients, 8 medical features (glucose, BMI, insulin, age, etc.). Target: diabetes diagnosis (0/1).

---

## Notebook Covers

- Sigmoid function diagram with decision boundary at 0.5
- EDA: Glucose and BMI distributions split by outcome; scatter colored by class
- Training with cross-entropy loss curve
- Predicted probability histogram split by true class
- Confusion matrix and classification report
