# Neural Networks (Multi-Layer Perceptron)

A dense neural network stacks multiple layers of neurons. Each layer applies a linear transformation followed by a non-linear activation. Weights are learned by backpropagation and stochastic gradient descent.

---

## Algorithm

**Forward pass** (layer $l$):

$$\mathbf{a}^{(l)} = \sigma\!\left(W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right)$$

**Backpropagation delta rule:**

$$\delta^{(l)} = \left(W^{(l+1)\top} \delta^{(l+1)}\right) \odot \sigma'(\mathbf{z}^{(l)})$$

**He initialization** (for ReLU-like activations):

$$W \sim \mathcal{N}\!\left(0,\, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

---

## Architecture

Default: `[n_features, 64, 32, n_classes]`

- Input layer: one neuron per feature
- Hidden layers: 64 and 32 neurons with sigmoid activation
- Output layer: softmax for multi-class, sigmoid for binary

---

## Dataset

**Obesity Levels** (`Obesity_levels.csv`) - 2111 samples, 16 features, 7 obesity categories. Falls back to sklearn digits if CSV not found.

---

## Notebook Covers

- Network architecture diagram (layered circles with connection lines)
- EDA: age and weight distributions by obesity class; height vs weight scatter
- Training cost curve per epoch
- Train vs test accuracy at epoch checkpoints (overfitting diagnostic)
- Confusion matrix for final model
