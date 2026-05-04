# Perceptron

The Perceptron is the simplest supervised learning model: a single artificial neuron that learns a linear decision boundary by correcting misclassified samples one at a time.

---

## Algorithm

Given a sample $(x_i, y_i)$ where $y_i \in \{-1, +1\}$, the perceptron predicts:

$$\hat{y} = \text{sign}(\mathbf{w} \cdot \mathbf{x} + b)$$

and updates weights only when a mistake is made:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta(\hat{y} - y)\mathbf{x}$$

The perceptron convergence theorem guarantees the algorithm halts in finite steps if the data is linearly separable.

---

## Dataset

**Heart Disease UCI** (`heart.csv`) - 1025 patients, 13 clinical features, binary target (heart disease: yes/no). Labels are converted from 0/1 to -1/+1.

---

## Notebook Covers

- Architecture diagram (input to weighted sum to step function to output)
- EDA: scatter of age vs max heart rate by class; boxplots of key features
- Training with convergence tracking (misclassification count per epoch)
- Confusion matrix evaluation
- Discussion of linear separability limitation
