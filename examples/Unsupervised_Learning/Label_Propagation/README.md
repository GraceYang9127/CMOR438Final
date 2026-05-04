# Label Propagation

Label Propagation is a semi-supervised learning method. A small number of labeled points spread their labels to unlabeled neighbors through a graph built on data similarity. It works under the manifold assumption: nearby points likely share the same label.

---

## Algorithm

**RBF affinity matrix:**

$$W_{ij} = \exp\!\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

**Row-normalized transition matrix:**

$$T = D^{-1} W, \quad D_{ii} = \sum_j W_{ij}$$

**Iterative update with clamping:**

$$F \leftarrow T \cdot F, \quad \text{then clamp labeled rows back to their true labels}$$

**Closed-form solution:**

$$F^* = (I - T_{uu})^{-1} T_{ul} F_l$$

The parameter $\gamma$ controls the RBF bandwidth: larger $\gamma$ means labels propagate only to very nearby points.

---

## Dataset

Synthetic two-cluster dataset: 100 points (50 per cluster), only 4 labeled (2 per cluster). The task is to propagate the 4 known labels to all 96 unlabeled points.

---

## Notebook Covers

- RBF affinity diagram (8 points with edge widths proportional to W_ij)
- EDA: labeled vs unlabeled points (stars vs circles); label rate pie chart
- Before/after propagation scatter plots
- Accuracy evaluation against true cluster membership
- Effect of gamma on propagation (4-panel sensitivity analysis)
