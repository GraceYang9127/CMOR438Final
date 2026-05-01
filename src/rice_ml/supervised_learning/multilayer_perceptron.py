"""
multilayer_perceptron.py

Dense Neural Network (Multi-Layer Perceptron) with backpropagation.

Forward pass:
    z_l = W_l @ a_{l-1} + b_l
    a_l = sigma(z_l)

Backpropagation:
    Output layer: delta_L = (a_L - y) * sigma'(z_L)     
    Hidden layers: delta_l = (W_{l+1}^T @ delta_{l+1}) * sigma'(z_l)

Weight update:
    W_l -= alpha * delta_l @ a_{l-1}^T
    B_l -= alpha * delta_l

W and B are lists indexed from 1. Index 0 holds a placeholder [0.0]

Classes
-------
DenseNetwork: Fully-connected neural network trained via stochastic gradient descent.
"""

import numpy as np

def _sigmoid(z):
    """
    Sigmoid activation: sigma(z) = 1 / (1 + e^{-z}).

    Clipped to prevent overflow for very large |z|.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _d_sigmoid(z):
    """
    Derivative of sigmoid: sigma'(z) = sigma(z) * (1 - sigma(z)).
    """
    s = _sigmoid(z)
    return s * (1.0 - s)


def _initialize_weights(layers):
    """
    Initialize weight matrices and bias vectors using He initialization.

    The initialization sets scale = sqrt(2 / n_in) to prevent vanishing
    or exploding gradients at the start of training.

    Parameters
    ----------
    layers: list of int
        Number of neurons per layer including input and output.

    Returns
    -------
    W: list
        W[i] is the weight matrix of shape (layers[i], layers[i-1]).
        W[0] is a placeholder [0.0].
    B: list
        B[i] is the bias vector of shape (layers[i], 1).
        B[0] is a placeholder [0.0].
    """
    W = [[0.0]]
    B = [[0.0]]
    for i in range(1, len(layers)):
        scale = np.sqrt(2.0 / layers[i - 1])
        W.append(np.random.randn(layers[i], layers[i - 1]) * scale)
        B.append(np.random.randn(layers[i], 1) * scale)
    return W, B

def _forward_pass(W, B, xi):
    """
    Run one forward pass through the network for a single sample.

    Parameters
    ----------
    W: list of ndarray
        Weight matrices (index 0 is placeholder).
    B: list of ndarray
        Bias vectors (index 0 is placeholder).
    xi: ndarray of shape (n_features, 1)
        Input sample as a column vector.

    Returns
    -------
    Z: list of ndarray
        Pre-activation values per layer (index 0 is placeholder).
    A: list of ndarray
        Post-activation values per layer (A[0] = xi).
    """
    Z = [[0.0]]
    A = [xi]
    L = len(W) - 1
    for i in range(1, L + 1):
        z = W[i] @ A[i - 1] + B[i]
        Z.append(z)
        A.append(_sigmoid(z))
    return Z, A


class DenseNetwork(object):
    """
    Fully-connected neural network trained with backpropagation.

    Supports multi-class classification (softmax-style via argmax on sigmoid
    output layer) and binary classification.

    Parameters
    ----------
    layers: list of int, default=[2, 4, 1]
        Architecture: [n_input, hidden1, ..., n_output].
        n_output > 1 triggers one-hot encoding and argmax prediction.

    Attributes
    ----------
    W: list of ndarray
        Weight matrices after fitting (index 0 is placeholder).
    B: list of ndarray
        Bias vectors after fitting (index 0 is placeholder).
    errors_: list of float
        Mean MSE cost per epoch.
    """
    def __init__(self, layers=[2, 4, 1]):
        self.layers = layers
        self.W, self.B = _initialize_weights(layers)

    def fit(self, X, y, alpha=0.01, epochs=100):
        """
        Train the network using stochastic gradient descent.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)
            Integer class labels (0-indexed).
        alpha: float, default=0.01
            Learning rate.
        epochs: int, default=100
            Number of full passes over the training data.

        Returns
        -------
        self
        """
        X = np.array(X, dtype=float)
        y = np.array(y)
        n_out = self.layers[-1]

        # One-hot encode targets if output layer has multiple nodes
        if n_out > 1:
            Y = np.zeros((len(y), n_out))
            for i, yi in enumerate(y.astype(int)):
                Y[i, yi] = 1.0
        else:
            Y = y.reshape(-1, 1).astype(float)

        L = len(self.layers) - 1
        self.errors_ = []

        for _ in range(epochs):
            total_error = 0.0
            for xi, yi in zip(X, Y):
                xi_col = xi.reshape(-1, 1)
                yi_col = yi.reshape(-1, 1)
                Z, A = _forward_pass(self.W, self.B, xi_col)

                deltas = {}
                # Output layer delta: (a_L - y) * sigma'(z_L)
                deltas[L] = (A[L] - yi_col) * _d_sigmoid(Z[L])
                # Hidden layer deltas via chain rule
                for i in range(L - 1, 0, -1):
                    deltas[i] = (self.W[i + 1].T @ deltas[i + 1]) * _d_sigmoid(Z[i])
                # Update weights and biases
                for i in range(1, L + 1):
                    self.W[i] -= alpha * deltas[i] @ A[i - 1].T
                    self.B[i] -= alpha * deltas[i]

                total_error += 0.5 * float(np.sum((A[L] - yi_col) ** 2))
            self.errors_.append(total_error / len(X))
        return self

    def predict(self, X):
        """
        Predict class labels for X.

        Uses argmax over output neurons for multi-class,
        or thresholds at 0.5 for binary output.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        X = np.array(X, dtype=float)
        preds = []
        for xi in X:
            xi_col = xi.reshape(-1, 1)
            _, A = _forward_pass(self.W, self.B, xi_col)
            if self.layers[-1] > 1:
                preds.append(int(np.argmax(A[-1])))
            else:
                preds.append(int(A[-1][0, 0] >= 0.5))
        return np.array(preds)

    def predict_proba(self, X):
        """Return raw output activations for X.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        proba: ndarray of shape (n_samples, n_output)
        """
        X = np.array(X, dtype=float)
        proba = []
        for xi in X:
            xi_col = xi.reshape(-1, 1)
            _, A = _forward_pass(self.W, self.B, xi_col)
            proba.append(A[-1].ravel())
        return np.array(proba)
