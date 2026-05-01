"""
gradient_descent.py

Gradient descent optimization utilities.

The derivative f'(w) gives the slope at w. Moving opposite the slope minimizes f.
Update rule: w_new = w - alpha * f'(w)

Classes
-------
GradientDescent: Generic gradient descent optimizer with convergence tracking.

Functions
---------
gradient_descent(f, df, w0, alpha, epochs): minimize f starting from w0.
"""

import numpy as np


def gradient_descent(f, df, w0, alpha=0.01, epochs=1000):
    """
    Minimize f starting from w0 using gradient descent.

    Parameters
    ----------
    f: callable
        Objective function f(w) to minimize.
    df: callable
        Derivative (gradient) of f.
    w0: float
        Initial value of w.
    alpha: float, default=0.01
        Learning rate.
    epochs: int, default=1000
        Number of update steps.

    Returns
    -------
    w: float
        Final value of w after optimization.
    history: list of float
        Value of w at each step including the initial value.
    """
    w = float(w0)
    history = [w]
    for _ in range(epochs):
        # w_new = w - alpha * f'(w)
        w = w - alpha * df(w)
        history.append(w)
    return w, history


class GradientDescent(object):
    """
    Generic gradient descent optimizer with early stopping.

    Parameters
    ----------
    alpha: float, default=0.01
        Learning rate (step size).
    epochs: int, default=1000
        Maximum number of iterations.
    tol: float, default=1e-6
        Stop early if the parameter update norm falls below this threshold.

    Attributes
    ----------
    history_: list of ndarray
        Parameter value w at each iteration.
    """

    def __init__(self, alpha=0.01, epochs=1000, tol=1e-6):
        self.alpha = alpha
        self.epochs = epochs
        self.tol = tol
        self.history_ = []

    def minimize(self, f, df, w0):
        """
        Run gradient descent to minimize f(w).

        Parameters
        ----------
        f: callable
            Objective function f(w).
        df: callable
            Gradient function df(w).
        w0: array-like
            Initial parameter vector.

        Returns
        -------
        w: ndarray
            Optimal parameter found.
        """
        w = np.array(w0, dtype=float)
        self.history_ = [w.copy()]
        for _ in range(self.epochs):
            grad = df(w)
            w_new = w - self.alpha * grad
            self.history_.append(w_new.copy())
            # Stop if the update step is smaller than the tolerance
            if np.linalg.norm(w_new - w) < self.tol:
                break
            w = w_new
        return w
