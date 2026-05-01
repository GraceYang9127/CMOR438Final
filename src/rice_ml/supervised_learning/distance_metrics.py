"""
distance_metrics.py

Distance functions used by KNN and other distance-based algorithms.

Functions
---------
euclidean(x1, x2): L2 distance = sqrt( sum((x1_i - x2_i)^2) )
manhattan(x1, x2): L1 distance = sum(|x1_i - x2_i|)
cosine(x1, x2): Cosine distance = 1 - (x1 · x2) / (||x1|| * ||x2||)

METRICS: dict
    Maps metric name strings to the corresponding function.
"""

import numpy as np

def euclidean(x1, x2):
    """
    Compute Euclidean (L2) distance between two vectors.

    d(x1, x2) = sqrt( sum( (x1_i - x2_i)^2 ) )

    Parameters
    ----------
    x1, x2: array-like of shape (n_features,)

    Returns
    -------
    float
    """
    return float(np.sqrt(np.sum((x1 - x2) ** 2)))


def manhattan(x1, x2):
    """
    Compute Manhattan (L1) distance between two vectors.

    d(x1, x2) = sum( |x1_i - x2_i| )

    Parameters
    ----------
    x1, x2: array-like of shape (n_features,)

    Returns
    -------
    float
    """
    return float(np.sum(np.abs(x1 - x2)))


def cosine(x1, x2):
    """
    Compute cosine distance between two vectors.

    d(x1, x2) = 1 - (x1 · x2) / ( ||x1|| * ||x2|| )

    Returns 0.0 if either vector is the zero vector.

    Parameters
    ----------
    x1, x2: array-like of shape (n_features,)

    Returns
    -------
    float in [0, 2]
    """
    denom = np.linalg.norm(x1) * np.linalg.norm(x2)
    if denom == 0:
        return 0.0
    return float(1.0 - np.dot(x1, x2) / denom)


METRICS = {
    "euclidean": euclidean,
    "manhattan": manhattan,
    "cosine": cosine,
}
