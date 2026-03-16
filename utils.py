"""
Utility functions for target transformation.

Softplus is used as a link function to guarantee strictly positive predictions.
"""

import numpy as np


def softplus(x):
    """Softplus link function: log(1 + exp(x)).

    Maps any real value to a strictly positive output.
    Clipping prevents overflow in exp().
    """
    return np.log1p(np.exp(np.clip(x, -500, 500)))


def inverse_softplus(y):
    """Inverse of softplus: log(exp(y) - 1).

    Used to transform the target into unconstrained space before fitting.
    Clipping prevents log(0).
    """
    return np.log(np.expm1(np.clip(y, 1e-7, 500)))
