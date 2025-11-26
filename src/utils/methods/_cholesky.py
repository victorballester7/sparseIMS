import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ..methods._method import Method


class Cholesky(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval(self, a: np.ndarray, b: np.ndarray):
        c, low = cho_factor(a, lower=True)
        x = cho_solve((c, low), b)
        return x
