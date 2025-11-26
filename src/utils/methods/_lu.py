import numpy as np
from scipy.linalg import lu_factor, lu_solve

from ..methods._method import Method


class LU(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval(self, a: np.ndarray, b: np.ndarray):
        lu, piv = lu_factor(a)
        x = lu_solve((lu, piv), b)
        return x
