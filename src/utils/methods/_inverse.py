import numpy as np
from scipy.linalg import inv

from ..methods._method import Method


class MatrixInverse(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval(self, a: np.ndarray, b: np.ndarray):
        x = inv(a) @ b
        return x
