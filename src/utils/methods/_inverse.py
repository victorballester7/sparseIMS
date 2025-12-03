import numpy as np
from scipy.linalg import inv

from scipy.sparse import spmatrix

from ..methods._method import Method
from typing import Union


class MatrixInverse(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval(self, a: Union[np.ndarray, spmatrix], b: np.ndarray):
        x = inv(a) @ b
        return x
