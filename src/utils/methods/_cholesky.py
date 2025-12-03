import numpy as np
from scipy.linalg import cho_factor, cho_solve

from scipy.sparse import spmatrix

from ..methods._method import Method
from typing import Union

class Cholesky(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval(self, a: Union[np.ndarray, spmatrix], b: np.ndarray):
        c, low = cho_factor(a, lower=True)
        x = cho_solve((c, low), b)
        return x
