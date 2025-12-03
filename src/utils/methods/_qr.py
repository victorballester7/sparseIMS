import numpy as np
import scipy.linalg as la

from scipy.sparse import spmatrix

from ..methods._method import Method
from typing import Union


class QR(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval(self, a: Union[np.ndarray, spmatrix], b: np.ndarray):
        q, r = la.qr(a)
        x = la.solve(r, q.T @ b)
        return x
