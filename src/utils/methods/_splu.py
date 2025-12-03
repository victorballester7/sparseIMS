import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import spmatrix

from ..methods._method import Method
from typing import Union



class SparseLU(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval(self, a: Union[np.ndarray, spmatrix], b: np.ndarray):
        lu = splu(a)
        x = lu.solve(b)
        return x

    def is_sparse(self) -> bool:
        return True
