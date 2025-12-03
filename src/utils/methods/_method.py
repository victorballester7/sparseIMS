from abc import ABC, abstractmethod

import numpy as np
from typing import Union
from scipy.sparse import spmatrix

class Method(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def __str__(self) -> str:
        return type(self).__name__

    @abstractmethod
    # a is sparse array or ndarray
    def eval(self, a: Union[np.ndarray, spmatrix], b: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def is_sparse(self) -> bool:
        return False
