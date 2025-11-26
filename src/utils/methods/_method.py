from abc import ABC, abstractmethod

import numpy as np


class Method(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def __str__(self) -> str:
        return type(self).__name__

    @abstractmethod
    def eval(self, a: np.ndarray, b: np.ndarray):
        raise NotImplementedError
