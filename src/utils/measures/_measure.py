from abc import ABC, abstractmethod
from typing import Callable



class Measure(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def __str__(self) -> str:
        return type(self).__name__

    @abstractmethod
    def measure(self, func: Callable):
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> float:
        raise NotImplementedError
