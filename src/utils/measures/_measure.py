from abc import ABC, abstractmethod


class Measure(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def __str__(self) -> str:
        return type(self).__name__

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        raise NotImplementedError

    @abstractmethod
    def measure(self) -> float:
        raise NotImplementedError
