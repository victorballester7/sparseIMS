import timeit
from typing import Callable
import numpy as np

from ._measure import Measure


class ExecutionTime(Measure):
    times: np.ndarray
    iterations: int = 100
    repetitions: int = 5

    def __init__(self):
        self.start_time = None
        self.stop_time = None

    def __str__(self) -> str:
        return super().__str__()

    def measure(self, func: Callable):
        self.times = (
            np.array(
                timeit.repeat(
                    stmt=func, repeat=self.repetitions, number=self.iterations
                )
            )
            / self.iterations
        )

    def compute(self) -> float:
        return np.min(self.times)
