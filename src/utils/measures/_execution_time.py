import time
from typing import Optional

from ._measure import Measure


class ExecutionTime(Measure):
    start_time: Optional[float]
    stop_time: Optional[float]

    def __init__(self):
        self.start_time = None
        self.stop_time = None

    def __str__(self) -> str:
        return super().__str__()

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.stop_time = time.perf_counter()

    def measure(self) -> float:
        if self.start_time is None:
            raise ValueError('Unable to evaluate - timer was not started.')
        if self.stop_time is None:
            raise ValueError('Unable to evaluate - timer was not stopped.')
        return self.stop_time - self.start_time
