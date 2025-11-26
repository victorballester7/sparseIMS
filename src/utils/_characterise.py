from functools import wraps
from typing import Callable, List, ParamSpec, Tuple, Type, TypeVar

import numpy as np
import polars as pl

from .measures._measure import Measure
from .methods._method import Method

P = ParamSpec('P')

A = TypeVar('A', bound=np.ndarray)
B = TypeVar('B', bound=np.ndarray)


def characterise(
    methods: List[Type['Method']],
    measures: List[Type['Measure']],
    iterations: int,
    realisations: int,
) -> Callable[[Callable[P, Tuple[A, B]]], Callable[P, pl.DataFrame]]:
    def decorator(func: Callable[P, Tuple[A, B]]) -> Callable[P, pl.DataFrame]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> pl.DataFrame:

            schema = {
                **{
                    'Method': pl.String,
                    'Realisation': pl.Int64,
                },
                **{str(m()): pl.Float64 for m in measures},
            }

            history = pl.DataFrame(schema=schema)

            _methods = [m() for m in methods]

            for method in _methods:
                for r in range(realisations):
                    a, b = func(*args, **kwargs)
                    _measures = [m() for m in measures]

                    for measure in _measures:
                        measure.start()
                        for _ in range(iterations):
                            _ = method.eval(a, b)
                        for measure in _measures:
                            measure.stop()

                    measurements = [
                        m.measure() / iterations for m in _measures
                    ]

                    row = pl.DataFrame(
                        [[str(method), r] + measurements],
                        schema=schema,
                        orient='row',
                    )

                    history = history.vstack(row)

            return history

        return wrapper

    return decorator
