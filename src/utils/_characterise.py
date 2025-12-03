from functools import wraps
from typing import Callable, List, ParamSpec, Tuple, Type, TypeVar

import numpy as np
import polars as pl
from scipy.sparse import csc_matrix

from .measures._measure import Measure
from .methods._method import Method

P = ParamSpec('P')

A = TypeVar('A', bound=np.ndarray)
B = TypeVar('B', bound=np.ndarray)


def characterise(
    methods: List[Type['Method']],
    measures: List[Type['Measure']],
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
                print(f"Solving with using method {method} with dimension {args[0]}")

                for r in range(realisations):
                    a, b = func(*args, **kwargs)
                    if method.is_sparse():
                        print("Converting to sparse matrix format.")
                        a = csc_matrix(a)
                    # a = csc_matrix(a) if method.is_sparse() else a
                    _measures = [m() for m in measures]

                    for m in _measures:
                        m.measure(lambda: method.eval(a, b))

                    measurements = [m.compute() for m in _measures]

                    row = pl.DataFrame(
                        [[str(method), r] + measurements],
                        schema=schema,
                        orient='row',
                    )

                    history = history.vstack(row)

            return history

        return wrapper

    return decorator
