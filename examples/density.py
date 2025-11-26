from typing import Tuple

import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt

from utils import characterise
from utils.measures import ExecutionTime
from utils.methods import LU, Cholesky, MatrixInverse


@characterise(
    methods=[Cholesky, LU, MatrixInverse],
    measures=[ExecutionTime],
    iterations=10,
    realisations=10,
)
def generate_sparse(
    dimension: int, rho: float
) -> Tuple[np.ndarray, np.ndarray]:
    ...


def main():
    ...


if __name__ == '__main__':
    main()
