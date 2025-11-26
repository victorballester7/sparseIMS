from typing import List, Tuple

import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.linalg as la

from utils import characterise
from utils.measures import ExecutionTime
from utils.methods import Cholesky, LU, MatrixInverse


@characterise(
    methods=[Cholesky],
    measures=[ExecutionTime],
    iterations=10,
    realisations=10,
)
def gen_sym(n, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    eps = 1
    x0 = 1
    x1 = 10.0
    blim = 15.0

    # number of negative eigenvalues
    # m = np.random.randint(0, n)
    # lambdas_neg = np.random.uniform(-x1, -x0, m)
    # lambdas_pos = np.random.uniform(x0, x1, n - m)
    # lambdas = np.concatenate((lambdas_neg, lambdas_pos))

    # all eigenvalues positive
    lambdas = np.random.uniform(x0, x1, n)

    V = np.random.uniform(0, 1, (n, n)) + eps * np.identity(n)
    V = np.tril(V)
    A = V.T @ np.diag(lambdas) @ V

    b = np.random.uniform(-blim, blim, n)

    return A, b

@characterise(
    methods=[MatrixInverse, LU],
    measures=[ExecutionTime],
    iterations=10,
    realisations=10,
)
def gen(n, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    eps = 1
    x0 = 1
    x1 = 10.0
    blim = 15.0

    # number of negative eigenvalues
    m = np.random.randint(0, n)
    lambdas_neg = np.random.uniform(-x1, -x0, m)
    lambdas_pos = np.random.uniform(x0, x1, n - m)
    lambdas = np.concatenate((lambdas_neg, lambdas_pos))

    V = np.random.uniform(0, 1, (n, n)) + eps * np.identity(n)
    V = np.tril(V)
    A = la.inv(V) @ np.diag(lambdas) @ V

    b = np.random.uniform(-blim, blim, n)

    return A, b


def main():
    dimensions = [int(2**i) for i in range(1, 10)]
    # ---- Run Cholesky experiments ----
    results_sym: List[pl.DataFrame] = [gen_sym(d) for d in dimensions]
    results_sym = [
        h.with_columns(pl.lit(d).alias("Dimension"))
        for h, d in zip(results_sym, dimensions)
    ]

    # ---- Run LU experiments ----
    results_lu: List[pl.DataFrame] = [gen(d) for d in dimensions]
    results_lu = [
        h.with_columns(pl.lit(d).alias("Dimension"))
        for h, d in zip(results_lu, dimensions)
    ]

    # Combine both
    result = pl.concat(results_sym + results_lu, how="vertical")

    # ---- Plot ----
    sns.scatterplot(result, x="Dimension", y="ExecutionTime", hue="Method")
    plt.gca().set_yscale("log")
    plt.gca().set_xscale("log", base=2)
    plt.show()
    print(result)


if __name__ == "__main__":
    main()
