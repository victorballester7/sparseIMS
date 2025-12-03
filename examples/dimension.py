from typing import Tuple

import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt

from utils import characterise
from utils.measures import ExecutionTime
from utils.methods import MatrixInverse, LU, Cholesky, QR 


@characterise(
    methods=[MatrixInverse, LU, Cholesky, QR],
    measures=[ExecutionTime],
    realisations=10,
)
def generate_symnetric(dimension: int) -> Tuple[np.ndarray, np.ndarray]:
    eps = 1
    x0 = 1
    x1 = 10.0
    blim = 15.0
    n = dimension

    lambdas = np.random.uniform(x0, x1, n)

    V = np.random.uniform(0, 1, (n, n)) + eps * np.identity(n)
    V = np.tril(V)
    A = V.T @ np.diag(lambdas) @ V

    b = np.random.uniform(-blim, blim, n)

    return A, b


def main():
    # Generate list of dimensions
    dimensions = [int(2**i) for i in range(3, 9)]

    # Evaluate Performance - due to @charectarise
    results_sym = [generate_symnetric(d) for d in dimensions]

    # Add dimension to outputs
    results_sym = [
        h.with_columns(pl.lit(d).alias("Dimension"))
        for h, d in zip(results_sym, dimensions)
    ]

    # Combine individual results into one object
    result = pl.concat(results_sym, how="vertical")

    # Scatter Dimension vs Execution Time and Color by Method
    sns.scatterplot(
        result,
        x="Dimension",
        y="ExecutionTime",
        hue="Method",
    )

    # Make axes lag scaled
    plt.gca().set_yscale("log")
    plt.gca().set_xscale("log", base=2)

    # Show
    plt.show()

    # Save results to file
    # results.write_csv(...)


if __name__ == "__main__":
    main()
