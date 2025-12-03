from typing import Tuple

import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt

from utils import characterise
from utils.measures import ExecutionTime
from utils.methods import MatrixInverse, LU, QR, SparseLU


@characterise(
    methods=[MatrixInverse, LU, QR, SparseLU],
    measures=[ExecutionTime],
    realisations=10,
)
def generate_sparse(
    dimension: int, rho: float, a=5, delta=0.01
) -> Tuple[np.ndarray, np.ndarray]:
    def sample_offdiagonal(n, num_nonzero):
        # all off-diagonal indices: (i, j) with i != j
        # length of rows and cols: n*n-n
        rows, cols = np.where(~np.identity(n, dtype=bool))  # ~ means negation

        # sample without replacement
        idx = np.random.choice(rows.size, size=num_nonzero, replace=False)

        return rows[idx], cols[idx]

    n = dimension
    # ensure rho is greater than 1/n
    if rho < 1 / n:
        raise ValueError(
            f"rho must be at least 1/n to ensure invertibility. In this case, rho = {rho}, 1/n = {1 / n}"
        )
    A = np.zeros((n, n))
    num_nonzero = int(
        rho * n * n - n
    )  # subtract n for diagonal elements (we will set them later)
    # make sure [i, i] are not chosen
    rows, cols = sample_offdiagonal(n, num_nonzero)

    # generate values from -a to a
    values = np.random.uniform(-a, a, size=num_nonzero)
    A[rows, cols] = values
    rowSums = np.sum(np.abs(A), axis=1)
    eps = np.random.uniform(delta, 1, size=n)
    diagonal_values = np.diag(rowSums + eps)
    
    A += diagonal_values

    # generate b of density as well rho
    num_nonzero_b = int(rho * n)
    indices_b = np.random.choice(n, size=num_nonzero_b, replace=False)
    b = np.zeros(n)
    b[indices_b] = np.random.uniform(-a, a, size=num_nonzero_b)

    return A, b


def main():
    # Generate list of dimensions
    dimensions = [2**i for i in range(7, 13)]  # 256 to 4096

    # fix rho
    rho = 0.01

    # Evaluate Performance - due to @charectarise
    results_sparse = [generate_sparse(d, rho) for d in dimensions]

    # Add dimension to outputs
    results_sparse = [
        h.with_columns(pl.lit(d).alias("Dimension"))
        for h, d in zip(results_sparse, dimensions)
    ]

    # Combine individual results into one object
    result = pl.concat(results_sparse, how="vertical")

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


if __name__ == "__main__":
    main()
