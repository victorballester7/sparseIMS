import numpy as np
import scipy.linalg as la


def sample_offdiagonal(n, num_nonzero):
    # all off-diagonal indices: (i, j) with i != j
    # length of rows and cols: n*n-n
    rows, cols = np.where(~np.identity(n, dtype=bool))  # ~ means negation

    # sample without replacement
    idx = np.random.choice(rows.size, size=num_nonzero, replace=False)

    return rows[idx], cols[idx]

def genSparse(n, rho, a=5, delta=0.01):
    # ensure rho is greater than 1/n
    if rho < 1 / n:
        raise ValueError(f"rho must be at least 1/n to ensure invertibility. In this case, rho = {rho}, 1/n = {1/n}")
    B = np.zeros((n, n))
    num_nonzero = int(
        rho * n * n - n
    )  # subtract n for diagonal elements (we will set them later)
    # make sure [i, i] are not chosen
    rows, cols = sample_offdiagonal(n, num_nonzero)

    # generate values from -a to a
    values = np.random.uniform(-a, a, size=num_nonzero)
    B[rows, cols] = values
    rowSums = np.sum(np.abs(B), axis=1)
    eps = np.random.uniform(delta, 1, size=n)
    diagonal_values = np.diag(rowSums + eps)
    B += diagonal_values

    return B

def main():
    n = 200
    rho = 0.1  # desired density
    B = genSparse(n, rho)

    print(f"\nMatrix B ({B.shape}):")
    print(f"Density of B: {np.count_nonzero(B) / B.size:.4f}")
    print(B)
    print(la.eig(B))


if __name__ == "__main__":
    main()
