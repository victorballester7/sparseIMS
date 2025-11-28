import numpy as np
import scipy.linalg as la


def main():
    A = np.zeros((5, 5))
    A[1, 0], A[3, 0], A[4, 3], A[3, 4] = -1, 1, 2, 1
    print(f"Matrix A ({A.shape}):")
    print(A)

    n = 200
    B1 = np.diag(7 * np.ones(n))
    B2 = np.diag(2 * np.ones(n - 1), k=1) # k=1 means 1st superdiagonal
    B3 = np.diag(3 * np.ones(n - 1), k=-1) # k=-1 means 1st subdiagonal
    B = B1 + B2 + B3
    print(f"\nMatrix B ({B.shape}):")
    print(B)


    rhoA = np.count_nonzero(A) / A.size
    rhoB = np.count_nonzero(B) / B.size

    print(f"\nDensity of A: {rhoA:.4f}")
    print(f"Density of B: {rhoB:.4f}")


if __name__ == "__main__":
    main()
