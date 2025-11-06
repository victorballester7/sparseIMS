import numpy as np
import scipy.linalg as la


def main():
    A = np.array([[7, 2], [3, 6]])
    B = np.array([[2, -2], [1, 3]])
    C = np.array([[1, 2, 3], [4, -6, 7], [1, 8, 9]])
    D = np.array([[4, 5, 6], [3, -1, 2], [1, 6, 4]])

    matrices = [A, B, C, D]
    labels = ["A", "B", "C", "D"]

    print("A + B =\n", A + B)
    print("AB =\n", A @ B)
    print("CD =\n", C @ D)
    print()

    # computing determinants and inverses
    for mat, lab in zip(matrices, labels):
        det_ = la.det(mat)
        print(f"det({lab}) is {det_}")
        if np.abs(det_) > 1e-10:
            inv_ = la.inv(mat)
            print(f"{lab}^-1 =\n", inv_)
        print()

    # computing eigenvalues and eigenvectors
    for mat, lab in zip(matrices, labels):
        eigvals, eigvecs = la.eig(mat)
        print(f"Eigenvalues of {lab} are:\n", eigvals)
        print(f"Eigenvectors of {lab} are:\n", eigvecs)
        eigenvecs_inv = la.inv(eigvecs)
        lambda_matrix = np.diag(eigvals)
        reconstructed = eigvecs @ lambda_matrix @ eigenvecs_inv
        print(f"Reconstructed {lab} from eigen decomposition:\n", reconstructed)
        print()
        

if __name__ == "__main__":
    main()
