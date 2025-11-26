import numpy as np
import scipy.linalg as la


def main():
    u = np.array([1, 2, -1])
    v = np.array([-2, 1, 8])
    w = np.array([4, 0, 1])

    vectors = [u, v, w]
    labels = ['u', 'v', 'w']

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dot_product = la.blas.ddot(
                vectors[i].flatten(), vectors[j].flatten()
            )
            print(f'Dot product of {labels[i]} and {labels[j]}: {dot_product}')

    M = np.column_stack((u, v, w))
    print('\nMatrix M with vectors u, v, w as columns:\n', M)
    M1 = np.column_stack((u + 4 * v, v, w - u))
    M2 = np.column_stack((u, v - 3 * u, w + 2 * v + 7 * u))

    print('\ndeterminant of matrix M:', la.det(M))
    print('determinant of matrix M1:', la.det(M1))
    print('determinant of matrix M2:', la.det(M2))

    # gram-schmidt process
    a = u
    b = v - np.dot(a, v) / np.dot(a, a) * a
    c = w - np.dot(a, w) / np.dot(a, a) * a - np.dot(b, w) / np.dot(b, b) * b

    a = a / la.norm(a)
    b = b / la.norm(b)
    c = c / la.norm(c)

    print('\nOrthogonal vectors after Gram-Schmidt process:')
    print('a (from u):', a)
    print('b (from v):', b)
    print('c (from w):', c)

    vectors_orthogonal = [a, b, c]
    vectors_labels = ['a', 'b', 'c']

    for i in range(len(vectors_orthogonal)):
        for j in range(i + 1, len(vectors_orthogonal)):
            dot_product = la.blas.ddot(
                vectors_orthogonal[i].flatten(),
                vectors_orthogonal[j].flatten(),
            )
            print(
                f'Dot product of {vectors_labels[i]} and {vectors_labels[j]}: {dot_product}'
            )

    N = np.column_stack((a, b, c))
    print('\nMatrix N with orthogonal vectors as columns:\n', N)
    print('\nDeterminant of matrix N:', la.det(N))
    N_inv = la.inv(N)
    print('\nInverse of matrix N:\n', N_inv)

    M = N @ np.diag([1, 2, 3]) @ N_inv
    print('\nMatrix M reconstructed from N and its inverse:\n', M)


if __name__ == '__main__':
    main()
