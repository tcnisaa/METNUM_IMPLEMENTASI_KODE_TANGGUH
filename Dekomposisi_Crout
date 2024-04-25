import numpy as np

def crout_method(A, b):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for j in range(n):
        U[j, j] = 1
        for i in range(j, n):
            sum1 = sum(U[k, j] * L[i, k] for k in range(j))
            L[i, j] = A[i, j] - sum1

        for i in range(j, n):
            sum2 = sum(U[k, j] * L[j, i] for k in range(j))
            U[j, i] = (A[j, i] - sum2) / L[j, j]

    y = np.linalg.solve(L, b)
    x = np.linalg.solve(U, y)
    return x

# Testing
A = np.array([[2, 1], [5, 7]])
b = np.array([11, 13])
print("Solusi menggunakan metode dekomposisi Crout:", crout_method(A, b))
