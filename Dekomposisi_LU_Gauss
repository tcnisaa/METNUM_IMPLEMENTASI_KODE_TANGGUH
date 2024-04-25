import numpy as np

def lu_gauss_method(A, b):
    n = len(A)
    L = np.eye(n)
    U = A.astype(float)  # Mengubah tipe data matriks U menjadi float

    for i in range(n-1):
        for j in range(i+1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j] -= factor * U[i]

    y = np.linalg.solve(L, b)
    x = np.linalg.solve(U, y)
    return x

# Testing
A = np.array([[2, 1], [5, 7]])
b = np.array([11, 13])
print("Solusi menggunakan metode dekomposisi LU Gauss:", lu_gauss_method(A, b))
