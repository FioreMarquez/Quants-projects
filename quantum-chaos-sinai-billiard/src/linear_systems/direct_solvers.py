from __future__ import annotations
"""
Solucionadores directos para sistemas lineales densos y de banda:
- Sustitución hacia adelante y hacia atrás
- Eliminación Gaussiana con pivoteo parcial (PA = LU)
- Factorización de Cholesky (A = L L^T) para matrices simétricas definidas positivas
"""

import numpy as np


def forward_substitution(L: np.ndarray, b: np.ndarray, unit_diag: bool = False) -> np.ndarray:
    """Resuelve el sistema triangular inferior L y = b."""
    n = L.shape[0]
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = np.dot(L[i, :i], y[:i])
        if unit_diag:
            y[i] = b[i] - s
        else:
            if abs(L[i, i]) < 1e-15:
                raise ZeroDivisionError(f"Pivote nulo en fila {i} durante sustitución hacia adelante.")
            y[i] = (b[i] - s) / L[i, i]
    return y


def backward_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Resuelve el sistema triangular superior U x = y."""
    n = U.shape[0]
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-15:
            raise ZeroDivisionError(f"Pivote nulo en fila {i} durante sustitución hacia atrás.")
        s = np.dot(U[i, i + 1 :], x[i + 1 :])
        x[i] = (y[i] - s) / U[i, i]
    return x


def lu_factorization(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Factorización PA = LU mediante eliminación Gaussiana con pivoteo parcial.
    Retorna (L, U, piv) donde L es triangular inferior unitaria, U triangular superior
    y piv es el vector de permutación de filas.
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz A debe ser cuadrada.")

    U = A.astype(np.float64, copy=True)
    L = np.eye(n, dtype=np.float64)
    piv = np.arange(n, dtype=np.int64)

    for k in range(n - 1):
        # Selección del pivote máximo en columna k desde fila k hasta n-1
        max_row = k + int(np.argmax(np.abs(U[k:, k])))
        if abs(U[max_row, k]) < 1e-15:
            raise np.linalg.LinAlgError("Matriz singular o numéricamente singular en PA = LU.")

        if max_row != k:
            U[[k, max_row], k:] = U[[max_row, k], k:]
            piv[[k, max_row]] = piv[[max_row, k]]
            if k > 0:
                L[[k, max_row], :k] = L[[max_row, k], :k]

        # Eliminación vectorizada por debajo del pivote
        multipliers = U[k + 1 :, k] / U[k, k]
        L[k + 1 :, k] = multipliers
        U[k + 1 :, k:] -= np.outer(multipliers, U[k, k:])

    return L, U, piv


def lu_solve(L: np.ndarray, U: np.ndarray, piv: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resuelve A x = b a partir de la factorización PA = LU."""
    y = forward_substitution(L, b[piv], unit_diag=True)
    return backward_substitution(U, y)


def cholesky_factorization(A: np.ndarray) -> np.ndarray:
    """
    Factorización de Cholesky A = L L^T (costo ~ n^3 / 3 flops).
    Lanza ValueError si A no es simétrica definida positiva (por ejemplo, en H - σI con σ > λ_1).
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz A debe ser cuadrada.")
    if not np.allclose(A, A.T, rtol=0.0, atol=1e-12):
        raise ValueError("La matriz A no es simétrica.")

    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            s = np.dot(L[i, :j], L[j, :j])
            if i == j:
                val = A[i, i] - s
                if val <= 0.0:
                    raise ValueError(
                        f"Falla de Cholesky en índice {i} (pivote {val:.3e} <= 0): "
                        "la matriz no es definida positiva."
                    )
                L[i, j] = np.sqrt(val)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L


def cholesky_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resuelve A x = L L^T x = b mediante dos sustituciones triangulares."""
    y = forward_substitution(L, b, unit_diag=False)
    return backward_substitution(L.T, y)