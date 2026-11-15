from __future__ import annotations
"""
Factorizaciones QR y resolución de mínimos cuadrados lineales:
- Gram-Schmidt Clásico (CGS) y Modificado (MGS)
- Reflexiones de Householder (A = Q R)
- Mínimos cuadrados lineales mediante QR (evitando ecuaciones normales)
"""

import numpy as np
from src.linear_systems.direct_solvers import backward_substitution


def gram_schmidt_clasico(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Factorización QR reducida mediante Gram-Schmidt Clásico (CGS).
    Se incluye para contrastar su pérdida de ortogonalidad en punto flotante frente a MGS.
    """
    m, n = A.shape
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    for j in range(n):
        v = A[:, j].astype(np.float64, copy=True)
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] < 1e-15:
            raise ValueError(f"Columnas linealmente dependientes detectadas en columna {j}.")
        Q[:, j] = v / R[j, j]

    return Q, R


def gram_schmidt_modificado(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Factorización QR reducida mediante Gram-Schmidt Modificado (MGS).
    Reortogonaliza paso a paso contra el vector actualizado para mejorar la estabilidad.
    """
    m, n = A.shape
    Q = A.astype(np.float64, copy=True)
    R = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        R[i, i] = np.linalg.norm(Q[:, i])
        if R[i, i] < 1e-15:
            raise ValueError(f"Columnas linealmente dependientes detectadas en columna {i}.")
        Q[:, i] = Q[:, i] / R[i, i]

        for j in range(i + 1, n):
            R[i, j] = np.dot(Q[:, i], Q[:, j])
            Q[:, j] = Q[:, j] - R[i, j] * Q[:, i]

    return Q, R


def qr_householder(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Factorización QR completa A = Q R (Q ∈ R^{m x m} ortogonal, R ∈ R^{m x n} triangular
    superior) mediante reflexiones elementales de Householder H_k = I - 2 v_k v_k^T.
    """
    m, n = A.shape
    R = A.astype(np.float64, copy=True)
    Q = np.eye(m, dtype=np.float64)

    pasos = min(m - 1, n)
    for k in range(pasos):
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-15:
            continue

        # Elegimos el signo opuesto a x[0] para evitar cancelación catastrófica
        signo = 1.0 if x[0] >= 0.0 else -1.0
        alpha = -signo * norm_x

        v = x.copy()
        v[0] = v[0] - alpha
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-15:
            continue
        v = v / norm_v

        # Aplicamos H_k por izquierda sobre R y por derecha sobre Q
        R[k:, k:] = R[k:, k:] - 2.0 * np.outer(v, v @ R[k:, k:])
        Q[:, k:] = Q[:, k:] - 2.0 * np.outer(Q[:, k:] @ v, v)

    return Q, R


def resolver_minimos_cuadrados_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Resuelve min_x ||A x - b||_2 mediante factorización QR de Householder:
        R_1 x = (Q^T b)_1
    Evita cuadrar el número de condición κ_2(A^T A) = κ_2(A)^2 de las ecuaciones normales.
    """
    m, n = A.shape
    if m < n:
        raise ValueError("El sistema sobredeterminado requiere m >= n.")

    Q, R = qr_householder(A)
    Qt_b = Q.T @ b
    R1 = R[:n, :n]
    c1 = Qt_b[:n]
    return backward_substitution(R1, c1)