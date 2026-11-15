from __future__ import annotations
"""
Métodos iterativos de subespacios de Krylov para sistemas lineales dispersos:
- Gradiente Conjugado (CG) para sistemas simétricos definidos positivos (H x = b)
- GMRES(m) con Arnoldi (Gram-Schmidt modificado) y rotaciones de Givens para sistemas
  simétricos indefinidos ((H - σI) y = v en Shift-and-Invert)
"""

import numpy as np
from scipy.sparse import spmatrix


def conjugate_gradient(
    A: np.ndarray | spmatrix,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int | None = None,
) -> tuple[np.ndarray, list[float]]:
    """
    Resuelve A x = b con A simétrica definida positiva mediante Gradiente Conjugado.
    Retorna la solución aproximada x y el historial de residuos relativos ||r_k||_2 / ||b||_2.
    """
    n = b.shape[0]
    if max_iter is None:
        max_iter = n

    x = np.zeros(n, dtype=np.float64) if x0 is None else x0.astype(np.float64, copy=True)
    norm_b = float(np.linalg.norm(b))
    if norm_b == 0.0:
        return np.zeros(n, dtype=np.float64), [0.0]

    r = b - (A @ x)
    p = r.copy()
    rs_old = float(np.dot(r, r))
    rel_residuals = [np.sqrt(rs_old) / norm_b]

    if rel_residuals[0] <= tol:
        return x, rel_residuals

    for _ in range(max_iter):
        Ap = A @ p
        pAp = float(np.dot(p, Ap))
        if pAp <= 0.0:
            raise ValueError("Dirección de curvatura no positiva en CG: A no es definida positiva.")

        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap

        rs_new = float(np.dot(r, r))
        rel_res = np.sqrt(rs_new) / norm_b
        rel_residuals.append(rel_res)

        if rel_res <= tol:
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, rel_residuals


def gmres(
    A: np.ndarray | spmatrix,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    restart: int = 60,
    tol: float = 1e-10,
    max_restarts: int = 30,
) -> tuple[np.ndarray, list[float]]:
    """
    Resuelve A x = b mediante GMRES(m) con recomienzos, ortogonalización de Arnoldi
    (Gram-Schmidt modificado) y reducción triangular superior por rotaciones de Givens.
    Apto para el sistema indefinido (H - σI) y = v en Shift-and-Invert.
    """
    n = b.shape[0]
    m = min(restart, n)
    x = np.zeros(n, dtype=np.float64) if x0 is None else x0.astype(np.float64, copy=True)

    norm_b = float(np.linalg.norm(b))
    if norm_b == 0.0:
        return np.zeros(n, dtype=np.float64), [0.0]

    r0 = b - (A @ x)
    beta = float(np.linalg.norm(r0))
    rel_residuals = [beta / norm_b]
    if rel_residuals[0] <= tol:
        return x, rel_residuals

    for _ in range(max_restarts):
        V = np.zeros((n, m + 1), dtype=np.float64)
        H_hess = np.zeros((m + 1, m), dtype=np.float64)
        cs = np.zeros(m, dtype=np.float64)
        sn = np.zeros(m, dtype=np.float64)
        g = np.zeros(m + 1, dtype=np.float64)

        V[:, 0] = r0 / beta
        g[0] = beta
        k_conv = m

        for j in range(m):
            # Paso de Arnoldi con Gram-Schmidt modificado
            w = A @ V[:, j]
            for i in range(j + 1):
                H_hess[i, j] = float(np.dot(w, V[:, i]))
                w -= H_hess[i, j] * V[:, i]

            H_hess[j + 1, j] = float(np.linalg.norm(w))
            if H_hess[j + 1, j] > 1e-15:
                V[:, j + 1] = w / H_hess[j + 1, j]

            # Aplicar rotaciones de Givens previas sobre la nueva columna j de H_hess
            for i in range(j):
                temp = cs[i] * H_hess[i, j] + sn[i] * H_hess[i + 1, j]
                H_hess[i + 1, j] = -sn[i] * H_hess[i, j] + cs[i] * H_hess[i + 1, j]
                H_hess[i, j] = temp

            # Calcular nueva rotación de Givens para anular el elemento subdiagonal H_hess[j+1, j]
            denom = float(np.hypot(H_hess[j, j], H_hess[j + 1, j]))
            if denom > 0.0:
                cs[j] = H_hess[j, j] / denom
                sn[j] = H_hess[j + 1, j] / denom
                H_hess[j, j] = denom
                H_hess[j + 1, j] = 0.0

                g[j + 1] = -sn[j] * g[j]
                g[j] = cs[j] * g[j]

            rel_res = abs(g[j + 1]) / norm_b
            rel_residuals.append(rel_res)

            if rel_res <= tol or H_hess[j + 1, j] <= 1e-15:
                k_conv = j + 1
                break

        # Resolver sistema triangular superior R_k y = g_k
        y = np.zeros(k_conv, dtype=np.float64)
        for i in range(k_conv - 1, -1, -1):
            s = float(np.dot(H_hess[i, i + 1 : k_conv], y[i + 1 : k_conv]))
            y[i] = (g[i] - s) / H_hess[i, i]

        x += V[:, :k_conv] @ y

        if rel_residuals[-1] <= tol:
            break

        r0 = b - (A @ x)
        beta = float(np.linalg.norm(r0))
        if beta / norm_b <= tol:
            break

    return x, rel_residuals