from __future__ import annotations
"""
Algoritmos densos para el problema algebraico de autovalores:
- Método de las Potencias y deflación simétrica
- Iteración del Cociente de Rayleigh (RQI)
- Reducción a forma de Hessenberg / tridiagonal mediante reflexiones de Householder
- Iteraciones QR con rotaciones planas de Givens y desplazamiento de Wilkinson
"""

import numpy as np
from src.linear_systems.direct_solvers import lu_factorization, lu_solve


def metodo_potencias(
    A: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> tuple[float, np.ndarray, int]:
    """
    Calcula el autovalor dominante (mayor módulo) y su autovector asociado.
    Retorna (lambda_dom, v, iteraciones).
    """
    n = A.shape[0]
    v = np.ones(n, dtype=np.float64) if x0 is None else x0.astype(np.float64, copy=True)
    v = v / np.linalg.norm(v)
    lam = float(v @ (A @ v))

    for it in range(1, max_iter + 1):
        w = A @ v
        norm_w = np.linalg.norm(w)
        if norm_w < 1e-15:
            return 0.0, v, it

        v_nuevo = w / norm_w
        lam_nuevo = float(v_nuevo @ (A @ v_nuevo))

        # Alinear signo para medir convergencia del autovector
        if np.dot(v_nuevo, v) < 0.0:
            v_nuevo = -v_nuevo

        residuo = np.linalg.norm(A @ v_nuevo - lam_nuevo * v_nuevo)
        v = v_nuevo
        lam = lam_nuevo

        if residuo <= tol * max(1.0, abs(lam)):
            return lam, v, it

    return lam, v, max_iter


def potencias_con_deflacion(
    A: np.ndarray,
    k: int,
    tol: float = 1e-10,
    max_iter: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae los k autovalores de mayor magnitud de una matriz real simétrica A
    aplicando el Método de las Potencias con deflación de Hotelling:
        A_{j+1} = A_j - λ_j v_j v_j^T
    """
    n = A.shape[0]
    A_def = A.astype(np.float64, copy=True)
    autovalores = np.zeros(k, dtype=np.float64)
    autovectores = np.zeros((n, k), dtype=np.float64)

    rng = np.random.default_rng(42)
    for j in range(k):
        x0 = rng.standard_normal(n)
        # Ortogonalizamos el vector inicial contra los autovectores ya hallados
        for prev in range(j):
            x0 = x0 - np.dot(autovectores[:, prev], x0) * autovectores[:, prev]

        lam, v, _ = metodo_potencias(A_def, x0=x0, tol=tol, max_iter=max_iter)
        autovalores[j] = lam
        autovectores[:, j] = v
        A_def = A_def - lam * np.outer(v, v)

    return autovalores, autovectores


def cociente_rayleigh(
    A: np.ndarray,
    x0: np.ndarray,
    tol: float = 1e-13,
    max_iter: int = 30,
) -> tuple[float, np.ndarray, int]:
    """
    Iteración del Cociente de Rayleigh (convergencia cúbica en matrices simétricas):
        ρ_k = (x_k^T A x_k) / (x_k^T x_k),   (A - ρ_k I) y = x_k,   x_{k+1} = y / ||y||_2
    Utiliza nuestra factorización PA = LU para resolver cada sistema desplazado.
    """
    n = A.shape[0]
    I_n = np.eye(n, dtype=np.float64)
    x = x0.astype(np.float64, copy=True)
    x = x / np.linalg.norm(x)
    rho = float(x @ (A @ x))

    for it in range(1, max_iter + 1):
        residuo = np.linalg.norm(A @ x - rho * x)
        if residuo <= tol * max(1.0, abs(rho)):
            return rho, x, it - 1

        try:
            L, U, piv = lu_factorization(A - rho * I_n)
            y = lu_solve(L, U, piv, x)
        except np.linalg.LinAlgError:
            # Si ρ ya coincide con un autovalor a precisión de máquina, el sistema es singular
            return rho, x, it

        x = y / np.linalg.norm(y)
        rho = float(x @ (A @ x))

    return rho, x, max_iter


def reducir_hessenberg_householder(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce una matriz cuadrada A a forma de Hessenberg superior H = Q^T A Q mediante
    transformaciones ortogonales de semejanza de Householder.
    Si A es simétrica, H resulta tridiagonal simétrica.
    """
    n = A.shape[0]
    H = A.astype(np.float64, copy=True)
    Q = np.eye(n, dtype=np.float64)

    for k in range(n - 2):
        x = H[k + 1 :, k]
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-15:
            continue

        signo = 1.0 if x[0] >= 0.0 else -1.0
        alpha = -signo * norm_x

        v = x.copy()
        v[0] = v[0] - alpha
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-15:
            continue
        v = v / norm_v

        # Semejanza ortogonal H <- P_k H P_k con P_k = I - 2 v v^T
        H[k + 1 :, k:] = H[k + 1 :, k:] - 2.0 * np.outer(v, v @ H[k + 1 :, k:])
        H[:, k + 1 :] = H[:, k + 1 :] - 2.0 * np.outer(H[:, k + 1 :] @ v, v)
        Q[:, k + 1 :] = Q[:, k + 1 :] - 2.0 * np.outer(Q[:, k + 1 :] @ v, v)

    return H, Q


def _desplazamiento_wilkinson(a: float, b: float, c: float, d: float) -> float:
    """
    Calcula el desplazamiento de Wilkinson a partir de la submatriz 2x2 inferior
    [[a, b], [c, d]]: elige el autovalor más cercano a d (o d si son complejos).
    """
    traza = a + d
    det = a * d - b * c
    disc = traza * traza - 4.0 * det
    if disc < 0.0:
        return d

    raiz = np.sqrt(disc)
    mu1 = 0.5 * (traza + raiz)
    mu2 = 0.5 * (traza - raiz)
    return mu1 if abs(mu1 - d) < abs(mu2 - d) else mu2


def iteracion_qr_hessenberg(
    H_in: np.ndarray,
    tol: float = 1e-12,
    max_iter_por_autovalor: int = 200,
) -> np.ndarray:
    """
    Calcula todos los autovalores reales de una matriz de Hessenberg superior (o tridiagonal)
    mediante iteraciones QR con desplazamiento de Wilkinson y rotaciones planas de Givens.
    """
    n = H_in.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([H_in[0, 0]], dtype=np.float64)

    H = H_in.astype(np.float64, copy=True)
    autovalores = np.zeros(n, dtype=np.float64)

    for m in range(n - 1, 0, -1):
        iters = 0
        while True:
            cota_cero = tol * (abs(H[m - 1, m - 1]) + abs(H[m, m]))
            if abs(H[m, m - 1]) <= max(cota_cero, 1e-15) or iters >= max_iter_por_autovalor:
                break

            mu = _desplazamiento_wilkinson(
                H[m - 1, m - 1], H[m - 1, m], H[m, m - 1], H[m, m]
            )

            # Desplazamiento H_m <- H_m - mu * I
            for i in range(m + 1):
                H[i, i] -= mu

            cs = np.zeros(m, dtype=np.float64)
            sn = np.zeros(m, dtype=np.float64)

            # Paso QR: 
            for i in range(m):
                r = float(np.hypot(H[i, i], H[i + 1, i]))
                if r == 0.0:
                    c, s = 1.0, 0.0
                else:
                    c = H[i, i] / r
                    s = H[i + 1, i] / r
                cs[i] = c
                sn[i] = s

                fila_i = c * H[i, i : m + 1] + s * H[i + 1, i : m + 1]
                fila_ip1 = -s * H[i, i : m + 1] + c * H[i + 1, i : m + 1]
                H[i, i : m + 1] = fila_i
                H[i + 1, i : m + 1] = fila_ip1

            # Completar transformación de semejanza R * Q aplicando G_i^T por derecha
            for i in range(m):
                c, s = cs[i], sn[i]
                lim = min(i + 3, m + 1)
                col_i = c * H[:lim, i] + s * H[:lim, i + 1]
                col_ip1 = -s * H[:lim, i] + c * H[:lim, i + 1]
                H[:lim, i] = col_i
                H[:lim, i + 1] = col_ip1

            # Restaurar desplazamiento H_m <- H_m + mu * I
            for i in range(m + 1):
                H[i, i] += mu

            iters += 1

        autovalores[m] = H[m, m]

    autovalores[0] = H[0, 0]
    return np.sort(autovalores)


def autovalores_qr_denso(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Calcula el espectro completo de una matriz real simétrica:
    1) Reducción a forma tridiagonal/Hessenberg por reflexiones de Householder.
    2) Iteraciones QR con desplazamiento de Wilkinson y rotaciones de Givens.
    """
    H_hess, _ = reducir_hessenberg_householder(A)
    return iteracion_qr_hessenberg(H_hess, tol=tol)
