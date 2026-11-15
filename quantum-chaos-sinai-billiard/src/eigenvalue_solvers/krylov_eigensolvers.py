from __future__ import annotations
"""
Cálculo de autovalores interiores de matrices dispersas grandes mediante:
- Método de Arnoldi con Gram-Schmidt modificado (con reortogonalización opcional)
- Transformación espectral Shift-and-Invert M_σ = (H - σ I)^(-1) resuelta con GMRES
- Proyección de Rayleigh-Ritz diagonalizada mediante iteraciones QR propias
"""

from typing import Callable
import numpy as np
from scipy.sparse import spmatrix, eye
from src.linear_systems.krylov_linear import gmres
from .dense_eigensolvers import autovalores_qr_denso


def arnoldi_mgs(
    operador_matvec: Callable[[np.ndarray], np.ndarray],
    v0: np.ndarray,
    m: int,
    reortogonalizar: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye una base ortonormal Q_m ∈ R^{N x m} del subespacio de Krylov K_m(M, v0)
    y la matriz de Hessenberg superior H_m ∈ R^{m x m} mediante Gram-Schmidt modificado.

    El parámetro `reortogonalizar` permite activar o desactivar la doble ortogonalización
    para estudiar experimentalmente la pérdida de ortogonalidad ||Q_m^T Q_m - I||_F
    y la aparición de autovalores espurios en punto flotante.
    """
    n = v0.shape[0]
    m_ef = min(m, n)

    Q = np.zeros((n, m_ef), dtype=np.float64)
    H_hess = np.zeros((m_ef, m_ef), dtype=np.float64)

    norm_v0 = float(np.linalg.norm(v0))
    if norm_v0 < 1e-15:
        raise ValueError("El vector inicial v0 no puede ser nulo.")
    Q[:, 0] = v0 / norm_v0

    for j in range(m_ef):
        w = operador_matvec(Q[:, j])

        # Ortogonalización por Gram-Schmidt modificado
        for i in range(j + 1):
            h_ij = float(np.dot(w, Q[:, i]))
            H_hess[i, j] = h_ij
            w = w - h_ij * Q[:, i]

        # Reortogonalización explícita para evitar pérdida de ortogonalidad en punto flotante
        if reortogonalizar:
            for i in range(j + 1):
                corr = float(np.dot(w, Q[:, i]))
                H_hess[i, j] += corr
                w = w - corr * Q[:, i]

        h_next = float(np.linalg.norm(w))
        if j + 1 < m_ef:
            if h_next < 1e-14:
                # Subespacio invariante exacto alcanzado
                return Q[:, : j + 1], H_hess[: j + 1, : j + 1]
            H_hess[j + 1, j] = h_next
            Q[:, j + 1] = w / h_next

    return Q, H_hess


def autovalores_interiores_shift_invert(
    H: spmatrix,
    sigma: float,
    m_krylov: int = 50,
    k_cercanos: int = 12,
    tol_gmres: float = 1e-11,
    reortogonalizar: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aproxima los `k_cercanos` autovalores de H más próximos al desplazamiento interior σ:
    1) Define el operador desplazado e invertido M_σ v = (H - σ I)^(-1) v vía GMRES.
    2) Proyecta M_σ con Arnoldi (MGS) obteniendo H_m ∈ R^{m x m}.
    3) Simetriza H_m (dado que M_σ es simétrico) y calcula sus valores de Ritz μ_j
       mediante nuestro algoritmo denso de Householder + iteraciones QR.
    4) Recupera los autovalores físicos λ_j = σ + 1 / μ_j ordenados por cercanía a σ.

    Retorna (lambdas_seleccionados, Q_m).
    """
    n = H.shape[0]
    H_desplazada = (H - sigma * eye(n, format="csr")).tocsr()

    def aplicar_inversa_desplazada(v: np.ndarray) -> np.ndarray:
        sol, _ = gmres(H_desplazada, v, restart=min(100, n), tol=tol_gmres, max_restarts=40)
        return sol

    rng = np.random.default_rng(123)
    v0 = rng.standard_normal(n)

    Q_m, H_m = arnoldi_mgs(
        operador_matvec=aplicar_inversa_desplazada,
        v0=v0,
        m=m_krylov,
        reortogonalizar=reortogonalizar,
    )

    # Dado que M_σ = (H - σ I)^(-1) es simétrico, su proyección de Rayleigh-Ritz es simétrica
    H_m_sim = 0.5 * (H_m + H_m.T)
    valores_ritz_mu = autovalores_qr_denso(H_m_sim, tol=1e-12)

    # Filtrar valores de Ritz no nulos y mapear μ -> λ = σ + 1 / μ
    mu_validos = valores_ritz_mu[np.abs(valores_ritz_mu) > 1e-12]
    lambdas_aprox = sigma + 1.0 / mu_validos

    # Seleccionar los k_cercanos autovalores más próximos al centro de ventana σ
    orden_cercania = np.argsort(np.abs(lambdas_aprox - sigma))
    seleccionados = np.sort(lambdas_aprox[orden_cercania[:k_cercanos]])
    return seleccionados, Q_m