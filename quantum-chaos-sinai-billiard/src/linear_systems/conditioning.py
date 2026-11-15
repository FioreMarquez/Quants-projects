from __future__ import annotations
"""
Análisis de error de redondeo, normas matriciales y localización espectral
mediante el Teorema de los Discos de Gershgorin.
"""

import numpy as np
from scipy.sparse import csr_matrix


def compute_machine_epsilon() -> float:
    """Determina empíricamente el épsilon de máquina ε_maq (2^-52 en IEEE 754 float64)."""
    eps = 1.0
    while 1.0 + 0.5 * eps > 1.0:
        eps *= 0.5
    return eps


def matrix_norms(A: np.ndarray) -> dict[str, float]:
    """Calcula las normas matriciales ||A||_1, ||A||_inf y ||A||_F desde su definición."""
    norm_1 = float(np.max(np.sum(np.abs(A), axis=0)))
    norm_inf = float(np.max(np.sum(np.abs(A), axis=1)))
    norm_fro = float(np.sqrt(np.sum(A * A)))
    return {"norm_1": norm_1, "norm_inf": norm_inf, "norm_fro": norm_fro}


def gershgorin_bounds(H: csr_matrix) -> tuple[float, float]:
    """
    Calcula las cotas inferior y superior del espectro de una matriz simétrica dispersa H
    aplicando el Teorema de los Discos de Gershgorin:
        λ ∈ ⋃_i [H_ii - R_i, H_ii + R_i],   con R_i = ∑_{j != i} |H_ij|

    Para el Laplaciano discretizado de Dirichlet, garantiza λ_min >= 0 y λ_max <= 8/h^2.
    """
    H_csr = H.tocsr()
    n = H_csr.shape[0]
    diag = H_csr.diagonal()

    radii = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start, end = H_csr.indptr[i], H_csr.indptr[i + 1]
        row_abs_sum = np.sum(np.abs(H_csr.data[start:end]))
        radii[i] = row_abs_sum - abs(diag[i])

    lower_bound = float(np.min(diag - radii))
    upper_bound = float(np.max(diag + radii))
    return lower_bound, upper_bound