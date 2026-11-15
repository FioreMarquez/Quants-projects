from __future__ import annotations
"""
Estadísticos de fluctuación espectral y comparación con Teoría de Matrices Aleatorias (GOE):
- Distribuciones teóricas P_Poisson(s) y P_Wigner(s)
- Momentos empíricos (varianza de P(s): 0.286 en GOE vs. 1.0 en Poisson)
- Rigidez espectral Δ_3(L) de Dyson-Mehta
- Generación y diagonalización de matrices GOE usando Householder + iteración QR propia
"""

import numpy as np
from src.eigenvalue_solvers.dense_eigensolvers import autovalores_qr_denso


def poisson_spacing_pdf(s: np.ndarray) -> np.ndarray:
    """Densidad teórica de primeros vecinos para sistemas integrables: P(s) = exp(-s)."""
    return np.exp(-s)


def wigner_surmise_pdf(s: np.ndarray) -> np.ndarray:
    """Conjetura de Wigner para el ensamble GOE: P(s) = (π/2) s exp(-π s^2 / 4)."""
    return 0.5 * np.pi * s * np.exp(-0.25 * np.pi * (s ** 2))


def spacing_statistics(spacings: np.ndarray, n_bins: int = 20, s_max: float = 3.5) -> dict:
    """
    Calcula la media, varianza empírica e histograma normalizado de P(s),
    junto con el error cuadrático medio (RMSE) respecto de Poisson y Wigner-GOE.
    """
    s = np.asarray(spacings, dtype=np.float64)
    mean_val = float(np.mean(s))
    var_val = float(np.var(s, ddof=1))

    hist_density, bin_edges = np.histogram(s, bins=n_bins, range=(0.0, s_max), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    rmse_poisson = float(np.sqrt(np.mean((hist_density - poisson_spacing_pdf(bin_centers)) ** 2)))
    rmse_goe = float(np.sqrt(np.mean((hist_density - wigner_surmise_pdf(bin_centers)) ** 2)))

    return {
        "mean": mean_val,
        "variance": var_val,
        "bin_centers": bin_centers,
        "hist_density": hist_density,
        "rmse_poisson": rmse_poisson,
        "rmse_goe": rmse_goe,
    }


def _delta3_single_interval(levels_in_window: np.ndarray, alpha: float, L: float) -> float:
    """
    Evalúa exactamente la desviación cuadrática media mínima
        min_{a, b} (1/L) ∫_α^{α+L} [N(ε) - (a ε + b)]^2 dε
    aprovechando que N(ε) es constante a trozos con saltos unitarios en ε_i ∈ [α, α+L].
    """
    n = levels_in_window.shape[0]
    if n == 0:
        return 0.0

    # Coordenadas centradas x_i = ε_i - (α + L/2) ∈ [-L/2, L/2]
    x = levels_in_window - (alpha + 0.5 * L)
    sum_x = float(np.sum(x))
    sum_x2 = float(np.sum(x * x))
    indices = np.arange(1, n + 1, dtype=np.float64)
    weighted_sum = float(np.sum((n - 2.0 * indices + 1.0) * x))

    val = (
        (n * n) / 16.0
        - (sum_x ** 2) / (L * L)
        + (1.5 * n * sum_x2) / (L * L)
        - (3.0 * (sum_x2 ** 2)) / (L ** 4)
        + weighted_sum / L
    )
    return max(0.0, val)


def spectral_rigidity_delta3(
    unfolded_levels: np.ndarray,
    L_values: np.ndarray,
    n_windows: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula la rigidez espectral promedio Δ_3(L) sobre `n_windows` intervalos solapados
    [α, α + L] dentro del espectro desplegado, y devuelve además las curvas teóricas
    de Poisson (L / 15) y asintótica GOE ((1/π^2) ln(L) - 0.0069).
    """
    eps = np.sort(np.asarray(unfolded_levels, dtype=np.float64))
    L_arr = np.asarray(L_values, dtype=np.float64)
    delta3_emp = np.zeros_like(L_arr)

    eps_min, eps_max = float(eps[0]), float(eps[-1])
    span = eps_max - eps_min

    for idx, L in enumerate(L_arr):
        if L >= 0.8 * span:
            raise ValueError(f"La longitud de ventana L={L} es demasiado grande para el espectro disponible.")

        alphas = np.linspace(eps_min, eps_max - L, n_windows)
        vals = np.empty(n_windows, dtype=np.float64)
        for w_i, alpha in enumerate(alphas):
            left = int(np.searchsorted(eps, alpha, side="left"))
            right = int(np.searchsorted(eps, alpha + L, side="right"))
            vals[w_i] = _delta3_single_interval(eps[left:right], alpha, L)

        delta3_emp[idx] = float(np.mean(vals))

    delta3_poisson = L_arr / 15.0
    delta3_goe_asymp = (1.0 / (np.pi ** 2)) * np.log(np.maximum(L_arr, 1e-12)) - 0.0069
    return delta3_emp, delta3_poisson, delta3_goe_asymp


def sample_goe_spacings(
    matrix_size: int = 80,
    n_realizations: int = 15,
    seed: int = 2026,
) -> np.ndarray:
    """
    Genera `n_realizations` matrices reales simétricas del ensamble GOE de tamaño N x N:
        M = (G + G^T) / sqrt(2 N),   G_ij ~ N(0, 1)
    Calcula sus autovalores usando nuestro solver denso de Householder + iteración QR,
    despliega el centro del espectro mediante la ley del semicírculo de Wigner y
    retorna los espaciamientos normalizados s_n.
    """
    rng = np.random.default_rng(seed)
    all_spacings: list[np.ndarray] = []

    # Descartamos el 25% de cada extremo para evitar efectos de borde del semicírculo
    lo = int(0.25 * matrix_size)
    hi = int(0.75 * matrix_size)

    for _ in range(n_realizations):
        G = rng.standard_normal((matrix_size, matrix_size))
        M_goe = (G + G.T) / np.sqrt(2.0 * matrix_size)

        eigs = autovalores_qr_denso(M_goe, tol=1e-12)
        bulk = eigs[lo:hi]

        # Función acumulada exacta del semicírculo de Wigner en [-2, 2]:
        # N_sc(E) = (N / π) * ( (E/2)*sqrt(1 - (E/2)^2) + arcsin(E/2) + π/2 )
        x = np.clip(0.5 * bulk, -1.0, 1.0)
        unfolded_bulk = (matrix_size / np.pi) * (x * np.sqrt(1.0 - x * x) + np.arcsin(x) + 0.5 * np.pi)

        s = np.diff(unfolded_bulk)
        s = s / np.mean(s)
        all_spacings.append(s)

    return np.concatenate(all_spacings)