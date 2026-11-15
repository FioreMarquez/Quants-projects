from __future__ import annotations
"""
Despliegue espectral (Unfolding) mediante la fórmula asintótica de Weyl en 2D:
    N_bar(E) = c_1 * E + c_2 * sqrt(E) + c_3
ajustada por mínimos cuadrados lineales mediante factorización QR de Householder
(sin formar las ecuaciones normales X^T X).
"""

from dataclasses import dataclass
import numpy as np
from src.qr_least_squares.qr_factorization import resolver_minimos_cuadrados_qr


@dataclass(frozen=True)
class UnfoldedSpectrum:
    """Resultado del despliegue espectral y espaciamientos normalizados."""

    raw_eigenvalues: np.ndarray      # Niveles ordenados E_1 <= E_2 <= ... <= E_M
    unfolded_levels: np.ndarray      # Niveles desplegados ε_n = N_bar(E_n)
    spacings: np.ndarray             # Espaciamientos normalizados s_n = ε_{n+1} - ε_n (<s > = 1)
    weyl_coeffs: np.ndarray          # Coeficientes ajustados [c_1, c_2, c_3]
    theoretical_coeffs: np.ndarray   # Coeficientes geométricos teóricos [A/(4π), -L/(4π)]


def theoretical_weyl_coefficients(Lx: float, Ly: float, R: float) -> np.ndarray:
    """
    Calcula los coeficientes asintóticos exactos de área y perímetro de la ley de Weyl
    para el cuarto de billar de Sinai Ω = ([0, Lx] x [0, Ly]) \ {x^2 + y^2 <= R^2}:
        c_1 = Área / (4π),    c_2 = -Perímetro / (4π)
    """
    area = Lx * Ly - 0.25 * np.pi * (R ** 2)
    perimeter = 2.0 * (Lx + Ly) - 2.0 * R + 0.5 * np.pi * R
    return np.array([area / (4.0 * np.pi), -perimeter / (4.0 * np.pi)], dtype=np.float64)


def unfold_spectrum_weyl(
    eigenvalues: np.ndarray,
    Lx: float = 1.0,
    Ly: float = 1.0,
    R: float = 0.35,
    index_offset: int = 0,
) -> UnfoldedSpectrum:
    """
    Realiza el despliegue espectral (unfolding) sobre un bloque ordenado de autovalores
    {E_1, ..., E_M} calculados en una ventana espectral.

    Parámetros
    ----------
    eigenvalues : np.ndarray
        Autovalores positivos ordenados de menor a mayor.
    Lx, Ly, R : float
        Parámetros geométricos del cuarto de billar (para comparar contra los coeficientes
        teóricos de Weyl).
    index_offset : int
        Índice acumulado previo si los autovalores provienen de una ventana interior
        (no afecta los espaciamientos s_n = ε_{n+1} - ε_n, ya que traslada c_3).
    """
    E = np.sort(np.asarray(eigenvalues, dtype=np.float64))
    if E.ndim != 1 or E.shape[0] < 10:
        raise ValueError("Se requieren al menos 10 autovalores ordenados para el unfolding.")
    if np.any(E <= 0.0):
        raise ValueError("Todos los autovalores del Laplaciano de Dirichlet deben ser > 0.")

    m = E.shape[0]
    # Función escalera empírica evaluada en el punto medio de cada salto: n - 0.5
    staircase_y = index_offset + np.arange(1, m + 1, dtype=np.float64) - 0.5

    # Matriz de diseño X = [E, sqrt(E), 1] (columnas con escalas muy dispares -> ideal para QR)
    X = np.column_stack((E, np.sqrt(E), np.ones(m, dtype=np.float64)))

    # Ajuste por mínimos cuadrados vía QR de Householder
    coeffs = resolver_minimos_cuadrados_qr(X, staircase_y)
    unfolded = X @ coeffs

    # Espaciamientos entre primeros vecinos y renormalización exacta a media unitaria
    raw_spacings = np.diff(unfolded)
    mean_s = float(np.mean(raw_spacings))
    if mean_s <= 0.0:
        raise ValueError("La tendencia ajustada de Weyl no es monótona creciente en este rango.")
    spacings = raw_spacings / mean_s
    unfolded = unfolded / mean_s

    theo_c = theoretical_weyl_coefficients(Lx, Ly, R)
    return UnfoldedSpectrum(
        raw_eigenvalues=E,
        unfolded_levels=unfolded,
        spacings=spacings,
        weyl_coeffs=coeffs,
        theoretical_coeffs=theo_c,
    )