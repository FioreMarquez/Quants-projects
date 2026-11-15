from __future__ import annotations
"""
Definición geométrica y discretización de malla para el cuarto de billar de Sinai:
    Ω = ([0, Lx] x [0, Ly]) \ {(x, y) : x^2 + y^2 <= R^2}
con condiciones de contorno homogéneas de Dirichlet (ψ = 0 en ∂Ω).
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SinaiBilliardMesh:
    """Estructura de datos inmutable con la topología de la malla y fracciones de borde."""

    Lx: float
    Ly: float
    R: float
    h: float
    nx: int
    ny: int
    n_active: int
    # Mapeo bidimensional (i, j) -> índice lineal k in {0, ..., n_active - 1} (-1 si es inactivo/borde)
    node_map: np.ndarray
    # Coordenadas (i, j) y físicas (x_i, y_j) de cada nodo activo k
    ij_coords: np.ndarray
    xy_coords: np.ndarray
    # Fracciones de paso cardinales θ_E, θ_W, θ_N, θ_S in (0, 1] para cada nodo activo
    theta_E: np.ndarray
    theta_W: np.ndarray
    theta_N: np.ndarray
    theta_S: np.ndarray
    # Indicadores booleanos: True si el vecino cardinal es otro nodo activo, False si es ∂Ω
    has_E: np.ndarray
    has_W: np.ndarray
    has_N: np.ndarray
    has_S: np.ndarray


def build_sinai_mesh(
    Lx: float = 1.0,
    Ly: float = 1.0,
    R: float = 0.35,
    nx: int = 80,
    theta_tol: float = 1e-8,
) -> SinaiBilliardMesh:
    """
    Construye la malla cartesiana uniforme de paso h = Lx / nx sobre el cuarto de billar.

    Parámetros
    ----------
    Lx, Ly : float
        Dimensiones del cuarto de rectángulo (se requiere Ly / h entero).
    R : float
        Radio del disco dispersivo centrado en el origen (R = 0 recupera el rectángulo integrable).
    nx : int
        Número de subintervalos en dirección x (h = Lx / nx).
    theta_tol : float
        Umbral mínimo admisible para fracciones de paso θ. Nodos activos a distancia
        menor que theta_tol * h del arco se proyectan sobre ∂Ω para preservar la estabilidad.
    """
    if Lx <= 0 or Ly <= 0:
        raise ValueError("Las dimensiones Lx y Ly deben ser estrictamente positivas.")
    if R < 0 or R >= min(Lx, Ly):
        raise ValueError("El radio R debe satisfacer 0 <= R < min(Lx, Ly).")
    if nx < 4:
        raise ValueError("Se requieren al menos nx >= 4 subintervalos.")

    h = Lx / nx
    ny = int(round(Ly / h))
    if not np.isclose(ny * h, Ly, rtol=0.0, atol=1e-12):
        raise ValueError(f"Ly={Ly} no es múltiplo entero de h={h} (Lx/nx).")

    x = np.linspace(0.0, Lx, nx + 1)
    y = np.linspace(0.0, Ly, ny + 1)

    # Identificación inicial de nodos estrictamente interiores a Ω
    node_map = np.full((nx + 1, ny + 1), -1, dtype=np.int64)
    active_list: list[tuple[int, int]] = []

    for i in range(1, nx):
        xi = x[i]
        for j in range(1, ny):
            yj = y[j]
            if xi * xi + yj * yj > R * R:
                # Verificación preventiva de distancia mínima al arco para evitar θ < theta_tol
                if R > 0.0:
                    if yj < R:
                        x_arc = np.sqrt(max(0.0, R * R - yj * yj))
                        if (xi - x_arc) / h < theta_tol:
                            continue
                    if xi < R:
                        y_arc = np.sqrt(max(0.0, R * R - xi * xi))
                        if (yj - y_arc) / h < theta_tol:
                            continue
                active_list.append((i, j))

    n_active = len(active_list)
    ij_coords = np.array(active_list, dtype=np.int64)
    for k, (i, j) in enumerate(active_list):
        node_map[i, j] = k

    xy_coords = np.column_stack((x[ij_coords[:, 0]], y[ij_coords[:, 1]]))

    theta_E = np.ones(n_active, dtype=np.float64)
    theta_W = np.ones(n_active, dtype=np.float64)
    theta_N = np.ones(n_active, dtype=np.float64)
    theta_S = np.ones(n_active, dtype=np.float64)

    has_E = np.zeros(n_active, dtype=bool)
    has_W = np.zeros(n_active, dtype=bool)
    has_N = np.zeros(n_active, dtype=bool)
    has_S = np.zeros(n_active, dtype=bool)

    for k, (i, j) in enumerate(active_list):
        xi, yj = x[i], y[j]

        # Vecino Este (i + 1, j): siempre más alejado del origen que (i, j)
        if node_map[i + 1, j] != -1:
            has_E[k] = True
            theta_E[k] = 1.0
        else:
            has_E[k] = False
            theta_E[k] = 1.0

        # Vecino Norte (i, j + 1): siempre más alejado del origen que (i, j)
        if node_map[i, j + 1] != -1:
            has_N[k] = True
            theta_N[k] = 1.0
        else:
            has_N[k] = False
            theta_N[k] = 1.0

        # Vecino Oeste (i - 1, j): puede estar activo, en x = 0, o dentro/sobre el arco R
        if node_map[i - 1, j] != -1:
            has_W[k] = True
            theta_W[k] = 1.0
        else:
            has_W[k] = False
            if R > 0.0 and (x[i - 1] ** 2 + yj ** 2 <= R * R):
                x_arc = np.sqrt(max(0.0, R * R - yj * yj))
                theta_W[k] = (xi - x_arc) / h
            else:
                theta_W[k] = 1.0

        # Vecino Sur (i, j - 1): puede estar activo, en y = 0, o dentro/sobre el arco R
        if node_map[i, j - 1] != -1:
            has_S[k] = True
            theta_S[k] = 1.0
        else:
            has_S[k] = False
            if R > 0.0 and (xi ** 2 + y[j - 1] ** 2 <= R * R):
                y_arc = np.sqrt(max(0.0, R * R - xi * xi))
                theta_S[k] = (yj - y_arc) / h
            else:
                theta_S[k] = 1.0

    return SinaiBilliardMesh(
        Lx=Lx,
        Ly=Ly,
        R=R,
        h=h,
        nx=nx,
        ny=ny,
        n_active=n_active,
        node_map=node_map,
        ij_coords=ij_coords,
        xy_coords=xy_coords,
        theta_E=theta_E,
        theta_W=theta_W,
        theta_N=theta_N,
        theta_S=theta_S,
        has_E=has_E,
        has_W=has_W,
        has_N=has_N,
        has_S=has_S,
    )