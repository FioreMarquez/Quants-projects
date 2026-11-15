from __future__ import annotations
"""
Formulación conservativa simétrica de Shortley-Weller y reducción al problema
estándar de autovalores real simétrico H w = λ w con H = D^(-1/2) B D^(-1/2).
"""

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix, diags
from .domain_mesh import SinaiBilliardMesh


@dataclass(frozen=True)
class SymmetricLaplacianSystem:
    """Contiene el operador estándar simétrico H y los factores D^(±1/2) para recuperar ψ."""

    H: csr_matrix          # Operador simétrico definido positivo: H = D^(-1/2) B D^(-1/2)
    B: csr_matrix          # Matriz de rigidez simétrica (B u = λ D u)
    cell_areas: np.ndarray # Diagonal d_p = Δx_p * Δy_p > 0
    inv_sqrt_d: np.ndarray # Vector d_p^(-1/2) para transformar autovectores u = D^(-1/2) w


def assemble_symmetric_laplacian(mesh: SinaiBilliardMesh) -> SymmetricLaplacianSystem:
    """
    Ensambla el operador discreto simétrico H ∈ R^{N x N} mediante volúmenes duales:
        Δx_p = ((θ_E,p + θ_W,p) / 2) h,   Δy_p = ((θ_N,p + θ_S,p) / 2) h,   d_p = Δx_p Δy_p

    En cada cara compartida entre nodos activos p y q se promedia la longitud transversal
    (ℓ_pq = (Δy_p + Δy_q)/2 en horizontal; ℓ_pr = (Δx_p + Δx_r)/2 en vertical), lo que
    garantiza B_pq = B_qp exactamente en aritmética de punto flotante y preserva el orden O(h^2).
    """
    n = mesh.n_active
    h = mesh.h

    dx = 0.5 * (mesh.theta_E + mesh.theta_W) * h
    dy = 0.5 * (mesh.theta_N + mesh.theta_S) * h
    cell_areas = dx * dy

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for k in range(n):
        i, j = mesh.ij_coords[k]

        # Conductancia Este
        if mesh.has_E[k]:
            q = int(mesh.node_map[i + 1, j])
            face_E = 0.5 * (dy[k] + dy[q])
            c_E = face_E / (mesh.theta_E[k] * h)
            rows.append(k)
            cols.append(q)
            vals.append(-c_E)
        else:
            c_E = dy[k] / (mesh.theta_E[k] * h)

        # Conductancia Oeste
        if mesh.has_W[k]:
            q = int(mesh.node_map[i - 1, j])
            face_W = 0.5 * (dy[k] + dy[q])
            c_W = face_W / (mesh.theta_W[k] * h)
            rows.append(k)
            cols.append(q)
            vals.append(-c_W)
        else:
            c_W = dy[k] / (mesh.theta_W[k] * h)

        # Conductancia Norte
        if mesh.has_N[k]:
            q = int(mesh.node_map[i, j + 1])
            face_N = 0.5 * (dx[k] + dx[q])
            c_N = face_N / (mesh.theta_N[k] * h)
            rows.append(k)
            cols.append(q)
            vals.append(-c_N)
        else:
            c_N = dx[k] / (mesh.theta_N[k] * h)

        # Conductancia Sur
        if mesh.has_S[k]:
            q = int(mesh.node_map[i, j - 1])
            face_S = 0.5 * (dx[k] + dx[q])
            c_S = face_S / (mesh.theta_S[k] * h)
            rows.append(k)
            cols.append(q)
            vals.append(-c_S)
        else:
            c_S = dx[k] / (mesh.theta_S[k] * h)

        # Entrada diagonal (suma de flujos salientes de la celda dual k)
        rows.append(k)
        cols.append(k)
        vals.append(c_E + c_W + c_N + c_S)

    B = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)

    inv_sqrt_d = 1.0 / np.sqrt(cell_areas)
    D_inv_sqrt = diags(inv_sqrt_d, offsets=0, format="csr")
    H = (D_inv_sqrt @ B @ D_inv_sqrt).tocsr()

    return SymmetricLaplacianSystem(
        H=H,
        B=B,
        cell_areas=cell_areas,
        inv_sqrt_d=inv_sqrt_d,
    )


def recover_physical_eigenfunction(
    w: np.ndarray,
    system: SymmetricLaplacianSystem,
    mesh: SinaiBilliardMesh,
) -> np.ndarray:
    """
    Transforma un autovector w de H (H w = λ w) a la función de onda física u = D^(-1/2) w,
    la normaliza en norma L^2 discreta (∑ u_k^2 d_k = 1) y la reconstruye sobre la grilla 2D.
    """
    u = system.inv_sqrt_d * w
    l2_norm = np.sqrt(np.sum((u * u) * system.cell_areas))
    if l2_norm > 0.0:
        u = u / l2_norm

    grid_psi = np.zeros((mesh.nx + 1, mesh.ny + 1), dtype=np.float64)
    grid_psi[mesh.ij_coords[:, 0], mesh.ij_coords[:, 1]] = u
    return grid_psi