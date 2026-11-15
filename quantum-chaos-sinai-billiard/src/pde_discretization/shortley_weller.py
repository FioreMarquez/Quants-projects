from __future__ import annotations
"""
Ensamblaje disperso del Laplaciano de Dirichlet (-∇²) mediante diferencias finitas
de 5 puntos con corrección de Shortley-Weller en fronteras curvas.
"""

import numpy as np
from scipy.sparse import csr_matrix
from .domain_mesh import SinaiBilliardMesh


def assemble_shortley_weller_nonsymmetric(mesh: SinaiBilliardMesh) -> csr_matrix:
    """
    Construye la matriz dispersa A ∈ R^{N x N} del esquema de Shortley-Weller directo:

        (Au)_p = (2 / h^2) * [ (1/(θ_E θ_W) + 1/(θ_N θ_S)) u_p
                             - u_E / (θ_E (θ_E + θ_W)) - u_W / (θ_W (θ_E + θ_W))
                             - u_N / (θ_N (θ_N + θ_S)) - u_S / (θ_S (θ_N + θ_S)) ]

    Nota numérica: En nodos adyacentes al arco curvo (θ_W < 1 o θ_S < 1), A resulta
    levemente asimétrica (A != A^T). Para algoritmos espectrales simétricos (Lanczos,
    Cholesky/LDL^T) debe utilizarse la formulación conservativa en `symmetrization.py`.
    """
    n = mesh.n_active
    inv_h2 = 1.0 / (mesh.h * mesh.h)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for k in range(n):
        i, j = mesh.ij_coords[k]
        tE, tW = mesh.theta_E[k], mesh.theta_W[k]
        tN, tS = mesh.theta_N[k], mesh.theta_S[k]

        diag = 2.0 * inv_h2 * (1.0 / (tE * tW) + 1.0 / (tN * tS))
        rows.append(k)
        cols.append(k)
        vals.append(diag)

        if mesh.has_E[k]:
            q = int(mesh.node_map[i + 1, j])
            rows.append(k)
            cols.append(q)
            vals.append(-2.0 * inv_h2 / (tE * (tE + tW)))

        if mesh.has_W[k]:
            q = int(mesh.node_map[i - 1, j])
            rows.append(k)
            cols.append(q)
            vals.append(-2.0 * inv_h2 / (tW * (tE + tW)))

        if mesh.has_N[k]:
            q = int(mesh.node_map[i, j + 1])
            rows.append(k)
            cols.append(q)
            vals.append(-2.0 * inv_h2 / (tN * (tN + tS)))

        if mesh.has_S[k]:
            q = int(mesh.node_map[i, j - 1])
            rows.append(k)
            cols.append(q)
            vals.append(-2.0 * inv_h2 / (tS * (tN + tS)))

    return csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)


def assemble_staircase_laplacian(mesh: SinaiBilliardMesh) -> csr_matrix:
    """
    Construye el Laplaciano discreto de 5 puntos sin corrección de frontera (θ ≡ 1).
    Sirve como caso de control para demostrar la degradación de convergencia de O(h^2)
    a O(h) causada por la aproximación escalonada del arco circular.
    """
    n = mesh.n_active
    inv_h2 = 1.0 / (mesh.h * mesh.h)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for k in range(n):
        i, j = mesh.ij_coords[k]
        rows.append(k)
        cols.append(k)
        vals.append(4.0 * inv_h2)

        for is_active, (ni, nj) in (
            (mesh.has_E[k], (i + 1, j)),
            (mesh.has_W[k], (i - 1, j)),
            (mesh.has_N[k], (i, j + 1)),
            (mesh.has_S[k], (i, j - 1)),
        ):
            if is_active:
                rows.append(k)
                cols.append(int(mesh.node_map[ni, nj]))
                vals.append(-inv_h2)

    return csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)