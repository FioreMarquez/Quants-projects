from __future__ import annotations


import numpy as np
import scipy.sparse as sp


def create_spatial_grid(
    K: float,
    S_max: float,
    n_nodes: int,
    concentrate_around_strike: bool = False,
) -> np.ndarray:
    """Construye la malla espacial 1D sobre [0, S_max] garantizando un nodo en S = K.

    Incluir exactamente el strike K como nodo de la malla evita errores de
    interpolacion en la condicion inicial no suave V*(S) = max(K - S, 0)
    (Forsyth & Vetzal, 2002, Seccion 7).

    Parameters
    ----------
    K : float
        Precio de ejercicio (strike) de la opcion.
    S_max : float
        Frontera superior truncada del dominio espacial.
    n_nodes : int
        Numero total de nodos de la malla (debe ser >= 3).
    concentrate_around_strike : bool, optional
        Si es True, genera una malla no uniforme mas fina cerca de S = K.

    Returns
    -------
    np.ndarray
        Vector ordenado de nodos espaciales S_0, S_1, ..., S_{N}.
    """
    if n_nodes < 3:
        raise ValueError("La malla requiere al menos 3 nodos.")

    if not concentrate_around_strike:
        # Malla uniforme ajustada para que K sea exactamente un nodo si es divisor
        S = np.linspace(0.0, S_max, n_nodes)
        idx_closest = int(np.argmin(np.abs(S - K)))
        S[idx_closest] = K
        return S

    # Malla graduada (sinh-grid) concentrada alrededor del strike K
    xi = np.linspace(-1.0, 1.0, n_nodes)
    alpha = 0.25  # Factor de concentracion
    c1 = np.arcsinh(-K / alpha)
    c2 = np.arcsinh((S_max - K) / alpha)
    S = K + alpha * np.sinh(c1 * (1.0 - (xi + 1.0) / 2.0) + c2 * ((xi + 1.0) / 2.0))
    S[0] = 0.0
    S[-1] = S_max
    idx_closest = int(np.argmin(np.abs(S - K)))
    S[idx_closest] = K
    return S


def assemble_fem_matrices(
    S: np.ndarray,
    r: float,
    sigma: float,
    use_mass_lumping: bool = True,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Ensambla las matrices globales M y K mediante Elementos Finitos P1 en 1D.

    Corresponde a la formulacion variacional de la EDP de Black-Scholes en
    tiempo hacia atras tau = T - t:
        M * dV/dtau + K * V = M * P(V) * (V* - V)

    Parameters
    ----------
    S : np.ndarray
        Vector de nodos espaciales ordenados en [0, S_max].
    r : float
        Tasa libre de riesgo anualizada.
    sigma : float
        Volatilidad del activo subyacente.
    use_mass_lumping : bool, optional
        Si es True, diagonaliza la matriz de masa sumando sus filas sobre la
        diagonal para preservar la estructura de M-matriz (default: True).

    Returns
    -------
    tuple[sp.csr_matrix, sp.csr_matrix]
        - M_mat: Matriz global de masa (diagonal si use_mass_lumping=True).
        - K_mat: Matriz global tridiagonal de rigidez, adveccion y reaccion.
    """
    n_nodes = len(S)
    n_elem = n_nodes - 1

    M_lil = sp.lil_matrix((n_nodes, n_nodes), dtype=float)
    K_lil = sp.lil_matrix((n_nodes, n_nodes), dtype=float)

    for e in range(n_elem):
        s0, s1 = S[e], S[e + 1]
        h = s1 - s0

        # 1. Matriz de masa elemental M^{(e)} (2x2)
        M_e = (h / 6.0) * np.array([
            [2.0, 1.0],
            [1.0, 2.0],
        ])

        # 2. Matriz de difusion elemental K_diff^{(e)} (integracion exacta de S^2)
        # Se usa (s1^2 + s1*s0 + s0^2) para evitar cancelacion numerica cuando h -> 0
        diff_coeff = (sigma**2) * (s1**2 + s1 * s0 + s0**2) / (6.0 * h)
        K_diff = diff_coeff * np.array([
            [ 1.0, -1.0],
            [-1.0,  1.0],
        ])

        # 3. Matriz de adveccion efectiva K_adv^{(e)} con coeficiente (sigma^2 - r)*S
        adv_coeff = (sigma**2 - r) / 6.0
        K_adv = adv_coeff * np.array([
            [-(2.0 * s0 + s1), +(2.0 * s0 + s1)],
            [-(s0 + 2.0 * s1), +(s0 + 2.0 * s1)],
        ])

        # 4. Matriz de reaccion elemental K_reac^{(e)} = r * M^{(e)}
        K_reac = r * M_e

        K_e = K_diff + K_adv + K_reac

        # Ensamblaje en las matrices globales
        for i_loc in range(2):
            for j_loc in range(2):
                M_lil[e + i_loc, e + j_loc] += M_e[i_loc, j_loc]
                K_lil[e + i_loc, e + j_loc] += K_e[i_loc, j_loc]

    # Condensacion de masa (Mass Lumping)
    if use_mass_lumping:
        row_sums = np.array(M_lil.sum(axis=1)).flatten()
        M_mat = sp.diags(row_sums, format="csr")
    else:
        M_lil[0, :] = 0.0
        M_lil[0, 0] = 0.5 * (S[1] - S[0])
        M_mat = M_lil.tocsr()

    # Condicion de borde en S = 0 (Ec. 2.6): V_tau + r*V = 0
    K_lil[0, :] = 0.0
    K_lil[0, 0] = r * M_mat[0, 0]

    return M_mat, K_lil.tocsr()


def check_m_matrix(
    M: sp.csr_matrix,
    K: sp.csr_matrix,
    dt: float,
    theta: float = 0.5,
) -> dict[str, bool | float]:
    """Verifica si el operador implicito A = M + (1 - theta)*dt*K es una M-matriz.

    Comprueba las condiciones suficientes del Teorema 6.1 de Forsyth & Vetzal:
    diagonales positivas, elementos extradiagonales no positivos y dominancia
    diagonal estricta por filas.
    """
    A = (M + (1.0 - theta) * dt * K).tocsr()
    # Excluir la ultima fila (donde se impone Dirichlet V(S_max) = 0)
    A_int = A[:-1, :-1].toarray()

    diag = np.diag(A_int)
    off_diag = A_int - np.diag(diag)

    positive_diag = bool(np.all(diag > 0.0))
    nonpositive_off_diag = bool(np.all(off_diag <= 1e-14))
    row_dominance = diag - np.sum(np.abs(off_diag), axis=1)
    strictly_diag_dominant = bool(np.all(row_dominance > 0.0))

    return {
        "is_m_matrix": positive_diag and nonpositive_off_diag and strictly_diag_dominant,
        "positive_diag": positive_diag,
        "nonpositive_off_diag": nonpositive_off_diag,
        "strictly_diag_dominant": strictly_diag_dominant,
        "max_off_diag": float(np.max(off_diag)),
    }


def compute_greeks(S: np.ndarray, V: np.ndarray) -> dict[str, np.ndarray]:
    """Calcula Delta (V_S) y Gamma (V_SS) sobre mallas uniformes o no uniformes.

    Utiliza diferencias finitas centradas de segundo orden adaptadas a pasos
    espaciales variables h_i = S_{i+1} - S_i.
    """
    delta = np.zeros_like(V)
    gamma = np.zeros_like(V)

    # Nodos interiores (1 hasta N-1)
    h_minus = S[1:-1] - S[:-2]
    h_plus = S[2:] - S[1:-1]

    delta[1:-1] = (
        -h_plus / (h_minus * (h_minus + h_plus)) * V[:-2]
        + (h_plus - h_minus) / (h_minus * h_plus) * V[1:-1]
        + h_minus / (h_plus * (h_minus + h_plus)) * V[2:]
    )

    gamma[1:-1] = 2.0 * (
        V[:-2] / (h_minus * (h_minus + h_plus))
        - V[1:-1] / (h_minus * h_plus)
        + V[2:] / (h_plus * (h_minus + h_plus))
    )

    # Extremos mediante diferencias laterales de primer orden
    delta[0] = (V[1] - V[0]) / (S[1] - S[0])
    delta[-1] = (V[-1] - V[-2]) / (S[-1] - S[-2])
    gamma[0] = gamma[1]
    gamma[-1] = gamma[-2]

    return {"delta": delta, "gamma": gamma}


if __name__ == "__main__":
    # Verificacion de ensamblado, propiedad M-matriz y convergencia europea
    from penalty import solve_penalty_step

    K_strike = 100.0
    T = 0.25
    r = 0.10
    sigma = 0.80
    S_max = 1000.0

    S = create_spatial_grid(K_strike, S_max, n_nodes=539, concentrate_around_strike=False)
    M, K_mat = assemble_fem_matrices(S, r, sigma, use_mass_lumping=True)

    dt = T / 200
    status = check_m_matrix(M, K_mat, dt=dt, theta=0.5)
    print("--- Diagnostico de Ensamblado FEM P1 ---")
    print(f"Propiedad M-matriz (theta=0.5): {status}")

    # Prueba con penalizacion desactivada (large=0.0 -> Put Europeo, Tabla 1 del paper)
    V_star = np.maximum(K_strike - S, 0.0)
    V = V_star.copy()
    for step in range(200):
        theta = 0.0 if step < 2 else 0.5  # Rannacher smoothing
        V, _ = solve_penalty_step(M, K_mat, V, V_star, dt=dt, theta=theta, large=0.0)

    val_at_K = float(np.interp(K_strike, S, V))
    print(f"Put Europeo FEM (N=539, steps=200): {val_at_K:.5f} (Ref. Tabla 1: 14.45138)")