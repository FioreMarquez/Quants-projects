from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from fem import assemble_fem_matrices, compute_greeks, create_spatial_grid
from penalty import compute_american_error, solve_penalty_step


def solve_explicit_step(
    M_lumped: sp.csr_matrix,
    K: sp.csr_matrix,
    V_n: np.ndarray,
    V_star: np.ndarray,
    dt: float,
    theta: float = 0.5,
    dirichlet_right: float = 0.0,
) -> np.ndarray:
    """Resuelve un paso temporal aplicando la restriccion americana de forma explicita.

    Implementa el Algoritmo (5.3) de Forsyth & Vetzal (2002) como linea base
    de comparacion (primer orden temporal O(dt) y oscilaciones en Gamma):
        [M + (1 - theta)*dt*K] V_hat = [M - theta*dt*K] V^n
        V^{n+1} = max(V_hat, V*)
    """
    A = (M_lumped + (1.0 - theta) * dt * K).tolil()
    rhs = (M_lumped - theta * dt * K) @ V_n

    # Condicion de borde de Dirichlet en S = S_max
    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    rhs[-1] = dirichlet_right

    V_hat = spsolve(A.tocsr(), rhs)
    return np.maximum(V_hat, V_star)


def solve_american_option_pde(
    S: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    adaptive: bool = True,
    n_steps: int = 100,
    dt_init: float = 1e-3,
    dnorm: float = 0.2,
    D_scale: float = 1.0,
    rannacher: bool = True,
    method: str = "penalty",
    large: float = 1e6,
    tol: float = 1e-6,
    use_mass_lumping: bool = True,
) -> dict[str, np.ndarray | float | int]:
    """Integra en el tiempo la EDP de Black-Scholes para una opcion Put americana.

    Combina Elementos Finitos P1 (con mass lumping), suavizado de Rannacher
    (2 pasos iniciales implicitos puros theta=0 seguidos de Crank-Nicolson
    theta=0.5) y selector adaptativo de paso temporal para lograr convergencia
    cuadratica O(h^2 + dtau^2).

    Parameters
    ----------
    S : np.ndarray
        Malla espacial ordenada en [0, S_max].
    K : float
        Precio de ejercicio (strike).
    T : float
        Tiempo hasta el vencimiento (en anos).
    r : float
        Tasa libre de riesgo anualizada.
    sigma : float
        Volatilidad del subyacente.
    adaptive : bool, optional
        Si es True, utiliza el selector adaptativo de paso temporal (Ec. 10.1).
        Si es False, usa pasos constantes dt = T / n_steps.
    n_steps : int, optional
        Numero de pasos temporales cuando adaptive=False.
    dt_init : float, optional
        Tamano del paso inicial Delta tau^0 cuando adaptive=True (default: 1e-3).
    dnorm : float, optional
        Cambio relativo objetivo por paso para el selector adaptativo (default: 0.2).
    D_scale : float, optional
        Escala de normalizacion D en el denominador de Ec. 10.1 (default: 1.0).
    rannacher : bool, optional
        Activa los 2 pasos iniciales implicitos puros (theta=0) para amortiguar
        la discontinuidad de la derivada del payoff en S=K (default: True).
    method : str, optional
        'penalty' para el metodo implicito penalizado (Newton no suave), o
        'explicit' para la proyeccion explicita del Algoritmo 5.3.
    large : float, optional
        Factor de penalizacion (default: 1e6). Fijar en 0.0 para Put europeo.
    tol : float, optional
        Tolerancia relativa de convergencia para Newton (default: 1e-6).
    use_mass_lumping : bool, optional
        Activa la condensacion de masa en FEM P1 (default: True).

    Returns
    -------
    dict
        Diccionario con la malla 'S', solucion 'V', griegas ('delta', 'gamma'),
        precio interpolado en el strike 'price_at_K', cantidad de 'steps',
        'total_iters' de Newton, 'total_flops' estimados y 'max_amer_err'.
    """
    if method not in {"penalty", "explicit"}:
        raise ValueError("method debe ser 'penalty' o 'explicit'.")

    M, K_mat = assemble_fem_matrices(S, r, sigma, use_mass_lumping=use_mass_lumping)
    V_star = np.maximum(K - S, 0.0)
    V = V_star.copy()
    n_nodes = len(S)

    tau = 0.0
    step = 0
    total_iters = 0
    total_flops = 0.0
    max_amer_err = 0.0
    dt_history = []

    dt = dt_init if adaptive else (T / n_steps)

    while tau < T - 1e-14:
        # Evitar sobrepasar el horizonte T en el ultimo paso
        if tau + dt > T:
            dt = T - tau

        # Suavizado de Rannacher: primeros 2 pasos con Implicito Puro (theta=0),
        # resto con Crank-Nicolson (theta=0.5) (Seccion 7 del paper)
        theta = 0.0 if (rannacher and step < 2) else 0.5

        if method == "penalty":
            V_next, iters = solve_penalty_step(
                M, K_mat, V, V_star, dt=dt, theta=theta, large=large, tol=tol
            )
            total_iters += iters
            # Conteo de operaciones de punto flotante (Seccion 5, pag. 2103):
            # 12*I (adaptativo) o 10*I (constante) en la 1ra iteracion + 7*I por extra
            first_iter_flops = 12 * n_nodes if adaptive else 10 * n_nodes
            total_flops += first_iter_flops + 7 * n_nodes * max(0, iters - 1)
        else:
            V_next = solve_explicit_step(M, K_mat, V, V_star, dt=dt, theta=theta)
            total_iters += 1
            # Conteo explicito (Seccion 5, pag. 2104): 10*I adaptativo, 6*I constante
            total_flops += (10 * n_nodes) if adaptive else (6 * n_nodes)

        # Auditoria a posteriori de la restriccion americana (Ec. 4.20)
        step_amer_err = compute_american_error(V_next, V_star)
        max_amer_err = max(max_amer_err, step_amer_err)

        tau += dt
        step += 1
        dt_history.append(dt)

        # Selector adaptativo de paso temporal (Ecuacion 10.1, pag. 2110)
        if adaptive and tau < T - 1e-14:
            denom = np.maximum(D_scale, np.maximum(np.abs(V_next), np.abs(V)))
            rel_change = np.max(np.abs(V_next - V) / denom)
            if rel_change > 1e-16:
                dt = (dnorm / rel_change) * dt

        V = V_next

    greeks = compute_greeks(S, V)

    return {
        "S": S,
        "V": V,
        "delta": greeks["delta"],
        "gamma": greeks["gamma"],
        "price_at_K": float(np.interp(K, S, V)),
        "steps": step,
        "total_iters": total_iters,
        "total_flops": total_flops,
        "max_amer_err": max_amer_err,
        "dt_history": np.array(dt_history),
    }


if __name__ == "__main__":
    # Reproduccion del estudio de convergencia cuadratica (Tabla 4 del paper)
    # Parametros: Put Americano, K=100, T=0.25, r=0.10, sigma=0.20, S_max=200
    K_strike = 100.0
    T_exp = 0.25
    r_rate = 0.10
    vol = 0.20
    S_max = 200.0

    print("=== Estudio de Convergencia: Metodo Penalizado FEM P1 + Timestep Adaptativo ===")
    print(f"{'Nodos':>6} | {'Pasos':>5} | {'Iters':>5} | {'Valor(K)':>10} | {'Cambio':>9} | {'Ratio':>6} | {'Flops':>9}")
    print("-" * 68)

    prev_val = None
    prev_diff = None

    for level, n_nodes in enumerate([55, 109, 217, 433, 865]):
        S_grid = create_spatial_grid(K_strike, S_max, n_nodes, concentrate_around_strike=True)
        # En cada refinamiento se divide dt_init por 4 y dnorm por 2 (Seccion 10)
        dt0 = 1e-3 / (4**level)
        dn = 0.2 / (2**level)

        res = solve_american_option_pde(
            S=S_grid,
            K=K_strike,
            T=T_exp,
            r=r_rate,
            sigma=vol,
            adaptive=True,
            dt_init=dt0,
            dnorm=dn,
            rannacher=True,
            method="penalty",
        )

        val = res["price_at_K"]
        diff = abs(val - prev_val) if prev_val is not None else np.nan
        ratio = (prev_diff / diff) if (prev_diff is not None and diff > 0) else np.nan

        diff_str = f"{diff:.5f}" if not np.isnan(diff) else "-"
        ratio_str = f"{ratio:.2f}" if not np.isnan(ratio) else "-"
        print(
            f"{n_nodes:6d} | {res['steps']:5d} | {res['total_iters']:5d} | "
            f"{val:10.5f} | {diff_str:>9} | {ratio_str:>6} | {res['total_flops']:9.2e}"
        )

        prev_val = val
        prev_diff = diff