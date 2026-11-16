from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def solve_penalty_step(
    M_lumped: sp.csr_matrix,
    K: sp.csr_matrix,
    V_n: np.ndarray,
    V_star: np.ndarray,
    dt: float,
    theta: float = 0.5,
    large: float = 1e6,
    tol: float = 1e-6,
    max_iter: int = 50,
    dirichlet_right: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Resuelve un paso temporal de la EDP penalizada mediante Newton no suave.

    Implementa el algoritmo (5.2) de Forsyth & Vetzal (2002) adaptado a Elementos
    Finitos con condensacion de masa (mass lumping):
        [M + (1 - theta)*dt*K + M*P^k] V^{k+1} = [M - theta*dt*K] V^n + M*P^k V*

    Parameters
    ----------
    M_lumped : sp.csr_matrix
        Matriz diagonal de masa condensada (n_nodes x n_nodes).
    K : sp.csr_matrix
        Matriz global de rigidez, adveccion y reaccion (n_nodes x n_nodes).
    V_n : np.ndarray
        Vector solucion en el nivel temporal actual tau^n.
    V_star : np.ndarray
        Vector del payoff intrinseco V*(S) = max(K - S, 0).
    dt : float
        Tamano del paso temporal Delta tau^{n+1}.
    theta : float, optional
        Parametro de ponderacion temporal (0.0: Implicito Puro, 0.5: Crank-Nicolson).
    large : float, optional
        Factor de penalizacion (default: 1e6). Fijar en 0.0 para opcion europea.
    tol : float, optional
        Tolerancia relativa para el criterio de parada de Newton (default: 1e-6).
    max_iter : int, optional
        Numero maximo de iteraciones permitidas por paso temporal.
    dirichlet_right : float, optional
        Condicion de contorno de Dirichlet en S = S_max (0.0 para un Put).

    Returns
    -------
    tuple[np.ndarray, int]
        - V_next: Vector solucion convergido en tau^{n+1}.
        - n_iter: Cantidad de iteraciones de Newton realizadas.
    """
    # Matriz base del operador implicito sin penalizacion: A_base = M + (1 - theta)*dt*K
    A_base = (M_lumped + (1.0 - theta) * dt * K).tolil()

    # Lado derecho base (constante durante todas las iteraciones k del paso temporal)
    rhs_base = (M_lumped - theta * dt * K) @ V_n

    # Diagonal de la matriz de masa condensada para escalar la penalizacion
    m_diag = M_lumped.diagonal()

    # Estimacion inicial de Newton: solucion del paso temporal previo (V^0 = V^n)
    V_k = V_n.copy()
    active_prev = V_k < V_star

    for k in range(1, max_iter + 1):
        # Construccion de la matriz diagonal de penalizacion M * P^k
        penalty_diag = np.where(active_prev, m_diag * large, 0.0)
        P_mat = sp.diags(penalty_diag, format="lil")

        # Ensamblado del sistema lineal tridiagonal de la iteracion k
        A_k = A_base + P_mat
        rhs_k = rhs_base + penalty_diag * V_star

        # Imposicion de condicion de borde de Dirichlet en S = S_max (ultimo nodo)
        A_k[-1, :] = 0.0
        A_k[-1, -1] = 1.0
        rhs_k[-1] = dirichlet_right

        # Resolucion del sistema disperso tridiagonal
        V_next = spsolve(A_k.tocsr(), rhs_k)

        # Evaluacion del nuevo conjunto activo de ejercicio anticipado
        active_curr = V_next < V_star

        # Criterio de parada dual (Ecuacion 5.2 de Forsyth & Vetzal):
        # 1) Cambio relativo menor a tol, o 2) Sin cambios en los nodos penalizados
        rel_diff = np.max(
            np.abs(V_next - V_k) / np.maximum(1.0, np.abs(V_next))
        )
        if rel_diff < tol or np.array_equal(active_curr, active_prev):
            return V_next, k

        V_k = V_next
        active_prev = active_curr

    raise RuntimeError(
        f"La iteracion de Newton no convergio tras {max_iter} iteraciones."
    )


def compute_american_error(V: np.ndarray, V_star: np.ndarray) -> float:
    """Calcula el error relativo maximo en la restriccion americana (Ec. 4.20).

    Mide cuanto cae la solucion por debajo del payoff debido a la aproximacion
    por penalizacion finita. Debe ser del orden de O(1 / Large).
    """
    violation = np.maximum(0.0, V_star - V)
    return float(np.max(violation / np.maximum(1.0, V_star)))


if __name__ == "__main__":
    # Prueba unitaria rapida de convergencia monotona y terminacion finita
    S = np.linspace(0.0, 200.0, 109)
    K_strike = 100.0
    r = 0.10
    sigma = 0.20
    dt = 0.01

    # Ensamblado basico FEM P1 con mass lumping para verificacion
    N = len(S) - 1
    M_lil = sp.lil_matrix((N + 1, N + 1))
    K_lil = sp.lil_matrix((N + 1, N + 1))

    for e in range(N):
        h = S[e + 1] - S[e]
        s_mid = 0.5 * (S[e] + S[e + 1])
        M_e = (h / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
        K_diff = 0.5 * (sigma**2) * (s_mid**2) / h * np.array([[1.0, -1.0], [-1.0, 1.0]])
        K_adv = (sigma**2 - r) * s_mid * 0.5 * np.array([[-1.0, 1.0], [-1.0, 1.0]])
        K_react = r * M_e
        K_e = K_diff + K_adv + K_react
        for i in range(2):
            for j in range(2):
                M_lil[e + i, e + j] += M_e[i, j]
                K_lil[e + i, e + j] += K_e[i, j]

    # Borde izquierdo S=0: dV/dtau + rV = 0 (Eq. 2.6)
    K_lil[0, :] = 0.0
    K_lil[0, 0] = r * np.sum(M_lil.rows[0] and M_lil.data[0])

    M_lumped = sp.diags(np.array(M_lil.sum(axis=1)).flatten(), format="csr")
    K_mat = K_lil.tocsr()

    V_star = np.maximum(K_strike - S, 0.0)
    V_next, iters = solve_penalty_step(
        M_lumped, K_mat, V_n=V_star, V_star=V_star, dt=dt, theta=0.0
    )
    amer_err = compute_american_error(V_next, V_star)

    print(f"--- Validacion de Paso de Penalizacion (Newton No Suave) ---")
    print(f"Iteraciones de Newton necesarias: {iters}")
    print(f"Error maximo restriccion americana (Ec. 4.20): {amer_err:.2e}")
    print(f"Valor en S=100 tras 1 paso: {np.interp(100.0, S, V_next):.5f}")