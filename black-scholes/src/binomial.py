from __future__ import annotations

import numpy as np

def binomial_american_put(S0: float, K: float, T: float, r: float, sigma: float, N: int) -> tuple[float, float]:
    """Valora un Put Americano mediante el arbol binomial CRR (Apendice C).

    Retorna el precio en t=0 y la cantidad aproximada de FLOPs (1.5 * N^2).
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Precios del subyacente en el vencimiento (n = N)
    j = np.arange(N + 1)
    S = S0 * (u ** (2 * j - N))

    # Payoff en t = T
    V = np.maximum(K - S, 0.0)

    # Retroceso en el tiempo (Backward induction)
    for n in range(N - 1, -1, -1):
        S = S[:-1] * u  # S_{m}^n = S_{m}^{n+1} * u (equivale a u^{2m - n})
        V_cont = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        V = np.maximum(K - S, V_cont)

    flops = 1.5 * (N ** 2)  # Estimacion de multiplicaciones del paper (Sec. 12)
    return float(V[0]), flops

if __name__ == "__main__":
    # Validacion con Tabla 5 del paper (T=0.25, r=0.10, K=100, S0=100, sigma=0.2)
    for steps in [50, 100, 200, 400, 800, 1600, 3200]:
        price, ops = binomial_american_put(100.0, 100.0, 0.25, 0.10, 0.20, steps)
        print(f"N={steps:4d} | Precio={price:.5f} | Flops={ops:.1e}")
