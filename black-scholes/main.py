import os
import sys
import time
import numpy as np

# Agregar src/ al path para que funcionen los imports internos entre modulos
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from binomial import binomial_american_put
from black_scholes import black_scholes_price
from fem import create_spatial_grid
from timestepping import solve_american_option_pde


def run_all_experiments():
    # Parametros base del caso de prueba (Tabla 1 del informe)
    K = 100.0
    r = 0.10
    sigma = 0.20
    T = 0.25
    S_max = 400.0
    D_scale = 1.0
    tol = 1e-8
    large = 1e6

    print("=" * 78)
    print("1. VALORES DE REFERENCIA (S0 = K = 100, T = 0.25, r = 0.10, sigma = 0.20)")
    print("=" * 78)
    euro_put = float(black_scholes_price(K, K, T, r, sigma, option_type="put"))
    bin_res = binomial_american_put(K, K, T, r, sigma, N=5000)
    amer_bin = float(bin_res[0]) if isinstance(bin_res, tuple) else float(bin_res)
    print(f"Put Europeo (Black-Scholes analitico): {euro_put:.8f}")
    print(f"Put Americano (Binomial N = 5000):     {amer_bin:.8f}\n")

  
    # TABLA 2: Estudio de convergencia espacio-temporal
   
    print("=" * 78)
    print("2. ESTUDIO DE CONVERGENCIA ESPACIO-TEMPORAL (TABLA 2)")
    print("=" * 78)
    m_values = [64, 128, 256, 512, 1024]
    dt0_base = 1e-3
    dnorm_base = 0.05

    conv_results = []
    for level, m in enumerate(m_values):
        S = create_spatial_grid(K=K, S_max=S_max, n_nodes=m + 1)
        dt_init = dt0_base / (2**level)
        dnorm = dnorm_base / (2**level)

        t0 = time.perf_counter()
        res = solve_american_option_pde(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            adaptive=True,
            dt_init=dt_init,
            dnorm=dnorm,
            D_scale=D_scale,
            rannacher=True,
            method="penalty",
            large=large,
            tol=tol,
            use_mass_lumping=True,
        )
        elapsed = time.perf_counter() - t0

        price = res["price_at_K"]
        steps = res["steps"]
        avg_iters = res["total_iters"] / steps
        conv_results.append((m, steps, avg_iters, price, res, elapsed))

    header = (
        f"{'m':>6} | {'Pasos':>6} | {'Iter/paso':>9} | {'V(K,T)':>12} | "
        f"{'|Vh - Vh/2|':>12} | {'Ratio':>7} | {'Err vs Bin':>11} | {'t (s)':>6}"
    )
    print(header)
    print("-" * len(header))

    diffs = []
    for idx, (m, steps, avg_iters, price, _, elapsed) in enumerate(conv_results):
        err_bin = abs(price - amer_bin)
        if idx == 0:
            print(
                f"{m:6d} | {steps:6d} | {avg_iters:9.2f} | {price:12.6f} | "
                f"{'---':>12} | {'---':>7} | {err_bin:11.6f} | {elapsed:6.2f}"
            )
        else:
            prev_price = conv_results[idx - 1][3]
            diff = abs(prev_price - price)
            diffs.append(diff)
            if idx == 1:
                ratio_str = "---"
            else:
                ratio = diffs[-2] / diffs[-1]
                ratio_str = f"{ratio:7.2f}"
            print(
                f"{m:6d} | {steps:6d} | {avg_iters:9.2f} | {price:12.6f} | "
                f"{diff:12.6f} | {ratio_str:>7} | {err_bin:11.6f} | {elapsed:6.2f}"
            )

    print("\n--- Filas listas para copiar en Tabla 2 de LaTeX ---")
    for idx, (m, steps, avg_iters, price, _, _) in enumerate(conv_results):
        if idx == 0:
            print(f"${m}$ & ${steps}$ & ${avg_iters:.2f}$ & ${price:.6f}$ & --- & --- \\\\")
        elif idx == 1:
            diff = abs(conv_results[idx - 1][3] - price)
            print(
                f"${m}$ & ${steps}$ & ${avg_iters:.2f}$ & ${price:.6f}$ & "
                f"${diff:.6f}$ & --- \\\\"
            )
        else:
            diff = abs(conv_results[idx - 1][3] - price)
            ratio = diffs[idx - 2] / diffs[idx - 1]
            print(
                f"${m}$ & ${steps}$ & ${avg_iters:.2f}$ & ${price:.6f}$ & "
                f"${diff:.6f}$ & ${ratio:.2f}$ \\\\"
            )

   
    # TABLA 3: Impacto del suavizado de Rannacher y paso adaptativo
 
    print("\n" + "=" * 78)
    print("3. COMPARACION DE ESQUEMAS TEMPORALES (TABLA 3)")
    print("=" * 78)
    schemes = [
        ("Crank-Nicolson (dt constante)", False, False),
        ("Rannacher + CN (dt constante)", True, False),
        ("Rannacher + CN + Paso adaptativo", True, True),
    ]
    m_comp = [64, 128, 256, 512]

    for name, use_rannacher, use_adaptive in schemes:
        prices_s = []
        min_gamma_near_K = 0.0
        gamma_at_K = 0.0

        for level, m in enumerate(m_comp):
            S = create_spatial_grid(K=K, S_max=S_max, n_nodes=m + 1)
            res = solve_american_option_pde(
                S=S,
                K=K,
                T=T,
                r=r,
                sigma=sigma,
                adaptive=use_adaptive,
                n_steps=25 * (2**level),
                dt_init=dt0_base / (2**level),
                dnorm=dnorm_base / (2**level),
                D_scale=D_scale,
                rannacher=use_rannacher,
                method="penalty",
                large=large,
                tol=tol,
                use_mass_lumping=True,
            )
            prices_s.append(res["price_at_K"])
            if m == 256:
                mask = (S >= K - 10.0) & (S <= K + 10.0)
                min_gamma_near_K = float(np.min(res["gamma"][mask]))
                gamma_at_K = float(np.interp(K, S, res["gamma"]))

        d1 = abs(prices_s[0] - prices_s[1])
        d2 = abs(prices_s[1] - prices_s[2])
        d3 = abs(prices_s[2] - prices_s[3])
        r_mid = d1 / d2
        r_fine = d2 / d3
        osc_flag = "Presentes (min Gamma < 0)" if min_gamma_near_K < -1e-4 else "Eliminadas"

        print(f"\nEsquema: {name}")
        print(f"  Precios V(K,T) [m=64..512]: {[round(p, 6) for p in prices_s]}")
        print(f"  Ratios sucesivos: {r_mid:.2f} -> {r_fine:.2f}")
        print(
            f"  Gamma(K,T) en m=256: {gamma_at_K:.6f} | "
            f"min Gamma en [90,110]: {min_gamma_near_K:.6f} -> {osc_flag}"
        )

   
    # SECCION 5.4: Frontera optima Sf(T), Griegas y sensibilidad a rho
  
    print("\n" + "=" * 78)
    print("4. FRONTERA DE EJERCICIO OPTIMO, GRIEGAS Y SENSIBILIDAD A RHO (m = 512)")
    print("=" * 78)
    res_512 = conv_results[3][4]  # m = 512
    S_512 = res_512["S"]
    V_512 = res_512["V"]
    V_star_512 = np.maximum(K - S_512, 0.0)

    active_nodes = S_512[V_512 <= V_star_512 + 1e-7]
    S_f = float(np.max(active_nodes)) if len(active_nodes) > 0 else np.nan
    delta_K = float(np.interp(K, S_512, res_512["delta"]))
    gamma_K = float(np.interp(K, S_512, res_512["gamma"]))

    print(f"V(K, T)       = {res_512['price_at_K']:.6f}")
    print(f"Delta(K, T)   = {delta_K:.6f}")
    print(f"Gamma(K, T)   = {gamma_K:.6f}")
    print(f"Frontera Sf(T)= {S_f:.4f}")

    print("\nSensibilidad al factor Large (rho = Large / dtau) en m = 512:")
    S_sens = create_spatial_grid(K=K, S_max=S_max, n_nodes=513)
    V_by_large = {}
    for lg in [1e4, 1e6, 1e8]:
        r_lg = solve_american_option_pde(
            S=S_sens,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            adaptive=True,
            dt_init=dt0_base / 8,
            dnorm=dnorm_base / 8,
            D_scale=D_scale,
            rannacher=True,
            method="penalty",
            large=lg,
            tol=tol,
            use_mass_lumping=True,
        )
        V_by_large[lg] = r_lg["V"]
        print(
            f"  Large = {lg:7.1e} -> V(K,T) = {r_lg['price_at_K']:.8f} | "
            f"max_amer_err = {r_lg['max_amer_err']:.2e}"
        )

    diff_1e6_1e8 = float(np.max(np.abs(V_by_large[1e6] - V_by_large[1e8])))
    diff_1e4_1e8 = float(np.max(np.abs(V_by_large[1e4] - V_by_large[1e8])))
    print(f"  max |V(1e4) - V(1e8)| = {diff_1e4_1e8:.2e}")
    print(f"  max |V(1e6) - V(1e8)| = {diff_1e6_1e8:.2e}")


if __name__ == "__main__":
    run_all_experiments()