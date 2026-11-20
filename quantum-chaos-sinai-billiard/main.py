from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

from src.pde_discretization import build_sinai_mesh, assemble_symmetric_laplacian
from src.linear_systems import gershgorin_bounds, lu_factorization, lu_solve
from src.eigenvalue_solvers import arnoldi_mgs, autovalores_qr_denso
from src.spectral_validation import (
    unfold_spectrum_weyl,
    spacing_statistics,
    spectral_rigidity_delta3,
    sample_goe_spacings,
    poisson_spacing_pdf,
    wigner_surmise_pdf,
)


def extraer_ventanas_lu(
    H_csr,
    sigmas: list[float],
    etiqueta: str,
    radio_confianza: float = 80.0,
    m_krylov: int = 34,
) -> np.ndarray:
    """
    Shift-and-Invert Arnoldi exacto: factoriza PA = LU de (H - σI) una vez por ventana
    y aplica Arnoldi con reortogonalización completa + QR de Householder.
    """
    H_densa = H_csr.toarray()
    n = H_densa.shape[0]
    rng = np.random.default_rng(42)
    niveles: list[float] = []

    for idx, sigma in enumerate(sigmas, start=1):
        A_desp = H_densa - sigma * np.eye(n, dtype=np.float64)
        L, U, piv = lu_factorization(A_desp)

        def aplicar_inversa(v: np.ndarray) -> np.ndarray:
            return lu_solve(L, U, piv, v)

        v0 = rng.standard_normal(n)
        _, H_m = arnoldi_mgs(aplicar_inversa, v0, m=m_krylov, reortogonalizar=True)
        H_sq = 0.5 * (H_m[:m_krylov, :m_krylov] + H_m[:m_krylov, :m_krylov].T)

        mus = autovalores_qr_denso(H_sq, tol=1e-12)
        mus_validos = mus[np.abs(mus) > 1e-12]
        lambdas = sigma + 1.0 / mus_validos

        # Retener únicamente los niveles convergidos cercanos al centro σ de la ventana
        cercanos = lambdas[np.abs(lambdas - sigma) <= radio_confianza]
        for lam in cercanos:
            if not niveles or np.min(np.abs(np.array(niveles) - lam)) > 1e-3:
                niveles.append(float(lam))

        print(f"  [{etiqueta}] Ventana {idx}/{len(sigmas)} (σ={sigma:.1f}) -> {len(niveles)} niveles")

    return np.sort(np.array(niveles, dtype=np.float64))


def main() -> None:
    os.makedirs("report/figures", exist_ok=True)

    # 1. Discretización Shortley-Weller simétrica (nx = 28 -> N ≈ 708 nodos activos)
    mesh_sinai = build_sinai_mesh(Lx=1.0, Ly=1.0, R=0.35, nx=28)
    sys_sinai = assemble_symmetric_laplacian(mesh_sinai)
    cota_inf_B, cota_sup_B = gershgorin_bounds(sys_sinai.B)

    mesh_rect = build_sinai_mesh(Lx=1.0, Ly=31.0 / 28.0, R=0.0, nx=28)
    sys_rect = assemble_symmetric_laplacian(mesh_rect)

    # 2. Extracción espectral en 11 ventanas hasta E ≈ 1700
    ventanas = [123.7 + 150.0 * k for k in range(11)]
    eigs_sinai = extraer_ventanas_lu(sys_sinai.H, sigmas=ventanas, etiqueta="Sinai")
    eigs_rect = extraer_ventanas_lu(sys_rect.H, sigmas=ventanas, etiqueta="Rectángulo")

    # 3. Unfolding mediante Ley de Weyl (mínimos cuadrados con QR de Householder)
    unf_sinai = unfold_spectrum_weyl(eigs_sinai, Lx=1.0, Ly=1.0, R=0.35)
    unf_rect = unfold_spectrum_weyl(eigs_rect, Lx=1.0, Ly=31.0 / 28.0, R=0.0)

    # 4. Estadísticos P(s) y Δ_3(L) + ensamble GOE diagonalizado con Householder QR
    stats_sinai = spacing_statistics(unf_sinai.spacings)
    stats_rect = spacing_statistics(unf_rect.spacings)
    goe_spacings = sample_goe_spacings(matrix_size=70, n_realizations=25)
    stats_goe = spacing_statistics(goe_spacings)

    L_vals = np.linspace(1.0, 12.0, 15)
    d3_sinai, d3_pois, d3_goe = spectral_rigidity_delta3(unf_sinai.unfolded_levels, L_vals)

    # 5. Resumen numérico para el Abstract y Sección 5 del informe
    print("\n-RESULTADOS FINALES ")
    print(f"Nodos activos Sinai: {mesh_sinai.n_active} | Gershgorin B: [{cota_inf_B:.2f}, {cota_sup_B:.2f}]")
    print(f"Niveles Sinai: {len(eigs_sinai)} | Varianza P(s): {stats_sinai['variance']:.3f} (GOE: {stats_goe['variance']:.3f})")
    print(f"Niveles Rectángulo: {len(eigs_rect)} | Varianza P(s): {stats_rect['variance']:.3f} (Poisson: 1.000)")
    print(f"Δ_3(L=10) Sinai: {d3_sinai[-3]:.3f} | GOE: {d3_goe[-3]:.3f} | Poisson: {d3_pois[-3]:.3f}")

    # 6. Guardar figura comparativa para LaTeX
    s_grid = np.linspace(0.0, 3.5, 200)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    axes[0].bar(stats_sinai["bin_centers"], stats_sinai["hist_density"], width=0.16, alpha=0.65, label="Sinai (R=0.35)")
    axes[0].plot(s_grid, wigner_surmise_pdf(s_grid), "k-", lw=1.8, label="Wigner GOE")
    axes[0].plot(s_grid, poisson_spacing_pdf(s_grid), "r--", lw=1.5, label="Poisson")
    axes[0].set_xlabel("s"); axes[0].set_ylabel("P(s)"); axes[0].legend()

    axes[1].plot(L_vals, d3_sinai, "o-", label="Sinai (R=0.35)")
    axes[1].plot(L_vals, d3_goe, "k-", lw=1.6, label="Asintótica GOE")
    axes[1].plot(L_vals, d3_pois, "r--", lw=1.5, label="Poisson (L/15)")
    axes[1].set_xlabel("L"); axes[1].set_ylabel(r"$\Delta_3(L)$"); axes[1].legend()

    plt.tight_layout()
    plt.savefig("report/figures/bgs_resultados.pdf", dpi=300)


if __name__ == "__main__":
    main()
