from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

from src.pde_discretization import build_sinai_mesh, assemble_symmetric_laplacian
from src.linear_systems import lu_factorization, lu_solve
from src.eigenvalue_solvers import metodo_potencias, arnoldi_mgs, autovalores_qr_denso


def shift_invert_lu(H_csr, sigma: float, m_krylov: int = 28) -> np.ndarray:
    H_d = H_csr.toarray()
    n = H_d.shape[0]
    L, U, piv = lu_factorization(H_d - sigma * np.eye(n, dtype=np.float64))
    v0 = np.random.default_rng(7).standard_normal(n)
    _, H_m = arnoldi_mgs(lambda v: lu_solve(L, U, piv, v), v0, m=m_krylov, reortogonalizar=True)
    H_sq = 0.5 * (H_m[:m_krylov, :m_krylov] + H_m[:m_krylov, :m_krylov].T)
    mus = autovalores_qr_denso(H_sq, tol=1e-12)
    return np.sort(sigma + 1.0 / mus[np.abs(mus) > 1e-12])


def main() -> None:
    os.makedirs("report/figures", exist_ok=True)

    # 1. Orden de convergencia O(h^p) frente al espectro exacto del rectángulo (R = 0)
    exactos = np.sort([np.pi**2 * (p**2 + q**2) for p in range(1, 7) for q in range(1, 7)])[:6]
    nxs = [12, 16, 20, 26]
    hs, errs = [], []
    for nx in nxs:
        mesh = build_sinai_mesh(Lx=1.0, Ly=1.0, R=0.0, nx=nx)
        H = assemble_symmetric_laplacian(mesh).H
        num = shift_invert_lu(H, sigma=55.0, m_krylov=26)[:6]
        hs.append(mesh.h)
        errs.append(float(np.max(np.abs(num - exactos) / exactos)))
    orden_p, _ = np.polyfit(np.log(hs), np.log(errs), 1)

    # 2. Mal condicionamiento κ_2(H) = λ_max / λ_min ~ O(h^-2) en el billar de Sinai (R = 0.35)
    hs_k, kappas = [], []
    for nx in [12, 16, 20, 24]:
        mesh = build_sinai_mesh(Lx=1.0, Ly=1.0, R=0.35, nx=nx)
        H = assemble_symmetric_laplacian(mesh).H
        res_max = metodo_potencias(H.toarray(), tol=1e-7, max_iter=3000)
        lam_max = float(res_max[0])
        lam_min = float(shift_invert_lu(H, sigma=0.0, m_krylov=15)[0])
        hs_k.append(mesh.h)
        kappas.append(lam_max / lam_min)
    exp_kappa, _ = np.polyfit(np.log(hs_k), np.log(kappas), 1)

    # 3. Pérdida de ortogonalidad ||Q_m^T Q_m - I_m||_F en Arnoldi (σ = 273.7)
    mesh_arn = build_sinai_mesh(Lx=1.0, Ly=1.0, R=0.35, nx=24)
    H_d = assemble_symmetric_laplacian(mesh_arn).H.toarray()
    n = H_d.shape[0]
    L, U, piv = lu_factorization(H_d - 273.7 * np.eye(n))
    v0 = np.random.default_rng(7).standard_normal(n)
    m_max = 35
    Q_sin, _ = arnoldi_mgs(lambda v: lu_solve(L, U, piv, v), v0, m=m_max, reortogonalizar=False)
    Q_con, _ = arnoldi_mgs(lambda v: lu_solve(L, U, piv, v), v0, m=m_max, reortogonalizar=True)
    ks = np.arange(1, m_max + 1)
    p_sin = np.array([np.linalg.norm(Q_sin[:, :k].T @ Q_sin[:, :k] - np.eye(k)) for k in ks])
    p_con = np.array([np.linalg.norm(Q_con[:, :k].T @ Q_con[:, :k] - np.eye(k)) for k in ks])

    print("\n--- DATOS PARA COMPLETAR EL INFORME ---")
    print(f"Orden de convergencia p = {orden_p:.2f}")
    print(f"Exponente condicionamiento κ_2(H) ~ h^({exp_kappa:.2f}) | κ_2(nx=24) = {kappas[-1]:.1f}")
    print(f"Ortogonalidad Arnoldi (m={m_max}): sin reort. = {p_sin[-1]:.2e} | con reort. = {p_con[-1]:.2e}")

    # Figura de 3 paneles para la Sección 5
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    axes[0].loglog(hs, errs, "o-", label="Shortley-Weller")
    axes[0].loglog(hs, errs[0] * (np.array(hs) / hs[0]) ** 2, "k--", label=r"$\mathcal{O}(h^2)$")
    axes[0].set_xlabel("Paso h"); axes[0].set_ylabel("Error relativo"); axes[0].legend()

    axes[1].loglog(hs_k, kappas, "o-", color="tab:green", label=r"$\kappa_2(H)$")
    axes[1].loglog(hs_k, kappas[0] * (np.array(hs_k) / hs_k[0]) ** (-2), "k--", label=r"$\mathcal{O}(h^{-2})$")
    axes[1].set_xlabel("Paso h"); axes[1].set_ylabel(r"$\kappa_2(H)$"); axes[1].legend()

    axes[2].semilogy(ks, np.maximum(p_sin, 1e-16), "r--", label="Sin reortogonalización")
    axes[2].semilogy(ks, np.maximum(p_con, 1e-16), "b-", label="Con reortogonalización")
    axes[2].set_xlabel("Dimensión m"); axes[2].set_ylabel(r"$\|Q_m^T Q_m - I_m\|_F$"); axes[2].legend()

    plt.tight_layout()
    plt.savefig("report/figures/benchmarks_numericos.pdf", dpi=300)


if __name__ == "__main__":
    main()