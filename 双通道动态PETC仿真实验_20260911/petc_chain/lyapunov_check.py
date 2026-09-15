"""
Discrete Lyapunov numerical check (B4).

ℓ_c, ℓ_u are proof parameters only. They never enter the simulator.
"""

from __future__ import annotations

import numpy as np

from petc_chain import params as P


def solve_discrete_lyapunov_ata(a: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Solve A^T P A - P = -Q via Kronecker product."""
    n = a.shape[0]
    kmat = np.kron(a.T, a.T)
    sys = np.eye(n * n) - kmat
    pvec = np.linalg.solve(sys, q.reshape(-1, order="F"))
    pmat = pvec.reshape((n, n), order="F")
    return 0.5 * (pmat + pmat.T)


def local_Ah():
    h, kp, kd = P.H, P.KP, P.KD
    return np.array(
        [
            [1.0 - 0.5 * h**2 * kp, h - 0.5 * h**2 * kd],
            [-h * kp, 1.0 - h * kd],
        ],
        dtype=float,
    )


def Bh():
    h = P.H
    return np.array([[0.5 * h**2], [h]], dtype=float)


def difference_matrix(n: int) -> np.ndarray:
    d = np.eye(n)
    for i in range(1, n):
        d[i, i - 1] = -1.0
    return d


def global_matrices():
    n = P.N
    ah = local_Ah()
    bh = Bh()
    fh = bh @ np.array([[P.KP, P.KD]], dtype=float)
    a_big = np.zeros((2 * n, 2 * n))
    for i in range(n):
        a_big[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = ah
        if i >= 1:
            a_big[2 * i : 2 * i + 2, 2 * i - 2 : 2 * i] = fh
    dn = difference_matrix(n)
    gh = np.kron(dn, bh)
    return a_big, gh, ah, bh, fh, dn


def run_check() -> dict:
    a_big, gh, ah, bh, fh, dn = global_matrices()
    eig = np.linalg.eigvals(ah)
    rho_local = float(np.max(np.abs(eig)))
    rho_global = float(np.max(np.abs(np.linalg.eigvals(a_big))))
    q = P.Q_SCALE * np.eye(2 * P.N)
    pmat = solve_discrete_lyapunov_ata(a_big, q)
    evals_p = np.real(np.linalg.eigvalsh(pmat))
    lam_min_p = float(evals_p.min())
    lam_max_p = float(evals_p.max())
    lam_min_q = float(P.Q_SCALE)
    eps = P.EPS_FRAC * lam_min_q
    alpha0 = lam_min_q - eps
    atp = a_big.T @ pmat
    cp = float(np.linalg.norm(atp, 2) ** 2 / eps + lam_max_p)
    gnorm = float(np.linalg.norm(gh, 2))
    beta_q = 9.0 * cp * gnorm**2
    ell_min = beta_q / P.CHI_C
    schur_lo = 0.5 * P.H * P.KP
    schur_hi = 2.0 / P.H
    chi_nu = P.CHI_C * P.NU_C
    ok_schur = (P.KP > 0.0) and (schur_lo < P.KD < schur_hi)
    ok_budget = (0.0 < P.RHO_C < 1.0) and (chi_nu <= P.RHO_C)
    ok_schur_num = (rho_local < 1.0 - 1e-12) and (rho_global < 1.0 - 1e-12)
    ok_p = lam_min_p > 0.0
    out = {
        "rho_local": rho_local,
        "rho_global": rho_global,
        "lam_min_P": lam_min_p,
        "lam_max_P": lam_max_p,
        "eps": eps,
        "alpha0": alpha0,
        "C_P": cp,
        "G_h_2norm": gnorm,
        "beta_q": beta_q,
        "ell_min": ell_min,
        "schur_lo": schur_lo,
        "schur_hi": schur_hi,
        "chi_nu": chi_nu,
        "ok_schur": ok_schur,
        "ok_schur_num": ok_schur_num,
        "ok_budget": ok_budget,
        "ok_P": ok_p,
        "det_Ah": float(np.linalg.det(ah)),
    }
    return out


def main():
    out = run_check()
    print("PETC chain Lyapunov / parameter-domain check")
    print(f"  |lambda(A_h)| max     = {out['rho_local']:.8f}")
    print(f"  |lambda(A_N)| max     = {out['rho_global']:.8f}")
    print(f"  det(A_h)              = {out['det_Ah']:.8f}")
    print(f"  Schur analytic        = {out['ok_schur']}  ({out['schur_lo']:.4g} < kd={P.KD} < {out['schur_hi']:.4g})")
    print(f"  chi*nu <= rho         = {out['ok_budget']}  ({out['chi_nu']:.4g} <= {P.RHO_C})")
    print(f"  lambda_min(P)         = {out['lam_min_P']:.6e}")
    print(f"  lambda_max(P)         = {out['lam_max_P']:.6e}")
    print(f"  ||G_h||_2             = {out['G_h_2norm']:.6e}")
    print(f"  C_P                   = {out['C_P']:.6e}")
    print(f"  beta_q                = {out['beta_q']:.6e}")
    print(f"  ell >= beta_q / chi   = {out['ell_min']:.6e}")
    print("  ell is a proof weight only; it is not used in the simulator.")
    passed = all(out[k] for k in ("ok_schur", "ok_schur_num", "ok_budget", "ok_P"))
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
