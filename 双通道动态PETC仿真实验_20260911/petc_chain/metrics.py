from __future__ import annotations

import numpy as np

from petc_chain import params as P
from petc_chain.sim_core import SimResult


def window_metrics(res: SimResult) -> dict:
    m = res.in_window
    e = res.e[m]
    z = res.znorm[m]
    rmse_e = float(np.sqrt(np.mean(e**2)))
    peak_e = float(np.max(np.abs(e)))
    rmse_z = float(np.sqrt(np.mean(z**2)))
    peak_z = float(np.max(z))
    n_c_mean = float(np.mean(res.n_c))
    n_u_mean = float(np.mean(res.n_u))
    r_c = 1.0 - n_c_mean / P.N_WIN
    r_u = 1.0 - n_u_mean / P.N_WIN
    per_edge_rmse = np.sqrt(np.mean(e**2, axis=0))
    comm_rate = n_c_mean / P.N_WIN
    ctrl_rate = n_u_mean / P.N_WIN
    return {
        "rmse_e": rmse_e,
        "peak_e": peak_e,
        "rmse_z": rmse_z,
        "peak_z": peak_z,
        "n_c": n_c_mean,
        "n_u": n_u_mean,
        "r_c": r_c,
        "r_u": r_u,
        "comm_rate": comm_rate,
        "ctrl_rate": ctrl_rate,
        "per_edge_rmse": per_edge_rmse,
        "eta_clip_count": res.eta_clip_count,
        "bound_violations": res.bound_violations,
        "min_comm_gap": res.min_comm_gap,
        "min_ctrl_gap": res.min_ctrl_gap,
        "max_znorm": float(np.max(res.znorm)),
        "min_eta_c": float(np.min(res.eta_c)),
        "min_eta_u": float(np.min(res.eta_u)),
    }


def gate_report(res: SimResult) -> tuple[bool, list[str]]:
    met = window_metrics(res)
    checks = []

    eta_ok = met["min_eta_c"] >= -1e-12 and met["min_eta_u"] >= -1e-12
    checks.append((eta_ok, f"eta >= 0 (min_c={met['min_eta_c']:.3e}, min_u={met['min_eta_u']:.3e}, clips={met['eta_clip_count']})"))

    gap_ok = met["min_comm_gap"] >= P.H - 1e-12 and met["min_ctrl_gap"] >= P.H - 1e-12
    checks.append((gap_ok, f"event gaps >= h (comm={met['min_comm_gap']:.4f}, ctrl={met['min_ctrl_gap']:.4f})"))

    z_ok = np.isfinite(met["max_znorm"]) and met["max_znorm"] < 50.0
    checks.append((z_ok, f"|z| bounded (max={met['max_znorm']:.4f})"))

    bound_ok = met["bound_violations"] == 0
    checks.append((bound_ok, f"residual bounds (violations={met['bound_violations']})"))

    lines = []
    passed = True
    for ok, msg in checks:
        lines.append(("PASS" if ok else "FAIL") + "  " + msg)
        passed = passed and ok
    lines.append(
        f"INFO  window comm_rate={met['comm_rate']:.3f}, ctrl_rate={met['ctrl_rate']:.3f} "
        f"(tune delta if a channel is >0.80 or <0.05)"
    )
    lines.append(
        f"INFO  RMSE_e={met['rmse_e']:.4f}, Peak_e={met['peak_e']:.4f}, "
        f"N_c={met['n_c']:.1f}, N_u={met['n_u']:.1f}"
    )
    return passed, lines
