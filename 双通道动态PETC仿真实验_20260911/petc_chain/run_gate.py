"""Gate run: proposed method, seed 0. No paper figures."""

from __future__ import annotations

from petc_chain.lyapunov_check import run_check
from petc_chain.metrics import gate_report
from petc_chain.sim_core import run_sim


def main():
    print("=== Lyapunov / parameter-domain check ===")
    lyap = run_check()
    lyap_ok = all(lyap[k] for k in ("ok_schur", "ok_schur_num", "ok_budget", "ok_P"))
    print(f"  beta_q = {lyap['beta_q']:.4e}, ell_min = {lyap['ell_min']:.4e}")
    print("  Lyapunov check:", "PASS" if lyap_ok else "FAIL")

    print("=== Simulator gate (dynamic_dual, seed 0) ===")
    res = run_sim("dynamic_dual", seed=0)
    passed, lines = gate_report(res)
    for line in lines:
        print(" ", line)
    ok = lyap_ok and passed
    print("GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
