"""Experiment A: proposed method figures for lemmas / Theorem 1."""

from __future__ import annotations

import json

from petc_chain.metrics import window_metrics
from petc_chain.plot_figures import DATADIR, plot_exp_a
from petc_chain.sim_core import run_sim


def main():
    res = run_sim("dynamic_dual", seed=0)
    plot_exp_a(res)
    met = window_metrics(res)
    DATADIR.mkdir(parents=True, exist_ok=True)
    payload = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in met.items()}
    (DATADIR / "exp_a_seed0.json").write_text(json.dumps(payload, indent=2))
    print("Experiment A written to figs/ and data/exp_a_seed0.json")
    print(
        f"  RMSE_e={met['rmse_e']:.4f}  Peak_e={met['peak_e']:.4f}  "
        f"N_c={met['n_c']:.1f}  N_u={met['n_u']:.1f}  "
        f"R_c={100*met['r_c']:.2f}%  R_u={100*met['r_u']:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
