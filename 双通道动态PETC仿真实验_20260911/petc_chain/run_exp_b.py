"""Experiment B: five-method ablation, 10 seeds, mean ± std."""

from __future__ import annotations

import csv
import json

import numpy as np

from petc_chain import params as P
from petc_chain.metrics import window_metrics
from petc_chain.plot_figures import DATADIR, plot_ablation_bars
from petc_chain.sim_core import run_sim

KEYS = ("rmse_e", "peak_e", "n_c", "n_u", "r_c", "r_u")


def main():
    DATADIR.mkdir(parents=True, exist_ok=True)
    rows = []
    raw = {}
    for method in P.METHODS:
        samples = {k: [] for k in KEYS}
        for seed in P.SEEDS:
            print(f"  {method} seed={seed}", flush=True)
            res = run_sim(method, seed=seed)
            met = window_metrics(res)
            for k in KEYS:
                samples[k].append(met[k])
        raw[method] = samples
        row = {"method": method, "label": P.METHOD_LABELS[method]}
        for k in KEYS:
            arr = np.array(samples[k], dtype=float)
            row[f"{k}_mean"] = float(arr.mean())
            row[f"{k}_std"] = float(arr.std(ddof=1))
        rows.append(row)
        print(
            f"{P.METHOD_LABELS[method]}: "
            f"RMSE={row['rmse_e_mean']:.4f}±{row['rmse_e_std']:.4f}  "
            f"N_c={row['n_c_mean']:.1f}±{row['n_c_std']:.1f}  "
            f"N_u={row['n_u_mean']:.1f}±{row['n_u_std']:.1f}"
        )

    plot_ablation_bars(rows)
    csv_path = DATADIR / "table_ii_ablation.csv"
    fieldnames = ["method", "label"] + [f"{k}_{s}" for k in KEYS for s in ("mean", "std")]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    (DATADIR / "table_ii_ablation.json").write_text(json.dumps({"rows": rows, "raw": raw}, indent=2))
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
