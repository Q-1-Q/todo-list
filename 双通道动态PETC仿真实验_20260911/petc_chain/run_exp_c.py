"""Experiment C: extra disturbance at nodes 1, 5, 10. Proposed method, 10 seeds."""

from __future__ import annotations

import csv
import json

import numpy as np

from petc_chain import params as P
from petc_chain.metrics import window_metrics
from petc_chain.plot_figures import DATADIR, plot_disturbance_location
from petc_chain.sim_core import run_sim


def main():
    DATADIR.mkdir(parents=True, exist_ok=True)
    curves = {}
    table = []
    for node in P.DIST_NODES:
        per_seed = []
        peaks = []
        tail = []
        print(f"  extra disturbance at node {node}", flush=True)
        for seed in P.SEEDS:
            res = run_sim("dynamic_dual", seed=seed, extra_node=node)
            met = window_metrics(res)
            per_seed.append(met["per_edge_rmse"])
            peaks.append(met["peak_e"])
            tail.append(met["per_edge_rmse"][-1])
        mean_edge = np.mean(np.stack(per_seed, axis=0), axis=0)
        std_edge = np.std(np.stack(per_seed, axis=0), axis=0, ddof=1)
        curves[node] = mean_edge
        table.append(
            {
                "inject_node": node,
                "peak_e_mean": float(np.mean(peaks)),
                "peak_e_std": float(np.std(peaks, ddof=1)),
                "tail_rmse_mean": float(np.mean(tail)),
                "tail_rmse_std": float(np.std(tail, ddof=1)),
                "per_edge_rmse_mean": mean_edge.tolist(),
                "per_edge_rmse_std": std_edge.tolist(),
            }
        )
        print(
            f"  node {node}: tail RMSE={np.mean(tail):.4f}±{np.std(tail, ddof=1):.4f}  "
            f"Peak={np.mean(peaks):.4f}±{np.std(peaks, ddof=1):.4f}"
        )

    plot_disturbance_location(curves)
    csv_path = DATADIR / "table_iii_location.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["inject_node"] + [f"e{i}_rmse" for i in range(1, P.N + 1)] + ["peak_e_mean", "tail_rmse_mean"])
        for row in table:
            w.writerow(
                [row["inject_node"]]
                + [f"{x:.6f}" for x in row["per_edge_rmse_mean"]]
                + [f"{row['peak_e_mean']:.6f}", f"{row['tail_rmse_mean']:.6f}"]
            )
    (DATADIR / "table_iii_location.json").write_text(json.dumps(table, indent=2))
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
