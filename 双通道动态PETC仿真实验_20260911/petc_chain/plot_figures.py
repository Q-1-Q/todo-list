from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

from petc_chain import params as P
from petc_chain.sim_core import SimResult

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figs"
DATADIR = ROOT / "data"


def _style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "figure.figsize": (7.2, 4.2),
        }
    )


def savefig(fig, name: str):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.png")
    fig.savefig(FIGDIR / f"{name}.pdf")
    plt.close(fig)


def plot_exp_a(res: SimResult):
    _style()
    t = res.t

    fig, ax = plt.subplots()
    ax.plot(t, res.p[:, 0], color="k", lw=1.8, label="leader")
    for i in range(P.N):
        ax.plot(t, res.p[:, i + 1], lw=1.0, label=f"i={i+1}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("position")
    ax.set_title("Positions (proposed)")
    ax.legend(ncol=3, fontsize=8)
    savefig(fig, "fig_a_positions")

    fig, ax = plt.subplots()
    for i in range(P.N):
        ax.plot(t, res.e[:, i], lw=1.0, label=f"e_{i+1}")
    ax.axvspan(P.T_WIN0, P.T_WIN1, color="0.85", zorder=0)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("edge position error")
    ax.set_title("Edge errors (shaded = stats window)")
    ax.legend(ncol=5, fontsize=8)
    savefig(fig, "fig_a_edge_error")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.2, 5.6))
    for i in (0, 4, 9):
        tc = t[res.comm_fire[:, i]]
        tu = t[res.ctrl_fire[:, i]]
        axes[0].vlines(tc, i + 0.7, i + 1.3, colors="C0", lw=0.8)
        axes[1].vlines(tu, i + 0.7, i + 1.3, colors="C1", lw=0.8)
    axes[0].set_ylabel("comm events")
    axes[1].set_ylabel("control events")
    axes[1].set_xlabel("t (s)")
    axes[0].set_yticks([1, 5, 10])
    axes[1].set_yticks([1, 5, 10])
    axes[0].set_title("Trigger instants (links / agents 1, 5, 10)")
    savefig(fig, "fig_a_triggers")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.2, 5.6))
    i = 4
    axes[0].plot(t, np.abs(res.c[:, i]), lw=1.0, label=r"|c_5|")
    axes[0].plot(t, res.c_bound[:, i], lw=1.0, ls="--", label=r"bound")
    axes[1].plot(t, np.abs(res.b[:, i]), lw=1.0, color="C1", label=r"|b_5|")
    axes[1].plot(t, res.b_bound[:, i], lw=1.0, ls="--", color="C1", label=r"bound")
    axes[0].legend()
    axes[1].legend()
    axes[0].set_ylabel("comm residual")
    axes[1].set_ylabel("control residual")
    axes[1].set_xlabel("t (s)")
    axes[0].set_title("Post-event residuals vs trigger bounds (agent 5)")
    savefig(fig, "fig_a_residual_bound")

    fig, ax = plt.subplots()
    ax.plot(t, res.eta_c[:, 4], label=r"eta_c,5")
    ax.plot(t, res.eta_u[:, 4], label=r"eta_u,5")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("dynamic budget")
    ax.set_title("Dynamic budgets (agent 5)")
    ax.legend()
    savefig(fig, "fig_a_eta")

    fig, ax = plt.subplots()
    ax.plot(t, res.w[:, 4], label="w_5")
    ax.plot(t, res.what[:, 4], ls="--", label="what_5")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("disturbance")
    ax.set_title("LESO on agent 5")
    ax.legend()
    savefig(fig, "fig_a_leso")

    fig, ax = plt.subplots()
    ax.plot(t, res.znorm, color="k")
    ax.axvspan(P.T_WIN0, P.T_WIN1, color="0.85", zorder=0)
    ax.set_xlabel("t (s)")
    ax.set_ylabel(r"||z||")
    ax.set_title("Edge-state norm")
    savefig(fig, "fig_a_znorm")


def plot_ablation_bars(rows: list[dict]):
    _style()
    labels = [P.METHOD_LABELS[r["method"]].replace(" (proposed)", "") for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots()
    ax.bar(x - 0.18, [r["r_c_mean"] * 100 for r in rows], 0.36, label="R_c")
    ax.bar(x + 0.18, [r["r_u_mean"] * 100 for r in rows], 0.36, label="R_u")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("resource saving (%)")
    ax.set_title("Window resource saving vs periodic baseline")
    ax.legend()
    savefig(fig, "fig_b_savings")

    fig, ax = plt.subplots()
    ax.bar(x, [r["rmse_e_mean"] for r in rows], yerr=[r["rmse_e_std"] for r in rows], capsize=3)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("RMSE of edge position error")
    ax.set_title("Window RMSE (mean ± std over 10 seeds)")
    savefig(fig, "fig_b_rmse")


def plot_disturbance_location(curves: dict[int, np.ndarray]):
    _style()
    edges = np.arange(1, P.N + 1)
    fig, ax = plt.subplots()
    for node, rmse in curves.items():
        ax.plot(edges, rmse, marker="o", label=f"inject at {node}")
    ax.set_xlabel("edge i")
    ax.set_ylabel("RMSE of e_i")
    ax.set_title("Disturbance location vs downstream edge error")
    ax.set_xticks(edges)
    ax.legend()
    savefig(fig, "fig_c_location")
