"""Table I — frozen first-version parameters.

Do not retune except the single allowed delta adjustment after the gate,
applied once to ALL methods and seeds.
"""

from __future__ import annotations

import numpy as np

PARAM_VERSION = "petc-chain-v1"
N = 10
D_STAR = 1.0
T_SIM = 20.0
T_WIN0 = 10.0
T_WIN1 = 20.0
DT = 1.0e-4
H = 0.01
SEEDS = tuple(range(10))

KP = 1.0
KD = 1.0

A0_AMP = 0.1
A0_W = 0.5
V0_INIT = 0.5
P0_INIT = 0.0

W_AMP = 0.1
W_OMEGA = 0.5
W_PHASE_STEP = 0.6
W_ADD_AMP = 0.5
W_ADD_OMEGA = 1.0
DIST_NODES = (1, 5, 10)

WO = 20.0
BETA1 = 3.0 * WO
BETA2 = 3.0 * WO ** 2
BETA3 = WO ** 3

RHO_C = 0.9
RHO_U = 0.9
NU_C = 1.0
NU_U = 1.0
CHI_C = 0.5
CHI_U = 0.5
DELTA_C = 1.0e-4
DELTA_U = 1.0e-4
ETA0 = 0.01

P_INIT_LOW, P_INIT_HIGH = -0.2, 0.2
V_INIT_LOW, V_INIT_HIGH = -0.2, 0.2
# Agent 1 applies sampled a0(k) (ZOH). Cancels the unmatched leader
# acceleration at detection instants; the intersample a0(t)-a0(k) stays in r_h.
LEADER_FF = True

Q_SCALE = 1.0
EPS_FRAC = 0.9

METHODS = (
    "periodic",
    "static_single",
    "static_dual",
    "dynamic_single",
    "dynamic_dual",
)

METHOD_LABELS = {
    "periodic": "Periodic",
    "static_single": "Static-Single",
    "static_dual": "Static-Dual",
    "dynamic_single": "Dynamic-Single",
    "dynamic_dual": "Dynamic-Dual (proposed)",
}

# comm/ctrl: "always" | "static" | "dynamic"
METHOD_SWITCH = {
    "periodic": ("always", "always"),
    "static_single": ("static", "always"),
    "static_dual": ("static", "static"),
    "dynamic_single": ("dynamic", "always"),
    "dynamic_dual": ("dynamic", "dynamic"),
}

H_STEPS = int(round(H / DT))
N_WIN = int(round((T_WIN1 - T_WIN0) / H))  # 1000 detections in [10, 20)


def leader_state(t: float):
    """Closed form for a0=A0_AMP*sin(A0_W t), v0(0)=V0_INIT, p0(0)=P0_INIT."""
    a0 = A0_AMP * np.sin(A0_W * t)
    v0 = V0_INIT + (A0_AMP / A0_W) * (1.0 - np.cos(A0_W * t))
    p0 = (
        P0_INIT
        + V0_INIT * t
        + (A0_AMP / A0_W) * t
        - (A0_AMP / A0_W**2) * np.sin(A0_W * t)
    )
    return float(p0), float(v0), float(a0)


def disturbance(i: int, t: float, extra_node: int | None = None) -> float:
    """Follower index i is 1..N."""
    w = W_AMP * np.sin(W_OMEGA * t + W_PHASE_STEP * i)
    if extra_node is not None and i == extra_node:
        w += W_ADD_AMP * np.sin(W_ADD_OMEGA * t)
    return float(w)
