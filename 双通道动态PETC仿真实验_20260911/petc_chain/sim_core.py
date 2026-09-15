"""PETC predecessor-following chain simulator.

Event order at each detection instant t_k = k h:
  communication decision -> candidate control -> control decision -> budget update.
Control and communication values are zero-order held between detections.
LESO is driven by the held applied input u, not the candidate.
k = 0 uses fresh initialization and is not counted as an event (scheme A).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from petc_chain import params as P


def _should_fire(mode: str, err2: float, eta: float, delta: float, nu: float) -> bool:
    if mode == "always":
        return True
    if mode == "static":
        return err2 > delta
    if mode == "dynamic":
        return err2 > nu * eta + delta
    raise ValueError(f"unknown trigger mode {mode}")


def _bound(mode: str, eta: float, delta: float, nu: float) -> float:
    if mode == "always":
        return 0.0
    if mode == "static":
        return float(np.sqrt(max(delta, 0.0)))
    return float(np.sqrt(max(nu * eta + delta, 0.0)))


@dataclass
class SimResult:
    t: np.ndarray
    p: np.ndarray
    v: np.ndarray
    e: np.ndarray
    edot: np.ndarray
    znorm: np.ndarray
    w: np.ndarray
    what: np.ndarray
    c: np.ndarray
    b: np.ndarray
    c_bound: np.ndarray
    b_bound: np.ndarray
    eta_c: np.ndarray
    eta_u: np.ndarray
    comm_fire: np.ndarray
    ctrl_fire: np.ndarray
    u: np.ndarray
    n_c: np.ndarray
    n_u: np.ndarray
    min_comm_gap: float
    min_ctrl_gap: float
    eta_clip_count: int
    bound_violations: int
    in_window: np.ndarray


def run_sim(
    method: str,
    seed: int = 0,
    extra_node: int | None = None,
    delta_c: float | None = None,
    delta_u: float | None = None,
) -> SimResult:
    comm_mode, ctrl_mode = P.METHOD_SWITCH[method]
    dc = P.DELTA_C if delta_c is None else delta_c
    du = P.DELTA_U if delta_u is None else delta_u

    rng = np.random.default_rng(seed)
    n_f = P.N
    n_steps = int(round(P.T_SIM / P.DT))
    h_steps = P.H_STEPS
    n_log = n_steps // h_steps + 1

    p0, v0, _ = P.leader_state(0.0)
    # Edge-scale random IC around the desired formation (not absolute U(-1,1),
    # which would make sequential chain settling far longer than T=20 s).
    p_f = p0 + P.D_STAR * np.arange(1, n_f + 1) + rng.uniform(
        P.P_INIT_LOW, P.P_INIT_HIGH, size=n_f
    )
    v_f = v0 + rng.uniform(P.V_INIT_LOW, P.V_INIT_HIGH, size=n_f)
    phat = p_f.copy()
    vhat = v_f.copy()
    what = np.zeros(n_f)

    p0, v0, _ = P.leader_state(0.0)
    phi_all = np.empty(n_f + 1)
    phi_all[0] = P.KP * p0 + P.KD * v0
    phi_all[1:] = P.KP * p_f + P.KD * v_f
    phi_bar = phi_all[:-1].copy()  # last sent by nodes 0..N-1
    nu = -P.KP * p_f - P.KD * v_f + phi_bar + P.KP * P.D_STAR - what
    u_hold = nu.copy()

    eta_c = np.full(n_f, P.ETA0)
    eta_u = np.full(n_f, P.ETA0)
    last_comm_k = np.full(n_f, -1)
    last_ctrl_k = np.full(n_f, -1)
    min_comm_gap = np.inf
    min_ctrl_gap = np.inf
    eta_clip_count = 0
    bound_violations = 0
    n_c = np.zeros(n_f, dtype=int)
    n_u = np.zeros(n_f, dtype=int)

    t_log = np.empty(n_log)
    p_log = np.empty((n_log, n_f + 1))
    v_log = np.empty((n_log, n_f + 1))
    e_log = np.empty((n_log, n_f))
    edot_log = np.empty((n_log, n_f))
    z_log = np.empty(n_log)
    w_log = np.empty((n_log, n_f))
    what_log = np.empty((n_log, n_f))
    c_log = np.empty((n_log, n_f))
    b_log = np.empty((n_log, n_f))
    cb_log = np.empty((n_log, n_f))
    bb_log = np.empty((n_log, n_f))
    etac_log = np.empty((n_log, n_f))
    etau_log = np.empty((n_log, n_f))
    fire_c = np.zeros((n_log, n_f), dtype=bool)
    fire_u = np.zeros((n_log, n_f), dtype=bool)
    u_log = np.empty((n_log, n_f))
    in_window = np.zeros(n_log, dtype=bool)

    dt = P.DT
    b1, b2, b3 = P.BETA1, P.BETA2, P.BETA3
    log_i = 0

    def w_vec(tt: float) -> np.ndarray:
        idx = np.arange(1, n_f + 1)
        w = P.W_AMP * np.sin(P.W_OMEGA * tt + P.W_PHASE_STEP * idx)
        if extra_node is not None:
            w[extra_node - 1] += P.W_ADD_AMP * np.sin(P.W_ADD_OMEGA * tt)
        return w

    def petc_step(k: int, t: float):
        nonlocal min_comm_gap, min_ctrl_gap, eta_clip_count, bound_violations, log_i
        p0t, v0t, _ = P.leader_state(t)
        phi = np.empty(n_f + 1)
        phi[0] = P.KP * p0t + P.KD * v0t
        phi[1:] = P.KP * p_f + P.KD * v_f

        c = np.empty(n_f)
        fired_c = np.zeros(n_f, dtype=bool)
        eta_c_used = eta_c.copy()
        for i in range(n_f):
            c_minus = phi_bar[i] - phi[i]
            if k == 0:
                fire = False
                phi_bar[i] = phi[i]
                c[i] = 0.0
            else:
                fire = _should_fire(comm_mode, c_minus * c_minus, eta_c[i], dc, P.NU_C)
                if fire:
                    phi_bar[i] = phi[i]
                    c[i] = 0.0
                    if last_comm_k[i] >= 0:
                        min_comm_gap = min(min_comm_gap, (k - last_comm_k[i]) * P.H)
                    last_comm_k[i] = k
                    if P.T_WIN0 <= t < P.T_WIN1:
                        n_c[i] += 1
                else:
                    c[i] = c_minus
            fired_c[i] = fire

        nu_c = -P.KP * p_f - P.KD * v_f + phi_bar + P.KP * P.D_STAR - what
        if P.LEADER_FF:
            nu_c[0] += P.leader_state(t)[2]
        b = np.empty(n_f)
        fired_u = np.zeros(n_f, dtype=bool)
        eta_u_used = eta_u.copy()
        for i in range(n_f):
            b_minus = u_hold[i] - nu_c[i]
            if k == 0:
                fire = False
                u_hold[i] = nu_c[i]
                b[i] = 0.0
            else:
                fire = _should_fire(ctrl_mode, b_minus * b_minus, eta_u[i], du, P.NU_U)
                if fire:
                    u_hold[i] = nu_c[i]
                    b[i] = 0.0
                    if last_ctrl_k[i] >= 0:
                        min_ctrl_gap = min(min_ctrl_gap, (k - last_ctrl_k[i]) * P.H)
                    last_ctrl_k[i] = k
                    if P.T_WIN0 <= t < P.T_WIN1:
                        n_u[i] += 1
                else:
                    b[i] = b_minus
            fired_u[i] = fire

        c_bnd = np.array([_bound(comm_mode, eta_c_used[i], dc, P.NU_C) for i in range(n_f)])
        b_bnd = np.array([_bound(ctrl_mode, eta_u_used[i], du, P.NU_U) for i in range(n_f)])
        if comm_mode != "always" and np.any(np.abs(c) > c_bnd + 1e-9):
            bound_violations += int(np.sum(np.abs(c) > c_bnd + 1e-9))
        if ctrl_mode != "always" and np.any(np.abs(b) > b_bnd + 1e-9):
            bound_violations += int(np.sum(np.abs(b) > b_bnd + 1e-9))

        if comm_mode == "dynamic":
            eta_next = P.RHO_C * eta_c + P.CHI_C * dc - P.CHI_C * c * c
            eta_clip_count += int(np.sum(eta_next < -1e-12))
            eta_c[:] = np.maximum(eta_next, 0.0)
        if ctrl_mode == "dynamic":
            eta_next = P.RHO_U * eta_u + P.CHI_U * du - P.CHI_U * b * b
            eta_clip_count += int(np.sum(eta_next < -1e-12))
            eta_u[:] = np.maximum(eta_next, 0.0)

        e = np.empty(n_f)
        edot = np.empty(n_f)
        e[0] = p_f[0] - p0t - P.D_STAR
        edot[0] = v_f[0] - v0t
        e[1:] = p_f[1:] - p_f[:-1] - P.D_STAR
        edot[1:] = v_f[1:] - v_f[:-1]
        z = np.empty(2 * n_f)
        z[0::2] = e
        z[1::2] = edot

        t_log[log_i] = t
        p_log[log_i, 0] = p0t
        p_log[log_i, 1:] = p_f
        v_log[log_i, 0] = v0t
        v_log[log_i, 1:] = v_f
        e_log[log_i] = e
        edot_log[log_i] = edot
        z_log[log_i] = float(np.linalg.norm(z))
        w_log[log_i] = w_vec(t)
        what_log[log_i] = what
        c_log[log_i] = c
        b_log[log_i] = b
        cb_log[log_i] = c_bnd
        bb_log[log_i] = b_bnd
        etac_log[log_i] = eta_c_used
        etau_log[log_i] = eta_u_used
        fire_c[log_i] = fired_c
        fire_u[log_i] = fired_u
        u_log[log_i] = u_hold
        in_window[log_i] = P.T_WIN0 <= t < P.T_WIN1
        log_i += 1

    def step_dyn(t: float):
        nonlocal p_f, v_f, phat, vhat, what
        u = u_hold
        wmid = w_vec(t + 0.5 * dt)
        acc = u + wmid
        p_f = p_f + dt * v_f + 0.5 * dt * dt * acc
        v_f = v_f + dt * acc
        err = p_f - phat
        phat = phat + dt * (vhat + b1 * err)
        vhat = vhat + dt * (u + what + b2 * err)
        what = what + dt * (b3 * err)

    for n in range(n_steps):
        t = n * dt
        if n % h_steps == 0:
            petc_step(n // h_steps, t)
        step_dyn(t)

    t_end = n_steps * dt
    petc_step(n_steps // h_steps, t_end)

    if min_comm_gap is np.inf:
        min_comm_gap = P.H
    if min_ctrl_gap is np.inf:
        min_ctrl_gap = P.H

    return SimResult(
        t=t_log[:log_i],
        p=p_log[:log_i],
        v=v_log[:log_i],
        e=e_log[:log_i],
        edot=edot_log[:log_i],
        znorm=z_log[:log_i],
        w=w_log[:log_i],
        what=what_log[:log_i],
        c=c_log[:log_i],
        b=b_log[:log_i],
        c_bound=cb_log[:log_i],
        b_bound=bb_log[:log_i],
        eta_c=etac_log[:log_i],
        eta_u=etau_log[:log_i],
        comm_fire=fire_c[:log_i],
        ctrl_fire=fire_u[:log_i],
        u=u_log[:log_i],
        n_c=n_c,
        n_u=n_u,
        min_comm_gap=float(min_comm_gap),
        min_ctrl_gap=float(min_ctrl_gap),
        eta_clip_count=int(eta_clip_count),
        bound_violations=int(bound_violations),
        in_window=in_window[:log_i],
    )
