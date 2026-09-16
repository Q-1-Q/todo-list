function met = run_sim_metrics(P, method, seed, varargin)
% Run one realization and return metrics over the configured statistics window.

[P, n_max_init] = apply_overrides(P, varargin{:});
[comm_mode, ctrl_mode] = method_modes(method);
rng(seed, 'twister');
n = P.N;
n_steps = round(P.T_SIM / P.DT);
idx = (1:n)';

[p0, v0, ~] = leader_state(0.0, P);
if isempty(n_max_init)
    dp = P.P_INIT_LOW + (P.P_INIT_HIGH - P.P_INIT_LOW) * rand(n, 1);
    dv = P.V_INIT_LOW + (P.V_INIT_HIGH - P.V_INIT_LOW) * rand(n, 1);
else
    if n_max_init < n
        error('NMaxInit must be greater than or equal to N.');
    end
    dp_all = P.P_INIT_LOW + (P.P_INIT_HIGH - P.P_INIT_LOW) * rand(n_max_init, 1);
    dv_all = P.V_INIT_LOW + (P.V_INIT_HIGH - P.V_INIT_LOW) * rand(n_max_init, 1);
    dp = dp_all(1:n);
    dv = dv_all(1:n);
end
p = p0 + P.D_STAR * idx + dp;
v = v0 + dv;
phat = p;
vhat = v;
what = zeros(n, 1);

phi_bar = zeros(n, 1);
u_hold = zeros(n, 1);
eta_c = P.ETA0 * ones(n, 1);
eta_u = P.ETA0 * ones(n, 1);
n_c = zeros(n, 1);
n_u = zeros(n, 1);

sum_e2 = 0.0;
sample_count = 0;
sample_times = 0;
sum_edge_e2 = zeros(n, 1);
peak_e = 0.0;

for step = 0:(n_steps - 1)
    t = step * P.DT;
    if mod(step, P.H_STEPS) == 0
        k = step / P.H_STEPS;
        [p0t, v0t, a0t] = leader_state(t, P);
        phi = [P.KP * p0t + P.KD * v0t; P.KP * p + P.KD * v];

        c_minus = phi_bar - phi(1:n);
        if k == 0
            fire_c = false(n, 1);
            phi_bar = phi(1:n);
            c = zeros(n, 1);
        else
            fire_c = should_fire(comm_mode, c_minus.^2, eta_c, P.DELTA_C, P.NU_C);
            predecessor_phi = phi(1:n);
            phi_bar(fire_c) = predecessor_phi(fire_c);
            c = c_minus;
            c(fire_c) = 0.0;
        end

        u_candidate = -P.KP * p - P.KD * v + phi_bar + P.KP * P.D_STAR - what;
        if P.LEADER_FF
            u_candidate(1) = u_candidate(1) + a0t;
        end
        b_minus = u_hold - u_candidate;
        if k == 0
            fire_u = false(n, 1);
            u_hold = u_candidate;
            b = zeros(n, 1);
        else
            fire_u = should_fire(ctrl_mode, b_minus.^2, eta_u, P.DELTA_U, P.NU_U);
            u_hold(fire_u) = u_candidate(fire_u);
            b = b_minus;
            b(fire_u) = 0.0;
        end

        if strcmp(comm_mode, 'dynamic')
            eta_c = max(P.RHO_C * eta_c + P.CHI_C * P.DELTA_C - P.CHI_C * c.^2, 0.0);
        end
        if strcmp(ctrl_mode, 'dynamic')
            eta_u = max(P.RHO_U * eta_u + P.CHI_U * P.DELTA_U - P.CHI_U * b.^2, 0.0);
        end

        in_window = (t >= P.T_WIN0) && (t < P.T_WIN1);
        if in_window
            n_c = n_c + double(fire_c);
            n_u = n_u + double(fire_u);
            e = p - [p0t; p(1:end-1)] - P.D_STAR;
            sum_e2 = sum_e2 + sum(e.^2);
            sample_count = sample_count + n;
            sample_times = sample_times + 1;
            sum_edge_e2 = sum_edge_e2 + e.^2;
            peak_e = max(peak_e, max(abs(e)));
        end
    end

    t_mid = t + 0.5 * P.DT;
    w_mid = P.W_AMP * sin(P.W_OMEGA * t_mid + P.W_PHASE_STEP * idx);
    acc = u_hold + w_mid;
    p_next = p + P.DT * v + 0.5 * P.DT^2 * acc;
    v_next = v + P.DT * acc;
    observer_error = p_next - phat;
    phat = phat + P.DT * (vhat + P.BETA1 * observer_error);
    vhat = vhat + P.DT * (u_hold + what + P.BETA2 * observer_error);
    what = what + P.DT * (P.BETA3 * observer_error);
    p = p_next;
    v = v_next;
end

met.rmse_e = sqrt(sum_e2 / sample_count);
met.peak_e = peak_e;
met.n_c = mean(n_c);
met.n_u = mean(n_u);
met.r_c = 1.0 - met.n_c / P.N_WIN;
met.r_u = 1.0 - met.n_u / P.N_WIN;
met.per_edge_rmse = sqrt(sum_edge_e2 / sample_times);
met.rmse_tail = met.per_edge_rmse(end);
end

function [P, n_max_init] = apply_overrides(P, varargin)
parser = inputParser;
addParameter(parser, 'N', P.N, @(x) isscalar(x) && x >= 1 && x == round(x));
addParameter(parser, 'T_SIM', P.T_SIM, @(x) isscalar(x) && x > 0);
addParameter(parser, 'T_WIN0', P.T_WIN0, @(x) isscalar(x) && x >= 0);
addParameter(parser, 'T_WIN1', P.T_WIN1, @(x) isscalar(x) && x > 0);
addParameter(parser, 'NMaxInit', [], @(x) isempty(x) || (isscalar(x) && x >= 1 && x == round(x)));
parse(parser, varargin{:});
P.N = parser.Results.N;
P.T_SIM = parser.Results.T_SIM;
P.T_WIN0 = parser.Results.T_WIN0;
P.T_WIN1 = parser.Results.T_WIN1;
if P.T_WIN0 >= P.T_WIN1 || P.T_WIN1 > P.T_SIM
    error('Require 0 <= T_WIN0 < T_WIN1 <= T_SIM.');
end
P.N_WIN = round((P.T_WIN1 - P.T_WIN0) / P.H);
n_max_init = parser.Results.NMaxInit;
end

function fire = should_fire(mode, err2, eta, delta, nu)
switch mode
    case 'always'
        fire = true(size(err2));
    case 'static'
        fire = err2 > delta;
    case 'dynamic'
        fire = err2 > nu * eta + delta;
    otherwise
        error('Unknown trigger mode: %s', mode);
end
end

function [comm_mode, ctrl_mode] = method_modes(method)
switch method
    case 'periodic'
        comm_mode = 'always'; ctrl_mode = 'always';
    case 'static_single'
        comm_mode = 'static'; ctrl_mode = 'always';
    case 'static_dual'
        comm_mode = 'static'; ctrl_mode = 'static';
    case 'dynamic_single'
        comm_mode = 'dynamic'; ctrl_mode = 'always';
    case 'dynamic_dual'
        comm_mode = 'dynamic'; ctrl_mode = 'dynamic';
    otherwise
        error('Unknown method: %s', method);
end
end
