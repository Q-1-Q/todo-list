function P = petc_params()
% Frozen parameters shared by all formal MATLAB experiments.

P.parameter_version = 'petc-chain-frozen-v2-matlab';
P.N = 10;
P.D_STAR = 1.0;
P.T_SIM = 20.0;
P.T_WIN0 = 10.0;
P.T_WIN1 = 20.0;
P.DT = 1.0e-4;
P.H = 0.01;
P.H_STEPS = round(P.H / P.DT);
P.N_WIN = round((P.T_WIN1 - P.T_WIN0) / P.H);

P.KP = 1.0;
P.KD = 3.0;

P.A0_AMP = 0.1;
P.A0_W = 0.5;
P.V0_INIT = 0.5;
P.P0_INIT = 0.0;

P.W_AMP = 0.1;
P.W_OMEGA = 0.5;
P.W_PHASE_STEP = 0.6;

P.WO = 20.0;
P.BETA1 = 3.0 * P.WO;
P.BETA2 = 3.0 * P.WO^2;
P.BETA3 = P.WO^3;

P.RHO_C = 0.9;
P.RHO_U = 0.9;
P.NU_C = 1.0;
P.NU_U = 1.0;
P.CHI_C = 0.5;
P.CHI_U = 0.5;
P.DELTA_C = 2.0e-4;
P.DELTA_U = 2.0e-4;
P.ETA0 = 0.01;

P.P_INIT_LOW = -0.2;
P.P_INIT_HIGH = 0.2;
P.V_INIT_LOW = -0.2;
P.V_INIT_HIGH = 0.2;
P.LEADER_FF = true;
end
