function run_n_sweep()
% Frozen-parameter N-sweep with nested initial conditions up to N=20.

P = petc_params();
Ns = [5, 10, 15, 20];
methods = {'periodic', 'dynamic_dual'};
labels = {'Periodic', 'Dynamic-Dual'};
seeds = 0:19;
T_SIM = 40.0;
T_WIN0 = 20.0;
T_WIN1 = 40.0;
N_MAX_INIT = max(Ns);

n_n = numel(Ns);
n_methods = numel(methods);
n_seeds = numel(seeds);
n_rows = n_n * n_methods * n_seeds;

N_col = zeros(n_rows, 1);
method_index = zeros(n_rows, 1);
method_col = cell(n_rows, 1);
seed_col = zeros(n_rows, 1);
row = 0;
for iN = 1:n_n
    for im = 1:n_methods
        for is = 1:n_seeds
            row = row + 1;
            N_col(row) = Ns(iN);
            method_index(row) = im;
            method_col{row} = methods{im};
            seed_col(row) = seeds(is);
        end
    end
end

rmse_all_col = zeros(n_rows, 1);
rmse_tail_col = zeros(n_rows, 1);
peak_col = zeros(n_rows, 1);
nc_col = zeros(n_rows, 1);
nu_col = zeros(n_rows, 1);
rc_col = zeros(n_rows, 1);
ru_col = zeros(n_rows, 1);
profiles = cell(n_rows, 1);

pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('Threads', min(8, feature('numcores')));
end
dq = parallel.pool.DataQueue;
afterEach(dq, @(~) report_n_progress(n_rows));
parfor i = 1:n_rows
    met = run_sim_metrics(P, methods{method_index(i)}, seed_col(i), ...
        'N', N_col(i), 'T_SIM', T_SIM, 'T_WIN0', T_WIN0, ...
        'T_WIN1', T_WIN1, 'NMaxInit', N_MAX_INIT);
    rmse_all_col(i) = met.rmse_e;
    rmse_tail_col(i) = met.rmse_tail;
    peak_col(i) = met.peak_e;
    nc_col(i) = met.n_c;
    nu_col(i) = met.n_u;
    rc_col(i) = met.r_c;
    ru_col(i) = met.r_u;
    profiles{i} = met.per_edge_rmse;
    send(dq, i);
end

raw = table(N_col, method_col, seed_col, rmse_all_col, rmse_tail_col, peak_col, ...
    nc_col, nu_col, rc_col, ru_col, 'VariableNames', ...
    {'N', 'method', 'seed', 'rmse_all', 'rmse_tail', 'peak', 'n_c', 'n_u', 'r_c', 'r_u'});

n_summary = n_n * n_methods;
sum_N = zeros(n_summary, 1);
sum_method = cell(n_summary, 1);
sum_label = cell(n_summary, 1);
metric_names = {'rmse_all', 'rmse_tail', 'peak', 'n_c', 'n_u', 'r_c', 'r_u'};
means = zeros(n_summary, numel(metric_names));
stds = zeros(n_summary, numel(metric_names));
row = 0;
for iN = 1:n_n
    for im = 1:n_methods
        row = row + 1;
        mask = raw.N == Ns(iN) & strcmp(raw.method, methods{im});
        sum_N(row) = Ns(iN);
        sum_method{row} = methods{im};
        sum_label{row} = labels{im};
        for k = 1:numel(metric_names)
            values = raw.(metric_names{k})(mask);
            means(row, k) = mean(values);
            stds(row, k) = std(values, 0);
        end
    end
end

summary = table(sum_N, sum_method, sum_label, ...
    means(:,1), stds(:,1), means(:,2), stds(:,2), means(:,3), stds(:,3), ...
    means(:,4), stds(:,4), means(:,5), stds(:,5), means(:,6), stds(:,6), ...
    means(:,7), stds(:,7), 'VariableNames', ...
    {'N', 'method', 'label', 'rmse_all_mean', 'rmse_all_std', ...
    'rmse_tail_mean', 'rmse_tail_std', 'peak_mean', 'peak_std', ...
    'n_c_mean', 'n_c_std', 'n_u_mean', 'n_u_std', ...
    'r_c_mean', 'r_c_std', 'r_u_mean', 'r_u_std'});

profile_N = [];
profile_method = {};
profile_seed = [];
profile_edge = [];
profile_rmse = [];
for i = 1:n_rows
    ni = N_col(i);
    profile_N = [profile_N; repmat(ni, ni, 1)]; %#ok<AGROW>
    profile_method = [profile_method; repmat(method_col(i), ni, 1)]; %#ok<AGROW>
    profile_seed = [profile_seed; repmat(seed_col(i), ni, 1)]; %#ok<AGROW>
    profile_edge = [profile_edge; (1:ni)']; %#ok<AGROW>
    profile_rmse = [profile_rmse; profiles{i}(:)]; %#ok<AGROW>
end
profile_raw = table(profile_N, profile_method, profile_seed, profile_edge, profile_rmse, ...
    'VariableNames', {'N', 'method', 'seed', 'edge', 'rmse_edge'});

profile_summary_N = [];
profile_summary_method = {};
profile_summary_edge = [];
profile_mean = [];
profile_std = [];
for iN = 1:n_n
    for im = 1:n_methods
        for edge = 1:Ns(iN)
            mask = profile_raw.N == Ns(iN) & strcmp(profile_raw.method, methods{im}) ...
                & profile_raw.edge == edge;
            values = profile_raw.rmse_edge(mask);
            profile_summary_N(end+1,1) = Ns(iN); %#ok<AGROW>
            profile_summary_method{end+1,1} = methods{im}; %#ok<AGROW>
            profile_summary_edge(end+1,1) = edge; %#ok<AGROW>
            profile_mean(end+1,1) = mean(values); %#ok<AGROW>
            profile_std(end+1,1) = std(values, 0); %#ok<AGROW>
        end
    end
end
profile_summary = table(profile_summary_N, profile_summary_method, profile_summary_edge, ...
    profile_mean, profile_std, 'VariableNames', ...
    {'N', 'method', 'edge', 'rmse_edge_mean', 'rmse_edge_std'});

root = fileparts(fileparts(mfilename('fullpath')));
data_dir = fullfile(root, 'data');
fig_dir = fullfile(root, 'figs');
if ~exist(data_dir, 'dir'), mkdir(data_dir); end
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
writetable(raw, fullfile(data_dir, 'n_sweep_20seeds_raw.csv'));
writetable(summary, fullfile(data_dir, 'table_n_sweep_final.csv'));
writetable(profile_raw, fullfile(data_dir, 'n_sweep_edge_profiles_raw.csv'));
writetable(profile_summary, fullfile(data_dir, 'n_sweep_edge_profiles_summary.csv'));

payload.parameter_version = P.parameter_version;
payload.rng = 'MATLAB twister with nested Nmax=20 initial conditions';
payload.N_values = Ns;
payload.methods = methods;
payload.seeds = seeds;
payload.T_SIM = T_SIM;
payload.T_WIN0 = T_WIN0;
payload.T_WIN1 = T_WIN1;
payload.frozen_parameters = P;
payload.raw = table2struct(raw);
payload.summary = table2struct(summary);
payload.profile_summary = table2struct(profile_summary);
fid = fopen(fullfile(data_dir, 'table_n_sweep_final.json'), 'w');
fprintf(fid, '%s', jsonencode(payload));
fclose(fid);

periodic_mask = strcmp(summary.method, 'periodic');
dynamic_mask = strcmp(summary.method, 'dynamic_dual');
periodic_rows = summary(periodic_mask, :);
dynamic_rows = summary(dynamic_mask, :);

f1 = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1250, 520]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
plot(periodic_rows.N, periodic_rows.rmse_all_mean, '-o', 'LineWidth', 1.7, 'DisplayName', 'Periodic all'); hold on;
plot(dynamic_rows.N, dynamic_rows.rmse_all_mean, '-s', 'LineWidth', 1.7, 'DisplayName', 'Dynamic-Dual all');
plot(periodic_rows.N, periodic_rows.rmse_tail_mean, '--o', 'LineWidth', 1.5, 'DisplayName', 'Periodic tail');
plot(dynamic_rows.N, dynamic_rows.rmse_tail_mean, '--s', 'LineWidth', 1.5, 'DisplayName', 'Dynamic-Dual tail');
xlabel('Chain length N'); ylabel('RMSE'); title('(a) Performance versus chain length');
set(gca, 'XTick', Ns); grid on; box on; legend('Location', 'northwest');
nexttile;
plot(dynamic_rows.N, 100 * dynamic_rows.r_c_mean, '-o', 'LineWidth', 1.7, 'DisplayName', 'R_c'); hold on;
plot(dynamic_rows.N, 100 * dynamic_rows.r_u_mean, '-s', 'LineWidth', 1.7, 'DisplayName', 'R_u');
xlabel('Chain length N'); ylabel('Resource saving (%)'); title('(b) Dynamic-Dual resource saving');
set(gca, 'XTick', Ns); ylim([0, 100]); grid on; box on; legend('Location', 'best');
exportgraphics(f1, fullfile(fig_dir, 'fig_n_sweep_performance_resource_final.png'), 'Resolution', 300);
exportgraphics(f1, fullfile(fig_dir, 'fig_n_sweep_performance_resource_final.pdf'), 'ContentType', 'vector');
close(f1);

f2 = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 900, 560]);
hold on;
colors = lines(n_n);
h_edge = gobjects(n_n, 1);
for iN = n_n:-1:1
    mask = profile_summary.N == Ns(iN) & strcmp(profile_summary.method, 'dynamic_dual');
    rows_i = profile_summary(mask, :);
    h_edge(iN) = plot(rows_i.edge, rows_i.rmse_edge_mean, '-o', 'LineWidth', 1.6, ...
        'Color', colors(iN, :), 'DisplayName', sprintf('N=%d', Ns(iN)));
end
xlabel('Edge index i'); ylabel('Per-edge RMSE');
title('Dynamic-Dual downstream error propagation');
grid on; box on;
legend(h_edge, arrayfun(@(N) sprintf('N=%d', N), Ns, ...
    'UniformOutput', false), 'Location', 'northwest');
exportgraphics(f2, fullfile(fig_dir, 'fig_n_sweep_edge_profile_final.png'), 'Resolution', 300);
exportgraphics(f2, fullfile(fig_dir, 'fig_n_sweep_edge_profile_final.pdf'), 'ContentType', 'vector');
close(f2);

disp(summary);
fprintf('N-sweep complete.\n');
end

function report_n_progress(total)
persistent completed
if isempty(completed), completed = 0; end
completed = completed + 1;
fprintf('completed %d/%d\n', completed, total);
end
