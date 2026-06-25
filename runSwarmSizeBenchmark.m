% RUNSWARMSIZEBENCHMARK Compare fleet sizes 1 vs 5 vs 10 on one fixed map.
%
%   Primary (comparable): BatteryAwareOrchestrator — same HPC target for every
%   fleet size; multi-sortie extends horizon so results are not mixed
%   "stopped early" vs "plan ended".
%
%   Secondary: single full plan (no stopHpcPct) — same definition of mission
%   time (plan horizon) and HPC% at end for apples-to-apples single-pass study.
%
%   SpatialDecomposition / PriorityPlanner fixes (n<=K targets, empty goals)
%   remove duplicate full-map assignment that caused 10-UAV energy/collision blow-ups.
%
%   Outputs: swarmSizeBenchmark.mat, swarmSizeBenchmark.fig
%   Also ranks tested fleet sizes via SwarmSizeOptimizer (recOptimal).
%   For a full 1:U sweep use runSwarmSizeOptimization.m.

clear;
close all;
clc;

projRoot = fileparts(mfilename('fullpath'));
addpath(projRoot);

%% Map (identical for all fleet sizes)
N = 54;
M = 72;
dx = 18;
mapOpts = struct('terrain', 'hilly', 'treeDensity', 'medium', ...
    'alpha', 1.0, 'beta', 0.55, 'gamma', 0.32, 'rngSeed', 91501);

mapData = MapGenerator.build(N, M, dx, mapOpts);

fleetSizes = [1, 5, 10];
hpcTargetFrac = 0.85;

nF = numel(fleetSizes);

%% --- A) Battery-aware: same HPC target, fair horizon ---
orchTime_s = zeros(nF, 1);
orchEnergy_J = zeros(nF, 1);
orchHpc_pct = zeros(nF, 1);
orchSorties = zeros(nF, 1);
orchEbit_J = nan(nF, 1);
orchColl = false(nF, 1);
orchHitTarget = false(nF, 1);
orchAggs = cell(nF, 1);

for k = 1:nF
    U = fleetSizes(k);
    dSafe = max(22, 17 + 1.35 * U);
    po = struct('planMode', 'blend', 'blendGamma', 0.42, ...
        'partitionMethod', 'voronoi_capacitated', ...
        'smoothPathIterations', 1, 'voronoiIterations', min(44, 16 + 2 * U));

    sc = struct('dt', 1.25, 'vMax', 26, 'dSafe', dSafe, ...
        'altitudeAGL', 32, 'usePowerModel3D', true, 'stepTolM', 38, ...
        'dynamicIoTEnable', false);

    orchCfg = struct();
    orchCfg.hpcTarget = hpcTargetFrac;
    orchCfg.maxSorties = 48;
    orchCfg.sortieEnergy_J = 8.8e5;
    orchCfg.rechargeTime_s = 85;
    orchCfg.dynamicIoTBetweenSorties = false;
    orchCfg.plannerOpts = po;
    orchCfg.simCfg = sc;

    agg = BatteryAwareOrchestrator.runMultiPass(mapData, U, orchCfg);
    met = Analytics.aggregateMissionMetrics(mapData, agg);

    orchAggs{k} = agg;
    orchTime_s(k) = met.total_mission_time_s;
    orchEnergy_J(k) = met.total_energy_J;
    orchHpc_pct(k) = met.HPC_pct;
    orchSorties(k) = met.sortie_count;
    orchEbit_J(k) = met.energy_per_bit_J;
    orchColl(k) = met.any_collision;
    orchHitTarget(k) = agg.hpc_final_pct >= 100 * hpcTargetFrac - 0.5;
end

summaryOrchestrated = table(fleetSizes(:), orchTime_s, orchEnergy_J / 1e6, ...
    (orchEnergy_J ./ fleetSizes(:)) / 1e6, orchHpc_pct, orchSorties, orchEbit_J, ...
    orchColl, orchHitTarget, ...
    'VariableNames', {'numUAV', 'mission_time_s', 'fleet_energy_MJ', ...
    'energy_per_UAV_MJ', 'HPC_pct', 'sorties', 'energy_per_bit_J', 'any_collision', 'met_hpc_target'});

fprintf('=== Battery-aware (same HPC target = %.0f%%) ===\n', 100 * hpcTargetFrac);
disp(summaryOrchestrated);

if ~all(orchHitTarget)
    warning(['Orchestrated runs did not all reach HPC target (see met_hpc_target). ', ...
        'Raise maxSorties or sortieEnergy_J.']);
end

%% --- A2) Optimal fleet size (weighted score on orchestrated metrics) ---
orchOptsForScore = struct('hpcTargetFrac', hpcTargetFrac, 'maxSorties', 48, ...
    'sortieEnergy_J', 8.8e5, 'rechargeTime_s', 85, 'blendGamma', 0.42);
scoreOptsBench = struct();

recOptimal = SwarmSizeOptimizer.recommendFromMetrics(summaryOrchestrated, ...
    orchOptsForScore, scoreOptsBench);

fprintf('\n=== Recommended fleet (lowest composite score among tested sizes) ===\n');
fprintf('Optimal numUAV: %d  (score = %.4f)\n', recOptimal.optimal_numUAV, recOptimal.best_composite_score);
disp(recOptimal.score_breakdown);
fprintf(['Weights: time=%.2f, energy=%.2f, energy/bit=%.2f; ', ...
    'penalties: collision=%.1f, HPC gap scale=%.1f/0.01 HPC miss. ', ...
    'Edit scoreOptsBench in runSwarmSizeBenchmark or use runSwarmSizeOptimization for sweeps.\n'], ...
    recOptimal.score_opts.w_time, recOptimal.score_opts.w_energy, ...
    recOptimal.score_opts.w_energy_per_bit, recOptimal.score_opts.penalty_collision, ...
    recOptimal.score_opts.penalty_hpc_per_unit_gap);

%% --- B) Single-pass full plan: same stopping rule (end of plan) for every U ---
fullTime_s = zeros(nF, 1);
fullEnergy_J = zeros(nF, 1);
fullHpc_pct = zeros(nF, 1);
fullEbit_J = nan(nF, 1);
fullColl = false(nF, 1);
fullLogs = cell(nF, 1);

for k = 1:nF
    U = fleetSizes(k);
    dSafe = max(22, 17 + 1.35 * U);
    plannerOpts = struct('planMode', 'blend', 'blendGamma', 0.42, ...
        'partitionMethod', 'voronoi_capacitated', ...
        'smoothPathIterations', 1, 'voronoiIterations', min(44, 16 + 2 * U));

    simCfg = struct('dt', 1.25, 'vMax', 26, 'dSafe', dSafe, ...
        'altitudeAGL', 32, 'usePowerModel3D', true, 'stepTolM', 38);

    plan = PriorityPlanner.buildPlan(mapData, U, plannerOpts);
    log = MissionSim.run(mapData, plan, simCfg);
    m = Analytics.computeMetrics(mapData, log);

    T = size(log.positions, 1);
    fullTime_s(k) = T * log.dt;
    fullEnergy_J(k) = m.total_energy_J;
    fullHpc_pct(k) = m.HPC_pct;
    fullEbit_J(k) = m.energy_per_bit_J;
    fullColl(k) = m.any_collision;
    fullLogs{k} = log;
end

speedup_vs_1 = fullTime_s(1) ./ max(fullTime_s, eps);

summaryFullPlan = table(fleetSizes(:), fullTime_s, fullHpc_pct, fullEnergy_J / 1e6, ...
    (fullEnergy_J ./ fleetSizes(:)) / 1e6, fullEbit_J, speedup_vs_1, fullColl, ...
    'VariableNames', {'numUAV', 'plan_horizon_s', 'HPC_pct_at_plan_end', ...
    'fleet_energy_MJ', 'energy_per_UAV_MJ', 'energy_per_bit_J', 'time_vs_1x_shorter_if_gt1', 'any_collision'});

fprintf('\n=== Single-pass (full plan, comparable horizon per run) ===\n');
disp(summaryFullPlan);

save(fullfile(projRoot, 'swarmSizeBenchmark.mat'), ...
    'mapOpts', 'fleetSizes', 'hpcTargetFrac', 'mapData', ...
    'summaryOrchestrated', 'summaryFullPlan', 'orchAggs', 'fullLogs', ...
    'recOptimal', 'orchOptsForScore', 'scoreOptsBench', '-v7.3');

fig = figure('Color', 'w', 'Position', [60, 60, 1000, 360]);
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
bar(fleetSizes, orchTime_s, 'FaceColor', [0.2 0.45 0.7]);
grid on;
xlabel('Fleet size');
ylabel('Time (s)');
title(sprintf('Orchestrated time (HPC target %.0f%%)', 100 * hpcTargetFrac));

nexttile;
bar(fleetSizes, orchEnergy_J / 1e6, 'FaceColor', [0.55 0.35 0.2]);
grid on;
xlabel('Fleet size');
ylabel('Fleet energy (MJ)');
title('Orchestrated total energy');

nexttile;
bar(fleetSizes, fullTime_s, 'FaceColor', [0.35 0.5 0.65]);
grid on;
xlabel('Fleet size');
ylabel('Plan horizon (s)');
title('Single-pass plan duration');

nexttile;
bar(fleetSizes, fullHpc_pct, 'FaceColor', [0.25 0.62 0.38]);
grid on;
xlabel('Fleet size');
ylabel('HPC (%)');
title('HPC at end of single plan');

savefig(fig, fullfile(projRoot, 'swarmSizeBenchmark.fig'));

fprintf('\nSaved swarmSizeBenchmark.mat / .fig\n');
fprintf(['Primary table: same HPC goal and multi-sortie horizon — use for swarm scaling.\n', ...
    'Full-plan table: same stop rule (end of synthesized path); HPC%% may differ by fleet.\n']);
