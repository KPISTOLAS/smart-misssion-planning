% RUNSWARMSIZEOPTIMIZATION Sweep fleet sizes and recommend optimal numUAV.
%   Uses SwarmSizeOptimizer + BatteryAwareOrchestrator (same HPC target for all).
%
%   Tune scoreOpts to emphasize time vs energy vs information efficiency.
%
%   Outputs: swarmSizeOptimization.mat, swarmSizeOptimization.fig

clear;
close all;
clc;

projRoot = fileparts(mfilename('fullpath'));
addpath(projRoot);

N = 54;
M = 72;
dx = 18;
mapOpts = struct('terrain', 'hilly', 'treeDensity', 'medium', ...
    'alpha', 1.0, 'beta', 0.55, 'gamma', 0.32, 'rngSeed', 91501);

mapData = MapGenerator.build(N, M, dx, mapOpts);

fleetSweep = 1:12;

orchOpts = struct('hpcTargetFrac', 0.85, 'maxSorties', 48, ...
    'sortieEnergy_J', 8.8e5, 'rechargeTime_s', 85, 'blendGamma', 0.42);

scoreOpts = struct( ...
    'w_time', 0.40, ...
    'w_energy', 0.30, ...
    'w_energy_per_bit', 0.30, ...
    'penalty_collision', 8, ...
    'penalty_hpc_per_unit_gap', 30, ...
    'feasible_only', false);

rec = SwarmSizeOptimizer.recommend(mapData, fleetSweep, orchOpts, scoreOpts);

fprintf('=== Recommended fleet size (lowest composite score) ===\n');
fprintf('Optimal numUAV: %d  (composite score = %.4f)\n\n', rec.optimal_numUAV, rec.best_composite_score);
disp(rec.score_breakdown);

save(fullfile(projRoot, 'swarmSizeOptimization.mat'), 'mapOpts', 'fleetSweep', ...
    'orchOpts', 'scoreOpts', 'rec', 'mapData', '-v7.3');

fig = figure('Color', 'w', 'Position', [80, 80, 880, 360]);
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
b = bar(rec.table.numUAV, rec.scores, 'FaceColor', [0.25 0.45 0.62]);
hold on;
xline(rec.optimal_numUAV, 'r--', 'LineWidth', 1.8);
grid on;
xlabel('Fleet size');
ylabel('Composite score (lower is better)');
title('Fleet ranking');

nexttile;
yyaxis left;
plot(rec.table.numUAV, rec.table.mission_time_s, '-o', 'LineWidth', 1.4);
ylabel('Mission time (s)');
yyaxis right;
plot(rec.table.numUAV, rec.table.fleet_energy_MJ, '-s', 'LineWidth', 1.4);
ylabel('Fleet energy (MJ)');
grid on;
xlabel('Fleet size');
title('Time vs energy sweep');

savefig(fig, fullfile(projRoot, 'swarmSizeOptimization.fig'));

fprintf('Saved swarmSizeOptimization.mat / .fig\n');
