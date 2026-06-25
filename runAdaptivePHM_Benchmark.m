% RUNADAPTIVEPHM_BENCHMARK Test matrix (3 terrains x 3 tree densities) comparing
% PriorityPlanner vs Lawnmower vs Greedy baselines on adaptive IoT-weighted CPP.
%
% Requires this folder on the MATLAB path.
%
% Outputs:
%   - metricsSummary.mat / metricsSummary.csv (optional export below)
%   - Figures for trajectories, energy curves, KPI bars

clear; close all; clc;

projRoot = fileparts(mfilename('fullpath'));
addpath(projRoot);

%% Simulation metadata
N = 54;
M = 72;
dx = 18;              % meters per cell edge
numUAV = 3;

terrains = {'flat', 'hilly', 'ridge'};
densities = {'low', 'medium', 'high'};

simCfg = struct('dt', 1.25, 'vMax', 26, 'dSafe', 22, 'altitudeAGL', 32, ...
    'P0', 165, 'kSpeed', 0.28, 'kAlt', 0.78, 'stepTolM', 38);

rows = struct('planner', {}, 'terrain', {}, 'density', {}, 'metrics', {}, 'log', {}, 'plan', {});

scenarioIdx = 0;

for iT = 1:numel(terrains)
    for iD = 1:numel(densities)
        scenarioIdx = scenarioIdx + 1;
        seedBase = 8100 + 17 * scenarioIdx;

        opts = struct('terrain', terrains{iT}, ...
            'treeDensity', densities{iD}, ...
            'alpha', 1.0, 'beta', 0.55, 'gamma', 0.32, ...
            'rngSeed', seedBase);

        mapData = MapGenerator.build(N, M, dx, opts);

        batch = BenchmarkComparison.runSingleMap(mapData, numUAV, simCfg);

        for k = 1:numel(batch)
            r = struct();
            r.planner = batch(k).name;
            r.terrain = terrains{iT};
            r.density = densities{iD};
            r.metrics = batch(k).metrics;
            r.log = batch(k).log;
            r.plan = batch(k).plan;
            rows(end + 1) = r; %#ok<AGROW>
        end

        fprintf('Scenario %s / %s completed.\n', terrains{iT}, densities{iD});
    end
end

summaryTbl = Analytics.summarizeBenchmark(rows);
disp(summaryTbl);

save(fullfile(projRoot, 'metricsSummary.mat'), 'rows', 'summaryTbl', '-v7.3');

try
    writetable(summaryTbl, fullfile(projRoot, 'metricsSummary.csv'));
catch
    % Some MATLAB installs restrict writetable without Excel license — MAT export persists.
end

%% Visualization — representative scenario (center of test matrix)
midOpts = struct('terrain', terrains{2}, 'treeDensity', densities{2}, ...
    'alpha', 1.0, 'beta', 0.55, 'gamma', 0.32, 'rngSeed', 8148);

mapShow = MapGenerator.build(N, M, dx, midOpts);
batchShow = BenchmarkComparison.runSingleMap(mapShow, numUAV, simCfg);

PlotPHM.trajectoryHeatmap(mapShow, batchShow(1).plan, "priority hybrid");
PlotPHM.trajectoryHeatmap(mapShow, batchShow(2).plan, "lawnmower baseline");

logs = {batchShow(1).log, batchShow(2).log, batchShow(3).log};
lbl = ["priority hybrid", "lawnmower", "greedy"];
PlotPHM.energyTime(logs, lbl);

PlotPHM.metricBars(summaryTbl);

fprintf('\nBenchmark finished. Inspect figures and metricsSummary.mat.\n');
