% RUNIEEESTUDY Paper-ready experiment sweep for multi-UAV PHM-CPP.
%   Runs repeated orchestrated missions across terrain/tree-density/seed/fleet
%   combinations and stores raw outputs for statistical analysis.
%
% Outputs:
%   - ieeeStudyRaw.mat  (raw per-run structs)
%   - ieeeStudyRaw.csv  (flat per-run table)

clear;
close all;
clc;

projRoot = fileparts(mfilename('fullpath'));
addpath(projRoot);

%% Study design (edit these values for your paper)
terrains = {'flat', 'hilly', 'ridge'};
densities = {'low', 'medium', 'high'};
fleetSweep = 1:12;
numSeedsPerScenario = 20;
seedBase = 24001;

%% Shared mission and optimizer settings
N = 54;
M = 72;
dx = 18;

orchOpts = struct( ...
    'hpcTargetFrac', 0.85, ...
    'maxSorties', 48, ...
    'sortieEnergy_J', 8.8e5, ...
    'rechargeTime_s', 85, ...
    'blendGamma', 0.42, ...
    'dt', 1.25, ...
    'vMax', 26, ...
    'altitudeAGL', 32, ...
    'stepTolM', 38);

scoreOpts = struct( ...
    'w_time', 0.40, ...
    'w_energy', 0.30, ...
    'w_energy_per_bit', 0.30, ...
    'penalty_collision', 8.0, ...
    'penalty_hpc_per_unit_gap', 30.0, ...
    'feasible_only', false);

%% Execution
rows = struct('terrain', {}, 'density', {}, 'seed', {}, 'obsFrac', {}, ...
    'numUAV', {}, 'mission_time_s', {}, 'fleet_energy_MJ', {}, ...
    'energy_per_UAV_MJ', {}, 'HPC_pct', {}, 'sorties', {}, ...
    'energy_per_bit_J', {}, 'any_collision', {}, 'met_hpc_target', {}, ...
    'composite_score', {}, 'is_optimal_for_seed', {});

scenarioCount = 0;
for iT = 1:numel(terrains)
    for iD = 1:numel(densities)
        for iS = 1:numSeedsPerScenario
            scenarioCount = scenarioCount + 1;
            seed = seedBase + 1000 * (iT - 1) + 100 * (iD - 1) + iS;

            mapOpts = struct('terrain', terrains{iT}, ...
                'treeDensity', densities{iD}, ...
                'alpha', 1.0, 'beta', 0.55, 'gamma', 0.32, ...
                'rngSeed', seed);

            mapData = MapGenerator.build(N, M, dx, mapOpts);

            rec = SwarmSizeOptimizer.recommend(mapData, fleetSweep, orchOpts, scoreOpts);
            tbl = rec.table;

            for k = 1:height(tbl)
                rr = struct();
                rr.terrain = string(terrains{iT});
                rr.density = string(densities{iD});
                rr.seed = seed;
                rr.obsFrac = mapData.obsFrac;
                rr.numUAV = tbl.numUAV(k);
                rr.mission_time_s = tbl.mission_time_s(k);
                rr.fleet_energy_MJ = tbl.fleet_energy_MJ(k);
                rr.energy_per_UAV_MJ = tbl.energy_per_UAV_MJ(k);
                rr.HPC_pct = tbl.HPC_pct(k);
                rr.sorties = tbl.sorties(k);
                rr.energy_per_bit_J = tbl.energy_per_bit_J(k);
                rr.any_collision = tbl.any_collision(k);
                rr.met_hpc_target = tbl.met_hpc_target(k);
                rr.composite_score = rec.scores(k);
                rr.is_optimal_for_seed = (tbl.numUAV(k) == rec.optimal_numUAV);
                rows(end + 1) = rr; %#ok<AGROW>
            end

            fprintf('[%03d] %s/%s seed=%d done. Best U=%d\n', scenarioCount, ...
                terrains{iT}, densities{iD}, seed, rec.optimal_numUAV);
        end
    end
end

rawTbl = struct2table(rows);

save(fullfile(projRoot, 'ieeeStudyRaw.mat'), ...
    'rows', 'rawTbl', 'terrains', 'densities', 'fleetSweep', ...
    'numSeedsPerScenario', 'seedBase', 'N', 'M', 'dx', ...
    'orchOpts', 'scoreOpts', '-v7.3');

try
    writetable(rawTbl, fullfile(projRoot, 'ieeeStudyRaw.csv'));
catch
    warning('Could not write ieeeStudyRaw.csv. MAT file was saved successfully.');
end

fprintf('\nSaved ieeeStudyRaw.mat and ieeeStudyRaw.csv\n');
