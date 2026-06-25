% RUNIEEECOMPARATIVESTUDY Gold-standard baseline comparison for IEEE narrative.
%   Algorithm X = PriorityPlanner hybrid under BatteryAwareOrchestrator (plannerMode 'priority').
%   Baseline A = myopic multi-UAV greedy (BaselinePlanners.greedy).
%   Baseline B = decentralized surrogate: nearest-depot Voronoi task assignment +
%                per-UAV weighted greedy tour with local A* stitching only
%                (BaselinePlanners.decentralizedVoronoiGreedy).
%
%   Outputs:
%     ieeeComparativeRaw.mat / ieeeComparativeRaw.csv
%
%   Defaults are smaller than runIEEEStudy for runtime; scale up for final paper.

clear;
close all;
clc;

projRoot = fileparts(mfilename('fullpath'));
addpath(projRoot);

terrains = {'hilly'};
densities = {'low', 'medium', 'high'};
planners = {'priority', 'greedy', 'decentralized_greedy'};
fleetSweep = 1:10;
numSeedsPerScenario = 6;
seedBase = 31001;

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

rows = struct('planner', {}, 'terrain', {}, 'density', {}, 'seed', {}, ...
    'obsFrac', {}, 'numUAV', {}, 'mission_time_s', {}, 'fleet_energy_MJ', {}, ...
    'energy_per_UAV_MJ', {}, 'HPC_pct', {}, 'sorties', {}, ...
    'energy_per_bit_J', {}, 'any_collision', {}, 'met_hpc_target', {}, ...
    'composite_score', {}, 'is_optimal_for_seed', {});

runId = 0;

for iP = 1:numel(planners)
    orchOpts.plannerMode = planners{iP};

    for iT = 1:numel(terrains)
        for iD = 1:numel(densities)
            for iS = 1:numSeedsPerScenario
                runId = runId + 1;
                seed = seedBase + 500 * (iP - 1) + 100 * (iT - 1) + 20 * (iD - 1) + iS;

                mapOpts = struct('terrain', terrains{iT}, ...
                    'treeDensity', densities{iD}, ...
                    'alpha', 1.0, 'beta', 0.55, 'gamma', 0.32, ...
                    'rngSeed', seed);

                mapData = MapGenerator.build(N, M, dx, mapOpts);
                rec = SwarmSizeOptimizer.recommend(mapData, fleetSweep, orchOpts, scoreOpts);
                tbl = rec.table;

                for k = 1:height(tbl)
                    rr = struct();
                    rr.planner = string(planners{iP});
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

                fprintf('[cmp %03d] %s | %s/%s seed=%d bestU=%d\n', runId, ...
                    planners{iP}, terrains{iT}, densities{iD}, seed, rec.optimal_numUAV);
            end
        end
    end
end

rawTbl = struct2table(rows);

save(fullfile(projRoot, 'ieeeComparativeRaw.mat'), ...
    'rows', 'rawTbl', 'terrains', 'densities', 'planners', 'fleetSweep', ...
    'numSeedsPerScenario', 'seedBase', 'N', 'M', 'dx', 'orchOpts', 'scoreOpts', '-v7.3');

try
    writetable(rawTbl, fullfile(projRoot, 'ieeeComparativeRaw.csv'));
catch
    warning('Could not write ieeeComparativeRaw.csv.');
end

fprintf('\nSaved ieeeComparativeRaw.mat / .csv\n');
