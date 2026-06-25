% RUNBATTERYAWAREPHM Multi-sortie battery-aware mission until HPC >= 90 percent.
%   Demonstrates replanning + virtual recharge minimizing mission time subject to sortie energy caps.

clear; close all; clc;

projRoot = fileparts(mfilename('fullpath'));
addpath(projRoot);

N = 54;
M = 72;
dx = 18;
numUAV = 3;

mapOpts = struct('terrain', 'hilly', 'treeDensity', 'medium', ...
    'alpha', 1.0, 'beta', 0.55, 'gamma', 0.32, 'rngSeed', 8244);

mapData = MapGenerator.build(N, M, dx, mapOpts);

orchCfg = struct();
orchCfg.hpcTarget = 0.90;
orchCfg.maxSorties = 28;
orchCfg.sortieEnergy_J = 8.5e5;
orchCfg.rechargeTime_s = 90;
orchCfg.dynamicIoTBetweenSorties = true;
orchCfg.plannerOpts = struct('planMode', 'blend', 'blendGamma', 0.40, ...
    'partitionMethod', 'voronoi_capacitated', ...
    'smoothPathIterations', 1, 'voronoiIterations', 16);

orchCfg.simCfg = struct('dt', 1.25, 'vMax', 26, 'dSafe', 24, ...
    'altitudeAGL', 32, 'usePowerModel3D', true, 'stepTolM', 38, ...
    'dynamicIoTEnable', false);

agg = BatteryAwareOrchestrator.runMultiPass(mapData, numUAV, orchCfg);

fprintf('Sorties: %d\n', agg.sorties);
fprintf('Final HPC: %.2f %%\n', agg.hpc_final_pct);
fprintf('Total mission time (incl. recharge overhead): %.1f s\n', agg.total_mission_time_s);
fprintf('Total fleet energy: %.3f MJ\n', agg.total_energy_J / 1e6);
fprintf('Energy per bit: %.2f J/bit\n', agg.metrics.energy_per_bit_J);

logs = agg.logs;
lbl = "sortie " + (1:numel(logs));

PlotPHM.energyTime(logs, lbl);

fprintf('\nDone. Inspect multi-sortie energy figure.\n');
