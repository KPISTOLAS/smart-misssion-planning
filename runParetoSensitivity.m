% RUNPARETOSENSITIVITY IoT weight sensitivity (\alpha,\beta) grid → energy vs HPC visualization.
%
% Interprets tunable mission emphasis between nominal stress fusion (MapGenerator.W) and GP/EIG cues.

clear; close all; clc;

projRoot = fileparts(mfilename('fullpath'));
addpath(projRoot);

N = 42;
M = 56;
dx = 18;
numUAV = 3;

alphaVec = 0:0.25:1;
betaVec = 0:0.25:1;

simCfg = struct('dt', 1.25, 'vMax', 26, 'dSafe', 22, 'altitudeAGL', 32, ...
    'usePowerModel3D', true, 'stepTolM', 38);

plannerOpts = struct('planMode', 'blend', 'blendGamma', 0.45, ...
    'partitionMethod', 'voronoi_capacitated', ...
    'smoothPathIterations', 1);

Na = numel(alphaVec);
Nb = numel(betaVec);
energyMat = nan(Na, Nb);
hpcMat = nan(Na, Nb);

for ia = 1:Na
    for ib = 1:Nb
        seed = 9300 + ia * 47 + ib * 13;

        opts = struct('terrain', 'hilly', 'treeDensity', 'medium', ...
            'alpha', alphaVec(ia), 'beta', betaVec(ib), 'gamma', 0.32, ...
            'rngSeed', seed);

        mapData = MapGenerator.build(N, M, dx, opts);
        plan = PriorityPlanner.buildPlan(mapData, numUAV, plannerOpts);
        log = MissionSim.run(mapData, plan, simCfg);
        m = Analytics.computeMetrics(mapData, log);

        energyMat(ia, ib) = m.total_energy_J;
        hpcMat(ia, ib) = m.HPC_pct;
    end
end

PlotPHM.paretoWeightSweep(alphaVec, betaVec, energyMat, hpcMat);

save(fullfile(projRoot, 'paretoSensitivity.mat'), 'alphaVec', 'betaVec', 'energyMat', 'hpcMat');

fprintf('Saved paretoSensitivity.mat and displayed trade-off figures.\n');
