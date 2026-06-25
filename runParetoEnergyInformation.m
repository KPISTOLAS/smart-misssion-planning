% RUNPARETOENERGYINFORMATION Constrained energy minimization visualization:
%   f(x) = min E_total  s.t.  HPC% >= tau (tau default 80).
%   Compares multi-pass adaptive planners vs lawnmower stopped at tau (fair baseline)
%   and optional full-grid lawnmower energy (context dashed line).
%
% Outputs figure via PlotPHM.paretoEnergyInformationFront + paretoEnergyInformation.mat

clear;
close all;
clc;

projRoot = fileparts(mfilename('fullpath'));
addpath(projRoot);

tau_pct = 80;
tau_frac = tau_pct / 100;

rng(7721);

N = 52;
M = 68;
dx = 18;
numUAV = 3;

mapOpts = struct('terrain', 'hilly', 'treeDensity', 'medium', ...
    'alpha', 1.0, 'beta', 0.52, 'gamma', 0.30, 'rngSeed', 90210);

mapData = MapGenerator.build(N, M, dx, mapOpts);

simCfg = struct('dt', 1.25, 'vMax', 26, 'dSafe', 22, ...
    'altitudeAGL', 32, 'usePowerModel3D', true, 'stepTolM', 38, ...
    'fleetEnergyBudget_J', []);

planLm = BaselinePlanners.lawnmower(mapData, numUAV);

cfgFull = simCfg;
logLmFull = MissionSim.run(mapData, planLm, cfgFull);
mFull = Analytics.computeMetrics(mapData, logLmFull);

cfgTau = simCfg;
cfgTau.stopHpcPct = tau_pct;
logLmTau = MissionSim.run(mapData, planLm, cfgTau);
mTau = Analytics.computeMetrics(mapData, logLmTau);

gammas = 0:0.12:1;
sortieEnergies_J = [4.8e5, 6e5, 7.2e5, 8.5e5, 10e5, 11.5e5];
smoothIts = [0, 1];

nRuns = numel(gammas) * numel(sortieEnergies_J) * numel(smoothIts);
E_J = zeros(nRuns, 1);
bits = zeros(nRuns, 1);
hpc_pct = zeros(nRuns, 1);
meta = repmat(struct('gamma', nan, 'sortieE_J', nan, 'smooth', nan), nRuns, 1);

ptr = 0;

for ig = 1:numel(gammas)
    for ie = 1:numel(sortieEnergies_J)
        for is = 1:numel(smoothIts)
            ptr = ptr + 1;

            orchCfg = struct();
            orchCfg.hpcTarget = tau_frac;
            orchCfg.maxSorties = 56;
            orchCfg.sortieEnergy_J = sortieEnergies_J(ie);
            orchCfg.rechargeTime_s = 75;
            orchCfg.dynamicIoTBetweenSorties = false;
            orchCfg.plannerOpts = struct('planMode', 'blend', ...
                'blendGamma', gammas(ig), ...
                'partitionMethod', 'voronoi_capacitated', ...
                'smoothPathIterations', smoothIts(is), ...
                'voronoiIterations', 16);
            orchCfg.simCfg = simCfg;

            agg = BatteryAwareOrchestrator.runMultiPass(mapData, numUAV, orchCfg);

            E_J(ptr) = agg.total_energy_J;
            bits(ptr) = agg.total_bits_information;
            hpc_pct(ptr) = agg.hpc_final_pct;
            meta(ptr).gamma = gammas(ig);
            meta(ptr).sortieE_J = sortieEnergies_J(ie);
            meta(ptr).smooth = smoothIts(is);
        end
    end
end

paretoOut = PlotPHM.paretoEnergyInformationFront(E_J, bits, hpc_pct, tau_pct, ...
    logLmTau.totalEnergy_J, logLmTau.total_bits_information, mTau.HPC_pct, ...
    "(precision-agriculture sweep)", logLmFull.totalEnergy_J);

fprintf('Full-grid lawnmower: E=%.3f MJ, HPC=%.1f%%\n', ...
    logLmFull.totalEnergy_J / 1e6, mFull.HPC_pct);
fprintf('Fair baseline (lawnmower stopped at tau=%.0f%%): E=%.3f MJ, HPC=%.1f%%\n', ...
    tau_pct, logLmTau.totalEnergy_J / 1e6, mTau.HPC_pct);

if ~isnan(paretoOut.E_star_J)
    fprintf(['Best adaptive feasible (min E | HPC>=tau): E*=%.3f MJ, HPC=%.1f%%, ', ...
        'I=%.1f kbits\n'], paretoOut.E_star_J / 1e6, paretoOut.hpc_star, paretoOut.bits_star / 1000);
    fprintf('Energy savings vs fair lawnmower baseline: %.2f%%\n', paretoOut.energy_savings_vs_ref_pct);

    if paretoOut.meets_30pct_savings
        fprintf('Narrative: >=30%% reduction achieved at target HPC (precision-ag headline).\n');
    end
end

save(fullfile(projRoot, 'paretoEnergyInformation.mat'), ...
    'tau_pct', 'mapOpts', 'E_J', 'bits', 'hpc_pct', 'meta', ...
    'logLmFull', 'logLmTau', 'mFull', 'mTau', 'paretoOut', '-v7.3');

fprintf('\nSaved paretoEnergyInformation.mat\n');
