% ANALYZEIEEESTUDY Build publishable statistics and figures from ieeeStudyRaw.
%
% Inputs:
%   - ieeeStudyRaw.mat produced by runIEEEStudy.m
%
% Outputs:
%   - ieeeStudySummary.mat
%   - ieeeSummaryByFleet.csv
%   - ieeeOptimalCounts.csv
%   - ieeeFigure_FleetScore.fig
%   - ieeeFigure_OptimalHistogram.fig

clear;
close all;
clc;

projRoot = fileparts(mfilename('fullpath'));
addpath(projRoot);

inFile = fullfile(projRoot, 'ieeeStudyRaw.mat');
if ~isfile(inFile)
    error('Missing ieeeStudyRaw.mat. Run runIEEEStudy.m first.');
end

S = load(inFile);
rawTbl = S.rawTbl;

fleetList = unique(rawTbl.numUAV);
nF = numel(fleetList);

% Aggregate by fleet size across all scenarios/seeds.
meanScore = zeros(nF, 1);
stdScore = zeros(nF, 1);
ci95Score = zeros(nF, 1);
meanTime = zeros(nF, 1);
meanEnergy = zeros(nF, 1);
meanHPC = zeros(nF, 1);
collisionRate = zeros(nF, 1);
targetHitRate = zeros(nF, 1);

for i = 1:nF
    U = fleetList(i);
    sel = rawTbl.numUAV == U;
    x = rawTbl.composite_score(sel);
    n = nnz(sel);

    meanScore(i) = mean(x);
    stdScore(i) = std(x);
    ci95Score(i) = 1.96 * stdScore(i) / max(sqrt(n), eps);
    meanTime(i) = mean(rawTbl.mission_time_s(sel));
    meanEnergy(i) = mean(rawTbl.fleet_energy_MJ(sel));
    meanHPC(i) = mean(rawTbl.HPC_pct(sel));
    collisionRate(i) = mean(double(rawTbl.any_collision(sel)));
    targetHitRate(i) = mean(double(rawTbl.met_hpc_target(sel)));
end

summaryByFleet = table(fleetList, meanScore, stdScore, ci95Score, ...
    meanTime, meanEnergy, meanHPC, collisionRate, targetHitRate, ...
    'VariableNames', {'numUAV', 'mean_composite', 'std_composite', 'ci95_composite', ...
    'mean_mission_time_s', 'mean_fleet_energy_MJ', 'mean_HPC_pct', ...
    'collision_rate', 'hpc_target_hit_rate'});

[~, bestIdx] = min(summaryByFleet.mean_composite);
recommendedByMean = summaryByFleet.numUAV(bestIdx);

% Distribution of per-seed optimal fleet counts.
seedKeys = strcat(string(rawTbl.terrain), "_", string(rawTbl.density), "_", string(rawTbl.seed));
[G, keyVals] = findgroups(seedKeys);
optPerSeed = splitapply(@(u, flag) u(find(flag, 1, 'first')), rawTbl.numUAV, rawTbl.is_optimal_for_seed, G);

[uOpt, ~, idxOpt] = unique(optPerSeed);
countOpt = accumarray(idxOpt, 1);
optimalCounts = table(uOpt, countOpt, ...
    'VariableNames', {'numUAV', 'count_as_optimal'});

% Simple pairwise effect vs recommended fleet (Cohen's d on composite score).
effectVsBest = nan(nF, 1);
xBest = rawTbl.composite_score(rawTbl.numUAV == recommendedByMean);
for i = 1:nF
    U = fleetList(i);
    x = rawTbl.composite_score(rawTbl.numUAV == U);
    sPooled = sqrt((var(xBest) + var(x)) / 2);
    effectVsBest(i) = (mean(x) - mean(xBest)) / max(sPooled, eps);
end
summaryByFleet.cohens_d_vs_best = effectVsBest;

%% Figures
fig1 = figure('Color', 'w', 'Position', [60, 60, 860, 360]);
yyaxis left;
errorbar(summaryByFleet.numUAV, summaryByFleet.mean_composite, ...
    summaryByFleet.ci95_composite, '-o', 'LineWidth', 1.6);
ylabel('Composite score (mean +- 95% CI)');
yyaxis right;
plot(summaryByFleet.numUAV, summaryByFleet.mean_HPC_pct, '-s', 'LineWidth', 1.4);
ylabel('Mean HPC (%)');
xlabel('Fleet size');
grid on;
title('Fleet-size ranking across all scenarios');

fig2 = figure('Color', 'w', 'Position', [80, 80, 640, 340]);
bar(optimalCounts.numUAV, optimalCounts.count_as_optimal, 'FaceColor', [0.25 0.50 0.70]);
grid on;
xlabel('Fleet size selected as optimal');
ylabel('Count across scenario-seed runs');
title('Optimal fleet-size frequency');

savefig(fig1, fullfile(projRoot, 'ieeeFigure_FleetScore.fig'));
savefig(fig2, fullfile(projRoot, 'ieeeFigure_OptimalHistogram.fig'));

save(fullfile(projRoot, 'ieeeStudySummary.mat'), ...
    'summaryByFleet', 'optimalCounts', 'recommendedByMean', 'keyVals', 'optPerSeed', '-v7.3');

try
    writetable(summaryByFleet, fullfile(projRoot, 'ieeeSummaryByFleet.csv'));
    writetable(optimalCounts, fullfile(projRoot, 'ieeeOptimalCounts.csv'));
catch
    warning('CSV export failed. MAT summary was saved.');
end

fprintf('Recommended fleet size by mean composite score: %d UAVs\n', recommendedByMean);
fprintf('Saved ieeeStudySummary.mat, csv tables, and figures.\n');
