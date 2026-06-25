% ANALYZEIEEECOMPARATIVESTUDY Tables + figures: baselines vs priority, capacity, operability.
%
% Inputs:  ieeeComparativeRaw.mat from runIEEEComparativeStudy.m
% Outputs: ieeeComparativeSummary.mat, CSVs, ieeeFigure_*.fig

clear;
close all;
clc;

projRoot = fileparts(mfilename('fullpath'));
addpath(projRoot);

inFile = fullfile(projRoot, 'ieeeComparativeRaw.mat');
if ~isfile(inFile)
    error('Missing ieeeComparativeRaw.mat. Run runIEEEComparativeStudy.m first.');
end

S = load(inFile);
T = S.rawTbl;

G = groupsummary(T, {'planner', 'numUAV'}, 'mean', ...
    {'composite_score', 'mission_time_s', 'fleet_energy_MJ', 'HPC_pct', 'any_collision'});

satThr = 0.45;
planners = unique(T.planner, 'stable');
satTable = table(planners, zeros(numel(planners), 1), ...
    'VariableNames', {'planner', 'saturation_numUAV_est'});

for ip = 1:numel(planners)
    selP = G.planner == planners(ip);
    Ulist = sort(unique(G.numUAV(selP)));
    satU = nan;

    for u = Ulist(:).'
        sel = selP & G.numUAV == u;
        if ~any(sel)
            continue;
        end
        cr = G.mean_any_collision(sel);

        if cr >= satThr
            satU = u;
            break;
        end
    end

    satTable.saturation_numUAV_est(ip) = satU;
end

%% Operability surface (priority / Algorithm X): mean composite vs obsFrac x fleet
selX = strcmpi(T.planner, 'priority');
obs = T.obsFrac(selX);
Uf = T.numUAV(selX);
Sc = T.composite_score(selX);

nBinObs = 10;
edgesObs = linspace(min(obs), max(obs), nBinObs + 1);
M = nan(nBinObs, max(Uf));

for ib = 1:nBinObs
    inO = obs >= edgesObs(ib) & obs < edgesObs(ib + 1);

    if ib == nBinObs
        inO = obs >= edgesObs(ib) & obs <= edgesObs(ib + 1);
    end

    for uu = 1:max(Uf)
        m = mean(Sc(inO & Uf == uu), 'omitnan');

        if ~isnan(m)
            M(ib, uu) = m;
        end
    end
end

centersObs = (edgesObs(1:end - 1) + edgesObs(2:end)) / 2;
[Ux, Oy] = meshgrid(1:size(M, 2), centersObs);

%% Figures
figA = figure('Color', 'w', 'Position', [40, 40, 900, 340]);
hold on;
cols = lines(numel(planners));

for ip = 1:numel(planners)
    selP = G.planner == planners(ip);
    [uSort, ord] = sort(G.numUAV(selP));
    y = G.mean_composite_score(selP);
    y = y(ord);
    plot(uSort, y, '-o', 'Color', cols(ip, :), 'LineWidth', 1.6);
end

grid on;
xlabel('Fleet size');
ylabel('Mean composite score (lower is better)');
title('Baseline comparison (orchestrated missions)');
legend(cellstr(planners), 'Location', 'best');

figB = figure('Color', 'w', 'Position', [60, 60, 900, 340]);
hold on;

for ip = 1:numel(planners)
    selP = G.planner == planners(ip);
    [uSort, ord] = sort(G.numUAV(selP));
    cr = G.mean_any_collision(selP);
    cr = cr(ord);
    plot(uSort, cr, '-s', 'Color', cols(ip, :), 'LineWidth', 1.5);
end

yline(satThr, 'k--', 'LineWidth', 1.2);
grid on;
xlabel('Fleet size');
ylabel('Mean collision indicator rate');
title(sprintf('Capacity curve (saturation threshold = %.2f)', satThr));
legend([cellstr(planners); {'threshold'}], 'Location', 'best');

figC = figure('Color', 'w', 'Position', [80, 80, 640, 420]);
surf(Ux, Oy, M, 'EdgeColor', 'none');
view(-35, 32);
xlabel('Fleet size');
ylabel('Obstacle fraction (map)');
zlabel('Mean composite score');
title('Operability map (priority planner)');
colorbar;

savefig(figA, fullfile(projRoot, 'ieeeFigure_ComparativeComposite.fig'));
savefig(figB, fullfile(projRoot, 'ieeeFigure_CollisionCapacity.fig'));
savefig(figC, fullfile(projRoot, 'ieeeFigure_OperabilitySurface.fig'));

save(fullfile(projRoot, 'ieeeComparativeSummary.mat'), ...
    'G', 'satTable', 'satThr', 'M', 'centersObs', 'Ux', 'Oy', '-v7.3');

try
    writetable(G, fullfile(projRoot, 'ieeeComparativeByPlannerFleet.csv'));
    writetable(satTable, fullfile(projRoot, 'ieeeComparativeSaturation.csv'));
catch
    warning('CSV export failed.');
end

disp(satTable);
fprintf('Saved ieeeComparativeSummary.mat and comparative figures.\n');
