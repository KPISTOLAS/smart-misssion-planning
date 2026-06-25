classdef PlotPHM
    % PLOTPHM Publication-oriented visualization helpers.

    methods (Static)

        function fig = trajectoryHeatmap(mapData, plan, tag)
            arguments
                mapData struct
                plan struct
                tag string = ""
            end

            fig = figure('Color', 'w', 'Position', [40, 40, 980, 420]);

            tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

            ax1 = nexttile;
            imagesc(ax1, mapData.H);
            axis(ax1, 'xy');
            hold(ax1, 'on');
            colormap(ax1, parula);
            cb = colorbar(ax1);
            cb.Label.String = 'Health stress H';

            U = size(plan.positions, 2);
            cols = lines(U);

            for u = 1:U
                rc = squeeze(plan.positions(:, u, :));
                plot(ax1, rc(:, 2), rc(:, 1), '-', 'Color', cols(u, :), 'LineWidth', 1.6);
                plot(ax1, rc(1, 2), rc(1, 1), 'o', 'MarkerFaceColor', cols(u, :), 'MarkerEdgeColor', 'k');
            end

            title(ax1, sprintf('Trajectories on IoT heatmap (%s)', tag));
            xlabel(ax1, 'Column index');
            ylabel(ax1, 'Row index');

            ax2 = nexttile;
            surf(ax2, mapData.Z_m, 'EdgeColor', 'none');
            hold(ax2, 'on');
            view(ax2, -35, 58);
            lighting(ax2, 'gouraud');
            camlight(ax2, 'headlight');

            for u = 1:U
                rc = squeeze(plan.positions(:, u, :));
                zs = zeros(size(rc, 1), 1);

                for k = 1:size(rc, 1)
                    r = max(1, min(mapData.N, round(rc(k, 1))));
                    c = max(1, min(mapData.M, round(rc(k, 2))));
                    zs(k) = mapData.Z_m(r, c) + 12;
                end

                plot3(ax2, rc(:, 2), rc(:, 1), zs, '-', 'LineWidth', 1.6, 'Color', cols(u, :));
            end

            title(ax2, '3D terrain overlay');
            xlabel(ax2, 'Column');
            ylabel(ax2, 'Row');
            zlabel(ax2, 'Elevation (m)');
        end

        function fig = energyTime(logs, labels)
            arguments
                logs cell
                labels string
            end

            fig = figure('Color', 'w', 'Position', [80, 520, 760, 340]);
            hold on;

            for k = 1:numel(logs)
                lg = logs{k};
                t = (1:numel(lg.energy_J)).' * lg.dt;
                plot(t, lg.energy_J / 1e6, 'LineWidth', 1.8);
            end

            grid on;
            xlabel('Time (s)');
            ylabel('Cumulative energy (MJ)');
            legend(labels, 'Location', 'northwest');
            title('Energy accumulation profiles');
        end

        function fig = metricBars(summaryTable)
            arguments
                summaryTable table
            end

            planners = unique(string(summaryTable.planner), 'stable');
            K = numel(planners);

            acrMean = zeros(K, 1);
            hpcMean = zeros(K, 1);
            egiMeanLog = zeros(K, 1);
            rmseMean = zeros(K, 1);

            for i = 1:K
                sel = strcmp(cellstr(string(summaryTable.planner)), char(planners(i)));
                acrMean(i) = mean(summaryTable.ACR_m2_s(sel), 'omitnan');
                hpcMean(i) = mean(summaryTable.HPC_pct(sel), 'omitnan');
                egiMeanLog(i) = mean(log10(summaryTable.energy_per_info_J(sel)), 'omitnan');
                rmseMean(i) = mean(summaryTable.cross_track_RMSE_m(sel), 'omitnan');
            end

            fig = figure('Color', 'w', 'Position', [120, 120, 900, 540]);
            tiledlayout(2, 2);

            ax = nexttile;
            bar(ax, acrMean);
            set(ax, 'XTickLabel', planners);
            title(ax, 'Mean ACR across scenarios');
            ylabel(ax, 'ACR (m^2/s)');
            grid(ax, 'on');

            ax = nexttile;
            bar(ax, hpcMean);
            set(ax, 'XTickLabel', planners);
            title(ax, 'Mean high-priority coverage %');
            ylabel(ax, 'HPC (%)');
            grid(ax, 'on');

            ax = nexttile;
            bar(ax, egiMeanLog);
            set(ax, 'XTickLabel', planners);
            title(ax, 'Mean log_{10} Energy / Information gain');
            ylabel(ax, 'log10(J per unit IG)');
            grid(ax, 'on');

            ax = nexttile;
            bar(ax, rmseMean);
            set(ax, 'XTickLabel', planners);
            title(ax, 'Mean cross-track RMSE vs lawnmower reference');
            ylabel(ax, 'RMSE (m)');
            grid(ax, 'on');
        end

        function fig = paretoWeightSweep(alphaVec, betaVec, energyMat, hpcMat)
            % Scatter / surface trade-off for sensitivity over (\alpha,\beta) IoT fusion weights.
            %   ENERGYMAT/HPCMAT rows index ALPHA, columns index BETA (matching runParetoSensitivity loops).

            arguments
                alphaVec (1, :) double
                betaVec (1, :) double
                energyMat (:, :) double
                hpcMat (:, :) double
            end

            Na = numel(alphaVec);
            Nb = numel(betaVec);
            nPts = Na * Nb;
            eFlat = zeros(nPts, 1);
            hFlat = zeros(nPts, 1);
            aFlat = zeros(nPts, 1);
            bFlat = zeros(nPts, 1);
            k = 0;

            for ia = 1:Na
                for ib = 1:Nb
                    k = k + 1;
                    eFlat(k) = energyMat(ia, ib);
                    hFlat(k) = hpcMat(ia, ib);
                    aFlat(k) = alphaVec(ia);
                    bFlat(k) = betaVec(ib);
                end
            end

            fig = figure('Color', 'w', 'Position', [90, 90, 920, 380]);
            tiledlayout(1, 2);

            ax = nexttile;
            scatter(ax, eFlat / 1e6, hFlat, 42, aFlat, 'filled');
            colorbar(ax);
            xlabel(ax, 'Single-sortie energy (MJ)');
            ylabel(ax, 'HPC (%)');
            title(ax, 'Pareto-style sweep (color = \alpha)');
            grid(ax, 'on');

            ax = nexttile;
            [Xg, Yg] = meshgrid(alphaVec, betaVec);
            surf(ax, Xg, Yg, hpcMat.', 'EdgeColor', 'none');
            view(ax, -28, 58);
            xlabel(ax, '\alpha');
            ylabel(ax, '\beta');
            zlabel(ax, 'HPC (%)');
            title(ax, 'High-priority coverage vs fusion weights');
            colorbar(ax);
        end

        function idx = nondominatedIndices(energyCol, hpcCol)
            % Maximizing HPC while minimizing energy — crude 2-D nondominated sorting.

            n = numel(energyCol);
            idx = false(n, 1);

            for i = 1:n
                dominated = false;

                for j = 1:n
                    if j == i
                        continue;
                    end

                    if energyCol(j) <= energyCol(i) && hpcCol(j) >= hpcCol(i) ...
                            && (energyCol(j) < energyCol(i) || hpcCol(j) > hpcCol(i))
                        dominated = true;
                        break;
                    end
                end

                if ~dominated
                    idx(i) = true;
                end
            end
        end

        function out = paretoEnergyInformationFront(E_J, bits, hpc_pct, tau_hpc, E_ref_J, bits_ref, hpc_ref, tag, E_full_reference_J)
            % PARETOENERGYINFORMATIONFRONT Energy (MJ) vs cumulative information (kbits) with HPC ≥ τ emphasis.
            %   Highlights $\min E_{\mathrm{total}}$ subject to $\mathrm{HPC}\%\geq\tau$ among swept tunings,
            %   baseline lawnmower, and the nondominated frontier (minimize energy, maximize bits).

            arguments
                E_J (:, 1) double
                bits (:, 1) double
                hpc_pct (:, 1) double
                tau_hpc (1, 1) double {mustBePositive}
                E_ref_J (1, 1) double
                bits_ref (1, 1) double
                hpc_ref (1, 1) double
                tag string = ""
                E_full_reference_J (1, 1) double = nan
            end

            E_MJ = E_J / 1e6;
            Eref_MJ = E_ref_J / 1e6;
            kbits = bits / 1000;
            kb_ref = bits_ref / 1000;

            tol = 0.75;
            feas = hpc_pct >= (tau_hpc - tol);

            % Constrained optimum: minimum energy among HPC-feasible sweep points.
            out = struct();
            out.tau_hpc = tau_hpc;
            out.feasible_mask = feas;

            if any(feas)
                [E_star_J, k_star] = min(E_J(feas));
                feasIdx = find(feas);
                out.star_idx_global = feasIdx(k_star);
                out.E_star_J = E_star_J;
                out.bits_star = bits(out.star_idx_global);
                out.hpc_star = hpc_pct(out.star_idx_global);
                out.energy_savings_vs_ref_pct = 100 * (E_ref_J - E_star_J) / max(eps, E_ref_J);
                out.meets_30pct_savings = out.energy_savings_vs_ref_pct >= 29.5 ...
                    && out.hpc_star >= (tau_hpc - tol);
            else
                out.star_idx_global = nan;
                out.E_star_J = nan;
                out.bits_star = nan;
                out.hpc_star = nan;
                out.energy_savings_vs_ref_pct = nan;
                out.meets_30pct_savings = false;
            end

            paretoFeas = PlotPHM.nondominatedEnergyBits(E_J, bits, feas);

            fig = figure('Color', 'w', 'Position', [60, 60, 1040, 480]);
            tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

            ax = nexttile;
            hold(ax, 'on');

            infeas = ~feas;
            scatter(ax, E_MJ(infeas), kbits(infeas), 36, hpc_pct(infeas), '^', ...
                'MarkerFaceColor', 'flat', 'MarkerEdgeColor', [0.65 0.65 0.65], 'LineWidth', 0.4);

            scatter(ax, E_MJ(feas), kbits(feas), 52, hpc_pct(feas), 'o', ...
                'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.35);

            colormap(ax, parula);
            cb = colorbar(ax);
            cb.Label.String = 'HPC (%)';

            if any(paretoFeas)
                pf = paretoFeas & feas;
                [Esrt, ord] = sort(E_MJ(pf));
                ksrt = kbits(pf);
                ksrt = ksrt(ord);
                plot(ax, Esrt, ksrt, 'k-', 'LineWidth', 1.85);
            end

            hLm = scatter(ax, Eref_MJ, kb_ref, 220, 'b', 'p', 'filled', ...
                'MarkerEdgeColor', 'k', 'DisplayName', 'Lawnmower @ $\tau$ (fair baseline)');

            hStar = [];

            if ~isnan(out.E_star_J)
                hStar = scatter(ax, out.E_star_J / 1e6, out.bits_star / 1000, 240, 'r', '*', ...
                    'LineWidth', 1.1, 'DisplayName', sprintf('min $E$ | HPC$\\geq%.0f$\\%%', tau_hpc));
            end

            if ~isnan(E_full_reference_J)
                xline(ax, E_full_reference_J / 1e6, '--', ...
                    'Color', [0.15 0.35 0.65], 'LineWidth', 1.35);
            end

            xlabel(ax, 'Total mission energy $E_{\mathrm{total}}$ (MJ)', 'Interpreter', 'latex');
            ylabel(ax, 'Cumulative information $\sum b_i$ (kbits)', 'Interpreter', 'latex');
            grid(ax, 'on');

            title(ax, {['Energy--information trade-off ', char(tag)], ...
                '$f(\mathbf{x})=\min E_{\mathrm{total}}$ s.t.\ $\mathrm{HPC}\geq\tau$', ...
                sprintf('$\\tau=%.0f$\\%% coverage threshold', tau_hpc)}, ...
                'Interpreter', 'latex');

            if isempty(hStar)
                legend(ax, [hLm], 'Location', 'southwest', 'Interpreter', 'latex');
            else
                legend(ax, [hLm, hStar], 'Location', 'southwest', 'Interpreter', 'latex');
            end

            ax2 = nexttile;
            axis(ax2, 'off');

            summaryLines = {
                'Precision-agriculture framing:'
                sprintf(['Fair baseline = lawnmower stopped at $\\tau$: ', ...
                    '$E_{\\mathrm{ref}}=%.2f$ MJ, HPC=%.1f\\%%, $I=%.1f$ kbits.'], ...
                Eref_MJ, hpc_ref, kb_ref)
                sprintf('Coverage constraint: HPC $\\geq %.0f$\\%% (filled circles = feasible sweep).', tau_hpc)
                };

            if any(feas)
                summaryLines{end + 1} = sprintf( ...
                    ['Constrained best (min energy among feasible sweep): ', ...
                    'E^\\star=%.2f MJ, HPC=%.1f%%, I=%.1f kbits.'], ...
                    out.E_star_J / 1e6, out.hpc_star, out.bits_star / 1000); %#ok<AGROW>

                summaryLines{end + 1} = sprintf( ...
                    'Energy vs baseline: %.1f%% savings (100\\cdot(E_{ref}-E^\\star)/E_{ref}).', ...
                    out.energy_savings_vs_ref_pct); %#ok<AGROW>

                if out.meets_30pct_savings
                    summaryLines{end + 1} = ...
                        '{\\bf Narrative check:} $\\geq 30\\%$ energy reduction vs lawnmower @ $\\tau$ - cite-worthy regime.'; %#ok<AGROW>
                else
                    summaryLines{end + 1} = ...
                        'Sweep did not yet exceed 30\\% savings vs lawnmower @ $\\tau$; extend grid (sortie budget / $\\gamma$).'; %#ok<AGROW>
                end
            else
                summaryLines{end + 1} = ...
                    'No sweep point reached \\tau — increase maxSorties or relax sortie energy cap.'; %#ok<AGROW>
            end

            text(ax2, 0.02, 0.96, summaryLines, 'Interpreter', 'latex', ...
                'VerticalAlignment', 'top', 'FontSize', 11);

            out.figure_handle = fig;
        end

        function idx = nondominatedEnergyBits(E_J, bits, feasMask)
            % Nondominated among feasible: minimize energy, maximize bits.

            n = numel(E_J);
            idx = false(n, 1);

            for i = 1:n
                if ~feasMask(i)
                    continue;
                end

                dominated = false;

                for j = 1:n
                    if ~feasMask(j) || j == i
                        continue;
                    end

                    if E_J(j) <= E_J(i) && bits(j) >= bits(i) ...
                            && (E_J(j) < E_J(i) || bits(j) > bits(i))
                        dominated = true;
                        break;
                    end
                end

                if ~dominated
                    idx(i) = true;
                end
            end
        end
    end
end
