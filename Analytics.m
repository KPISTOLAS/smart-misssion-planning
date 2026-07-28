classdef Analytics
    % ANALYTICS KPIs for adaptive PHM-CPP evaluation.

    methods (Static)

        function m = computeMetrics(mapData, log)
            arguments
                mapData struct
                log struct
            end

            dt = log.dt;
            tAxis = (1:numel(log.coveredArea_m2)).' * dt;
            instRate = [0; diff(log.coveredArea_m2) / max(dt, eps)];
            acr_m2_s = mean(instRate(instRate > 0));

            highCells = nnz(log.highMask);
            coveredHigh = nnz(log.visited & log.highMask);

            if highCells == 0
                hpc_pct = nan;
            else
                hpc_pct = 100 * coveredHigh / highCells;
            end

            if log.totalInformationGain > eps
                egi_J_per_unit = log.totalEnergy_J / log.totalInformationGain;
            else
                egi_J_per_unit = inf;
            end

            if isfield(log, 'total_bits_information') && log.total_bits_information > eps
                ebits = log.totalEnergy_J / log.total_bits_information;
            else
                ebits = inf;
            end

            refPlan = BaselinePlanners.lawnmower(mapData, size(log.positions, 2));
            refSeg = refPlan.segment{1};
            rmse_m = Analytics.crossTrackRMSE(log.positions(:, 1, :), refSeg, mapData.dx);

            m = struct();
            m.ACR_m2_s = acr_m2_s;
            m.HPC_pct = hpc_pct;
            m.energy_to_information_J = egi_J_per_unit;
            m.energy_per_bit_J = ebits;
            m.cross_track_RMSE_m = rmse_m;
            m.total_energy_J = log.totalEnergy_J;
            m.total_information_gain = log.totalInformationGain;

            if isfield(log, 'total_bits_information')
                m.total_bits_information = log.total_bits_information;
            else
                m.total_bits_information = nan;
            end

            m.final_coverage_area_m2 = log.coveredArea_m2(end);
            m.any_collision = any(log.collision);

            if isfield(log, 'truncatedSteps')
                m.sim_truncated = log.truncatedSteps;
            else
                m.sim_truncated = false;
            end
        end

        function m = aggregateMissionMetrics(mapData, agg)
            % Aggregate KPIs for BatteryAwareOrchestrator multi-pass missions.

            arguments
                mapData struct
                agg struct
            end

            trav = ~mapData.obstacle;

            covArea = nnz(agg.finalVisited & trav) * mapData.cellArea_m2;

            m = struct();
            m.HPC_pct = agg.hpc_final_pct;
            m.total_energy_J = agg.total_energy_J;
            m.total_mission_time_s = agg.total_mission_time_s;
            m.sortie_count = agg.sorties;
            m.final_coverage_area_m2 = covArea;
            m.ACR_m2_s = covArea / max(eps, agg.total_mission_time_s);

            if agg.total_bits_information > eps
                m.energy_per_bit_J = agg.total_energy_J / agg.total_bits_information;
            else
                m.energy_per_bit_J = inf;
            end

            if agg.total_information_gain > eps
                m.energy_to_information_J = agg.total_energy_J / agg.total_information_gain;
            else
                m.energy_to_information_J = inf;
            end

            m.cross_track_RMSE_m = nan;

            if isfield(agg, 'logs') && ~isempty(agg.logs)
                m.any_collision = any(cellfun(@(lg) any(lg.collision), agg.logs));
            else
                m.any_collision = false;
            end
        end

        function rmse = crossTrackRMSE(posTensor, refRC, dx)
            % POS TENSOR: T x 1 x 2 for primary UAV; REFRC: K x [row col].

            if isempty(refRC)
                rmse = nan;
                return;
            end

            T = size(posTensor, 1);
            P = zeros(T, 2);

            for t = 1:T
                rc = squeeze(posTensor(t, 1, :));
                P(t, :) = [(rc(2) - 1) * dx, (rc(1) - 1) * dx];
            end

            Q = [(refRC(:, 2) - 1) * dx, (refRC(:, 1) - 1) * dx];

            d = zeros(T, 1);
            for t = 1:T
                d(t) = Analytics.pointToPolylineDistance(P(t, :), Q);
            end

            rmse = sqrt(mean(d.^2));
        end

        function dist = pointToPolylineDistance(p, Q)
            if size(Q, 1) == 1
                dist = hypot(p(1) - Q(1, 1), p(2) - Q(1, 2));
                return;
            end

            best = inf;

            for k = 1:size(Q, 1) - 1
                a = Q(k, :);
                b = Q(k + 1, :);
                dd = Analytics.pointToSegmentDistance(p, a, b);
                if dd < best
                    best = dd;
                end
            end

            dist = best;
        end

        function d = pointToSegmentDistance(p, a, b)
            ab = b - a;
            ap = p - a;
            t = dot(ap, ab) / max(eps, dot(ab, ab));
            t = max(0, min(1, t));
            proj = a + t * ab;
            d = hypot(p(1) - proj(1), p(2) - proj(2));
        end

        function tbl = summarizeBenchmark(rows)
            % ROWS: struct array with fields name, terrain, density, metrics struct.

            n = numel(rows);
            tbl = table();

            names = cell(n, 1);
            terr = cell(n, 1);
            dens = cell(n, 1);
            acr = zeros(n, 1);
            hpc = zeros(n, 1);
            egi = zeros(n, 1);
            epb = nan(n, 1);
            rmse = zeros(n, 1);
            Etot = zeros(n, 1);

            for i = 1:n
                names{i} = char(string(rows(i).planner));
                terr{i} = char(string(rows(i).terrain));
                dens{i} = char(string(rows(i).density));
                mm = rows(i).metrics;
                acr(i) = mm.ACR_m2_s;
                hpc(i) = mm.HPC_pct;
                egi(i) = mm.energy_to_information_J;
                rmse(i) = mm.cross_track_RMSE_m;
                Etot(i) = mm.total_energy_J;

                if isfield(mm, 'energy_per_bit_J')
                    epb(i) = mm.energy_per_bit_J;
                end
            end

            tbl.planner = string(names);
            tbl.terrain = string(terr);
            tbl.treeDensity = string(dens);
            tbl.ACR_m2_s = acr;
            tbl.HPC_pct = hpc;
            tbl.energy_per_info_J = egi;
            tbl.energy_per_bit_J = epb;
            tbl.cross_track_RMSE_m = rmse;
            tbl.total_energy_J = Etot;
        end
    end
end
