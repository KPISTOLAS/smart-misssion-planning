classdef SwarmSizeOptimizer
    % SWARMSIZEOPTIMIZER Sweep fleet sizes on a fixed map and pick a recommended
    %   count using a weighted, min-max normalized score (lower is better).
    %
    %   Default scoring favors: short mission time, low fleet energy, low
    %   energy/bit, with large penalties for collisions and for missing the HPC
    %   target under BatteryAwareOrchestrator.
    %
    %   See also runSwarmSizeOptimization, runSwarmSizeBenchmark.

    methods (Static)

        function tbl = sweepOrchestrated(mapData, fleetList, orchOpts)
            arguments
                mapData struct
                fleetList (1, :) double {mustBeVector}
                orchOpts struct = struct()
            end

            orchOpts = SwarmSizeOptimizer.fillOrchSweepOpts(orchOpts);
            nF = numel(fleetList);

            mission_time_s = zeros(nF, 1);
            fleet_energy_J = zeros(nF, 1);
            energy_per_UAV_J = zeros(nF, 1);
            HPC_pct = zeros(nF, 1);
            sorties = zeros(nF, 1);
            energy_per_bit_J = nan(nF, 1);
            any_collision = false(nF, 1);
            met_hpc_target = false(nF, 1);
            aggs = cell(nF, 1);

            for k = 1:nF
                U = fleetList(k);
                orchCfg = SwarmSizeOptimizer.buildOrchCfgForFleet(mapData, U, orchOpts);

                agg = BatteryAwareOrchestrator.runMultiPass(mapData, U, orchCfg);
                met = Analytics.aggregateMissionMetrics(mapData, agg);

                aggs{k} = agg;
                mission_time_s(k) = met.total_mission_time_s;
                fleet_energy_J(k) = met.total_energy_J;
                energy_per_UAV_J(k) = met.total_energy_J / max(U, 1);
                HPC_pct(k) = met.HPC_pct;
                sorties(k) = met.sortie_count;
                energy_per_bit_J(k) = met.energy_per_bit_J;
                any_collision(k) = met.any_collision;
                met_hpc_target(k) = agg.hpc_final_pct >= 100 * orchOpts.hpcTargetFrac - 0.5;
            end

            tbl = table(fleetList(:), mission_time_s, fleet_energy_J / 1e6, ...
                energy_per_UAV_J / 1e6, HPC_pct, sorties, energy_per_bit_J, ...
                any_collision, met_hpc_target, ...
                'VariableNames', {'numUAV', 'mission_time_s', 'fleet_energy_MJ', ...
                'energy_per_UAV_MJ', 'HPC_pct', 'sorties', 'energy_per_bit_J', ...
                'any_collision', 'met_hpc_target'});
            tbl.Properties.UserData = struct('aggs', {aggs}, 'orchOpts', orchOpts);
        end

        function rec = recommend(mapData, fleetList, orchOpts, scoreOpts)
            arguments
                mapData struct
                fleetList (1, :) double {mustBeVector}
                orchOpts struct = struct()
                scoreOpts struct = struct()
            end

            orchOpts = SwarmSizeOptimizer.fillOrchSweepOpts(orchOpts);
            tbl = SwarmSizeOptimizer.sweepOrchestrated(mapData, fleetList, orchOpts);
            rec = SwarmSizeOptimizer.pickFromTable(tbl, orchOpts, scoreOpts);
        end

        function rec = recommendFromMetrics(metricsTbl, orchOpts, scoreOpts)
            % RECOMMENDFROMMETRICS Score an existing sweep table (no re-simulation).
            %   metricsTbl must include: numUAV, mission_time_s, fleet_energy_MJ,
            %   energy_per_bit_J, any_collision, met_hpc_target, HPC_pct.

            arguments
                metricsTbl table
                orchOpts struct = struct()
                scoreOpts struct = struct()
            end

            orchOpts = SwarmSizeOptimizer.fillOrchSweepOpts(orchOpts);
            rec = SwarmSizeOptimizer.pickFromTable(metricsTbl, orchOpts, scoreOpts);
        end

        function rec = pickFromTable(tbl, orchOpts, scoreOpts)
            arguments
                tbl table
                orchOpts struct
                scoreOpts struct = struct()
            end

            scoreOpts = SwarmSizeOptimizer.fillScoreOpts(scoreOpts);

            T = tbl.mission_time_s;
            E = tbl.fleet_energy_MJ;
            B = tbl.energy_per_bit_J;
            B = SwarmSizeOptimizer.finiteOrCap(B);

            Tn = SwarmSizeOptimizer.minmax01(T);
            En = SwarmSizeOptimizer.minmax01(E);
            Bn = SwarmSizeOptimizer.minmax01(B);

            penColl = double(tbl.any_collision) * scoreOpts.penalty_collision;
            hpcTar = 100 * orchOpts.hpcTargetFrac;
            gap = max(0, hpcTar - tbl.HPC_pct) / 100;
            penHpc = gap * scoreOpts.penalty_hpc_per_unit_gap;

            comp = scoreOpts.w_time .* Tn + scoreOpts.w_energy .* En ...
                + scoreOpts.w_energy_per_bit .* Bn + penColl + penHpc;

            feasible = tbl.met_hpc_target & ~tbl.any_collision;

            if scoreOpts.feasible_only && any(feasible)
                comp(~feasible) = inf;
            end

            if all(isinf(comp))
                warning('SwarmSizeOptimizer:NoFeasibleFleet', ...
                    'No fleet passed feasibility filters; ranking by unconstrained composite.');
                comp = scoreOpts.w_time .* Tn + scoreOpts.w_energy .* En ...
                    + scoreOpts.w_energy_per_bit .* Bn + penColl + penHpc;
            end

            [bestScore, idx] = min(comp);

            rec = struct();
            rec.optimal_numUAV = tbl.numUAV(idx);
            rec.optimal_row_index = idx;
            rec.best_composite_score = bestScore;
            rec.scores = comp;
            rec.table = tbl;
            rec.score_opts = scoreOpts;
            rec.orch_opts = orchOpts;
            rec.feasible_mask = feasible;
            rec.T_norm = Tn;
            rec.E_norm = En;
            rec.Ebit_norm = Bn;
            rec.penalty_collision = penColl;
            rec.penalty_hpc = penHpc;

            detail = table(tbl.numUAV, Tn, En, Bn, penColl, penHpc, comp, feasible, ...
                'VariableNames', {'numUAV', 'T_norm', 'E_norm', 'Ebit_norm', ...
                'pen_coll', 'pen_hpc', 'composite_score', 'feasible'});
            rec.score_breakdown = detail;
        end

        function orchCfg = buildOrchCfgForFleet(mapData, numUAV, orchOpts)
            U = numUAV;

            if isfield(mapData, 'obsFrac') && ~isempty(mapData.obsFrac)
                rho = double(mapData.obsFrac);
            else
                rho = mean(mapData.obstacle(:));
            end

            dSafe = max(22, 17 + 1.35 * U + 32 * rho);
            po = struct('planMode', 'blend', 'blendGamma', orchOpts.blendGamma, ...
                'partitionMethod', 'voronoi_capacitated', ...
                'smoothPathIterations', 1, 'voronoiIterations', min(44, 16 + 2 * U));

            sc = struct('dt', orchOpts.dt, 'vMax', orchOpts.vMax, 'dSafe', dSafe, ...
                'altitudeAGL', orchOpts.altitudeAGL, 'usePowerModel3D', true, ...
                'stepTolM', orchOpts.stepTolM, 'dynamicIoTEnable', false);

            orchCfg = struct();
            orchCfg.hpcTarget = orchOpts.hpcTargetFrac;
            orchCfg.maxSorties = orchOpts.maxSorties;
            orchCfg.sortieEnergy_J = orchOpts.sortieEnergy_J;
            orchCfg.rechargeTime_s = orchOpts.rechargeTime_s;
            orchCfg.dynamicIoTBetweenSorties = false;
            orchCfg.plannerOpts = po;
            orchCfg.simCfg = sc;
            orchCfg.plannerMode = char(string(orchOpts.plannerMode));
        end
    end

    methods (Static, Access = private)

        function orchOpts = fillOrchSweepOpts(orchOpts)
            d = struct( ...
                'hpcTargetFrac', 0.85, ...
                'maxSorties', 48, ...
                'sortieEnergy_J', 8.8e5, ...
                'rechargeTime_s', 85, ...
                'blendGamma', 0.42, ...
                'plannerMode', 'priority', ...
                'dt', 1.25, ...
                'vMax', 26, ...
                'altitudeAGL', 32, ...
                'stepTolM', 38);

            fn = fieldnames(d);
            for k = 1:numel(fn)
                if ~isfield(orchOpts, fn{k}) || isempty(orchOpts.(fn{k}))
                    orchOpts.(fn{k}) = d.(fn{k});
                end
            end
        end

        function s = fillScoreOpts(scoreOpts)
            d = struct( ...
                'w_time', 0.35, ...
                'w_energy', 0.30, ...
                'w_energy_per_bit', 0.35, ...
                'penalty_collision', 6, ...
                'penalty_hpc_per_unit_gap', 25, ...
                'feasible_only', false);

            fn = fieldnames(d);
            for k = 1:numel(fn)
                if ~isfield(scoreOpts, fn{k}) || isempty(scoreOpts.(fn{k}))
                    scoreOpts.(fn{k}) = d.(fn{k});
                end
            end
            s = scoreOpts;
        end

        function v = minmax01(x)
            mn = min(x);
            mx = max(x);
            v = (x - mn) / max(mx - mn, eps);
        end

        function b = finiteOrCap(b)
            fin = isfinite(b);
            if ~any(fin)
                b(:) = 1;
                return;
            end
            cap = max(b(fin)) * 3 + 1;
            b(~fin) = cap;
        end
    end
end
