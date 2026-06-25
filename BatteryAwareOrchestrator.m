classdef BatteryAwareOrchestrator
    % BATTERYAWAREORCHESTRATOR Multi-sortie replanning with virtual recharge and HPC termination.
    %   Objective emphasis: time-to-(HPC $\geq$ target) under per-sortie energy ceilings — not single-pass energy alone.
    %
    %   Each sortie applies MissionSim with fleetEnergyBudget_J; visited cells accumulate across rounds.

    methods (Static)

        function out = runMultiPass(mapData, numUAV, orchCfg)
            arguments
                mapData struct
                numUAV (1,1) double {mustBePositive, mustBeInteger}
                orchCfg struct
            end

            orchCfg = BatteryAwareOrchestrator.fillOrchDefaults(orchCfg);

            [N, M] = size(mapData.obstacle);
            trav = ~mapData.obstacle;
            highRef = mapData.highPriorityMask & trav;

            visited = false(N, M);
            md = mapData;

            logs = cell(0);
            energyTot = 0;
            bitsTot = 0;
            igTot = 0;
            timeTot = 0;
            stagnation = 0;

            pass = 0;

            while pass < orchCfg.maxSorties
                pass = pass + 1;

                denHi = nnz(highRef);

                if denHi == 0
                    break;
                end

                hpcNow = nnz(visited & highRef) / denHi;

                if hpcNow >= orchCfg.hpcTarget
                    break;
                end

                prevHighCov = nnz(visited & highRef);

                po = orchCfg.plannerOpts;
                po.remainingHighMask = ~visited;

                plan = BatteryAwareOrchestrator.buildPlan(md, numUAV, po, orchCfg.plannerMode);

                sc = orchCfg.simCfg;
                sc.fleetEnergyBudget_J = orchCfg.sortieEnergy_J;

                if ~isfield(sc, 'dynamicIoTEnable')
                    sc.dynamicIoTEnable = false;
                end

                log = MissionSim.run(md, plan, sc);

                visited = visited | log.visited;
                energyTot = energyTot + log.totalEnergy_J;
                bitsTot = bitsTot + log.total_bits_information;
                igTot = igTot + log.totalInformationGain;
                timeTot = timeTot + size(log.positions, 1) * log.dt + orchCfg.rechargeTime_s;

                logs{end + 1} = log; %#ok<AGROW>

                if orchCfg.dynamicIoTBetweenSorties
                    md = BatteryAwareOrchestrator.applyIoTDrift(md, orchCfg);
                    highRef = md.highPriorityMask & trav;
                end

                postHighCov = nnz(visited & highRef);

                if postHighCov <= prevHighCov && log.truncatedSteps
                    stagnation = stagnation + 1;
                else
                    stagnation = 0;
                end

                if stagnation >= orchCfg.maxStagnationRounds
                    break;
                end
            end

            out = struct();
            out.finalVisited = visited;
            out.hpc_final_pct = 100 * nnz(visited & highRef) / max(eps, nnz(highRef));
            out.total_energy_J = energyTot;
            out.total_mission_time_s = timeTot;
            out.total_bits_information = bitsTot;
            out.total_information_gain = igTot;
            out.sorties = numel(logs);
            out.logs = logs;
            out.metrics = Analytics.aggregateMissionMetrics(mapData, out);
        end

        function plan = buildPlan(mapData, numUAV, plannerOpts, plannerMode)
            arguments
                mapData struct
                numUAV (1, 1) double {mustBePositive, mustBeInteger}
                plannerOpts struct
                plannerMode = "priority"
            end

            pm = lower(strtrim(char(string(plannerMode))));

            switch pm
                case 'greedy'
                    plan = BaselinePlanners.greedy(mapData, numUAV);
                case 'decentralized_greedy'
                    plan = BaselinePlanners.decentralizedVoronoiGreedy(mapData, numUAV);
                otherwise
                    plan = PriorityPlanner.buildPlan(mapData, numUAV, plannerOpts);
            end
        end

        function orchCfg = fillOrchDefaults(orchCfg)
            d = struct( ...
                'hpcTarget', 0.90, ...
                'maxSorties', 36, ...
                'sortieEnergy_J', 9e5, ...
                'rechargeTime_s', 120, ...
                'plannerMode', 'priority', ...
                'plannerOpts', struct(), ...
                'simCfg', struct(), ...
                'dynamicIoTBetweenSorties', false, ...
                'dynamicDriftH', 0.06, ...
                'dynamicDriftSigma', 0.012, ...
                'maxStagnationRounds', 4);

            fn = fieldnames(d);
            for k = 1:numel(fn)
                if ~isfield(orchCfg, fn{k}) || isempty(orchCfg.(fn{k}))
                    orchCfg.(fn{k}) = d.(fn{k});
                end
            end
        end

        function md = applyIoTDrift(md, orchCfg)
            mask = ~md.obstacle;

            driftH = orchCfg.dynamicDriftH .* (rand(size(md.H)) - 0.45) .* double(mask);
            md.H = md.H + driftH;
            md.H = min(4, max(0, round(md.H)));

            md.sigma = md.sigma + orchCfg.dynamicDriftSigma .* randn(size(md.sigma)) .* double(mask);
            md.sigma = min(1, max(0.06, md.sigma));

            if isfield(md.meta, 'highHealthThr')
                thr = md.meta.highHealthThr;
            else
                thr = 3;
            end

            md.highPriorityMask = (md.H >= thr);
            md.O = MapGenerator.obstacleProximity(md.obstacle, md.meta.sigmaObsKernel);
            md.W = md.meta.alpha .* md.H + md.meta.beta .* md.sigma - md.meta.gamma .* md.O;
            md = GPInformationField.attach(md, struct());
        end
    end
end
