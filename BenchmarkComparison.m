classdef BenchmarkComparison
    % BENCHMARKCOMPARISON Runs Priority vs Lawnmower vs Greedy on a common map instance.

    methods (Static)

        function batch = runSingleMap(mapData, numUAV, simCfg)
            arguments
                mapData struct
                numUAV (1,1) double {mustBePositive, mustBeInteger}
                simCfg struct = struct()
            end

            planners = {@() PriorityPlanner.buildPlan(mapData, numUAV, struct()), ...
                        @() BaselinePlanners.lawnmower(mapData, numUAV), ...
                        @() BaselinePlanners.greedy(mapData, numUAV)};

            names = {'priority_hybrid', 'lawnmower', 'greedy'};
            batch = struct('name', {}, 'plan', {}, 'log', {}, 'metrics', {});

            for k = 1:numel(planners)
                plan = planners{k}();
                log = MissionSim.run(mapData, plan, simCfg);
                met = Analytics.computeMetrics(mapData, log);

                batch(k).name = names{k};
                batch(k).plan = plan;
                batch(k).log = log;
                batch(k).metrics = met;
            end
        end
    end
end
