classdef PriorityPlanner
    % PRIORITYPLANNER Hybrid IoT-aware CPP:
    %   (1) Weighted region assignment (DARP-inspired partitioning on high-H cells)
    %   (2) Intra-region goal sequencing via weighted greedy insertion
    %   (3) Collision-aware stitching with grid A* and short receding-horizon lookahead.
    %
    %   See also BaselinePlanners, MissionSim, astarGrid.

    methods (Static)

        function plan = buildPlan(mapData, numUAV, opts)
            arguments
                mapData struct
                numUAV (1,1) double {mustBePositive, mustBeInteger}
                opts struct = struct()
            end

            opts = PriorityPlanner.defaultOpts(opts);

            trav = ~mapData.obstacle;
            [N, M] = size(trav);
            W = mapData.W;

            if isempty(opts.remainingHighMask)
                remMask = true(N, M);
            else
                remMask = logical(opts.remainingHighMask);
            end

            highMask = mapData.highPriorityMask & trav & remMask;

            targetLinear = find(highMask);
            if isempty(targetLinear)
                targetLinear = find(mapData.highPriorityMask & trav);
            end

            if isempty(targetLinear)
                targetLinear = find(trav);
            end

            [tr, tc] = ind2sub([N, M], targetLinear);
            coords = [tr, tc];
            wgt = PriorityPlanner.plannerWeights(mapData, targetLinear, opts);

            starts = PriorityPlanner.pickDepots(trav, numUAV);

            if strcmp(opts.partitionMethod, 'voronoi_capacitated')
                assigns = SpatialDecomposition.capacitatedVoronoi(coords, wgt, starts, mapData.dx, ...
                    opts.voronoiIterations);
            else
                assigns = PriorityPlanner.weightedPartition(coords, wgt, numUAV);
            end

            seg = cell(numUAV, 1);
            maxLen = 0;

            for u = 1:numUAV
                idxU = assigns == u;
                goals = coords(idxU, :);
                gw = wgt(idxU);

                if isempty(goals)
                    % Do not duplicate the full target set (causes n<=K swarm blow-ups).
                    p0 = round(starts(u, :));
                    p0(1) = max(1, min(size(trav, 1), p0(1)));
                    p0(2) = max(1, min(size(trav, 2), p0(2)));
                    if ~trav(p0(1), p0(2))
                        [r0, c0] = find(trav, 1, 'first');
                        p0 = [r0, c0];
                    end
                    seg{u} = double(p0);
                    maxLen = max(maxLen, 1);
                    continue;
                end

                order = PriorityPlanner.weightedGreedyOrder(starts(u, :), goals, gw);
                orderedGoals = goals(order, :);

                path = PriorityPlanner.stitchedAStar(trav, starts(u, :), orderedGoals, ...
                    mapData.Z_m, opts);

                path = PriorityPlanner.applyHorizonShortcut(trav, path, ...
                    mapData.Z_m, opts.horizonCells);

                if opts.smoothPathIterations > 0
                    path = PathSmoother.smoothGridPath(path, mapData.dx, trav, opts.smoothPathIterations);
                end

                seg{u} = path;
                maxLen = max(maxLen, size(path, 1));
            end

            plan = struct();
            plan.segment = seg;
            plan.positions = PriorityPlanner.padPaths(seg, maxLen);
            plan.starts = starts;
            plan.meta = opts;
            plan.name = sprintf('priority_%s_%s', opts.planMode, opts.partitionMethod);
        end

        function w = plannerWeights(mapData, targetLinear, opts)
            arguments
                mapData struct
                targetLinear (:, 1) double
                opts struct
            end

            Wh = max(0, mapData.W(targetLinear));

            if ~isfield(mapData, 'activeSensingScore')
                mapData = GPInformationField.attach(mapData, struct());
            end

            We = max(0, mapData.activeSensingScore(targetLinear));

            switch opts.planMode
                case 'heuristic'
                    w = Wh;
                case 'eig'
                    w = We;

                    if ~any(w)
                        w = Wh;
                    end

                case 'blend'
                    nh = Wh / max(eps, max(Wh));
                    ne = We / max(eps, max(We));
                    w = opts.blendGamma .* nh + (1 - opts.blendGamma) .* ne;

                    if ~any(w)
                        w = nh;
                    end

                otherwise
                    w = Wh;
            end
        end

        function opts = defaultOpts(opts)
            d = struct( ...
                'horizonCells', 12, ...
                'lambdaSlope', 0.08, ...
                'planMode', 'blend', ...
                'blendGamma', 0.45, ...
                'partitionMethod', 'voronoi_capacitated', ...
                'voronoiIterations', 14, ...
                'remainingHighMask', [], ...
                'smoothPathIterations', 0);
            fn = fieldnames(d);
            for k = 1:numel(fn)
                if ~isfield(opts, fn{k}) || isempty(opts.(fn{k}))
                    opts.(fn{k}) = d.(fn{k});
                end
            end
        end

        function starts = pickDepots(trav, numUAV)
            [N, M] = size(trav);
            row = max(2, min(N - 1, 3));
            cols = find(trav(row, :));
            if numel(cols) < numUAV
                cols = find(any(trav, 1));
            end

            starts = zeros(numUAV, 2);
            if isempty(cols)
                starts(:, 1) = 2;
                starts(:, 2) = (1:numUAV).' + 1;
                return;
            end

            picks = round(linspace(1, numel(cols), numUAV));
            picks = max(1, min(numel(cols), picks));

            for u = 1:numUAV
                starts(u, :) = [row, cols(picks(u))];
                if ~trav(starts(u, 1), starts(u, 2))
                    starts(u, 2) = cols(max(1, min(numel(cols), picks(u))));
                end
            end

            % Guarantee traversable seeds (nudge along row).
            for u = 1:numUAV
                if trav(starts(u, 1), starts(u, 2))
                    continue;
                end
                [rF, cF] = find(trav, 1, 'first');
                starts(u, :) = [rF, cF];
            end
        end

        function assigns = weightedPartition(coords, wgt, K)
            % Capacitated weighted assignment via soft k-means on plane coordinates.

            n = size(coords, 1);
            assigns = ones(n, 1);

            if K <= 1
                return;
            end

            if n == 0
                assigns = zeros(0, 1);
                return;
            end

            if n <= K
                assigns = SpatialDecomposition.exclusiveTargetToUAV(coords, K);
                return;
            end

            % Weighted Lloyd clustering (no Statistics Toolbox dependency).
            [~, ordW] = sort(wgt, 'descend');
            seedIdx = ordW(1:min(K, n));
            ctr = coords(seedIdx, :);

            if size(ctr, 1) < K
                pad = coords(randperm(n, K - size(ctr, 1)), :);
                ctr = [ctr; pad];
            end

            maxIter = 40;
            for it = 1:maxIter
                lbl = zeros(n, 1);
                for i = 1:n
                    best = 1;
                    bestD = inf;
                    for kk = 1:K
                        d = norm(coords(i, :) - ctr(kk, :));
                        if d < bestD
                            bestD = d;
                            best = kk;
                        end
                    end
                    lbl(i) = best;
                end

                moved = 0;
                for kk = 1:K
                    sel = lbl == kk;
                    if ~any(sel)
                        ctr(kk, :) = coords(randi(n), :);
                        moved = moved + 1;
                        continue;
                    end
                    wloc = wgt(sel);
                    pts = coords(sel, :);
                    newC = sum(pts .* wloc, 1) / sum(wloc);
                    moved = moved + norm(newC - ctr(kk, :));
                    ctr(kk, :) = newC;
                end

                if moved < 1e-3
                    break;
                end
            end

            assigns = lbl;
        end

        function order = weightedGreedyOrder(startRC, goals, gw)
            % NN heuristic with priority bias on remaining goals.

            Ng = size(goals, 1);
            order = zeros(Ng, 1);
            unvisited = true(Ng, 1);
            cur = startRC;

            for k = 1:Ng
                cand = find(unvisited);
                score = zeros(numel(cand), 1);

                for j = 1:numel(cand)
                    g = cand(j);
                    dist = hypot(goals(g, 1) - cur(1), goals(g, 2) - cur(2));
                    score(j) = gw(g) / max(eps, dist);
                end

                [~, pickLocal] = max(score);
                pick = cand(pickLocal);
                order(k) = pick;
                unvisited(pick) = false;
                cur = goals(pick, :);
            end
        end

        function path = stitchedAStar(trav, startRC, orderedGoals, Z, opts)
            pathCells = zeros(0, 2);
            cur = startRC;

            for g = 1:size(orderedGoals, 1)
                goal = orderedGoals(g, :);
                costFn = @(r1, c1, r2, c2) PriorityPlanner.edgeCost(r1, c1, r2, c2, Z, opts);
                seg = astarGrid(trav, cur, goal, costFn);

                if isempty(seg)
                    continue;
                end

                if isempty(pathCells)
                    pathCells = seg;
                else
                    pathCells = [pathCells; seg(2:end, :)]; %#ok<AGROW>
                end

                cur = goal;
            end

            if isempty(pathCells)
                pathCells = startRC;
            end

            path = pathCells;
        end

        function c = edgeCost(r1, c1, r2, c2, Z, opts)
            base = hypot(r2 - r1, c2 - c1);
            slope = abs(Z(r2, c2) - Z(r1, c1));
            c = base * (1 + opts.lambdaSlope * slope);
        end

        function path = applyHorizonShortcut(trav, path, Z, horizonCells)
            % Lightweight receding-horizon smoothing: skip redundant detours within window.

            if size(path, 1) <= horizonCells + 2
                return;
            end

            opts = PriorityPlanner.defaultOpts(struct('lambdaSlope', 0.08));
            out = path(1, :);
            i = 1;
            n = size(path, 1);

            while i < n
                j = min(n, i + horizonCells);
                costFn = @(r1, c1, r2, c2) PriorityPlanner.edgeCost(r1, c1, r2, c2, Z, opts);
                shortcut = astarGrid(trav, path(i, :), path(j, :), costFn);

                if ~isempty(shortcut) && size(shortcut, 1) < (j - i + 1)
                    out = [out; shortcut(2:end, :)]; %#ok<AGROW>
                    i = j;
                else
                    out = [out; path(i + 1, :)]; %#ok<AGROW>
                    i = i + 1;
                end
            end

            path = out;
        end

        function P = padPaths(seg, maxLen)
            numUAV = numel(seg);
            P = zeros(maxLen, numUAV, 2);

            for u = 1:numUAV
                pu = seg{u};
                Lu = size(pu, 1);
                P(1:Lu, u, :) = pu;

                if Lu < maxLen && Lu > 0
                    tail = repmat(pu(end, :), maxLen - Lu, 1);
                    P(Lu + 1:end, u, :) = tail;
                end
            end
        end
    end
end
