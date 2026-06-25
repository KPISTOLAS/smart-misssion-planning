classdef BaselinePlanners
    % BASELINEPLANNERS — Lawnmower (boustrophedon), greedy IoT-weighted CPP, and a
    %   decentralized Voronoi-partitioned greedy surrogate for swarm comparisons.

    methods (Static)

        function plan = lawnmower(mapData, numUAV)
            trav = ~mapData.obstacle;
            [N, M] = size(trav);
            order = BaselinePlanners.serpentineOrder(trav);
            starts = PriorityPlanner.pickDepots(trav, numUAV);

            chunks = BaselinePlanners.splitSequence(order, numUAV);
            seg = cell(numUAV, 1);
            maxLen = 0;

            for u = 1:numUAV
                seq = chunks{u};
                path = BaselinePlanners.connectSequence(trav, starts(u, :), seq);
                seg{u} = path;
                maxLen = max(maxLen, size(path, 1));
            end

            plan = struct('segment', {seg}, ...
                'positions', PriorityPlanner.padPaths(seg, maxLen), ...
                'starts', starts, ...
                'meta', struct(), ...
                'name', 'lawnmower');
        end

        function plan = decentralizedVoronoiGreedy(mapData, numUAV)
            % DECENTRALIZEDVORONOIGREEDY Engineering surrogate for decentralized swarms:
            %   nearest-depot Voronoi assignment of high-priority targets, then each
            %   UAV runs a weighted greedy visit order with local A* stitching only
            %   inside its cell (no global capacitated refinement — contrast with
            %   PriorityPlanner hybrid). Suitable baseline for "partitioned myopic"
            %   multi-agent coverage.

            trav = ~mapData.obstacle;
            W = mapData.W;
            starts = PriorityPlanner.pickDepots(trav, numUAV);

            highMask = mapData.highPriorityMask & trav;
            [tr, tc] = find(highMask);

            if isempty(tr)
                [tr, tc] = find(trav);
            end

            coords = [tr, tc];
            nT = size(coords, 1);
            assigns = ones(nT, 1);

            if numUAV > 1 && nT > 0
                D = zeros(nT, numUAV);

                for u = 1:numUAV
                    dr = coords(:, 1) - starts(u, 1);
                    dc = coords(:, 2) - starts(u, 2);
                    D(:, u) = hypot(dr, dc);
                end

                [~, assigns] = min(D, [], 2);
            end

            seg = cell(numUAV, 1);
            maxLen = 0;

            for u = 1:numUAV
                mask = assigns == u;
                goals = coords(mask, :);

                if isempty(goals)
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

                gw = zeros(size(goals, 1), 1);

                for i = 1:size(goals, 1)
                    gw(i) = W(goals(i, 1), goals(i, 2));
                end

                order = PriorityPlanner.weightedGreedyOrder(starts(u, :), goals, gw);
                orderedGoals = goals(order, :);
                path = BaselinePlanners.connectSequence(trav, starts(u, :), orderedGoals);
                seg{u} = path;
                maxLen = max(maxLen, size(path, 1));
            end

            plan = struct('segment', {seg}, ...
                'positions', PriorityPlanner.padPaths(seg, maxLen), ...
                'starts', starts, ...
                'meta', struct(), ...
                'name', 'decentralized_voronoi_greedy');
        end

        function plan = greedy(mapData, numUAV)
            trav = ~mapData.obstacle;
            [N, M] = size(trav);
            W = mapData.W;
            starts = PriorityPlanner.pickDepots(trav, numUAV);

            seg = cell(numUAV, 1);
            maxLen = 0;

            for u = 1:numUAV
                path = BaselinePlanners.greedyWalk(trav, W, starts(u, :), ...
                    ceil(N * M / max(numUAV, 1)));
                seg{u} = path;
                maxLen = max(maxLen, size(path, 1));
            end

            plan = struct('segment', {seg}, ...
                'positions', PriorityPlanner.padPaths(seg, maxLen), ...
                'starts', starts, ...
                'meta', struct(), ...
                'name', 'greedy');
        end

        function order = serpentineOrder(trav)
            [N, M] = size(trav);
            order = zeros(0, 2);

            for r = 1:N
                cols = 1:M;
                if mod(r, 2) == 0
                    cols = fliplr(cols);
                end

                for c = cols
                    if trav(r, c)
                        order(end + 1, :) = [r, c]; %#ok<AGROW>
                    end
                end
            end
        end

        function chunks = splitSequence(order, K)
            chunks = cell(K, 1);
            L = size(order, 1);
            if L == 0
                return;
            end

            edges = round(linspace(1, L + 1, K + 1));

            for u = 1:K
                a = edges(u);
                b = edges(u + 1) - 1;
                if b >= a
                    chunks{u} = order(a:b, :);
                else
                    chunks{u} = order(min(L, a), :);
                end
            end
        end

        function path = connectSequence(trav, startRC, seq)
            pathCells = zeros(0, 2);
            cur = startRC;

            if isempty(seq)
                path = startRC;
                return;
            end

            block = [startRC; seq];
            for k = 2:size(block, 1)
                goal = block(k, :);
                seg = astarGrid(trav, cur, goal);
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

        function path = greedyWalk(trav, W, startRC, maxSteps)
            [N, M] = size(trav);
            visited = false(N, M);
            path = zeros(maxSteps + 16, 2);
            path(1, :) = startRC;
            visited(startRC(1), startRC(2)) = true;
            cnt = 1;

            nbr = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];

            for step = 1:maxSteps
                r = path(cnt, 1);
                c = path(cnt, 2);
                bestScore = -inf;
                nxt = [];

                for h = 1:size(nbr, 1)
                    nr = r + nbr(h, 1);
                    nc = c + nbr(h, 2);
                    if nr < 1 || nr > N || nc < 1 || nc > M
                        continue;
                    end
                    if ~trav(nr, nc)
                        continue;
                    end

                    bonus = 3 * (~visited(nr, nc));
                    score = W(nr, nc) + bonus;

                    if score > bestScore
                        bestScore = score;
                        nxt = [nr, nc];
                    elseif score == bestScore && ~isempty(nxt)
                        if hypot(nr - r, nc - c) < hypot(nxt(1) - r, nxt(2) - c)
                            nxt = [nr, nc];
                        end
                    end
                end

                if isempty(nxt)
                    cand = find(~visited & trav);
                    if isempty(cand)
                        break;
                    end
                    [gr, gc] = ind2sub([N, M], cand(randi(numel(cand))));
                    seg = astarGrid(trav, [r, c], [gr, gc]);
                    if isempty(seg)
                        break;
                    end

                    for k = 2:size(seg, 1)
                        cnt = cnt + 1;
                        path(cnt, :) = seg(k, :);
                        visited(seg(k, 1), seg(k, 2)) = true;
                    end

                    continue;
                end

                cnt = cnt + 1;
                path(cnt, :) = nxt;
                visited(nxt(1), nxt(2)) = true;
            end

            path = path(1:cnt, :);
        end
    end
end
