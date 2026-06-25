classdef MapGenerator
    % MAPGENERATOR Synthetic environment for adaptive plant-health CPP.
    %   Produces: digital elevation Z (m), discrete health H in [0,4],
    %   uncertainty sigma, obstacle mask, obstacle proximity O in [0,1],
    %   and composite priority weight W = alpha*H + beta*sigma - gamma*O.
    %
    %   Usage:
    %       opts = struct('terrain','hilly','treeDensity','medium',...
    %                     'alpha',1,'beta',0.5,'gamma',0.3,'rngSeed',42);
    %       mapData = MapGenerator.build(N,M,dx,opts);
    %
    %   See also PriorityPlanner, MissionSim, Analytics.

    methods (Static)

        function mapData = build(N, M, dx, opts)
            % MAPDATA = BUILD(N,M,dx,OPTS) — assemble full map structure.

            arguments
                N (1,1) double {mustBePositive, mustBeInteger}
                M (1,1) double {mustBePositive, mustBeInteger}
                dx (1,1) double {mustBePositive}
                opts struct = struct()
            end

            opts = MapGenerator.fillDefaultOpts(opts);

            if opts.rngSeed >= 0
                rng(opts.rngSeed, 'twister');
            end

            mapData = struct();
            mapData.N = N;
            mapData.M = M;
            mapData.dx = dx;
            mapData.cellArea_m2 = dx^2;

            mapData.Z_m = MapGenerator.terrainSurface(N, M, dx, opts.terrain);
            mapData.obstacle = MapGenerator.obstacleMask(N, M, opts.treeDensity);
            mapData.obsFrac = mean(mapData.obstacle(:));
            [mapData.H, mapData.sigma] = MapGenerator.iotHeatmap(N, M, mapData.obstacle);

            mapData.O = MapGenerator.obstacleProximity(mapData.obstacle, opts.sigmaObsKernel);
            mapData.W = opts.alpha .* mapData.H + opts.beta .* mapData.sigma ...
                - opts.gamma .* mapData.O;

            mapData.highPriorityMask = (mapData.H >= opts.highHealthThr);
            mapData.meta = opts;

            % GP / information-theoretic priors (EIG proxy in bits; active sensing score).
            mapData = GPInformationField.attach(mapData, struct());
        end

        function opts = fillDefaultOpts(opts)
            d = struct( ...
                'terrain', 'flat', ... % 'flat' | 'hilly' | 'ridge'
                'treeDensity', 'medium', ...
                'alpha', 1.0, ...
                'beta', 0.5, ...
                'gamma', 0.3, ...
                'highHealthThr', 3, ...
                'sigmaObsKernel', 2.5, ...
                'rngSeed', -1);
            fn = fieldnames(d);
            for k = 1:numel(fn)
                if ~isfield(opts, fn{k}) || isempty(opts.(fn{k}))
                    opts.(fn{k}) = d.(fn{k});
                end
            end
            opts.terrain = lower(char(opts.terrain));
            opts.treeDensity = lower(char(opts.treeDensity));
        end

        function Z = terrainSurface(N, M, dx, terrainType)
            % Piecewise-smooth elevation (m). Flat baseline + optional hills.

            [jj, ii] = meshgrid(1:M, 1:N);
            switch terrainType
                case 'flat'
                    Z = zeros(N, M);
                case 'hilly'
                    % Summed radial bumps + gentle plane tilt for variety.
                    ctr = [N, M] / 2;
                    Z = 18 * exp(-((ii - ctr(1)).^2 + (jj - ctr(2)).^2) / (0.22 * (N + M)));
                    Z = Z + 12 * exp(-((ii - 0.35 * N).^2 + (jj - 0.62 * M).^2) / (0.08 * (N + M)));
                    Z = Z + 0.008 * dx * (ii - jj);
                case 'ridge'
                    % Elongated ridge: directional slope + transverse Gaussian crest.
                    ridgeAxis = 0.65 * ii + 0.35 * jj;
                    Z = 22 * exp(-(ridgeAxis - 0.55 * (N + M)).^2 / (0.12 * (N + M)));
                    Z = Z + 0.012 * dx * (jj - ii);
                otherwise
                    error('MapGenerator:terrainSurface:UnknownTerrain', ...
                        'terrain must be ''flat'', ''hilly'', or ''ridge''.');
            end
        end

        function obs = obstacleMask(N, M, densityTag)
            % Binary obstacle field (trees / no-fly patches).

            switch densityTag
                case 'low'
                    p = 0.06;
                    rMin = 4;
                case 'medium'
                    p = 0.12;
                    rMin = 3;
                case 'high'
                    p = 0.20;
                    rMin = 2;
                otherwise
                    error('MapGenerator:obstacleMask:Density', ...
                        'treeDensity must be ''low'', ''medium'', or ''high''.');
            end

            obs = false(N, M);
            trials = rand(N, M);
            seeds = trials < p;
            [rs, cs] = find(seeds);

            for k = 1:numel(rs)
                if obs(rs(k), cs(k))
                    continue;
                end
                rad = randi([rMin, rMin + 2]);
                MapGenerator.paintDisk(obs, rs(k), cs(k), rad);
            end

            % Ensure border strip mostly free for depot-style starts.
            obs(1:2, :) = false;
            obs(end-1:end, :) = false;
            obs(:, 1:2) = false;
            obs(:, end-1:end) = false;
        end

        function paintDisk(obs, r0, c0, rad)
            [N, M] = size(obs);
            r1 = max(1, r0 - rad);
            r2 = min(N, r0 + rad);
            c1 = max(1, c0 - rad);
            c2 = min(M, c0 + rad);
            for r = r1:r2
                for c = c1:c2
                    if (r - r0)^2 + (c - c0)^2 <= rad^2
                        obs(r, c) = true;
                    end
                end
            end
        end

        function [H, sigma] = iotHeatmap(N, M, obstacle)
            % Discrete health H in {0,...,4} and uncertainty sigma in (0,1].

            base = smoothField(N, M, 8);
            stress = smoothField(N, M, 5);

            H = zeros(N, M);
            thr = MapGenerator.percentileLinear(base(:), [20 40 60 80]);
            H(base <= thr(1)) = 0;
            H(base > thr(1) & base <= thr(2)) = 1;
            H(base > thr(2) & base <= thr(3)) = 2;
            H(base > thr(3) & base <= thr(4)) = 3;
            H(base > thr(4)) = 4;

            % Elevated stress in clustered regions (pest / drought pockets).
            pocket = stress > MapGenerator.percentileLinear(stress(:), 75);
            H(pocket) = min(4, H(pocket) + 1);

            % Obstacles obscure sensing -> raise uncertainty nearby.
            sigma = 0.15 + 0.85 * rescale01(stress);
            dil = MapGenerator.dilateDisk(obstacle, 2);
            sigma(dil) = min(1, sigma(dil) + 0.25);

            % Inside obstacles: undefined; mask handled by planner (non-traversable).
            H(obstacle) = 0;
            sigma(obstacle) = 1;

            function u = smoothField(n2, m2, wx)
                u = conv2(rand(n2, m2), ones(wx, wx), 'same');
                u = u / max(u(:));
            end

            function y = rescale01(x)
                y = (x - min(x(:))) / max(eps, max(x(:)) - min(x(:)));
            end
        end

        function O = obstacleProximity(obstacle, sigmaBlurPx)
            % O in [0,1]: normalized distance transform proximity to obstacles.

            if ~any(obstacle(:))
                O = zeros(size(obstacle));
                return;
            end

            invDist = MapGenerator.manhattanDistToObstacles(obstacle);
            O = exp(-invDist.^2 / (2 * sigmaBlurPx^2));
        end

        function out = dilateDisk(bin, radiusPx)
            arguments
                bin (:,:) logical
                radiusPx (1,1) double {mustBeNonnegative}
            end

            r = radiusPx;
            if r == 0
                out = bin;
                return;
            end

            [JJ, II] = meshgrid(-r:r, -r:r);
            se = (II.^2 + JJ.^2) <= r^2;
            out = conv2(double(bin), double(se), 'same') > 0;
        end

        function D = manhattanDistToObstacles(obstacle)
            % Multi-source BFS (8-neighborhood, unit steps): distance to nearest obstacle cell.

            [N, M] = size(obstacle);
            D = inf(N, M);
            cap = N * M * 4;
            qr = zeros(cap, 1);
            qc = zeros(cap, 1);
            tail = 0;

            [rs, cs] = find(obstacle);
            for k = 1:numel(rs)
                D(rs(k), cs(k)) = 0;
                tail = tail + 1;
                qr(tail) = rs(k);
                qc(tail) = cs(k);
            end

            if tail == 0
                D(:) = max(N, M);
                return;
            end

            head = 1;
            nbr = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];

            while head <= tail
                r = qr(head);
                c = qc(head);
                head = head + 1;
                base = D(r, c);
                for h = 1:size(nbr, 1)
                    nr = r + nbr(h, 1);
                    nc = c + nbr(h, 2);
                    if nr < 1 || nr > N || nc < 1 || nc > M
                        continue;
                    end
                    cand = base + 1;
                    if cand < D(nr, nc)
                        D(nr, nc) = cand;
                        tail = tail + 1;
                        qr(tail) = nr;
                        qc(tail) = nc;
                    end
                end
            end
        end

        function q = percentileLinear(x, p)
            % PERCENTILELINEAR Toolbox-free percentiles (linear interpolation on sorted samples).
            %   Matches the common MATLAB prctile definition for vector X and percent levels P in [0,100].

            x = sort(double(x(:)));
            n = numel(x);
            p = double(p(:)).';
            q = zeros(size(p));

            for i = 1:numel(p)
                pi = max(0, min(100, p(i)));
                if n == 0
                    q(i) = nan;
                elseif n == 1
                    q(i) = x(1);
                else
                    k = (pi / 100) * (n - 1) + 1;
                    f = floor(k);
                    c = min(n, ceil(k));
                    q(i) = x(f) + (k - f) * (x(c) - x(f));
                end
            end
        end
    end
end
