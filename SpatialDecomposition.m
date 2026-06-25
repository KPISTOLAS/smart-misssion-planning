classdef SpatialDecomposition
    % SPATIALDECOMPOSITION Heterogeneous spatial decomposition via weighted / capacitated Voronoi
    %   Lloyd refinement — assigns informative targets to UAV seeds without clustering overlap.

    methods (Static)

        function assigns = capacitatedVoronoi(coords, wgt, depotRC, dx, numIter)
            arguments
                coords (:, 2) double
                wgt (:, 1) double
                depotRC (:, 2) double
                dx (1, 1) double
                numIter (1, 1) double {mustBePositive, mustBeInteger} = 12
            end

            n = size(coords, 1);
            K = size(depotRC, 1);
            assigns = ones(n, 1);

            if K <= 1
                return;
            end

            if n == 0
                assigns = zeros(0, 1);
                return;
            end

            % Fewer targets than UAVs (or equal): classic Voronoi/Lloyd cannot split
            % mass sensibly. Returning all-ones used to make every "empty" UAV fall
            % back to the full goal list in PriorityPlanner (duplicate coverage,
            % collisions, energy explosion). Assign targets exclusively to UAVs 1..n.
            if n <= K
                assigns = SpatialDecomposition.exclusiveTargetToUAV(coords, K);
                return;
            end

            D = zeros(n, K);

            for u = 1:K
                dr = coords(:, 1) - depotRC(u, 1);
                dc = coords(:, 2) - depotRC(u, 2);
                D(:, u) = hypot(dr, dc) * dx;
            end

            lam = ones(K, 1);
            massTarget = max(sum(wgt), eps) / K;

            for it = 1:numIter
                for i = 1:n
                    costs = lam(:).' .* (D(i, :) .^ 2);
                    [~, assigns(i)] = min(costs);
                end

                mass = zeros(K, 1);

                for u = 1:K
                    sel = assigns == u;
                    mass(u) = sum(wgt(sel));
                end

                lam = lam .* sqrt(max(eps, mass / massTarget));
                lam = lam / max(eps, mean(lam));
            end
        end

        function assigns = exclusiveTargetToUAV(coords, K)
            % EXCLUSIVETARGETTOUAV One distinct UAV per target when n <= K (stable order).

            arguments
                coords (:, 2) double
                K (1, 1) double {mustBePositive, mustBeInteger}
            end

            n = size(coords, 1);
            assigns = zeros(n, 1);

            if n == 0
                return;
            end

            [~, ord] = sort(coords(:, 1) * 1e6 + coords(:, 2));

            for ii = 1:n
                assigns(ord(ii)) = min(ii, K);
            end
        end
    end
end
