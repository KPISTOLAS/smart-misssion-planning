classdef PathSmoother
    % PATHSMOOTHER Chaikin corner cutting in metric space + sparse waypoint snapping.
    %   Reduces zig-zag grid artifacts for lower modeled energy and cross-track variance vs. raw grid polyline.

    methods (Static)

        function pathRC = smoothGridPath(pathRC, dx, trav, iterations)
            arguments
                pathRC (:, 2) double
                dx (1, 1) double
                trav (:, :) logical
                iterations (1, 1) double {mustBeNonnegative, mustBeInteger} = 2
            end

            if size(pathRC, 1) <= 2 || iterations == 0
                return;
            end

            xy = [(pathRC(:, 2) - 1) * dx, (pathRC(:, 1) - 1) * dx];
            xy = PathSmoother.chaikinClosed(xy, iterations);

            knots = round(linspace(1, size(xy, 1), max(3, ceil(size(pathRC, 1) / 2))));
            knots = unique(knots, 'stable');
            wayRC = PathSmoother.snapXYtoTraversable(xy(knots, :), dx, trav);

            pathRC = PathSmoother.stitchWaypoints(trav, wayRC);
        end

        function xy = chaikinClosed(xy, iters)
            for k = 1:iters
                n = size(xy, 1);
                if n < 3
                    return;
                end

                newPts = zeros(2 * n, 2);
                ptr = 0;

                for i = 1:n - 1
                    p = xy(i, :);
                    q = xy(i + 1, :);
                    ptr = ptr + 1;
                    newPts(ptr, :) = 0.75 * p + 0.25 * q;
                    ptr = ptr + 1;
                    newPts(ptr, :) = 0.25 * p + 0.75 * q;
                end

                xy = newPts(1:ptr, :);
            end
        end

        function rc = snapXYtoTraversable(xy, dx, trav)
            [N, M] = size(trav);
            k = size(xy, 1);
            rc = zeros(k, 2);

            for i = 1:k
                cc = round(xy(i, 1) / max(dx, eps) + 1);
                rr = round(xy(i, 2) / max(dx, eps) + 1);
                rr = max(1, min(N, rr));
                cc = max(1, min(M, cc));

                if trav(rr, cc)
                    rc(i, :) = [rr, cc];
                else
                    [rr2, cc2] = PathSmoother.nearestTraversable(trav, rr, cc);
                    rc(i, :) = [rr2, cc2];
                end
            end
        end

        function [rr, cc] = nearestTraversable(trav, r0, c0)
            [N, M] = size(trav);
            bestD = inf;
            rr = r0;
            cc = c0;

            radMax = max(N, M);

            for rad = 0:radMax
                r1 = max(1, r0 - rad);
                r2 = min(N, r0 + rad);
                c1 = max(1, c0 - rad);
                c2 = min(M, c0 + rad);

                for r = r1:r2
                    for c = c1:c2
                        if trav(r, c)
                            d = (r - r0)^2 + (c - c0)^2;

                            if d < bestD
                                bestD = d;
                                rr = r;
                                cc = c;
                            end
                        end
                    end
                end

                if bestD < inf
                    return;
                end
            end
        end

        function path = stitchWaypoints(trav, wayRC)
            path = wayRC(1, :);

            for k = 2:size(wayRC, 1)
                seg = astarGrid(trav, path(end, :), wayRC(k, :));

                if isempty(seg)
                    continue;
                end

                if size(path, 1) == 1
                    path = seg;
                else
                    path = [path; seg(2:end, :)]; %#ok<AGROW>
                end
            end

            if size(path, 1) < 2
                path = wayRC;
            end
        end
    end
end
