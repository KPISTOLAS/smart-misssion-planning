function path = astarGrid(trav, startRC, goalRC, costFn)
    % ASTARGRID — 8-connected A* on logical traversable grid TRAV.
    %   STARTRC, GOALRC are [row, col]. COSTFN(R1,C1,R2,C2) returns edge cost.
    %   PATH is Kx2 list of nodes including start and goal; empty if unreachable.

    arguments
        trav (:,:) logical
        startRC (1,2) double
        goalRC (1,2) double
        costFn function_handle = @(r1, c1, r2, c2) hypot(r2 - r1, c2 - c1)
    end

    [N, M] = size(trav);
    sr = startRC(1);
    sc = startRC(2);
    gr = goalRC(1);
    gc = goalRC(2);

    if ~trav(sr, sc) || ~trav(gr, gc)
        path = zeros(0, 2);
        return;
    end

    gScore = inf(N, M);
    cameR = zeros(N, M);
    cameC = zeros(N, M);

    gScore(sr, sc) = 0;

    openMask = false(N, M);
    openMask(sr, sc) = true;
    closed = false(N, M);

    nbr = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];

    while any(openMask(:))
        [r, c] = popLowestOpen(openMask, gScore, gr, gc);
        if isempty(r)
            path = zeros(0, 2);
            return;
        end

        openMask(r, c) = false;
        closed(r, c) = true;

        if r == gr && c == gc
            path = reconstructPath(cameR, cameC, sr, sc, gr, gc);
            return;
        end

        for h = 1:size(nbr, 1)
            nr = r + nbr(h, 1);
            nc = c + nbr(h, 2);
            if nr < 1 || nr > N || nc < 1 || nc > M
                continue;
            end
            if ~trav(nr, nc) || closed(nr, nc)
                continue;
            end

            tentative = gScore(r, c) + costFn(r, c, nr, nc);
            if tentative < gScore(nr, nc)
                cameR(nr, nc) = r;
                cameC(nr, nc) = c;
                gScore(nr, nc) = tentative;
                openMask(nr, nc) = true;
            end
        end
    end

    path = zeros(0, 2);
end

function h = heuristic(r, c, gr, gc)
    h = hypot(gr - r, gc - c);
end

function [r, c] = popLowestOpen(openMask, gScore, gr, gc)
    idx = find(openMask);
    if isempty(idx)
        r = [];
        c = [];
        return;
    end

    bestF = inf;
    bestLinear = idx(1);

    for k = 1:numel(idx)
        linearIdx = idx(k);
        [rr, cc] = ind2sub(size(openMask), linearIdx);
        f = gScore(rr, cc) + heuristic(rr, cc, gr, gc);
        if f < bestF
            bestF = f;
            bestLinear = linearIdx;
        end
    end

    [r, c] = ind2sub(size(openMask), bestLinear);
end

function path = reconstructPath(cameR, cameC, sr, sc, gr, gc)
    path = zeros(1024, 2);
    cnt = 0;
    r = gr;
    c = gc;

    while true
        cnt = cnt + 1;
        path(cnt, :) = [r, c];
        if r == sr && c == sc
            break;
        end
        pr = cameR(r, c);
        pc = cameC(r, c);
        if pr == 0 && pc == 0
            path = zeros(0, 2);
            return;
        end
        r = pr;
        c = pc;
        if cnt > numel(cameR(:))
            path = zeros(0, 2);
            return;
        end
    end

    path = flipud(path(1:cnt, :));
end
