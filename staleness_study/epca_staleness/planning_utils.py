"""Shared planning utilities: A*, depots, path stitching, capacitated Voronoi.

Notation follows the paper: traversable cells carry priority weight W_i,
obstacle-proximity O_i in [0,1], safety separation d_safe (m), grid spacing dx.
"""

from __future__ import annotations

import heapq
import numpy as np

_NBR = [(-1, -1, np.sqrt(2)), (-1, 0, 1.0), (-1, 1, np.sqrt(2)),
        (0, -1, 1.0), (0, 1, 1.0),
        (1, -1, np.sqrt(2)), (1, 0, 1.0), (1, 1, np.sqrt(2))]


def pick_depots(trav: np.ndarray, num_uav: int) -> np.ndarray:
    """Place U depots along a near-top traversable row (shared across planners)."""
    n, m = trav.shape
    row = min(max(2, 2), n - 2)
    cols = np.where(trav[row])[0]
    if cols.size < num_uav:
        cols = np.where(trav.any(axis=0))[0]
    starts = np.zeros((num_uav, 2), dtype=int)
    if cols.size == 0:
        starts[:, 0] = 1
        starts[:, 1] = np.arange(num_uav) + 1
        return starts
    picks = np.clip(np.round(np.linspace(0, cols.size - 1, num_uav)).astype(int),
                    0, cols.size - 1)
    for u in range(num_uav):
        starts[u] = [row, cols[picks[u]]]
        if not trav[starts[u, 0], starts[u, 1]]:
            fr, fc = np.argwhere(trav)[0]
            starts[u] = [fr, fc]
    return starts


def edge_cost(r1, c1, r2, c2, O=None, Z=None,
              lambda_slope: float = 0.08,
              lambda_cong: float = 0.35) -> float:
    """Elevation- and congestion-aware edge cost for A*."""
    base = np.hypot(r2 - r1, c2 - c1)
    slope = 0.0
    if Z is not None:
        slope = abs(Z[r2, c2] - Z[r1, c1])
    cong = 0.0
    if O is not None:
        cong = 0.5 * (O[r1, c1] + O[r2, c2])
    return base * (1.0 + lambda_slope * slope + lambda_cong * cong)


def astar_grid(trav: np.ndarray, start, goal,
               O: np.ndarray | None = None,
               Z: np.ndarray | None = None,
               lambda_slope: float = 0.08,
               lambda_cong: float = 0.35):
    """8-connected A* with optional elevation / congestion edge costs."""
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    n, m = trav.shape
    if not (0 <= goal[0] < n and 0 <= goal[1] < m) or not trav[goal]:
        return [start]
    if start == goal:
        return [start]

    def h(a):
        return np.hypot(a[0] - goal[0], a[1] - goal[1])

    open_heap = [(h(start), 0.0, start)]
    came, gscore, closed = {}, {start: 0.0}, set()
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if cur in closed:
            continue
        closed.add(cur)
        for dr, dc, _ in _NBR:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < n and 0 <= nc < m) or not trav[nr, nc]:
                continue
            step = edge_cost(cur[0], cur[1], nr, nc, O, Z, lambda_slope, lambda_cong)
            ng = g + step
            nb = (nr, nc)
            if ng < gscore.get(nb, np.inf):
                gscore[nb] = ng
                came[nb] = cur
                heapq.heappush(open_heap, (ng + h(nb), ng, nb))
    return [start]


def manhattan_stitch(trav: np.ndarray, start, goal):
    """Greedy 8-connected Manhattan walk without A* (ablation: no A* refinement)."""
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    path = [start]
    cur = list(start)
    guard = 0
    while tuple(cur) != goal and guard < 5000:
        guard += 1
        dr = int(np.sign(goal[0] - cur[0]))
        dc = int(np.sign(goal[1] - cur[1]))
        moved = False
        for drr, dcc in [(dr, dc), (dr, 0), (0, dc)]:
            if drr == 0 and dcc == 0:
                continue
            nr, nc = cur[0] + drr, cur[1] + dcc
            if 0 <= nr < trav.shape[0] and 0 <= nc < trav.shape[1] and trav[nr, nc]:
                cur = [nr, nc]
                path.append(tuple(cur))
                moved = True
                break
        if not moved:
            break
    return path


def stitch_goals(trav, start, ordered_goals, O=None, Z=None,
                 use_astar: bool = True, lambda_cong: float = 0.35):
    """Connect ordered goals into one polyline."""
    path = []
    cur = tuple(start)
    for g in ordered_goals:
        g = tuple(g)
        if use_astar:
            seg = astar_grid(trav, cur, g, O=O, Z=Z, lambda_cong=lambda_cong)
        else:
            seg = manhattan_stitch(trav, cur, g)
        if not seg:
            continue
        path.extend(seg if not path else seg[1:])
        cur = g
    return path if path else [tuple(start)]


def capacitated_voronoi(coords: np.ndarray, wgt: np.ndarray,
                        depots: np.ndarray, dx: float = 18.0,
                        eta: float = 1.0, num_iter: int = 14) -> np.ndarray:
    """Classic DARP-style capacitated Voronoi (ported from SpatialDecomposition.m).

    eta controls workload-balancing strength (eta=0 -> nearest-depot only).
    """
    n, K = coords.shape[0], depots.shape[0]
    if K <= 1 or n == 0:
        return np.zeros(n, dtype=int)
    if n <= K:
        order = np.argsort(coords[:, 0] * 1e6 + coords[:, 1])
        lbl = np.zeros(n, dtype=int)
        for ii, idx in enumerate(order):
            lbl[idx] = min(ii, K - 1)
        return lbl

    D = np.zeros((n, K))
    for u in range(K):
        D[:, u] = np.hypot(coords[:, 0] - depots[u, 0],
                           coords[:, 1] - depots[u, 1]) * dx
    if eta <= 1e-9:
        return np.argmin(D, axis=1)

    lam = np.ones(K)
    mass_target = max(wgt.sum(), 1e-9) / K
    lbl = np.zeros(n, dtype=int)
    for _ in range(num_iter):
        for i in range(n):
            costs = lam * (D[i] ** 2)
            lbl[i] = int(np.argmin(costs))
        mass = np.array([wgt[lbl == u].sum() for u in range(K)])
        lam *= np.sqrt(np.maximum(mass / mass_target, 1e-9))
        lam /= max(lam.mean(), 1e-9)
    return lbl


def weighted_lloyd_partition(coords: np.ndarray, wgt: np.ndarray, K: int,
                           iterations: int = 40) -> np.ndarray:
    """Weighted Lloyd clustering (fallback partition without capacitated Voronoi)."""
    n = coords.shape[0]
    if K <= 1:
        return np.zeros(n, dtype=int)
    order = np.argsort(-wgt)
    ctr = coords[order[:K]].astype(float)
    lbl = np.zeros(n, dtype=int)
    for _ in range(iterations):
        d = np.linalg.norm(coords[:, None, :] - ctr[None, :, :], axis=2)
        lbl = np.argmin(d, axis=1)
        for k in range(K):
            sel = lbl == k
            if not sel.any():
                ctr[k] = coords[np.random.randint(n)]
                continue
            wl = wgt[sel][:, None]
            ctr[k] = (coords[sel] * wl).sum(axis=0) / max(wl.sum(), 1e-9)
    return lbl


def horizon_shortcut(trav, path, Z=None, O=None, horizon: int = 12,
                     lambda_cong: float = 0.35):
    """Receding-horizon shortcut smoothing (Algorithm 1 final step)."""
    if len(path) <= horizon + 2:
        return path
    out = [path[0]]
    i, n = 0, len(path)
    while i < n - 1:
        j = min(n - 1, i + horizon)
        shortcut = astar_grid(trav, path[i], path[j], O=O, Z=Z, lambda_cong=lambda_cong)
        if shortcut and len(shortcut) < j - i + 1:
            out.extend(shortcut[1:])
            i = j
        else:
            out.append(path[i + 1])
            i += 1
    return out
