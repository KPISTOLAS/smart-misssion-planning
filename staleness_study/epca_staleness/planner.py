"""IUEF-EM priority planner (Algorithm 1) - Python port.

Faithful, dependency-light re-implementation of the MATLAB ``PriorityPlanner``
(the paper's Algorithm 1, "IoT-Uncertainty-Enhanced Field - Energy-aware
Multi-UAV" hybrid coverage planner).  Pipeline:

  1. Pick ``U`` traversable depot seeds along a near-top row.
  2. Select high-priority target cells from the (possibly stale) estimated
     weight map.
  3. Partition targets across UAVs with weighted Lloyd clustering
     (DARP-inspired capacitated decomposition).
  4. Order each UAV's targets with a priority-biased nearest-neighbour rule
     (weighted greedy insertion).
  5. Stitch an 8-connected A* path through the ordered targets.

The planner consumes an **estimate** of the priority weight (the digital-twin
map), so staleness directly changes which hotspots are prioritised.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import numpy as np


@dataclass
class PlannerOptions:
    plan_mode: str = "blend"        # 'heuristic' | 'eig' | 'blend'
    blend_gamma: float = 0.45       # weight of health vs. uncertainty in 'blend'
    lloyd_iterations: int = 40
    max_targets: int = 260          # cap targets for tractable A* stitching


# ---------------------------------------------------------------------- #
# A* on an 8-connected grid
# ---------------------------------------------------------------------- #
_NBR = [(-1, -1, np.sqrt(2)), (-1, 0, 1.0), (-1, 1, np.sqrt(2)),
        (0, -1, 1.0), (0, 1, 1.0),
        (1, -1, np.sqrt(2)), (1, 0, 1.0), (1, 1, np.sqrt(2))]


def astar_grid(trav: np.ndarray, start, goal):
    """Return a list of ``(r, c)`` cells from ``start`` to ``goal`` (inclusive).

    ``trav`` is a boolean traversability mask.  Returns ``[start]`` if no path.
    """
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
    came = {}
    gscore = {start: 0.0}
    closed = set()
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
        for dr, dc, w in _NBR:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < n and 0 <= nc < m) or not trav[nr, nc]:
                continue
            ng = g + w
            nb = (nr, nc)
            if ng < gscore.get(nb, np.inf):
                gscore[nb] = ng
                came[nb] = cur
                heapq.heappush(open_heap, (ng + h(nb), ng, nb))
    return [start]


class IUEFEMPlanner:
    """The IUEF-EM hybrid coverage planner (Algorithm 1)."""

    def __init__(self, options: PlannerOptions | None = None):
        self.opts = options or PlannerOptions()

    # ------------------------------------------------------------------ #
    def pick_depots(self, trav: np.ndarray, num_uav: int) -> np.ndarray:
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
        picks = np.clip(np.round(np.linspace(0, cols.size - 1, num_uav)).astype(int), 0, cols.size - 1)
        for u in range(num_uav):
            starts[u] = [row, cols[picks[u]]]
            if not trav[starts[u, 0], starts[u, 1]]:
                fr, fc = np.argwhere(trav)[0]
                starts[u] = [fr, fc]
        return starts

    # ------------------------------------------------------------------ #
    def _weights(self, W_est: np.ndarray, sigma: np.ndarray, H: np.ndarray, idx) -> np.ndarray:
        """Planner target weights from the (stale) estimate, per ``plan_mode``."""
        Wh = np.maximum(0.0, W_est[idx[:, 0], idx[:, 1]])
        # EIG proxy in bits from uncertainty (see GPInformationField.attach).
        v = sigma[idx[:, 0], idx[:, 1]] ** 2
        We = 0.5 * np.log2(1.0 + v / (0.12 ** 2))
        We = We * (H[idx[:, 0], idx[:, 1]] >= 3)
        mode = self.opts.plan_mode
        if mode == "heuristic":
            w = Wh
        elif mode == "eig":
            w = We if We.any() else Wh
        else:  # blend
            nh = Wh / max(Wh.max(), 1e-9)
            ne = We / max(We.max(), 1e-9)
            w = self.opts.blend_gamma * nh + (1 - self.opts.blend_gamma) * ne
            if not w.any():
                w = nh
        return w

    def _weighted_partition(self, coords: np.ndarray, wgt: np.ndarray, K: int) -> np.ndarray:
        """Weighted Lloyd (k-means) partition of target coordinates."""
        n = coords.shape[0]
        if K <= 1 or n == 0:
            return np.zeros(n, dtype=int)
        if n <= K:
            return np.arange(n) % K
        order = np.argsort(-wgt)
        ctr = coords[order[:K]].astype(float)
        lbl = np.zeros(n, dtype=int)
        for _ in range(self.opts.lloyd_iterations):
            d = np.linalg.norm(coords[:, None, :] - ctr[None, :, :], axis=2)
            lbl = np.argmin(d, axis=1)
            moved = 0.0
            for k in range(K):
                sel = lbl == k
                if not sel.any():
                    ctr[k] = coords[np.random.randint(n)]
                    moved += 1
                    continue
                wl = wgt[sel][:, None]
                newc = (coords[sel] * wl).sum(axis=0) / max(wl.sum(), 1e-9)
                moved += np.linalg.norm(newc - ctr[k])
                ctr[k] = newc
            if moved < 1e-3:
                break
        return lbl

    def _weighted_order(self, start, goals: np.ndarray, gw: np.ndarray) -> np.ndarray:
        """Priority-biased nearest-neighbour ordering (weighted greedy)."""
        ng = goals.shape[0]
        order = np.zeros(ng, dtype=int)
        unvisited = np.ones(ng, dtype=bool)
        cur = np.asarray(start, dtype=float)
        for k in range(ng):
            cand = np.where(unvisited)[0]
            dist = np.hypot(goals[cand, 0] - cur[0], goals[cand, 1] - cur[1])
            score = gw[cand] / np.maximum(dist, 1e-9)
            pick = cand[np.argmax(score)]
            order[k] = pick
            unvisited[pick] = False
            cur = goals[pick].astype(float)
        return order

    # ------------------------------------------------------------------ #
    def build_plan(self, field, W_est: np.ndarray, num_uav: int,
                   starts: np.ndarray | None = None) -> dict:
        """Build per-UAV waypoint paths from the (stale) estimated weight map.

        Parameters
        ----------
        field:
            :class:`~epca_staleness.environment.PriorityField` (ground truth
            geometry: traversability, sigma, H).
        W_est:
            The *estimated* (possibly stale) priority weight the planner sees.
        num_uav:
            Number of UAVs.
        starts:
            Optional fixed depots (keeps UAVs consistent across replans).
        """
        trav = field.traversable
        n, m = trav.shape
        if starts is None:
            starts = self.pick_depots(trav, num_uav)

        # Target selection: cells whose *estimated* (possibly faded) weight still
        # exceeds the field's hotspot threshold.  Because fade scales W_est down,
        # faded hotspots drop below this threshold and are no longer targeted.
        # NOTE: there is intentionally NO "cover everything" fallback - if the
        # stale belief no longer shows a hotspot, the planner cannot target it.
        w_hi = field.meta.get("w_hi", None)
        if w_hi is None:
            w_hi = field.meta.get("high_health_thr", 3) * field.meta.get("alpha", 1.0) * 0.55
        cand_mask = (W_est >= w_hi) & trav

        # Grid-centre patrol cell (used when a UAV has no detectable hotspots:
        # it loiters toward the centre, modelling loss of situational awareness).
        center = np.array([n // 2, m // 2])
        if not trav[center[0], center[1]]:
            free = np.argwhere(trav)
            center = free[np.argmin(np.linalg.norm(free - center, axis=1))]

        idx = np.argwhere(cand_mask)
        if idx.shape[0] == 0:
            # No detectable hotspots at all -> every UAV patrols the centre.
            segments = []
            for u in range(num_uav):
                seg = astar_grid(trav, tuple(starts[u]), tuple(center))
                segments.append(seg if seg else [tuple(starts[u])])
            return {"segments": segments, "starts": starts, "num_uav": num_uav}
        # Cap number of targets (keep the strongest) for tractable stitching.
        if idx.shape[0] > self.opts.max_targets:
            vals = W_est[idx[:, 0], idx[:, 1]]
            keep = np.argsort(-vals)[: self.opts.max_targets]
            idx = idx[keep]

        wgt = self._weights(W_est, field.sigma, field.H, idx)
        assigns = self._weighted_partition(idx.astype(float), wgt, num_uav)

        segments = []
        for u in range(num_uav):
            sel = assigns == u
            goals = idx[sel]
            gw = wgt[sel]
            if goals.shape[0] == 0:
                # This UAV's belief shows no hotspots -> patrol toward centre.
                seg = astar_grid(trav, tuple(starts[u]), tuple(center))
                segments.append(seg if seg else [tuple(starts[u])])
                continue
            order = self._weighted_order(starts[u], goals.astype(float), gw)
            ordered = goals[order]
            path = []
            cur = tuple(starts[u])
            for g in ordered:
                seg = astar_grid(trav, cur, tuple(g))
                if not seg:
                    continue
                path.extend(seg if not path else seg[1:])
                cur = tuple(g)
            if not path:
                path = [tuple(starts[u])]
            segments.append(path)

        return {"segments": segments, "starts": starts, "num_uav": num_uav}
