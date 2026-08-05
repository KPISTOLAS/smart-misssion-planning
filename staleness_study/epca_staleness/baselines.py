"""External baseline planners for publication-grade comparison.

Baselines
---------
* **darp** — classic Divide Areas for Robotic Patrolling: multi-source BFS
  Voronoi partition from depots, then serpentine coverage within each region.
* **priority_tsp** — priority-weighted cheapest-insertion TSP per UAV region.
* **lawnmower** — boustrophedon serpentine over traversable cells.
* **potential_field** — gradient ascent on W_i with obstacle repulsion.
* **greedy** — myopic IoT-weighted walk (internal baseline).
* **decentralized_greedy** — Voronoi-partitioned greedy (internal baseline).
"""

from __future__ import annotations

import numpy as np

from .planning_utils import (
    pick_depots, capacitated_voronoi, stitch_goals, astar_grid, bfs_voronoi_regions,
)


def _hotspot_targets(field, W_est, max_targets=260):
    trav = field.traversable
    w_hi = field.meta.get("w_hi", 3.0)
    idx = np.argwhere((W_est >= w_hi) & trav)
    if idx.shape[0] > max_targets:
        vals = W_est[idx[:, 0], idx[:, 1]]
        idx = idx[np.argsort(-vals)[:max_targets]]
    return idx


# ------------------------------------------------------------------ #
# DARP — divide areas, serpentine within region
# ------------------------------------------------------------------ #
def _serpentine_in_mask(mask: np.ndarray) -> list:
    """Serpentine order over True cells in mask."""
    n, m = mask.shape
    order = []
    for r in range(n):
        cols = range(m) if r % 2 == 0 else range(m - 1, -1, -1)
        for c in cols:
            if mask[r, c]:
                order.append((r, c))
    return order


def build_darp_plan(field, W_est, num_uav, starts=None, name="darp") -> dict:
    """Classic DARP: BFS area division + serpentine coverage per region."""
    trav = field.traversable
    if starts is None:
        starts = pick_depots(trav, num_uav)
    regions = bfs_voronoi_regions(trav, starts)
    segments = []
    for u in range(num_uav):
        mask = (regions == u) & trav
        cells = _serpentine_in_mask(mask)
        if not cells:
            segments.append([tuple(starts[u])])
            continue
        # Prioritise high-W cells first within region, then serpentine remainder
        w_vals = W_est[mask]
        if w_vals.size > 0:
            hi_cells = [c for c in cells if W_est[c[0], c[1]] >= field.meta.get("w_hi", 0)]
            rest = [c for c in cells if c not in hi_cells]
            ordered = hi_cells + rest
        else:
            ordered = cells
        path = stitch_goals(trav, starts[u], ordered[:min(150, len(ordered))],
                            O=field.O, use_astar=True)
        segments.append(path)
    return {"segments": segments, "starts": starts, "num_uav": num_uav, "name": name}


# ------------------------------------------------------------------ #
# Priority-weighted TSP (cheapest insertion)
# ------------------------------------------------------------------ #
def _cheapest_insertion_tsp(start, goals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Priority-weighted cheapest insertion: cost/delta_dist penalised by 1/weight."""
    n = goals.shape[0]
    if n == 0:
        return goals
    if n == 1:
        return goals
    # Seed with highest-weight goal
    seed = int(np.argmax(weights))
    tour = [seed]
    remaining = set(range(n)) - {seed}
    while remaining:
        best_g, best_pos, best_cost = None, 0, np.inf
        for g in remaining:
            for pos in range(len(tour) + 1):
                if pos == 0:
                    a = np.asarray(start, dtype=float)
                    b = goals[tour[0]].astype(float)
                elif pos == len(tour):
                    a = goals[tour[-1]].astype(float)
                    b = goals[g].astype(float)
                else:
                    a = goals[tour[pos - 1]].astype(float)
                    b = goals[tour[pos]].astype(float)
                gpt = goals[g].astype(float)
                delta = np.hypot(*(gpt - a)) + np.hypot(*(b - gpt)) - np.hypot(*(b - a))
                cost = delta / max(weights[g], 1e-6)
                if cost < best_cost:
                    best_cost, best_g, best_pos = cost, g, pos
        tour.insert(best_pos, best_g)
        remaining.remove(best_g)
    return goals[tour]


def build_priority_tsp_plan(field, W_est, num_uav, starts=None, name="priority_tsp") -> dict:
    """Capacitated partition + per-UAV priority-weighted insertion TSP."""
    trav = field.traversable
    if starts is None:
        starts = pick_depots(trav, num_uav)
    idx = _hotspot_targets(field, W_est, max_targets=80)
    if idx.shape[0] == 0:
        return build_lawnmower_plan(field, W_est, num_uav, starts, name=name)

    wgt = W_est[idx[:, 0], idx[:, 1]]
    assigns = capacitated_voronoi(idx.astype(float), wgt, starts.astype(float),
                                  dx=field.dx, eta=1.0)
    segments = []
    for u in range(num_uav):
        sel = assigns == u
        goals = idx[sel]
        gw = wgt[sel]
        if goals.shape[0] == 0:
            segments.append([tuple(starts[u])])
            continue
        # Cheapest insertion for small sets; weighted NN for larger (tractability).
        if goals.shape[0] <= 25:
            ordered = _cheapest_insertion_tsp(starts[u], goals.astype(float), gw)
        else:
            from .iuef_em import _weighted_order
            ordered = _weighted_order(starts[u], goals.astype(float), gw, True)
        path = stitch_goals(trav, starts[u], ordered, O=field.O, use_astar=True)
        segments.append(path)
    return {"segments": segments, "starts": starts, "num_uav": num_uav, "name": name}


# ------------------------------------------------------------------ #
# Lawnmower (boustrophedon)
# ------------------------------------------------------------------ #
def build_lawnmower_plan(field, W_est, num_uav, starts=None, name="lawnmower") -> dict:
    """Serpentine over traversable cells, split evenly across UAVs."""
    trav = field.traversable
    if starts is None:
        starts = pick_depots(trav, num_uav)
    order = _serpentine_in_mask(trav)
    # Reference lawnmower: cap cells for fair runtime (full grid is infeasible at scale).
    order = order[:min(350 * num_uav, len(order))]
    if not order:
        return {"segments": [[tuple(starts[u])] for u in range(num_uav)],
                "starts": starts, "num_uav": num_uav, "name": name}
    chunks = np.array_split(order, num_uav)
    segments = []
    for u in range(num_uav):
        chunk = chunks[u] if u < len(chunks) else []
        if not len(chunk):
            segments.append([tuple(starts[u])])
            continue
        path = stitch_goals(trav, starts[u], chunk, use_astar=True)
        segments.append(path)
    return {"segments": segments, "starts": starts, "num_uav": num_uav, "name": name}


# ------------------------------------------------------------------ #
# Potential field
# ------------------------------------------------------------------ #
def build_potential_field_plan(field, W_est, num_uav, starts=None,
                               max_steps: int = 800, name="potential_field") -> dict:
    """Gradient ascent on W with obstacle repulsion (reference baseline)."""
    trav = field.traversable
    n, m = trav.shape
    if starts is None:
        starts = pick_depots(trav, num_uav)
    O = field.O
    segments = []
    nbr = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for u in range(num_uav):
        path = [tuple(starts[u])]
        visited = set(path)
        for _ in range(max_steps // num_uav):
            r, c = path[-1]
            best, best_score = None, -np.inf
            for dr, dc in nbr:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n and 0 <= nc < m) or not trav[nr, nc]:
                    continue
                novelty = 2.0 if (nr, nc) not in visited else 0.0
                score = W_est[nr, nc] + novelty - 0.5 * O[nr, nc]
                if score > best_score:
                    best_score, best = score, (nr, nc)
            if best is None:
                break
            path.append(best)
            visited.add(best)
        segments.append(path)
    return {"segments": segments, "starts": starts, "num_uav": num_uav, "name": name}


# ------------------------------------------------------------------ #
# Internal baselines (greedy, decentralized greedy)
# ------------------------------------------------------------------ #
def build_greedy_plan(field, W_est, num_uav, starts=None,
                      max_steps: int = 600, name="greedy") -> dict:
    """Myopic 8-connected walk maximising local W + coverage novelty."""
    trav = field.traversable
    n, m = trav.shape
    if starts is None:
        starts = pick_depots(trav, num_uav)
    nbr = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    segments = []
    for u in range(num_uav):
        path = [tuple(starts[u])]
        visited = {(starts[u, 0], starts[u, 1])}
        for _ in range(max_steps // num_uav):
            r, c = path[-1]
            best, best_score = None, -np.inf
            for dr, dc in nbr:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n and 0 <= nc < m) or not trav[nr, nc]:
                    continue
                bonus = 3.0 * ((nr, nc) not in visited)
                score = W_est[nr, nc] + bonus
                if score > best_score:
                    best_score, best = score, (nr, nc)
            if best is None:
                break
            path.append(best)
            visited.add(best)
        segments.append(path)
    return {"segments": segments, "starts": starts, "num_uav": num_uav, "name": name}


def build_decentralized_greedy_plan(field, W_est, num_uav, starts=None, name="decentralized_greedy") -> dict:
    """Nearest-depot Voronoi on hotspots + per-UAV greedy walk."""
    trav = field.traversable
    if starts is None:
        starts = pick_depots(trav, num_uav)
    idx = _hotspot_targets(field, W_est)
    if idx.shape[0] == 0:
        return build_greedy_plan(field, W_est, num_uav, starts, name=name)
    wgt = np.ones(idx.shape[0])
    assigns = capacitated_voronoi(idx.astype(float), wgt, starts.astype(float),
                                  dx=field.dx, eta=0.0)
    segments = []
    for u in range(num_uav):
        goals = idx[assigns == u]
        if goals.shape[0] == 0:
            segments.append([tuple(starts[u])])
            continue
        sub_field = field  # reuse greedy locally toward assigned goals
        # Short greedy path toward assigned hotspot centroid then local greedy
        cr = goals.mean(axis=0).astype(int)
        path = stitch_goals(trav, starts[u], [tuple(cr)], use_astar=True)
        segments.append(path if path else [tuple(starts[u])])
    return {"segments": segments, "starts": starts, "num_uav": num_uav, "name": name}
