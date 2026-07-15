"""IUEF-EM planner (Algorithm 1) with systematic ablation variants.

Full pipeline
-------------
1. Priority pruning  (targets with W_est >= w_hi)
2. Capacitated Voronoi partitioning (eta > 0) or nearest-depot (eta = 0)
3. Weighted greedy insertion for goal ordering (or pure distance if no priority)
4. A* stitching with elevation / congestion costs (or Manhattan if no A*)
5. Horizon shortcut refinement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .planning_utils import (
    pick_depots, capacitated_voronoi, stitch_goals, horizon_shortcut, astar_grid,
)


class AblationMode(Enum):
    FULL = "iuef_em"
    NO_BALANCE = "ablation_no_balance"       # eta = 0
    NO_CONGESTION = "ablation_no_congestion" # lambda_cong = 0
    NO_PRIORITY = "ablation_no_priority"     # pure distance ordering
    NO_ASTAR = "ablation_no_astar"           # Manhattan stitch only


@dataclass
class IUEFEMOptions:
    plan_mode: str = "blend"
    blend_gamma: float = 0.45
    max_targets: int = 260
    eta: float = 1.0                 # workload-balancing weight (0 = off)
    lambda_cong: float = 0.35        # congestion penalty on O_i
    lambda_slope: float = 0.08
    use_priority: bool = True
    use_astar: bool = True
    use_balancing: bool = True
    use_congestion: bool = True
    horizon_cells: int = 12
    voronoi_iterations: int = 14

    @classmethod
    def from_ablation(cls, mode: AblationMode) -> "IUEFEMOptions":
        opts = cls()
        if mode is AblationMode.NO_BALANCE:
            opts.use_balancing = False
            opts.eta = 0.0
        elif mode is AblationMode.NO_CONGESTION:
            opts.use_congestion = False
            opts.lambda_cong = 0.0
        elif mode is AblationMode.NO_PRIORITY:
            opts.use_priority = False
        elif mode is AblationMode.NO_ASTAR:
            opts.use_astar = False
        return opts


def _eig_weights(sigma, H, idx):
    v = sigma[idx[:, 0], idx[:, 1]] ** 2
    We = 0.5 * np.log2(1.0 + v / (0.12 ** 2))
    return We * (H[idx[:, 0], idx[:, 1]] >= 3)


def _target_weights(W_est, sigma, H, idx, opts: IUEFEMOptions) -> np.ndarray:
    Wh = np.maximum(0.0, W_est[idx[:, 0], idx[:, 1]])
    if not opts.use_priority:
        return np.ones(idx.shape[0])
    We = _eig_weights(sigma, H, idx)
    if opts.plan_mode == "heuristic":
        return Wh
    if opts.plan_mode == "eig":
        return We if We.any() else Wh
    nh = Wh / max(Wh.max(), 1e-9)
    ne = We / max(We.max(), 1e-9)
    w = opts.blend_gamma * nh + (1 - opts.blend_gamma) * ne
    return w if w.any() else nh


def _weighted_order(start, goals, gw, use_priority: bool):
    ng = goals.shape[0]
    order = np.zeros(ng, dtype=int)
    unvisited = np.ones(ng, dtype=bool)
    cur = np.asarray(start, dtype=float)
    for k in range(ng):
        cand = np.where(unvisited)[0]
        dist = np.hypot(goals[cand, 0] - cur[0], goals[cand, 1] - cur[1])
        if use_priority:
            score = gw[cand] / np.maximum(dist, 1e-9)
        else:
            score = -dist  # pure nearest-neighbour
        pick = cand[np.argmax(score)]
        order[k] = pick
        unvisited[pick] = False
        cur = goals[pick].astype(float)
    return order


def build_iuef_em_plan(field, W_est: np.ndarray, num_uav: int,
                       starts: np.ndarray | None = None,
                       opts: IUEFEMOptions | None = None,
                       name: str = "iuef_em") -> dict:
    """Build IUEF-EM plan (Algorithm 1) with optional ablation flags."""
    opts = opts or IUEFEMOptions()
    trav = field.traversable
    n, m = trav.shape
    O = field.O if opts.use_congestion else None
    Z = getattr(field, "Z_m", None)
    lam_cong = opts.lambda_cong if opts.use_congestion else 0.0

    if starts is None:
        starts = pick_depots(trav, num_uav)

    w_hi = field.meta.get("w_hi", 3.0)
    cand_mask = (W_est >= w_hi) & trav
    center = np.array([n // 2, m // 2])
    if not trav[center[0], center[1]]:
        center = np.argwhere(trav)[0]

    idx = np.argwhere(cand_mask)
    if idx.shape[0] == 0:
        segs = [astar_grid(trav, tuple(starts[u]), tuple(center), O=O, Z=Z,
                           lambda_cong=lam_cong) or [tuple(starts[u])]
                for u in range(num_uav)]
        return {"segments": segs, "starts": starts, "num_uav": num_uav, "name": name}

    if idx.shape[0] > opts.max_targets:
        vals = W_est[idx[:, 0], idx[:, 1]]
        idx = idx[np.argsort(-vals)[:opts.max_targets]]

    wgt = _target_weights(W_est, field.sigma, field.H, idx, opts)
    eta = opts.eta if opts.use_balancing else 0.0
    assigns = capacitated_voronoi(idx.astype(float), wgt, starts.astype(float),
                                  dx=field.dx, eta=eta, num_iter=opts.voronoi_iterations)

    segments = []
    for u in range(num_uav):
        sel = assigns == u
        goals = idx[sel]
        gw = wgt[sel]
        if goals.shape[0] == 0:
            seg = astar_grid(trav, tuple(starts[u]), tuple(center), O=O, Z=Z,
                             lambda_cong=lam_cong)
            segments.append(seg if seg else [tuple(starts[u])])
            continue
        order = _weighted_order(starts[u], goals.astype(float), gw, opts.use_priority)
        ordered = goals[order]
        path = stitch_goals(trav, starts[u], ordered, O=O, Z=Z,
                            use_astar=opts.use_astar, lambda_cong=lam_cong)
        if opts.use_astar and opts.horizon_cells > 0:
            path = horizon_shortcut(trav, path, Z=Z, O=O,
                                    horizon=opts.horizon_cells, lambda_cong=lam_cong)
        segments.append(path)

    return {"segments": segments, "starts": starts, "num_uav": num_uav, "name": name}
