"""IUEF-EM planner (Algorithm 1) with systematic ablation variants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np

from .planning_utils import (
    pick_depots, capacitated_voronoi, stitch_goals, horizon_shortcut, astar_grid,
    balanced_hotspot_assign,
)


class AblationMode(Enum):
    FULL = "iuef_em"
    NO_BALANCE = "ablation_no_balance"
    NO_CONGESTION = "ablation_no_congestion"
    NO_PRIORITY = "ablation_no_priority"
    NO_ASTAR = "ablation_no_astar"


@dataclass
class IUEFEMOptions:
    plan_mode: str = "blend"
    blend_gamma: float = 0.45
    max_targets: int = 500
    eta: float = 1.0
    lambda_cong: float = 0.35
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


def _weighted_order(start, goals: np.ndarray, gw: np.ndarray, use_priority: bool,
                    O: np.ndarray | None = None, lambda_cong: float = 0.35) -> np.ndarray:
    """Priority-biased nearest-neighbour tour through regional hotspots."""
    ng = goals.shape[0]
    if ng == 0:
        return goals
    order_idx = np.zeros(ng, dtype=int)
    unvisited = np.ones(ng, dtype=bool)
    cur = np.asarray(start, dtype=float)
    for k in range(ng):
        cand = np.where(unvisited)[0]
        dist = np.hypot(goals[cand, 0] - cur[0], goals[cand, 1] - cur[1])
        if use_priority:
            cong = 0.0
            if O is not None:
                cong = lambda_cong * O[goals[cand, 0].astype(int), goals[cand, 1].astype(int)]
            score = gw[cand] / np.maximum(dist, 0.5) - cong
        else:
            score = -dist
        pick = cand[int(np.argmax(score))]
        order_idx[k] = pick
        unvisited[pick] = False
        cur = goals[pick].astype(float)
    return goals[order_idx]


def _partition_hotspots(field, W_est, starts, num_uav, opts: IUEFEMOptions):
    """Partition hotspots with stable workload balancing (eta>0) or nearest-depot."""
    trav = field.traversable
    w_hi = field.meta.get("w_hi", 3.0)
    high_mask = (W_est >= w_hi) & trav
    if not high_mask.any():
        high_mask = field.high_mask & trav

    idx = np.argwhere(high_mask)
    if idx.shape[0] == 0:
        return idx, np.zeros(0, dtype=int)

    if idx.shape[0] > opts.max_targets:
        vals = W_est[idx[:, 0], idx[:, 1]]
        idx = idx[np.argsort(-vals)[: opts.max_targets]]

    wgt = _target_weights(W_est, field.sigma, field.H, idx, opts)
    eta = opts.eta if opts.use_balancing else 0.0
    assigns = balanced_hotspot_assign(
        idx.astype(float), wgt, starts.astype(float), eta=eta)
    return idx, assigns


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

    center = np.array([n // 2, m // 2])
    if not trav[center[0], center[1]]:
        center = np.argwhere(trav)[0]

    idx, assigns = _partition_hotspots(field, W_est, starts, num_uav, opts)
    if idx.shape[0] == 0:
        segs = [astar_grid(trav, tuple(starts[u]), tuple(center), O=O, Z=Z,
                           lambda_cong=lam_cong) or [tuple(starts[u])]
                for u in range(num_uav)]
        return {"segments": segs, "starts": starts, "num_uav": num_uav, "name": name}

    wgt = _target_weights(W_est, field.sigma, field.H, idx, opts)
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
        ordered = _weighted_order(starts[u], goals.astype(float), gw, opts.use_priority,
                                  O=O, lambda_cong=lam_cong)
        path = stitch_goals(trav, starts[u], ordered, O=O, Z=Z,
                            use_astar=opts.use_astar, lambda_cong=lam_cong)
        # Ensure every goal cell appears on the path (shortcut must not skip hotspots).
        if path:
            path_set = {tuple(p) for p in path}
            for g in ordered:
                gt = (int(g[0]), int(g[1]))
                if gt not in path_set:
                    seg = astar_grid(trav, path[-1], gt, O=O, Z=Z, lambda_cong=lam_cong)
                    if seg and len(seg) > 1:
                        path.extend(seg[1:])
                    else:
                        path.append(gt)
                    path_set.add(gt)
        segments.append(path)

    return {"segments": segments, "starts": starts, "num_uav": num_uav, "name": name}
