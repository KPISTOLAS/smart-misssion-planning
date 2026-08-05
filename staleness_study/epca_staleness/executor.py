"""Mission executor and KPI computation for fair planner comparison.

All planners are evaluated under identical constraints:
  * same priority field (ground-truth W or stale W_est),
  * same depots, fleet size U, safety margin d_safe,
  * same time step Delta t = 1.0 s and grid spacing dx.

Reported metrics
----------------
* HPC (%) — high-priority cell coverage
* total coverage (%) — all traversable cells visited
* mission duration (steps)
* total energy (J, normalised per UAV-hour)
* collision rate — fraction of steps with pairwise distance < d_safe
* near-miss rate — fraction of steps with distance < d_near
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .planning_utils import pick_depots


@dataclass
class ExecConfig:
    horizon: int = 1200
    dt: float = 1.0
    d_safe_m: float = 25.0
    d_near_m: float = 45.0
    d_avoid_cells: float = 0.0   # measure planner-induced congestion (no reactive hold)
    # Simplified multicopter power model (MissionSim defaults)
    P0: float = 180.0
    k_speed: float = 0.35
    dx: float = 18.0


@dataclass
class MissionMetrics:
    planner: str
    hpc_pct: float
    whpc_pct: float              # priority-weighted HPC
    hpc_early_pct: float         # HPC at 50% horizon (priority ordering benefit)
    coverage_pct: float
    duration_steps: int
    total_energy_J: float
    energy_per_uav_hour: float
    collision_rate: float
    near_miss_rate: float
    mission_score: float         # composite: WHPC * (1-coll) / energy_norm
    n_uav: int
    seed: int = 0


def _pairwise_violation(positions, u, U, dx, thresh_m):
    pa = positions[u]
    for v in range(U):
        if v == u:
            continue
        d = np.hypot((pa[0] - positions[v][0]) * dx, (pa[1] - positions[v][1]) * dx)
        if d < thresh_m:
            return True
    return False


def execute_plan(field, plan: dict, cfg: ExecConfig | None = None) -> MissionMetrics:
    """Simulate parallel UAV execution along precomputed paths."""
    cfg = cfg or ExecConfig(dx=field.dx)
    trav = field.traversable
    n, m = trav.shape
    U = plan["num_uav"]
    segments = plan["segments"]
    high_mask = field.high_mask
    W = field.W
    n_high = max(1, int(high_mask.sum()))
    w_total = max(float(W[high_mask].sum()), 1e-9)
    n_trav = max(1, int(trav.sum()))

    d_safe_c = cfg.d_safe_m / cfg.dx
    d_near_c = cfg.d_near_m / cfg.dx

    positions = np.array([[seg[0][0], seg[0][1]] for seg in segments], dtype=float)
    path_idx = [0] * U
    visited = np.zeros((n, m), dtype=bool)
    for u in range(U):
        r, c = int(round(positions[u, 0])), int(round(positions[u, 1]))
        if 0 <= r < n and 0 <= c < m:
            visited[r, c] = True

    energy = 0.0
    coll_steps = near_steps = 0
    prev_pos = positions.copy()
    hpc_early = 0.0
    half_t = cfg.horizon // 2
    total_path = max(sum(max(len(s) - 1, 1) for s in segments), 1)

    for t in range(cfg.horizon):
        for u in range(U):
            seg = segments[u]
            if path_idx[u] >= len(seg) - 1:
                continue
            nxt = seg[path_idx[u] + 1]
            hold = False
            for v in range(U):
                if v == u:
                    continue
                dgc = np.hypot(nxt[0] - positions[v][0], nxt[1] - positions[v][1])
                if dgc < cfg.d_avoid_cells:
                    hold = True
                    break
            if not hold:
                step_m = np.hypot(nxt[0] - positions[u][0], nxt[1] - positions[u][1]) * cfg.dx
                energy += cfg.P0 * cfg.dt * 0.05 + cfg.k_speed * (step_m / max(cfg.dt, 1e-9)) ** 3 * cfg.dt
                positions[u] = [nxt[0], nxt[1]]
                path_idx[u] += 1
                r, c = int(round(nxt[0])), int(round(nxt[1]))
                if 0 <= r < n and 0 <= c < m and trav[r, c]:
                    visited[r, c] = True

        coll = any(_pairwise_violation(positions, u, U, cfg.dx, cfg.d_safe_m) for u in range(U))
        near = any(_pairwise_violation(positions, u, U, cfg.dx, cfg.d_near_m) for u in range(U))
        coll_steps += int(coll)
        near_steps += int(near)

        if t == half_t:
            hpc_early = 100.0 * np.count_nonzero(visited & high_mask) / n_high

        if all(path_idx[u] >= len(segments[u]) - 1 for u in range(U)):
            steps = t + 1
            break
    else:
        steps = cfg.horizon
    hpc = 100.0 * np.count_nonzero(visited & high_mask) / n_high
    whpc = 100.0 * float(W[visited & high_mask].sum()) / w_total
    cov = 100.0 * np.count_nonzero(visited & trav) / n_trav
    eph = energy / max(steps * cfg.dt / 3600.0 * U, 1e-9)
    coll_r = float(coll_steps / steps)
    mission_score = float(whpc * (1.0 - coll_r) / max(eph / 10000.0, 1e-9))

    return MissionMetrics(
        planner=plan.get("name", "unknown"),
        hpc_pct=float(hpc),
        whpc_pct=float(whpc),
        hpc_early_pct=float(hpc_early),
        coverage_pct=float(cov),
        duration_steps=int(steps),
        total_energy_J=float(energy),
        energy_per_uav_hour=float(eph),
        collision_rate=coll_r,
        near_miss_rate=float(near_steps / steps),
        mission_score=mission_score,
        n_uav=U,
    )


def evaluate_planner(planner_name: str, field, num_uav: int,
                     W_est=None, cfg: ExecConfig | None = None,
                     seed: int = 0) -> MissionMetrics:
    """Build plan + execute under fair shared inputs."""
    from .registry import build_plan
    from .planning_utils import pick_depots as _pd

    if W_est is None:
        W_est = field.W
    starts = pick_depots(field.traversable, num_uav)
    plan = build_plan(planner_name, field, W_est, num_uav, starts=starts)
    m = execute_plan(field, plan, cfg)
    m.planner = planner_name
    m.seed = seed
    return m
