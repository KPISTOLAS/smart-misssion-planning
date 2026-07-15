"""Mission runner for sensitivity analysis (staleness + optional imperfect W)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from epca_staleness.channel import NTNChannel
from epca_staleness.environment import PriorityField
from epca_staleness.iuef_em import IUEFEMOptions, build_iuef_em_plan
from epca_staleness.planning_utils import pick_depots
from epca_staleness.staleness import StalenessModel, StalenessParams, kappa, interval_retention
from epca_staleness.mission import SyncPolicy
from epca_staleness.executor import ExecConfig, execute_plan

from .imperfect_priority import ImperfectPriorityConfig, corrupt_priority_field


@dataclass
class SensitivityMissionConfig:
    num_uav: int = 3
    horizon: int = 600
    d_safe_m: float = 25.0
    d_near_m: float = 45.0
    d_avoid_cells: float = 2.2
    policy: SyncPolicy = SyncPolicy.PERIODIC
    tau_ref: float | None = None
    base_tau: float = 45.0
    link: str = "medium"
    perfect_priority: bool = True
    imperfect: ImperfectPriorityConfig | None = None
    disable_staleness: bool = False


@dataclass
class SensitivityMissionResult:
    hpc_pct: float
    whpc_pct: float
    coverage_pct: float
    collision_rate: float
    near_miss_rate: float
    duration_steps: int
    total_energy_J: float
    energy_per_uav_hour: float
    mission_score: float
    mean_tau: float
    inference_mae: float
    targeting_error_pct: float   # visits to planner-high cells not in ground-truth hotspots
    n_syncs: int
    seed: int = 0


def _plan_belief(W: np.ndarray, staleness: StalenessModel, tau_plan: float) -> np.ndarray:
    R = interval_retention(staleness.params.beta_M, tau_plan)
    k = float(kappa(max(tau_plan, 1.0)))
    noise = staleness.rng.normal(0.0, staleness.params.sigma_M * np.sqrt(max(k, 0.0)), size=W.shape)
    return np.maximum(0.0, W * R + noise)


def run_sensitivity_mission(field: PriorityField,
                            mission_cfg: SensitivityMissionConfig,
                            staleness_params: StalenessParams,
                            channel: NTNChannel | None = None,
                            rng=None) -> SensitivityMissionResult:
    """Run one mission with synthetic map + optional imperfect priority + staleness."""
    rng = np.random.default_rng(rng)
    if channel is None:
        channel = NTNChannel(mission_cfg.link, rng=rng, base_tau_override=mission_cfg.base_tau)
    staleness = StalenessModel(staleness_params, rng=rng)
    imperfect_cfg = mission_cfg.imperfect or ImperfectPriorityConfig()

    trav = field.traversable
    n, m = field.N, field.M
    dx = field.dx
    U = mission_cfg.num_uav
    tau_ref = mission_cfg.tau_ref if mission_cfg.tau_ref is not None else channel.expected_tau(2000)
    tau_ref = max(1.0, float(tau_ref))

    starts = pick_depots(trav, U)
    positions = starts.astype(float).copy()
    last_synced = positions.copy()

    def _sync_w() -> tuple[np.ndarray, float]:
        if mission_cfg.perfect_priority:
            return field.W.copy(), 0.0
        W_hat, _, diag = corrupt_priority_field(field, imperfect_cfg, rng=int(rng.integers(0, 2**31)))
        return W_hat, float(diag.get("mae_hotspots", 0.0))

    W_sync, mae0 = _sync_w()
    inference_maes = [mae0]
    tau_plan = channel.sample_tau()
    W_plan = _plan_belief(W_sync, staleness, tau_plan) if not mission_cfg.disable_staleness else W_sync.copy()

    inferred = PriorityField(
        N=n, M=m, dx=dx, H=field.H, sigma=field.sigma, obstacle=field.obstacle,
        O=field.O, W=W_plan, high_mask=field.high_mask, Z_m=field.Z_m, meta=field.meta,
    )
    plan = build_iuef_em_plan(inferred, W_plan, U, starts, opts=IUEFEMOptions())
    paths = [list(seg) for seg in plan["segments"]]
    path_idx = [0] * U
    visited = np.zeros((n, m), dtype=bool)
    for u in range(U):
        r, c = int(round(positions[u, 0])), int(round(positions[u, 1]))
        if 0 <= r < n and 0 <= c < m:
            visited[r, c] = True

    steps_since_sync = 0
    tau = tau_plan
    tau_samples = [tau]
    n_syncs = 0
    coll_steps = near_steps = 0
    n_high = max(1, int(field.high_mask.sum()))
    d_safe_c = mission_cfg.d_safe_m / dx
    d_near_c = mission_cfg.d_near_m / dx
    AGE_CAP = 4.0

    for t in range(mission_cfg.horizon):
        age = min(AGE_CAP, steps_since_sync / max(tau_ref, 1.0))
        do_sync = t > 0 and steps_since_sync >= tau

        if do_sync:
            n_syncs += 1
            steps_since_sync = 0
            age = 0.0
            last_synced = positions.copy()
            tau = channel.sample_tau()
            tau_samples.append(tau)
            W_sync, mae = _sync_w()
            inference_maes.append(mae)
            W_plan = _plan_belief(W_sync, staleness, tau) if not mission_cfg.disable_staleness else W_sync.copy()
            inferred = PriorityField(
                N=n, M=m, dx=dx, H=field.H, sigma=field.sigma, obstacle=field.obstacle,
                O=field.O, W=W_plan, high_mask=field.high_mask, Z_m=field.Z_m, meta=field.meta,
            )
            plan = build_iuef_em_plan(inferred, W_plan, U, starts, opts=IUEFEMOptions())
            paths = [list(seg) for seg in plan["segments"]]
            path_idx = []
            for u in range(U):
                seg = np.array(paths[u], dtype=float)
                d = np.hypot(seg[:, 0] - positions[u, 0], seg[:, 1] - positions[u, 1])
                path_idx.append(int(np.argmin(d)))

        ghosts = (last_synced if mission_cfg.disable_staleness
                  else staleness.ghost_positions(last_synced, age, int(round(tau_ref))))

        for u in range(U):
            seg = paths[u]
            if path_idx[u] >= len(seg) - 1:
                continue
            nxt = seg[path_idx[u] + 1]
            hold = any(
                np.hypot(nxt[0] - ghosts[v, 0], nxt[1] - ghosts[v, 1]) < mission_cfg.d_avoid_cells
                for v in range(U) if v != u
            )
            if not hold:
                positions[u] = [nxt[0], nxt[1]]
                path_idx[u] += 1
                r, c = int(round(nxt[0])), int(round(nxt[1]))
                if 0 <= r < n and 0 <= c < m and trav[r, c]:
                    visited[r, c] = True

        coll = any(
            np.hypot(positions[a, 0] - positions[b, 0], positions[a, 1] - positions[b, 1]) < d_safe_c
            for a in range(U) for b in range(a + 1, U)
        )
        near = any(
            np.hypot(positions[a, 0] - positions[b, 0], positions[a, 1] - positions[b, 1]) < d_near_c
            for a in range(U) for b in range(a + 1, U)
        )
        coll_steps += int(coll)
        near_steps += int(near)
        steps_since_sync += 1

    exec_cfg = ExecConfig(horizon=mission_cfg.horizon, dx=dx, d_safe_m=mission_cfg.d_safe_m,
                          d_near_m=mission_cfg.d_near_m, d_avoid_cells=0.0)
    em = execute_plan(field, plan, exec_cfg)
    hpc = 100.0 * np.count_nonzero(visited & field.high_mask) / n_high
    whpc = 100.0 * float(field.W[visited & field.high_mask].sum()) / max(field.W[field.high_mask].sum(), 1e-9)
    cov = 100.0 * np.count_nonzero(visited & trav) / max(1, int(trav.sum()))
    # Targeting error: fraction of visited cells that are planner-high (W_plan at sync) but not true hotspots.
    planner_high = W_plan >= field.meta.get("w_hi", np.quantile(field.W[trav], 0.88))
    mis_visits = visited & planner_high & ~field.high_mask
    all_visits = visited & trav
    targ_err = 100.0 * np.count_nonzero(mis_visits) / max(1, np.count_nonzero(all_visits))

    return SensitivityMissionResult(
        hpc_pct=float(hpc),
        whpc_pct=float(whpc),
        coverage_pct=float(cov),
        collision_rate=float(coll_steps / mission_cfg.horizon),
        near_miss_rate=float(near_steps / mission_cfg.horizon),
        duration_steps=mission_cfg.horizon,
        total_energy_J=em.total_energy_J,
        energy_per_uav_hour=em.energy_per_uav_hour,
        mission_score=em.mission_score,
        mean_tau=float(np.mean(tau_samples)),
        inference_mae=float(np.mean(inference_maes)),
        targeting_error_pct=float(targ_err),
        n_syncs=n_syncs,
        seed=int(rng.integers(0, 2**31)) if hasattr(rng, "integers") else 0,
    )
