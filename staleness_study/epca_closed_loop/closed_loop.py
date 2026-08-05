"""Closed-loop EPCA-M mission simulator (sensing -> inference -> planning).

Each synchronization event:
  1. Capture UAV RGB + IoT windows (synthetic or loaded).
  2. Run Tier-2 inference (EPCA-Det-s + MLP forecaster).
  3. Fuse priority field W_i (Eq. 3).
  4. Degrade belief under staleness; replan with IUEF-EM.
Between syncs the digital twin fades and ghost positions drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from epca_staleness.channel import NTNChannel
from epca_staleness.environment import PriorityField, build_priority_field
from epca_staleness.executor import ExecConfig, execute_plan
from epca_staleness.iuef_em import IUEFEMOptions, build_iuef_em_plan
from epca_staleness.planning_utils import pick_depots
from epca_staleness.staleness import (
    StalenessModel, StalenessParams, retention,
)
from epca_staleness.experiments import default_params

from .data_synth import sample_iot_windows, sample_uav_images
from .detector import DetectorConfig, EPCADetector
from .forecaster import ForecasterConfig, MLPForecaster
from .priority_field import PriorityFusionConfig, fuse_priority_field


class SyncPolicy(Enum):
    PERIODIC = "periodic"
    ADAPTIVE = "adaptive"


@dataclass
class ClosedLoopConfig:
    num_uav: int = 3
    horizon: int = 600
    dt: float = 1.0
    d_safe_m: float = 25.0
    d_near_m: float = 45.0
    d_avoid_cells: float = 2.2
    policy: SyncPolicy = SyncPolicy.PERIODIC
    adapt_age_steps: float = 30.0
    adapt_retention_drop: float = 0.28
    adapt_tau_plan_factor: float = 0.65
    tau_ref: float | None = None
    # Modes
    perfect_info: bool = False       # planner sees ground-truth W (upper bound)
    disable_staleness: bool = False  # no map fade / ghost drift
    # Grid
    grid_n: int = 50
    grid_m: int = 50
  # Fusion
    fusion: PriorityFusionConfig = field(default_factory=PriorityFusionConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    forecaster: ForecasterConfig = field(default_factory=ForecasterConfig)
    forecaster_mode: str = "mlp"  # mlp | persistence | ar1
    views_per_uav: int = 6
    iot_stations: int = 16


@dataclass
class ClosedLoopResult:
    hpc_pct: float
    whpc_pct: float
    coverage_pct: float
    collision_rate: float
    near_miss_rate: float
    mission_duration: int
    total_energy_J: float
    energy_per_uav_hour: float
    mission_score: float
    uplink_cost: float
    n_syncs: int
    mean_tau: float
    mean_age: float
    retained_high_frac: float
    inference_mae: float          # mean |W_hat - W_true| on hotspots
    mode: str
    seed: int = 0
    # Optional snapshots for plotting
    W_snapshots: list | None = None
    traj_snapshots: list | None = None


def _plan_belief(W_sync: np.ndarray, staleness: StalenessModel, tau_plan: float) -> np.ndarray:
    return staleness.degraded_map(W_sync, tau_plan)


def _run_tier2_inference(field_gt: PriorityField,
                         positions: np.ndarray,
                         cfg: ClosedLoopConfig,
                         detector: EPCADetector,
                         forecaster: MLPForecaster,
                         rng) -> tuple[np.ndarray, np.ndarray, float]:
    """Tier-2 sensing stack -> fused W and hotspot mask."""
    from .forecaster_baselines import predict_persistence, predict_ar1

    uav_batch = sample_uav_images(field_gt, positions, views_per_uav=cfg.views_per_uav, rng=rng)
    iot_seed = int(rng.integers(0, 1_000_000)) if hasattr(rng, "integers") else int(rng) + 7
    iot_batch = sample_iot_windows(field_gt, window_len=cfg.forecaster.window_len,
                                   n_stations=cfg.iot_stations, rng=iot_seed)
    dets = detector.predict(uav_batch, (field_gt.N, field_gt.M))
    mode = (cfg.forecaster_mode or "mlp").lower()
    if mode == "persistence":
        risks = predict_persistence(iot_batch)
    elif mode == "ar1":
        risks = predict_ar1(iot_batch)
    else:
        risks = forecaster.predict(iot_batch)
    W_inf, high_mask, meta = fuse_priority_field(
        field_gt, dets, iot_batch.station_coords, risks, cfg.fusion,
    )
    # Inference error on true hotspots (diagnostic).
    hm = field_gt.high_mask
    mae = float(np.mean(np.abs(W_inf[hm] - field_gt.W[hm]))) if hm.any() else 0.0
    return W_inf, high_mask, mae


def run_closed_loop(field_gt: PriorityField | None = None,
                    config: ClosedLoopConfig | None = None,
                    staleness_params: StalenessParams | None = None,
                    channel: NTNChannel | None = None,
                    rng=None,
                    record_snapshots: bool = False) -> ClosedLoopResult:
    """Execute one closed-loop mission rollout."""
    config = config or ClosedLoopConfig()
    rng = np.random.default_rng(rng)
    if field_gt is None:
        field_gt = build_priority_field(config.grid_n, config.grid_m, seed=int(rng.integers(0, 1_000_000)))
    staleness_params = staleness_params or default_params(rng=rng)
    if channel is None:
        channel = NTNChannel("medium", rng=rng)

    detector = EPCADetector(config.detector)
    forecaster = MLPForecaster(config.forecaster)
    staleness = StalenessModel(staleness_params, rng=rng)

    trav = field_gt.traversable
    n, m = field_gt.N, field_gt.M
    dx = field_gt.dx
    U = config.num_uav
    tau_ref = config.tau_ref if config.tau_ref is not None else channel.expected_tau(3000)
    tau_ref = max(1.0, float(tau_ref))

    starts = pick_depots(trav, U)
    positions = starts.astype(float).copy()
    last_synced_positions = positions.copy()

    W_snapshots, traj_snapshots = [], []
    inference_maes = []

    # --- initial sync: Tier-2 inference or perfect info ---
    if config.perfect_info:
        W_belief = field_gt.W.copy()
        high_mask = field_gt.high_mask.copy()
        inference_maes.append(0.0)
    else:
        W_belief, high_mask, mae = _run_tier2_inference(
            field_gt, positions, config, detector, forecaster, int(rng.integers(0, 1_000_000)),
        )
        inference_maes.append(mae)

    if config.policy is SyncPolicy.PERIODIC:
        tau_plan_init = channel.sample_tau()
    else:
        tau_plan_init = tau_ref * config.adapt_tau_plan_factor

    if not config.disable_staleness:
        W_plan = _plan_belief(W_belief, staleness, tau_plan_init)
    else:
        W_plan = W_belief.copy()

    inferred_field = PriorityField(
        N=n, M=m, dx=dx, H=field_gt.H, sigma=field_gt.sigma,
        obstacle=field_gt.obstacle, O=field_gt.O, W=W_plan,
        high_mask=high_mask, Z_m=field_gt.Z_m, meta=dict(field_gt.meta, w_hi=field_gt.meta.get("w_hi", 3.0)),
    )
    plan = build_iuef_em_plan(inferred_field, W_plan, U, starts, opts=IUEFEMOptions())
    paths = [list(seg) for seg in plan["segments"]]
    path_idx = [0] * U

    visited = np.zeros((n, m), dtype=bool)
    for u in range(U):
        r, c = int(round(positions[u, 0])), int(round(positions[u, 1]))
        if 0 <= r < n and 0 <= c < m:
            visited[r, c] = True

    steps_since_sync = 0
    tau = tau_plan_init if config.policy is SyncPolicy.PERIODIC else config.adapt_age_steps
    n_syncs = 0
    tau_samples = [tau]
    age_series, retained_fracs = [], []
    collision_steps = near_steps = 0
    n_high = max(1, int(field_gt.high_mask.sum()))
    w_hi = field_gt.meta.get("w_hi", 3.0)
    d_safe_c = config.d_safe_m / dx
    d_near_c = config.d_near_m / dx
    if record_snapshots:
        W_snapshots.append(W_plan.copy())
        traj_snapshots.append(positions.copy())

    W_synced = W_belief.copy()

    for t in range(config.horizon):
        age = steps_since_sync

        # --- sync decision ---
        do_sync = False
        if t > 0:
            if config.policy is SyncPolicy.PERIODIC:
                do_sync = steps_since_sync >= tau
            else:
                pred_drop = 1.0 - retention(max(steps_since_sync, 1), staleness_params.beta_M)
                do_sync = (steps_since_sync >= config.adapt_age_steps) or \
                          (pred_drop >= config.adapt_retention_drop)

        if do_sync:
            n_syncs += 1
            steps_since_sync = 0
            age = 0.0
            last_synced_positions = positions.copy()
            if config.policy is SyncPolicy.PERIODIC:
                tau = channel.sample_tau()
                tau_samples.append(tau)
                tau_plan = tau
            else:
                tau_plan = tau_ref * config.adapt_tau_plan_factor

            if config.perfect_info:
                W_belief = field_gt.W.copy()
                high_mask = field_gt.high_mask.copy()
                inference_maes.append(0.0)
            else:
                W_belief, high_mask, mae = _run_tier2_inference(
                    field_gt, positions, config, detector, forecaster, int(rng.integers(0, 1_000_000)),
                )
                inference_maes.append(mae)

            W_synced = W_belief.copy()
            if not config.disable_staleness:
                W_plan = _plan_belief(W_belief, staleness, tau_plan)
            else:
                W_plan = W_belief.copy()

            inferred_field = PriorityField(
                N=n, M=m, dx=dx, H=field_gt.H, sigma=field_gt.sigma,
                obstacle=field_gt.obstacle, O=field_gt.O, W=W_plan,
                high_mask=high_mask, Z_m=field_gt.Z_m,
                meta={**field_gt.meta, "w_hi": w_hi},
            )
            plan = build_iuef_em_plan(inferred_field, W_plan, U, starts, opts=IUEFEMOptions())
            paths = [list(seg) for seg in plan["segments"]]
            path_idx = []
            for u in range(U):
                seg = np.array(paths[u], dtype=float)
                d = np.hypot(seg[:, 0] - positions[u, 0], seg[:, 1] - positions[u, 1])
                path_idx.append(int(np.argmin(d)))

            if record_snapshots:
                W_snapshots.append(W_plan.copy())
                traj_snapshots.append(positions.copy())

        if not config.disable_staleness:
            W_exec = staleness.degraded_map(W_synced, age)
        else:
            W_exec = W_plan

        retained_fracs.append(float(np.mean(W_exec[field_gt.high_mask] >= w_hi)) if field_gt.high_mask.any() else 0.0)

        ghosts = (last_synced_positions if config.disable_staleness
                  else staleness.ghost_positions(last_synced_positions, age))

        for u in range(U):
            seg = paths[u]
            if path_idx[u] >= len(seg) - 1:
                continue
            nxt = seg[path_idx[u] + 1]
            hold = False
            for v in range(U):
                if v == u:
                    continue
                if np.hypot(nxt[0] - ghosts[v, 0], nxt[1] - ghosts[v, 1]) < config.d_avoid_cells:
                    hold = True
                    break
            if not hold:
                positions[u] = [nxt[0], nxt[1]]
                path_idx[u] += 1
                r, c = int(round(nxt[0])), int(round(nxt[1]))
                if 0 <= r < n and 0 <= c < m and trav[r, c]:
                    visited[r, c] = True

        coll = near = False
        for a in range(U):
            for b in range(a + 1, U):
                d = np.hypot(positions[a, 0] - positions[b, 0], positions[a, 1] - positions[b, 1])
                if d < d_safe_c:
                    coll = True
                if d < d_near_c:
                    near = True
        collision_steps += int(coll)
        near_steps += int(near)
        age_series.append(age)
        steps_since_sync += 1

    # Energy via executor replay on final plan segment (fair normalisation).
    exec_cfg = ExecConfig(horizon=config.horizon, dt=config.dt, dx=dx,
                          d_safe_m=config.d_safe_m, d_near_m=config.d_near_m,
                          d_avoid_cells=0.0)
    # Rebuild a static plan for energy estimate from initial segments.
    energy_field = PriorityField(
        N=n, M=m, dx=dx, H=field_gt.H, sigma=field_gt.sigma,
        obstacle=field_gt.obstacle, O=field_gt.O, W=field_gt.W,
        high_mask=field_gt.high_mask, Z_m=field_gt.Z_m, meta=field_gt.meta,
    )
    em = execute_plan(energy_field, plan, exec_cfg)

    hpc = 100.0 * np.count_nonzero(visited & field_gt.high_mask) / n_high
    whpc = 100.0 * float(field_gt.W[visited & field_gt.high_mask].sum()) / max(field_gt.W[field_gt.high_mask].sum(), 1e-9)
    cov = 100.0 * np.count_nonzero(visited & trav) / max(1, int(trav.sum()))

    mode = "perfect_info" if config.perfect_info else ("no_staleness" if config.disable_staleness else "closed_loop")

    return ClosedLoopResult(
        hpc_pct=float(hpc),
        whpc_pct=float(whpc),
        coverage_pct=float(cov),
        collision_rate=float(collision_steps / config.horizon),
        near_miss_rate=float(near_steps / config.horizon),
        mission_duration=config.horizon,
        total_energy_J=em.total_energy_J,
        energy_per_uav_hour=em.energy_per_uav_hour,
        mission_score=em.mission_score,
        uplink_cost=float(n_syncs / config.horizon),
        n_syncs=n_syncs,
        mean_tau=float(np.mean(tau_samples)),
        mean_age=float(np.mean(age_series)) if age_series else 0.0,
        retained_high_frac=float(np.mean(retained_fracs)) if retained_fracs else 0.0,
        inference_mae=float(np.mean(inference_maes)) if inference_maes else 0.0,
        mode=mode,
        seed=int(rng.integers(0, 1_000_000)) if hasattr(rng, "integers") else 0,
        W_snapshots=W_snapshots if record_snapshots else None,
        traj_snapshots=traj_snapshots if record_snapshots else None,
    )
