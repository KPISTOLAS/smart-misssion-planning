"""Mission execution engine coupling staleness to the IUEF-EM planner.

Causal model (how staleness degrades the mission)
-------------------------------------------------
* **Map fade -> HPC loss.**  The digital twin holds a belief of the priority
  field that is refreshed only at synchronization events.  At each sync the
  belief is reset to ground truth and then *planned under expected staleness*:
  the planner sees the truth degraded by a single application of the fade model
  at the mean Age-of-Information ``kappa(tau) = (tau-1)/(2 tau)``::

      belief = max(0, W_true * (1 - beta_M * kappa) + N(0, sigma_M^2 * kappa))

  Larger ``tau`` -> larger ``kappa`` -> faded hotspots drop below the planner's
  selection threshold -> fewer truly-high-priority cells are targeted -> the
  high-priority coverage (HPC) plateau falls.

* **Ghost drift -> collisions / near-misses.**  Between syncs each UAV's belief
  of its teammates' positions is *frozen* at the last synced position plus a
  drift term ``N(0, sigma_g^2 * age * log(1+tau))`` (the paper's ghost model).
  Reactive collision avoidance uses these stale *ghost* positions, so as the
  interval ages the avoidance targets the wrong location and true near-misses /
  collisions occur.

Synchronization policies
------------------------
* ``PERIODIC`` : sync every ``tau`` steps, with ``tau`` drawn per event from the
  NTN channel model.
* ``ADAPTIVE`` : event-triggered - sync when the estimated age exceeds a step
  threshold OR when the predicted HPC-relevant retention drop exceeds a
  fraction.  The trigger is tunable so it can be matched to the periodic policy
  at equal average uplink cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .channel import NTNChannel
from .staleness import StalenessModel, StalenessParams, age_of, kappa, interval_retention
from .planner import IUEFEMPlanner, PlannerOptions


class SyncPolicy(Enum):
    PERIODIC = "periodic"
    ADAPTIVE = "adaptive"


@dataclass
class MissionConfig:
    """Configuration for one mission rollout."""

    num_uav: int = 3
    horizon: int = 600               # total simulation steps
    dt: float = 1.0                  # seconds per step
    dx: float = 18.0                 # meters per cell (from field, overridden)
    d_safe_m: float = 25.0           # true collision threshold (m)
    d_near_m: float = 45.0           # near-miss threshold (m)
    d_avoid_cells: float = 2.2       # reactive-avoidance radius on GHOST positions
    policy: SyncPolicy = SyncPolicy.PERIODIC
    # Adaptive-policy triggers:
    adapt_age_steps: float = 30.0    # sync if steps_since_sync exceeds this
    adapt_retention_drop: float = 0.28  # sync if predicted retention drop > this
    adapt_tau_plan_factor: float = 0.65  # adaptive plans on tau_ref * factor (not age cap)
    # Staleness normalisation reference (mean interval for the regime).  Using a
    # single reference for both policies isolates the *timing* effect of the
    # policy from the channel statistics for a fair comparison.
    tau_ref: float | None = None
    planner_options: PlannerOptions = field(default_factory=PlannerOptions)


@dataclass
class MissionResult:
    hpc_pct: float
    collision_rate: float
    near_miss_rate: float
    uplink_cost: float               # syncs per step (average uplink rate)
    n_syncs: int
    mean_tau: float
    mean_age: float
    retained_high_frac: float        # mean fraction of hotspots kept in belief
    steps: int


def _plan_belief(field, staleness: StalenessModel, tau_plan: float):
    """Ground-truth priority field degraded over the *planned* interval.

    The plan made at a sync must serve the upcoming interval of length
    ``tau_plan`` (the sampled channel interval for PERIODIC, or the age cap for
    ADAPTIVE).  We apply the *cumulative* interval retention ``R(tau_plan)``
    (monotone-decreasing in the interval length) plus map process noise scaled
    by the mean AoI ``kappa(tau_plan)``.  Shorter planned intervals therefore
    retain more hotspots - this is how an adaptive policy that syncs early can
    plan on fresher data.
    """
    R = interval_retention(staleness.params.beta_M, tau_plan)
    k = float(kappa(max(tau_plan, 1.0)))
    p = staleness.params
    noise = staleness.rng.normal(0.0, p.sigma_M * np.sqrt(max(k, 0.0)), size=field.W.shape)
    belief = field.W * R + noise
    return np.maximum(0.0, belief)


def _predicted_retention(field, tau_ref: float, params: StalenessParams) -> float:
    """Predicted cumulative retention of hotspot weight over the interval."""
    return interval_retention(params.beta_M, max(tau_ref, 1.0))


def run_mission(field,
                config: MissionConfig | None = None,
                staleness_params: StalenessParams | None = None,
                channel: NTNChannel | None = None,
                rng=None) -> MissionResult:
    """Run one staleness-coupled EPCA-M mission and return summary metrics.

    Parameters
    ----------
    field:
        A :class:`~epca_staleness.environment.PriorityField` (ground truth).
    config:
        :class:`MissionConfig`.
    staleness_params:
        :class:`~epca_staleness.staleness.StalenessParams` degradation params.
    channel:
        :class:`~epca_staleness.channel.NTNChannel`; if ``None`` a medium link
        is used.
    rng:
        Seed / generator for reproducibility (drives staleness + channel if the
        latter is not supplied).
    """
    config = config or MissionConfig()
    rng = np.random.default_rng(rng)
    staleness_params = staleness_params or StalenessParams()
    if channel is None:
        channel = NTNChannel("medium", rng=rng)

    staleness = StalenessModel(staleness_params, rng=rng)
    planner = IUEFEMPlanner(config.planner_options)

    tau_ref = config.tau_ref if config.tau_ref is not None else channel.expected_tau(4000)
    tau_ref = max(1.0, float(tau_ref))

    trav = field.traversable
    N, M = trav.shape
    dx = field.dx
    U = config.num_uav

    starts = planner.pick_depots(trav, U)
    positions = starts.astype(float).copy()
    last_synced_positions = positions.copy()

    # Planned interval used to degrade the planning belief: the sampled channel
    # interval for PERIODIC, or the adaptive age cap for ADAPTIVE.
    if config.policy is SyncPolicy.PERIODIC:
        tau_plan_init = channel.sample_tau()
    else:
        tau_plan_init = tau_ref * config.adapt_tau_plan_factor

    # Initial plan on a fresh sync.
    belief = _plan_belief(field, staleness, tau_plan_init)
    plan = planner.build_plan(field, belief, U, starts=starts)
    paths = [list(seg) for seg in plan["segments"]]
    path_idx = [0] * U

    visited = np.zeros((N, M), dtype=bool)
    for u in range(U):
        r, c = int(round(positions[u, 0])), int(round(positions[u, 1]))
        if 0 <= r < N and 0 <= c < M:
            visited[r, c] = True

    steps_since_sync = 0
    # Current interval length that governs the next sync (PERIODIC) and the plan.
    tau = tau_plan_init if config.policy is SyncPolicy.PERIODIC else config.adapt_age_steps
    n_syncs = 0
    tau_samples = [tau]
    age_series = []
    collision_steps = 0
    near_miss_steps = 0
    retained_fracs = []

    high_mask = field.high_mask
    n_high = max(1, int(np.count_nonzero(high_mask)))

    d_safe_cells = config.d_safe_m / dx
    d_near_cells = config.d_near_m / dx

    # Peak-AoI cap: during a channel outage a *periodic* policy can run far past
    # tau_ref, so we let the normalized age grow beyond 1 (staleness keeps
    # worsening) up to a generous safety cap.  An adaptive policy avoids this by
    # forcing a sync at its age threshold.
    AGE_CAP = 4.0
    for t in range(config.horizon):
        age = min(AGE_CAP, steps_since_sync / max(tau_ref, 1.0))

        # ---- synchronization decision -----------------------------------
        do_sync = False
        if t > 0:
            if config.policy is SyncPolicy.PERIODIC:
                do_sync = steps_since_sync >= tau
            else:  # ADAPTIVE
                pred_drop = 1.0 - _predicted_retention(field, max(steps_since_sync, 1), staleness_params)
                do_sync = (steps_since_sync >= config.adapt_age_steps) or \
                          (pred_drop >= config.adapt_retention_drop)

        if do_sync:
            n_syncs += 1
            steps_since_sync = 0
            age = 0.0
            last_synced_positions = positions.copy()
            if config.policy is SyncPolicy.PERIODIC:
                tau = channel.sample_tau()          # upcoming interval length
                tau_samples.append(tau)
                tau_plan = tau
            else:
                tau_plan = tau_ref * config.adapt_tau_plan_factor
            belief = _plan_belief(field, staleness, tau_plan)
            plan = planner.build_plan(field, belief, U, starts=starts)
            paths = [list(seg) for seg in plan["segments"]]
            # Resume each path near the UAV's current cell (closest waypoint).
            path_idx = []
            for u in range(U):
                seg = np.array(paths[u], dtype=float)
                d = np.hypot(seg[:, 0] - positions[u, 0], seg[:, 1] - positions[u, 1])
                path_idx.append(int(np.argmin(d)))

        # Track belief retention over hotspots (reporting): fraction of true
        # hotspots whose faded belief still exceeds the targeting threshold.
        w_hi = field.meta.get("w_hi", field.meta.get("high_health_thr", 3)
                              * field.meta.get("alpha", 1.0) * 0.55)
        retained_fracs.append(float(np.mean(belief[high_mask] >= w_hi)))

        # ---- ghost (stale) teammate positions ---------------------------
        ghosts = staleness.ghost_positions(last_synced_positions, age, int(round(tau_ref)))

        # ---- move each UAV one step along its path w/ reactive avoidance --
        for u in range(U):
            seg = paths[u]
            if path_idx[u] >= len(seg) - 1:
                continue  # reached end of assigned path -> idle
            nxt = seg[path_idx[u] + 1]
            # Predicted min distance to any teammate using GHOST beliefs.
            hold = False
            for v in range(U):
                if v == u:
                    continue
                dgc = np.hypot(nxt[0] - ghosts[v, 0], nxt[1] - ghosts[v, 1])
                if dgc < config.d_avoid_cells:
                    hold = True
                    break
            if not hold:
                positions[u] = [nxt[0], nxt[1]]
                path_idx[u] += 1
                r, c = int(round(nxt[0])), int(round(nxt[1]))
                if 0 <= r < N and 0 <= c < M and trav[r, c]:
                    visited[r, c] = True

        # ---- true collision / near-miss check ---------------------------
        collided = False
        near = False
        for a in range(U):
            for b in range(a + 1, U):
                d = np.hypot(positions[a, 0] - positions[b, 0],
                             positions[a, 1] - positions[b, 1])
                if d < d_safe_cells:
                    collided = True
                if d < d_near_cells:
                    near = True
        collision_steps += int(collided)
        near_miss_steps += int(near)

        age_series.append(age)
        steps_since_sync += 1

    hpc = 100.0 * np.count_nonzero(visited & high_mask) / n_high
    steps = config.horizon
    return MissionResult(
        hpc_pct=float(hpc),
        collision_rate=float(collision_steps / steps),
        near_miss_rate=float(near_miss_steps / steps),
        uplink_cost=float(n_syncs / steps),
        n_syncs=int(n_syncs),
        mean_tau=float(np.mean(tau_samples)),
        mean_age=float(np.mean(age_series)) if age_series else 0.0,
        retained_high_frac=float(np.mean(retained_fracs)) if retained_fracs else 0.0,
        steps=steps,
    )
