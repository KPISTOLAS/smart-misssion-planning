"""Methodology validation: CIs, Wilcoxon tests, and comparison tables."""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from .environment import build_priority_field
from .channel import NTNChannel, LINK_PRESETS
from .mission import MissionConfig, SyncPolicy, run_mission
from .experiments import default_params, run_single_trial
from .publication import ablation_under_staleness
from .planner_evaluation import ablation_study, monte_carlo, EvalScenario
from .registry import ABLATION_PLANNERS
from .stats_export import aggregate_metric, compare_variants, StatRow

SEED_SYNC_POLICY = 5000
SEED_R1_ABLATION = 1000
SEED_R2_ABLATION = 11000
SEED_DX_SENSITIVITY = 20000


def _add_aoi_threshold_policy():
    """Register AOI_THRESHOLD on SyncPolicy if not present."""
    if not hasattr(SyncPolicy, "AOI_THRESHOLD"):
        SyncPolicy.AOI_THRESHOLD = "aoi_threshold"  # type: ignore[attr-defined]


def _run_mission_policy(
    seed: int,
    policy: SyncPolicy,
    link: str = "medium",
    base_tau: float = 45.0,
    aoi_kappa_thr: float = 25.0,
):
    params = default_params()
    field = build_priority_field(50, 50, seed=seed)
    channel = NTNChannel(link, rng=seed + 17, base_tau_override=base_tau)
    tau_ref = channel.expected_tau(2000)
    cfg = MissionConfig(
        num_uav=3, horizon=600, policy=policy, tau_ref=tau_ref,
        adapt_age_steps=30.0, adapt_retention_drop=1.1,
    )
    if policy is SyncPolicy.ADAPTIVE:
        pass  # uses adapt_age_steps from compare_policies tuning below
    return run_mission(field, cfg, staleness_params=params, channel=channel, rng=seed + 31)


def run_sync_policy_comparison(
    n_mc: int = 150,
    link: str = "medium",
    base_tau: float = 45.0,
    seed_base: int = SEED_SYNC_POLICY,
) -> dict:
    """R2-S3: periodic vs adaptive vs AoI-threshold with CI + Wilcoxon."""
    from .experiments import compare_policies

    # Extend compare_policies pattern for 3-way with paired seeds.
    params = default_params()
    periodic_hpc, periodic_coll = [], []
    adaptive_hpc, adaptive_coll = [], []
    aoi_hpc, aoi_coll = [], []

    # Calibrate adaptive age cap via existing cost-matching search.
    full_cmp = compare_policies(link, n_mc=min(20, n_mc), params=params, seed_base=seed_base)
    age_cap = float(full_cmp["adaptive"].get("age_cap", 30.0))

    aoi_sync_steps = int(round(2.0 * 25.0 + 1))  # κ threshold = 25 → age ≈ 51

    for j in range(n_mc):
        seed = seed_base + j
        r_per = run_single_trial(seed, link, base_tau, params, policy=SyncPolicy.PERIODIC)
        periodic_hpc.append(r_per.hpc_pct)
        periodic_coll.append(r_per.collision_rate)

        field = build_priority_field(50, 50, seed=seed)
        channel = NTNChannel(link, rng=seed + 17, base_tau_override=base_tau)
        tau_ref = channel.expected_tau(2000)
        cfg_ad = MissionConfig(
            num_uav=3, horizon=600, policy=SyncPolicy.ADAPTIVE, tau_ref=tau_ref,
            adapt_age_steps=age_cap, adapt_retention_drop=1.1,
        )
        r_ad = run_mission(field, cfg_ad, staleness_params=params, channel=channel, rng=seed + 31)
        adaptive_hpc.append(r_ad.hpc_pct)
        adaptive_coll.append(r_ad.collision_rate)

        # AoI-threshold: sync when κ(age) >= 25
        field2 = build_priority_field(50, 50, seed=seed)
        channel2 = NTNChannel(link, rng=seed + 19, base_tau_override=base_tau)
        r_aoi = _run_aoi_threshold_mission(field2, channel2, params, seed + 33, kappa_thr=25.0)
        aoi_hpc.append(r_aoi.hpc_pct)
        aoi_coll.append(r_aoi.collision_rate)

    rows = {
        "periodic": aggregate_metric("periodic", periodic_hpc),
        "adaptive": aggregate_metric("adaptive", adaptive_hpc),
        "aoi_threshold": aggregate_metric("aoi_threshold", aoi_hpc),
    }
    coll_rows = {
        "periodic": aggregate_metric("periodic_coll", periodic_coll),
        "adaptive": aggregate_metric("adaptive_coll", adaptive_coll),
        "aoi_threshold": aggregate_metric("aoi_threshold_coll", aoi_coll),
    }

    ref = rows["periodic"]
    comparisons_hpc = compare_variants(ref, {k: v for k, v in rows.items() if k != "periodic"}, "HPC_pct")
    ref_c = coll_rows["periodic"]
    comparisons_coll = compare_variants(ref_c, {k: v for k, v in coll_rows.items() if k != "periodic"}, "collision_rate")

    return {
        "n_mc": n_mc,
        "seed_base": seed_base,
        "link": link,
        "base_tau": base_tau,
        "hpc_rows": rows,
        "collision_rows": coll_rows,
        "comparisons_hpc": comparisons_hpc,
        "comparisons_collision": comparisons_coll,
        "adaptive_age_cap": age_cap,
        "aoi_kappa_threshold": 25.0,
    }


def _run_aoi_threshold_mission(field, channel, params, rng_seed, kappa_thr: float = 25.0):
    """Mission with AoI-threshold sync: sync when (age-1)/2 >= kappa_thr."""
    from .staleness import StalenessModel, retention
    from .deconflict import SpaceTimeReservation, apply_spacetime_hold, pairwise_collision_rate
    from .registry import build_plan as build_planner_plan
    from .planning_utils import pick_depots

    rng = np.random.default_rng(rng_seed)
    staleness = StalenessModel(params, rng=rng)
    config = MissionConfig(num_uav=3, horizon=600, policy=SyncPolicy.PERIODIC)
    trav = field.traversable
    N, M = trav.shape
    U = 3
    starts = pick_depots(trav, U)
    positions = starts.astype(float).copy()
    last_synced_positions = positions.copy()
    W_synced = field.W.copy()
    tau_plan = channel.sample_tau()
    belief = staleness.degraded_map(W_synced, tau_plan)
    plan = build_planner_plan("iuef_em", field, belief, U, starts=starts)
    paths = [list(seg) for seg in plan["segments"]]
    path_idx = [0] * U
    visited = np.zeros((N, M), dtype=bool)
    for u in range(U):
        r, c = int(round(positions[u, 0])), int(round(positions[u, 1]))
        if 0 <= r < N and 0 <= c < M:
            visited[r, c] = True
    steps_since_sync = 0
    n_syncs = 0
    collision_steps = 0
    high_mask = field.high_mask
    n_high = max(1, int(np.count_nonzero(high_mask)))
    d_safe_cells = 25.0 / field.dx
    d_near_cells = 45.0 / field.dx
    st_res = SpaceTimeReservation(horizon=600, n_rows=N, n_cols=M)
    sync_age_steps = int(round(2.0 * kappa_thr + 1))

    for t in range(600):
        age = steps_since_sync
        do_sync = t > 0 and steps_since_sync >= sync_age_steps
        if do_sync:
            n_syncs += 1
            steps_since_sync = 0
            last_synced_positions = positions.copy()
            W_synced = field.W.copy()
            tau_plan = channel.sample_tau()
            belief = staleness.degraded_map(W_synced, tau_plan)
            plan = build_planner_plan("iuef_em", field, belief, U, starts=starts)
            paths = [list(seg) for seg in plan["segments"]]
            st_res = SpaceTimeReservation(horizon=600, n_rows=N, n_cols=M)
            for u, seg in enumerate(paths):
                st_res.reserve_path(u, seg, start_t=t)
            path_idx = []
            for u in range(U):
                seg = np.array(paths[u], dtype=float)
                d = np.hypot(seg[:, 0] - positions[u, 0], seg[:, 1] - positions[u, 1])
                path_idx.append(int(np.argmin(d)))

        belief = staleness.degraded_map(W_synced, age)
        ghosts = staleness.ghost_positions(last_synced_positions, age)
        for u in range(U):
            seg = paths[u]
            if path_idx[u] >= len(seg) - 1:
                continue
            nxt = seg[path_idx[u] + 1]
            hold = apply_spacetime_hold(u, t, tuple(nxt), ghosts, st_res, 2.2)
            if not hold:
                positions[u] = [nxt[0], nxt[1]]
                path_idx[u] += 1
                r, c = int(round(nxt[0])), int(round(nxt[1]))
                if 0 <= r < N and 0 <= c < M and trav[r, c]:
                    visited[r, c] = True
        coll, _ = pairwise_collision_rate(positions, U, d_safe_cells, d_near_cells)
        collision_steps += int(coll)
        steps_since_sync += 1

    from .mission import MissionResult
    hpc = 100.0 * np.count_nonzero(visited & high_mask) / n_high
    return MissionResult(
        hpc_pct=float(hpc),
        collision_rate=float(collision_steps / 600),
        near_miss_rate=0.0,
        collision_rate_pre=0.0,
        near_miss_rate_pre=0.0,
        uplink_cost=float(n_syncs / 600),
        n_syncs=n_syncs,
        mean_tau=float(channel.expected_tau(500)),
        mean_age=0.0,
        retained_high_frac=0.0,
        steps=600,
    )


def run_r1_ablation_with_stats(n_mc: int = 150, seed_base: int = SEED_R1_ABLATION) -> dict:
    """R1 component ablations with CI + Wilcoxon vs full IUEF-EM."""
    results = {}
    raw_hpc = {}
    for pname in ABLATION_PLANNERS:
        metrics = []
        for j in range(n_mc):
            sc = EvalScenario(seed=seed_base + j)
            from .planner_evaluation import run_trial
            metrics.append(run_trial(pname, sc))
        from .planner_evaluation import aggregate
        results[pname] = aggregate(metrics)
        raw_hpc[pname] = [m.hpc_pct for m in metrics]

    rows = {k: aggregate_metric(k, v) for k, v in raw_hpc.items()}
    ref = rows["iuef_em"]
    comparisons = compare_variants(ref, {k: v for k, v in rows.items() if k != "iuef_em"}, "HPC_pct")
    return {"n_mc": n_mc, "seed_base": seed_base, "stats": results, "comparisons": comparisons, "rows": rows}


def run_r2_ablation_tau50_with_stats(n_mc: int = 150, seed_base: int = SEED_R2_ABLATION) -> dict:
    """R2 component ablations at τ̄=50 with CI + Wilcoxon."""
    abl = ablation_under_staleness(tau_bar=50.0, n_mc=n_mc, seed_base=seed_base)
    raw = {}
    for pname in ABLATION_PLANNERS:
        raw[pname] = []
    # Re-run to capture per-seed (ablation_under_staleness only returns aggregates)
    params = default_params()
    for pname in ABLATION_PLANNERS:
        for j in range(n_mc):
            seed = seed_base + j
            field = build_priority_field(50, 50, seed=seed)
            channel = NTNChannel("medium", rng=seed + 17, base_tau_override=50.0)
            cfg = MissionConfig(
                num_uav=3, horizon=600, planner_name=pname,
                policy=SyncPolicy.PERIODIC, tau_ref=channel.expected_tau(2000),
            )
            r = run_mission(field, cfg, staleness_params=params, channel=channel, rng=seed + 31)
            raw[pname].append(r.hpc_pct)

    rows = {k: aggregate_metric(k, v) for k, v in raw.items()}
    ref = rows["iuef_em"]
    comparisons = compare_variants(ref, {k: v for k, v in rows.items() if k != "iuef_em"}, "HPC_pct")
    return {"n_mc": n_mc, "seed_base": seed_base, "tau_bar": 50.0, "aggregates": abl, "comparisons": comparisons, "rows": rows}


def run_dx_sensitivity_with_stats(
    n_mc: int = 150,
    dx_values: tuple[float, ...] = (5.0, 10.0, 18.0),
    seed_base: int = SEED_DX_SENSITIVITY,
) -> dict:
    """Grid-resolution sensitivity Δx ∈ {5,10,18} m with CI."""
    raw = {dx: [] for dx in dx_values}
    for dx in dx_values:
        for j in range(n_mc):
            field = build_priority_field(50, 50, dx=dx, seed=seed_base + j)
            from .executor import ExecConfig, evaluate_planner
            m = evaluate_planner("iuef_em", field, 3, seed=seed_base + j,
                                 cfg=ExecConfig(dx=dx, d_safe_m=25.0))
            raw[dx].append(m.hpc_pct)

    rows = {f"dx_{int(dx)}m": aggregate_metric(f"dx_{int(dx)}m", raw[dx]) for dx in dx_values}
    ref = rows[f"dx_{18}m"]
    variants = {k: v for k, v in rows.items() if k != f"dx_{18}m"}
    comparisons = compare_variants(ref, variants, "HPC_pct")
    return {"n_mc": n_mc, "seed_base": seed_base, "dx_values": list(dx_values), "rows": rows, "comparisons": comparisons}


def _write_comparison_csv(path: Path, rows: dict[str, StatRow], comparisons: list, extra_cols: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n_seeds", "mean", "std", "ci95_low", "ci95_high"])
        for name, row in rows.items():
            w.writerow([name, row.n_seeds, f"{row.mean:.4f}", f"{row.std:.4f}",
                        f"{row.ci95_low:.4f}", f"{row.ci95_high:.4f}"])
        w.writerow([])
        w.writerow(["comparison", "reference", "metric", "mean_diff", "wilcoxon_p", "cliffs_delta", "fdr_p"])
        for c in comparisons:
            w.writerow([c.variant, c.reference, c.metric, f"{c.mean_diff:.4f}",
                        f"{c.wilcoxon_p:.4g}", f"{c.cliffs_delta:.4f}",
                        f"{c.fdr_p:.4g}" if c.fdr_p is not None else ""])


def export_all_comparison_tables(out_dir: Path | str, n_mc: int = 150, quick: bool = False) -> dict:
    """Run all CI/significance comparisons and write CSVs."""
    out_dir = Path(out_dir)
    if quick:
        n_mc = min(n_mc, 20)

    sync = run_sync_policy_comparison(n_mc=n_mc)
    _write_comparison_csv(out_dir / "table_R2S3_sync_policy.csv", sync["hpc_rows"], sync["comparisons_hpc"])
    _write_comparison_csv(out_dir / "table_R2S3_sync_policy_collision.csv",
                        sync["collision_rows"], sync["comparisons_collision"])

    r1 = run_r1_ablation_with_stats(n_mc=n_mc)
    _write_comparison_csv(out_dir / "table_R1_ablation.csv", r1["rows"], r1["comparisons"])

    r2 = run_r2_ablation_tau50_with_stats(n_mc=n_mc)
    _write_comparison_csv(out_dir / "table_R2_ablation_tau50.csv", r2["rows"], r2["comparisons"])

    dx = run_dx_sensitivity_with_stats(n_mc=n_mc)
    _write_comparison_csv(out_dir / "table_grid_resolution_dx.csv", dx["rows"], dx["comparisons"])

    return {
        "n_mc": n_mc,
        "seeds": {
            "sync_policy": SEED_SYNC_POLICY,
            "r1_ablation": SEED_R1_ABLATION,
            "r2_ablation": SEED_R2_ABLATION,
            "dx_sensitivity": SEED_DX_SENSITIVITY,
        },
        "sync_policy": sync,
        "r1_ablation": r1,
        "r2_ablation": r2,
        "dx_sensitivity": dx,
    }
