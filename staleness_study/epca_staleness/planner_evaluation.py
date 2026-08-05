"""Monte-Carlo planner evaluation framework (Section V experiments).

Systematic comparison of IUEF-EM, ablations, and external baselines under:
  * N independent seeds (default 50)
  * fleet size U in {1,2,3,4,5,6,8,10}
  * grid sizes 50x50 and 54x72
  * hotspot density and d_safe sweeps
  * staleness-degraded priority fields (optional)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import numpy as np

from .environment import build_priority_field
from .executor import ExecConfig, MissionMetrics, evaluate_planner
from .registry import ABLATION_PLANNERS, BASELINE_PLANNERS, PLANNER_REGISTRY
from .staleness import StalenessParams, retention
from .experiments import default_params as staleness_default_params


@dataclass
class EvalScenario:
    N: int = 50
    M: int = 50
    dx: float = 18.0
    num_uav: int = 3
    hotspot_frac: float = 0.12
    d_safe_m: float = 25.0
    seed: int = 0


@dataclass
class AggregateStats:
    planner: str
    hpc_mean: float
    hpc_std: float
    hpc_lo: float
    hpc_hi: float
    whpc_mean: float
    hpc_early_mean: float
    mission_score_mean: float
    coverage_mean: float
    duration_mean: float
    energy_mean: float
    collision_mean: float
    collision_std: float
    near_miss_mean: float
    n_trials: int


def _ci95(samples: np.ndarray):
    m = float(np.mean(samples))
    s = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
    se = s / max(np.sqrt(len(samples)), 1.0)
    return m, s, m - 1.96 * se, m + 1.96 * se


def aggregate(metrics: list[MissionMetrics]) -> AggregateStats:
    hpc = np.array([m.hpc_pct for m in metrics])
    hm, hs, hlo, hhi = _ci95(hpc)
    coll = np.array([m.collision_rate for m in metrics])
    cm, cs, _, _ = _ci95(coll)
    return AggregateStats(
        planner=metrics[0].planner if metrics else "",
        hpc_mean=hm, hpc_std=hs, hpc_lo=hlo, hpc_hi=hhi,
        whpc_mean=float(np.mean([m.whpc_pct for m in metrics])),
        hpc_early_mean=float(np.mean([m.hpc_early_pct for m in metrics])),
        mission_score_mean=float(np.mean([m.mission_score for m in metrics])),
        coverage_mean=float(np.mean([m.coverage_pct for m in metrics])),
        duration_mean=float(np.mean([m.duration_steps for m in metrics])),
        energy_mean=float(np.mean([m.energy_per_uav_hour for m in metrics])),
        collision_mean=cm, collision_std=cs,
        near_miss_mean=float(np.mean([m.near_miss_rate for m in metrics])),
        n_trials=len(metrics),
    )


def degraded_belief(field, tau: float, params: StalenessParams, rng) -> np.ndarray:
    """Stale priority field at mean interval tau (for planner input W_est)."""
    R = retention(tau, params.beta_M)
    noise = rng.normal(0.0, params.sigma_M * np.sqrt(max(tau, 0.0)), size=field.W.shape)
    return np.maximum(0.0, field.W * R + noise)


def run_trial(planner: str, scenario: EvalScenario,
              W_est=None, horizon: int = 1200) -> MissionMetrics:
    field = build_priority_field(
        scenario.N, scenario.M, dx=scenario.dx,
        hotspot_frac=scenario.hotspot_frac, seed=scenario.seed,
    )
    cfg = ExecConfig(horizon=horizon, d_safe_m=scenario.d_safe_m, dx=scenario.dx)
    return evaluate_planner(planner, field, scenario.num_uav, W_est=W_est, cfg=cfg, seed=scenario.seed)


def monte_carlo(planners: list[str], scenario_template: EvalScenario,
                n_mc: int = 50, seed_base: int = 1000,
                tau: float | None = None,
                staleness_params: StalenessParams | None = None) -> dict[str, AggregateStats]:
    """Run N_MC trials per planner; optional staleness via degraded W_est."""
    results = {}
    params = staleness_params or staleness_default_params()
    for pname in planners:
        trials = []
        for j in range(n_mc):
            sc = EvalScenario(**{**asdict(scenario_template), "seed": seed_base + j})
            W_est = None
            if tau is not None:
                field = build_priority_field(sc.N, sc.M, dx=sc.dx,
                                             hotspot_frac=sc.hotspot_frac, seed=sc.seed)
                rng = np.random.default_rng(sc.seed + 7)
                W_est = degraded_belief(field, tau, params, rng)
            trials.append(run_trial(pname, sc, W_est=W_est))
        results[pname] = aggregate(trials)
    return results


def sweep_fleet_size(planners: list[str], U_values=None, n_mc: int = 50,
                     grid: tuple = (50, 50), seed_base: int = 2000) -> dict:
    if U_values is None:
        U_values = [1, 2, 3, 4, 5, 6, 8, 10]
    out = {}
    for U in U_values:
        tmpl = EvalScenario(N=grid[0], M=grid[1], num_uav=U)
        out[U] = monte_carlo(planners, tmpl, n_mc=n_mc, seed_base=seed_base + U * 10000)
    return out


def sweep_grid_sizes(planners: list[str], grids=None, n_mc: int = 30,
                     num_uav: int = 3, seed_base: int = 3000) -> dict:
    if grids is None:
        grids = [(50, 50), (54, 72)]
    out = {}
    for g in grids:
        tmpl = EvalScenario(N=g[0], M=g[1], num_uav=num_uav)
        out[f"{g[0]}x{g[1]}"] = monte_carlo(planners, tmpl, n_mc=n_mc, seed_base=seed_base)
    return out


def sweep_hotspot_density(planners: list[str], fracs=None, n_mc: int = 30,
                          seed_base: int = 4000) -> dict:
    if fracs is None:
        fracs = [0.08, 0.10, 0.12, 0.15, 0.18]
    out = {}
    for hf in fracs:
        tmpl = EvalScenario(hotspot_frac=hf)
        out[hf] = monte_carlo(planners, tmpl, n_mc=n_mc, seed_base=seed_base + int(hf * 1000))
    return out


def sweep_d_safe(planners: list[str], d_values=None, n_mc: int = 30,
                 seed_base: int = 5000) -> dict:
    if d_values is None:
        d_values = [18.0, 22.0, 25.0, 30.0, 35.0]
    out = {}
    for d in d_values:
        tmpl = EvalScenario(d_safe_m=d)
        out[d] = monte_carlo(planners, tmpl, n_mc=n_mc, seed_base=seed_base + int(d))
    return out


def ablation_study(n_mc: int = 50, seed_base: int = 6000) -> dict[str, AggregateStats]:
    """Systematic ablation of IUEF-EM components."""
    tmpl = EvalScenario(num_uav=3)
    return monte_carlo(ABLATION_PLANNERS, tmpl, n_mc=n_mc, seed_base=seed_base)


def baseline_comparison(n_mc: int = 50, seed_base: int = 7000) -> dict[str, AggregateStats]:
    """Full proposed planner vs external + internal baselines."""
    tmpl = EvalScenario(num_uav=3)
    return monte_carlo(BASELINE_PLANNERS, tmpl, n_mc=n_mc, seed_base=seed_base)


def staleness_planner_sweep(planners: list[str] | None = None,
                            tau_grid=None, n_mc: int = 30,
                            seed_base: int = 8000) -> dict:
    """Planner performance vs mean tau under degraded priority fields."""
    if planners is None:
        planners = ["iuef_em", "darp", "priority_tsp", "lawnmower", "greedy"]
    if tau_grid is None:
        tau_grid = [15, 25, 35, 45, 60, 80, 100]
    params = staleness_default_params()
    out = {}
    for tau in tau_grid:
        tmpl = EvalScenario()
        out[tau] = monte_carlo(planners, tmpl, n_mc=n_mc, seed_base=seed_base + tau * 100,
                               tau=float(tau), staleness_params=params)
    return out


def stats_to_table(stats: dict[str, AggregateStats]) -> list[dict]:
    """Convert aggregate stats to rows for CSV / LaTeX."""
    rows = []
    for name, s in stats.items():
        rows.append({
            "planner": name,
            "HPC_mean": round(s.hpc_mean, 2),
            "HPC_std": round(s.hpc_std, 2),
            "WHPC_mean": round(s.whpc_mean, 2),
            "HPC_early_mean": round(s.hpc_early_mean, 2),
            "mission_score": round(s.mission_score_mean, 3),
            "HPC_CI95": f"[{s.hpc_lo:.1f}, {s.hpc_hi:.1f}]",
            "coverage_%": round(s.coverage_mean, 2),
            "duration_steps": round(s.duration_mean, 1),
            "energy_J_per_UAVh": round(s.energy_mean, 1),
            "collision_mean": round(s.collision_mean, 4),
            "collision_std": round(s.collision_std, 4),
            "near_miss_mean": round(s.near_miss_mean, 4),
            "N": s.n_trials,
        })
    return rows


def save_results(payload: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "planner_evaluation.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
