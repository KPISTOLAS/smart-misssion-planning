"""Publication-grade experiments: regime separation, staleness ablations, operating envelope.

Three explicit evaluation regimes (must be reported separately in tables/figures):

  R1 — Planner-perfect: ground-truth W, no staleness (planning upper bound).
  R2 — Staleness-moderate: full mission loop, τ̄ ∈ {5,10,20,40,80,160,320}.
  R3 — Closed-loop: Tier-2 inference + staleness + replanning.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
from pathlib import Path

from .environment import build_priority_field
from .registry import ABLATION_PLANNERS
from .mission import MissionConfig, SyncPolicy, run_mission
from .channel import NTNChannel
from .experiments import sweep_tau, operating_bounds, _ci95
from .planner_evaluation import ablation_study, baseline_comparison, AggregateStats
from .sweep_config import TAU_SWEEP_GRID, TAU_SWEEP_QUICK, TAU_SWEEP_N_MC, TAU_SWEEP_N_MC_SMOKE
from .staleness import calibrated_defaults


class Regime(str, Enum):
    R1_PLANNER_PERFECT = "R1_planner_perfect"
    R2_STALENESS_MODERATE = "R2_staleness_moderate"
    R3_CLOSED_LOOP = "R3_closed_loop"


@dataclass
class StalenessAblationStats:
    planner: str
    tau_bar: float
    hpc_mean: float
    hpc_std: float
    hpc_lo: float
    hpc_hi: float
    collision_mean: float
    collision_pre_mean: float
    near_miss_mean: float
    uplink_mean: float
    retained_mean: float
    n_syncs_mean: float
    duration_mean: float
    n_trials: int


def _aggregate_mission(trials, planner: str, tau_bar: float) -> StalenessAblationStats:
    hpc = np.array([t.hpc_pct for t in trials])
    m, lo, hi = _ci95(hpc)
    s = float(np.std(hpc, ddof=1)) if len(hpc) > 1 else 0.0
    return StalenessAblationStats(
        planner=planner, tau_bar=tau_bar,
        hpc_mean=m, hpc_std=s, hpc_lo=lo, hpc_hi=hi,
        collision_mean=float(np.mean([t.collision_rate for t in trials])),
        collision_pre_mean=float(np.mean([getattr(t, "collision_rate_pre", t.collision_rate) for t in trials])),
        near_miss_mean=float(np.mean([t.near_miss_rate for t in trials])),
        uplink_mean=float(np.mean([t.uplink_cost for t in trials])),
        retained_mean=float(np.mean([t.retained_high_frac for t in trials])),
        n_syncs_mean=float(np.mean([t.n_syncs for t in trials])),
        duration_mean=float(np.mean([t.steps for t in trials])),
        n_trials=len(trials),
    )


def ablation_under_staleness(tau_bar: float = 50.0,
                             n_mc: int = 50,
                             seed_base: int = 11000,
                             link: str = "medium") -> dict[str, StalenessAblationStats]:
    """Full mission-loop ablation at fixed mean τ (moderate staleness)."""
    params = calibrated_defaults()
    out = {}
    for pname in ABLATION_PLANNERS:
        trials = []
        for j in range(n_mc):
            seed = seed_base + j
            field = build_priority_field(50, 50, seed=seed)
            channel = NTNChannel(link, rng=seed + 17, base_tau_override=tau_bar)
            cfg = MissionConfig(
                num_uav=3, horizon=600, planner_name=pname,
                policy=SyncPolicy.PERIODIC, tau_ref=channel.expected_tau(2000),
            )
            trials.append(run_mission(field, cfg, staleness_params=params,
                                    channel=channel, rng=seed + 31))
        out[pname] = _aggregate_mission(trials, pname, tau_bar)
    return out


def ablation_staleness_sweep(tau_values=None,
                             n_mc: int = 50,
                             seed_base: int = 12000) -> dict[float, dict[str, StalenessAblationStats]]:
    if tau_values is None:
        tau_values = list(TAU_SWEEP_GRID)
    return {tau: ablation_under_staleness(tau, n_mc, seed_base + int(tau * 10))
            for tau in tau_values}


def run_regime_r1(n_mc: int = 50) -> dict:
    """R1: planner-perfect (no staleness). HPC saturates — lead with secondary KPIs."""
    abl = ablation_study(n_mc=n_mc)
    bl = baseline_comparison(n_mc=n_mc)
    return {
        "regime": Regime.R1_PLANNER_PERFECT.value,
        "description": "Ground-truth W, no staleness — planning upper bound",
        "ablation_stats": {k: _stats_dict(v) for k, v in abl.items()},
        "baselines": {k: _stats_dict(v) for k, v in bl.items()},
        "note": "HPC often saturates at 100%; compare energy, duration, mission_score, collision",
    }


def _stats_dict(s: AggregateStats) -> dict:
    return dict(
        hpc_mean=s.hpc_mean, hpc_std=s.hpc_std, hpc_lo=s.hpc_lo, hpc_hi=s.hpc_hi,
        whpc_mean=s.whpc_mean, mission_score=s.mission_score_mean,
        coverage_mean=s.coverage_mean, duration_mean=s.duration_mean,
        energy_mean=s.energy_mean, collision_mean=s.collision_mean,
        collision_std=s.collision_std, near_miss_mean=s.near_miss_mean, n=s.n_trials,
    )


def run_regime_r2(n_mc: int = 50, tau_values=None) -> dict:
    """R2: ablations under moderate staleness."""
    sweep = ablation_staleness_sweep(tau_values, n_mc)
    return {
        "regime": Regime.R2_STALENESS_MODERATE.value,
        "description": f"Full staleness mission loop, τ̄ ∈ {list(TAU_SWEEP_GRID)}",
        "tau_sweep": {
            str(int(tau)): {k: asdict(v) for k, v in planners.items()}
            for tau, planners in sweep.items()
        },
    }


def _compare_modes_extended(link: str = "medium",
                            base_tau: float = 45.0,
                            n_mc: int = 50,
                            seed_base: int = 8000) -> dict:
    """Closed-loop vs perfect-info vs no-staleness with secondary KPIs."""
    from epca_closed_loop.experiments import run_single_trial
    from epca_closed_loop.closed_loop import SyncPolicy

    modes = [
        ("closed_loop", dict(perfect_info=False, disable_staleness=False)),
        ("perfect_info", dict(perfect_info=True, disable_staleness=False)),
        ("no_staleness", dict(perfect_info=False, disable_staleness=True)),
    ]
    out = {}
    for name, kw in modes:
        hpc, coll, near, score = [], [], [], []
        mae, energy, duration, retained = [], [], [], []
        for j in range(n_mc):
            r = run_single_trial(seed_base + j, link=link, base_tau=base_tau,
                                 policy=SyncPolicy.PERIODIC, **kw)
            hpc.append(r.hpc_pct)
            coll.append(r.collision_rate)
            near.append(r.near_miss_rate)
            score.append(r.mission_score)
            mae.append(r.inference_mae)
            energy.append(r.energy_per_uav_hour)
            duration.append(r.mission_duration)
            retained.append(r.retained_high_frac)
        hm, hlo, hhi = _ci95(np.array(hpc))
        cm, clo, chi = _ci95(np.array(coll))
        out[name] = dict(
            hpc_mean=hm, hpc_lo=hlo, hpc_hi=hhi,
            coll_mean=cm, coll_lo=clo, coll_hi=chi,
            near_mean=float(np.mean(near)),
            mission_score_mean=float(np.mean(score)),
            inference_mae_mean=float(np.mean(mae)),
            energy_mean=float(np.mean(energy)),
            duration_mean=float(np.mean(duration)),
            retained_mean=float(np.mean(retained)),
        )
    return out


def run_regime_r3(n_mc: int = 50) -> dict:
    """R3: closed-loop end-to-end."""
    return {
        "regime": Regime.R3_CLOSED_LOOP.value,
        "description": "Detector + forecaster → W_i → staleness → IUEF-EM",
        "modes": _compare_modes_extended("medium", 45.0, n_mc),
        "note": "Lead with inference_mae (targeting error), collision, energy where HPC plateaus",
    }


def build_operating_envelope(n_mc: int = 50, quick: bool = False) -> dict:
    """Integrate τ bounds across link, fleet, hotspot, and regime."""
    from epca_sensitivity.experiments import (
        sweep_hotspot_density, sweep_fleet_size, ExperimentConfig,
    )

    tau_grid = TAU_SWEEP_QUICK if quick else TAU_SWEEP_GRID
    env_mc = TAU_SWEEP_N_MC_SMOKE if quick else min(TAU_SWEEP_N_MC, n_mc)
    exp = ExperimentConfig(n_mc=env_mc, tau_grid=tau_grid)

    envelope = {"criteria": {"hpc_min_pct": 65.0, "coll_max": 0.4},
                "envelope_n_mc": env_mc, "bounds": []}

    # Link quality (staleness mission model)
    link_tau = tau_grid
    params = calibrated_defaults()
    for link in ("good", "medium", "poor"):
        sw = sweep_tau(link, link_tau, env_mc, params)
        b = operating_bounds(sw)
        envelope["bounds"].append({
            "factor": "link_quality", "setting": link,
            "tau_max_steps": b.get("max_tau"), "tau_max_s": b.get("max_tau"),
            "hpc_at_bound": b.get("hpc_at_bound"), "coll_at_bound": b.get("coll_at_bound"),
            "feasible": b.get("feasible", False), "regime": "R2_staleness",
        })

    # Hotspot density + fleet (sensitivity package)
    hd = sweep_hotspot_density(exp)
    for k, v in hd.items():
        b = v["operating_bound"]
        envelope["bounds"].append({
            "factor": "hotspot_density", "setting": k,
            "tau_max_steps": b.get("max_tau"), "tau_max_s": b.get("max_tau"),
            "hpc_at_bound": b.get("hpc_at_bound"), "coll_at_bound": b.get("coll_at_bound"),
            "feasible": b.get("feasible", False), "regime": "R2_staleness",
        })

    fs = sweep_fleet_size(exp)
    for k, v in fs.items():
        b = v["operating_bound"]
        envelope["bounds"].append({
            "factor": "fleet_size_U", "setting": k,
            "tau_max_steps": b.get("max_tau"), "tau_max_s": b.get("max_tau"),
            "hpc_at_bound": b.get("hpc_at_bound"), "coll_at_bound": b.get("coll_at_bound"),
            "feasible": b.get("feasible", False), "regime": "R2_staleness",
        })

    # R1 note: no τ bound (instant perfect W)
    envelope["bounds"].append({
        "factor": "regime", "setting": "R1_planner_perfect",
        "tau_max_steps": None, "tau_max_s": None,
        "hpc_at_bound": None, "coll_at_bound": None,
        "feasible": True, "regime": "R1_planner_perfect",
        "note": "HPC saturates; compare energy, duration, mission_score",
    })

    return envelope


def run_publication_study(out_dir: Path,
                          n_mc: int = 50,
                          quick: bool = False) -> dict:
    """Full publication pipeline."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if quick:
        n_mc = min(n_mc, TAU_SWEEP_N_MC_SMOKE)

    summary = dict(n_mc=n_mc, quick=quick)

    print("  R1 — planner-perfect regime ...")
    summary["R1"] = run_regime_r1(n_mc)

    print("  R2 — ablation under staleness (extended τ̄ grid) ...")
    summary["R2"] = run_regime_r2(n_mc, list(TAU_SWEEP_QUICK) if quick else list(TAU_SWEEP_GRID))

    print("  R3 — closed-loop regime ...")
    summary["R3"] = run_regime_r3(n_mc)

    print("  Operating envelope integration ...")
    summary["operating_envelope"] = build_operating_envelope(n_mc, quick)

    (out_dir / "publication_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    from .publication_plots import generate_publication_figures
    generate_publication_figures(summary, out_dir)

    return summary
