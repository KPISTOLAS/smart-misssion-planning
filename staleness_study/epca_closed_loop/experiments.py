"""Monte-Carlo experiment drivers for the closed-loop EPCA-M simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from epca_staleness.channel import NTNChannel, LINK_PRESETS
from epca_staleness.environment import build_priority_field
from epca_staleness.experiments import default_params

from .closed_loop import ClosedLoopConfig, ClosedLoopResult, SyncPolicy, run_closed_loop


def _ci95(samples: np.ndarray) -> tuple[float, float, float]:
    m = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / max(np.sqrt(len(samples)), 1.0))
    return m, m - 1.96 * se, m + 1.96 * se


@dataclass
class SweepResult:
    x: np.ndarray
    hpc_mean: np.ndarray
    hpc_lo: np.ndarray
    hpc_hi: np.ndarray
    coll_mean: np.ndarray
    coll_lo: np.ndarray
    coll_hi: np.ndarray
    raw: dict


def run_single_trial(seed: int,
                     link: str = "medium",
                     base_tau: float | None = None,
                     policy: SyncPolicy = SyncPolicy.PERIODIC,
                     perfect_info: bool = False,
                     disable_staleness: bool = False,
                     horizon: int = 600) -> ClosedLoopResult:
    """One Monte-Carlo closed-loop rollout."""
    params = default_params(rng=seed)
    field = build_priority_field(50, 50, seed=seed)
    channel = NTNChannel(link, rng=seed + 17, base_tau_override=base_tau)
    tau_ref = channel.expected_tau(2000)
    cfg = ClosedLoopConfig(
        horizon=horizon,
        policy=policy,
        tau_ref=tau_ref,
        perfect_info=perfect_info,
        disable_staleness=disable_staleness,
        adapt_age_steps=30.0,
        adapt_retention_drop=1.1 if policy is SyncPolicy.ADAPTIVE else 0.28,
    )
    return run_closed_loop(field, cfg, staleness_params=params,
                           channel=NTNChannel(link, rng=seed + 17, base_tau_override=base_tau),
                           rng=seed + 31)


def sweep_tau(link: str = "medium",
              tau_grid=None,
              n_mc: int = 50,
              with_staleness: bool = True,
              seed_base: int = 5000) -> SweepResult:
    """Sweep mean tau; compare closed-loop with/without staleness."""
    if tau_grid is None:
        tau_grid = np.array([15, 20, 30, 40, 50, 60, 75, 90, 110], dtype=float)
    hpc = np.zeros((len(tau_grid), n_mc))
    coll = np.zeros((len(tau_grid), n_mc))
    for i, bt in enumerate(tau_grid):
        for j in range(n_mc):
            r = run_single_trial(seed_base + i * n_mc + j, link=link, base_tau=float(bt),
                                 disable_staleness=not with_staleness)
            hpc[i, j] = r.hpc_pct
            coll[i, j] = r.collision_rate
    hm, hl, hh = zip(*[_ci95(hpc[i]) for i in range(len(tau_grid))])
    cm, cl, ch = zip(*[_ci95(coll[i]) for i in range(len(tau_grid))])
    return SweepResult(
        x=np.array(tau_grid), hpc_mean=np.array(hm), hpc_lo=np.array(hl), hpc_hi=np.array(hh),
        coll_mean=np.array(cm), coll_lo=np.array(cl), coll_hi=np.array(ch),
        raw={"hpc": hpc.tolist(), "coll": coll.tolist(), "staleness": with_staleness},
    )


def compare_modes(link: str = "medium",
                  base_tau: float = 45.0,
                  n_mc: int = 50,
                  seed_base: int = 8000) -> dict:
    """Closed-loop vs perfect-info vs no-staleness at fixed link quality."""
    modes = [
        ("closed_loop", dict(perfect_info=False, disable_staleness=False)),
        ("perfect_info", dict(perfect_info=True, disable_staleness=False)),
        ("no_staleness", dict(perfect_info=False, disable_staleness=True)),
    ]
    out = {}
    for name, kw in modes:
        hpc, coll, near, score = [], [], [], []
        for j in range(n_mc):
            r = run_single_trial(seed_base + j, link=link, base_tau=base_tau, **kw)
            hpc.append(r.hpc_pct)
            coll.append(r.collision_rate)
            near.append(r.near_miss_rate)
            score.append(r.mission_score)
        m, lo, hi = _ci95(np.array(hpc))
        cm, cl, ch = _ci95(np.array(coll))
        out[name] = dict(
            hpc_mean=m, hpc_lo=lo, hpc_hi=hi,
            coll_mean=cm, coll_lo=cl, coll_hi=ch,
            near_mean=float(np.mean(near)), mission_score_mean=float(np.mean(score)),
        )
    return out


def compare_policies(link: str = "medium",
                     base_tau: float = 45.0,
                     n_mc: int = 50,
                     seed_base: int = 9000) -> dict:
    """Periodic vs adaptive synchronization under closed loop."""
    out = {}
    for pol, name in [(SyncPolicy.PERIODIC, "periodic"), (SyncPolicy.ADAPTIVE, "adaptive")]:
        hpc, coll = [], []
        for j in range(n_mc):
            r = run_single_trial(seed_base + j, link=link, base_tau=base_tau, policy=pol)
            hpc.append(r.hpc_pct)
            coll.append(r.collision_rate)
        m, lo, hi = _ci95(np.array(hpc))
        cm, cl, ch = _ci95(np.array(coll))
        out[name] = dict(hpc_mean=m, hpc_lo=lo, hpc_hi=hi, coll_mean=cm, coll_lo=cl, coll_hi=ch)
    return out


def run_full_study(out_dir: Path,
                   n_mc: int = 50,
                   quick: bool = False) -> dict:
    """Run all sweeps and write JSON summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if quick:
        n_mc = 12
        tau_grid = np.array([20, 40, 60, 90], dtype=float)
    else:
        tau_grid = None

    summary = dict(n_mc=n_mc, link_presets={k: v.base_tau for k, v in LINK_PRESETS.items()})

    summary["tau_sweep_staleness"] = sweep_tau("medium", tau_grid, n_mc, with_staleness=True).__dict__
    summary["tau_sweep_no_staleness"] = sweep_tau("medium", tau_grid, n_mc, with_staleness=False).__dict__
    summary["mode_comparison_medium"] = compare_modes("medium", 45.0, n_mc)
    summary["policy_comparison"] = compare_policies("medium", 45.0, n_mc)

    for link in ("good", "medium", "poor"):
        bt = LINK_PRESETS[link].base_tau
        summary[f"mode_comparison_{link}"] = compare_modes(link, bt, n_mc, seed_base=10000 + hash(link) % 1000)

    # Convert numpy arrays for JSON
    def _sanitize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(x) for x in obj]
        return obj

    summary = _sanitize(summary)
    (out_dir / "closed_loop_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
