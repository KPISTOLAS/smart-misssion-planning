"""Monte Carlo sensitivity experiment drivers for EPCA-M."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np

from epca_staleness.channel import NTNChannel
from epca_staleness.experiments import default_params, operating_bounds, SweepResult, _ci95

from .map_generator import SyntheticMapConfig, SyntheticMapGenerator, HotspotDensity, TerrainRoughness
from .imperfect_priority import ImperfectPriorityConfig
from .mission_runner import SensitivityMissionConfig, run_sensitivity_mission
from .analysis import extract_operating_bound, summarise_trials


TAU_GRID_DEFAULT = np.array([10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110], dtype=float)
TAU_GRID_QUICK = np.array([15, 30, 45, 60, 90], dtype=float)

GRID_PRESETS = {
    "50x50": (50, 50),
    "54x72": (54, 72),
}

FLEET_SIZES = [1, 2, 3, 4, 5, 6, 8]
D_SAFE_VALUES = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0]


@dataclass
class ExperimentConfig:
    n_mc: int = 50
    horizon: int = 600
    link: str = "medium"
    seed_base: int = 20000
    tau_grid: np.ndarray | None = None
    hpc_thr: float = 65.0
    coll_thr: float = 0.40


def _run_tau_sweep(map_cfg: SyntheticMapConfig,
                   mission_kw: dict,
                   tau_grid: np.ndarray,
                   n_mc: int,
                   params,
                   seed_base: int,
                   link: str) -> SweepResult:
    """Sweep base_tau for a fixed map/mission configuration."""
    hpc_all = np.zeros((len(tau_grid), n_mc))
    coll_all = np.zeros((len(tau_grid), n_mc))
    near_all = np.zeros((len(tau_grid), n_mc))
    gen = SyntheticMapGenerator()

    for i, bt in enumerate(tau_grid):
        for j in range(n_mc):
            seed = seed_base + i * 1000 + j
            field = gen.generate(SyntheticMapConfig(**{**asdict(map_cfg), "seed": seed}))
            mcfg = SensitivityMissionConfig(
                link=link, base_tau=float(bt), num_uav=mission_kw.get("num_uav", 3),
                d_safe_m=mission_kw.get("d_safe_m", 25.0),
                perfect_priority=mission_kw.get("perfect_priority", True),
                imperfect=mission_kw.get("imperfect"),
                disable_staleness=mission_kw.get("disable_staleness", False),
                horizon=mission_kw.get("horizon", 600),
            )
            channel = NTNChannel(link, rng=seed + 17, base_tau_override=float(bt))
            res = run_sensitivity_mission(field, mcfg, params, channel=channel, rng=seed + 31)
            hpc_all[i, j] = res.hpc_pct
            coll_all[i, j] = res.collision_rate
            near_all[i, j] = res.near_miss_rate

    hpc_m, hpc_lo, hpc_hi, coll_m, coll_lo, coll_hi, near_m = [], [], [], [], [], [], []
    for i in range(len(tau_grid)):
        m, lo, hi = _ci95(hpc_all[i])
        hpc_m.append(m); hpc_lo.append(lo); hpc_hi.append(hi)
        m, lo, hi = _ci95(coll_all[i])
        coll_m.append(m); coll_lo.append(lo); coll_hi.append(hi)
        near_m.append(float(np.mean(near_all[i])))

    return SweepResult(
        tau_mean=np.asarray(tau_grid, dtype=float),
        hpc_mean=np.asarray(hpc_m), hpc_lo=np.asarray(hpc_lo), hpc_hi=np.asarray(hpc_hi),
        coll_mean=np.asarray(coll_m), coll_lo=np.asarray(coll_lo), coll_hi=np.asarray(coll_hi),
        near_mean=np.asarray(near_m),
        raw=dict(hpc=hpc_all, coll=coll_all, near=near_all),
    )


def sweep_hotspot_density(exp: ExperimentConfig | None = None) -> dict:
    """Operating τ bound vs hotspot density (low / medium / high)."""
    exp = exp or ExperimentConfig()
    tau_grid = exp.tau_grid if exp.tau_grid is not None else TAU_GRID_DEFAULT
    params = default_params(rng=exp.seed_base)
    out = {}
    for dens in HotspotDensity:
        map_cfg = SyntheticMapConfig(hotspot_density=dens)
        sw = _run_tau_sweep(map_cfg, dict(num_uav=3), tau_grid, exp.n_mc, params,
                            exp.seed_base + hash(dens.value) % 5000, exp.link)
        out[dens.value] = dict(
            sweep=sw,
            operating_bound=extract_operating_bound(sw, exp.hpc_thr, exp.coll_thr),
        )
    return out


def sweep_fleet_size(exp: ExperimentConfig | None = None,
                     hotspot_density: str = "medium") -> dict:
    exp = exp or ExperimentConfig()
    tau_grid = exp.tau_grid if exp.tau_grid is not None else TAU_GRID_DEFAULT
    params = default_params(rng=exp.seed_base)
    out = {}
    for U in FLEET_SIZES:
        map_cfg = SyntheticMapConfig(hotspot_density=hotspot_density)
        sw = _run_tau_sweep(map_cfg, dict(num_uav=U), tau_grid, exp.n_mc, params,
                            exp.seed_base + U * 100, exp.link)
        out[str(U)] = dict(
            sweep=sw,
            operating_bound=extract_operating_bound(sw, exp.hpc_thr, exp.coll_thr),
        )
    return out


def sweep_d_safe(exp: ExperimentConfig | None = None) -> dict:
    exp = exp or ExperimentConfig()
    tau_grid = exp.tau_grid if exp.tau_grid is not None else TAU_GRID_DEFAULT
    params = default_params(rng=exp.seed_base)
    out = {}
    for d in D_SAFE_VALUES:
        map_cfg = SyntheticMapConfig()
        sw = _run_tau_sweep(map_cfg, dict(d_safe_m=d), tau_grid, exp.n_mc, params,
                            exp.seed_base + int(d), exp.link)
        out[str(int(d))] = dict(
            sweep=sw,
            operating_bound=extract_operating_bound(sw, exp.hpc_thr, exp.coll_thr),
        )
    return out


def sweep_terrain(exp: ExperimentConfig | None = None) -> dict:
    exp = exp or ExperimentConfig()
    tau_grid = exp.tau_grid if exp.tau_grid is not None else TAU_GRID_DEFAULT
    params = default_params(rng=exp.seed_base)
    out = {}
    for rough in TerrainRoughness:
        map_cfg = SyntheticMapConfig(terrain_roughness=rough)
        sw = _run_tau_sweep(map_cfg, dict(num_uav=3), tau_grid, exp.n_mc, params,
                            exp.seed_base + hash(rough.value) % 3000, exp.link)
        out[rough.value] = dict(
            sweep=sw,
            operating_bound=extract_operating_bound(sw, exp.hpc_thr, exp.coll_thr),
        )
    return out


def sweep_grid_resolution(exp: ExperimentConfig | None = None) -> dict:
    exp = exp or ExperimentConfig()
    tau_grid = exp.tau_grid if exp.tau_grid is not None else TAU_GRID_DEFAULT
    params = default_params(rng=exp.seed_base)
    out = {}
    for label, (n, m) in GRID_PRESETS.items():
        map_cfg = SyntheticMapConfig(N=n, M=m)
        sw = _run_tau_sweep(map_cfg, dict(num_uav=3), tau_grid, exp.n_mc, params,
                            exp.seed_base + hash(label) % 2000, exp.link)
        out[label] = dict(
            sweep=sw,
            operating_bound=extract_operating_bound(sw, exp.hpc_thr, exp.coll_thr),
        )
    return out


def compare_priority_quality(exp: ExperimentConfig | None = None,
                             base_tau: float = 45.0) -> dict:
    """Perfect vs imperfect priority fields at fixed τ under staleness."""
    exp = exp or ExperimentConfig()
    params = default_params(rng=exp.seed_base)
    gen = SyntheticMapGenerator()
    modes = {
        "perfect": dict(perfect_priority=True),
        "imperfect_mild": dict(perfect_priority=False, imperfect=ImperfectPriorityConfig(
            false_positive_rate=0.10, false_negative_rate=0.10, enabled=True)),
        "imperfect_severe": dict(perfect_priority=False, imperfect=ImperfectPriorityConfig(
            false_positive_rate=0.20, false_negative_rate=0.20, enabled=True)),
    }
    out = {}
    for name, kw in modes.items():
        trials = []
        for j in range(exp.n_mc):
            seed = exp.seed_base + j
            field = gen.generate(SyntheticMapConfig(seed=seed))
            mcfg = SensitivityMissionConfig(
                link=exp.link, base_tau=base_tau, num_uav=3, **kw, horizon=exp.horizon,
            )
            channel = NTNChannel(exp.link, rng=seed + 17, base_tau_override=base_tau)
            trials.append(run_sensitivity_mission(field, mcfg, params, channel=channel, rng=seed + 31))
        out[name] = summarise_trials(trials)
    return out


def run_full_sensitivity_study(out_dir: Path,
                             n_mc: int = 50,
                             quick: bool = False) -> dict:
    """Execute all sensitivity sweeps and write JSON summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if quick:
        tau_grid = np.array([20.0, 40.0, 60.0], dtype=float)
        n_mc = min(n_mc, 8)
    else:
        tau_grid = TAU_GRID_DEFAULT
    exp = ExperimentConfig(n_mc=n_mc, tau_grid=tau_grid)

    summary = dict(n_mc=n_mc, quick=quick, hpc_thr=exp.hpc_thr, coll_thr=exp.coll_thr)

    print("  hotspot density sweep ...")
    summary["hotspot_density"] = _serialise_sweep_dict(sweep_hotspot_density(exp))

    print("  fleet size sweep ...")
    if quick:
        # Reduced fleet sweep for smoke test.
        exp_fs = ExperimentConfig(n_mc=n_mc, tau_grid=tau_grid, seed_base=exp.seed_base + 100)
        params = default_params(rng=exp_fs.seed_base)
        fs_out = {}
        for U in [2, 3, 4, 6]:
            map_cfg = SyntheticMapConfig()
            sw = _run_tau_sweep(map_cfg, dict(num_uav=U), tau_grid, n_mc, params,
                                exp_fs.seed_base + U * 100, exp_fs.link)
            fs_out[str(U)] = dict(sweep=sw, operating_bound=extract_operating_bound(sw))
        summary["fleet_size"] = _serialise_sweep_dict(fs_out)
    else:
        summary["fleet_size"] = _serialise_sweep_dict(sweep_fleet_size(exp))

    print("  d_safe sweep ...")
    if quick:
        exp_d = ExperimentConfig(n_mc=n_mc, tau_grid=tau_grid, seed_base=exp.seed_base + 200)
        params = default_params(rng=exp_d.seed_base)
        d_out = {}
        for d in [20.0, 25.0, 35.0]:
            map_cfg = SyntheticMapConfig()
            sw = _run_tau_sweep(map_cfg, dict(d_safe_m=d), tau_grid, n_mc, params,
                                exp_d.seed_base + int(d), exp_d.link)
            d_out[str(int(d))] = dict(sweep=sw, operating_bound=extract_operating_bound(sw))
        summary["d_safe"] = _serialise_sweep_dict(d_out)
    else:
        summary["d_safe"] = _serialise_sweep_dict(sweep_d_safe(exp))

    if not quick:
        print("  terrain sweep ...")
        summary["terrain"] = _serialise_sweep_dict(sweep_terrain(exp))
        print("  grid resolution sweep ...")
        summary["grid_resolution"] = _serialise_sweep_dict(sweep_grid_resolution(exp))

    print("  perfect vs imperfect priority ...")
    summary["priority_quality"] = compare_priority_quality(exp)

    (out_dir / "sensitivity_summary.json").write_text(json.dumps(_sanitize(summary), indent=2))
    return summary


def _serialise_sweep_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        sw = v["sweep"]
        out[k] = dict(
            operating_bound=v["operating_bound"],
            tau_mean=sw.tau_mean.tolist(),
            hpc_mean=sw.hpc_mean.tolist(),
            coll_mean=sw.coll_mean.tolist(),
        )
    return out


def _sanitize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(x) for x in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, int, float, bool)):
        return _sanitize(vars(obj))
    return obj
