"""Monte-Carlo experiment drivers for the enhanced EPCA-M staleness study.

Runs reproducible sweeps over:
  * link quality (good / medium / poor) and mean synchronization interval,
  * periodic vs. adaptive synchronization at matched uplink cost,
  * parameter sensitivity (beta_M, sigma_g, hotspot fraction),
  * operating-bound extraction (HPC > 65 %, collision < 0.4).

All figures are written to ``output/`` as PNG + PDF.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from .channel import NTNChannel, LINK_PRESETS, kappa as channel_kappa
from .environment import build_priority_field
from .mission import MissionConfig, MissionResult, SyncPolicy, run_mission
from .sweep_config import TAU_SWEEP_GRID, TAU_SWEEP_QUICK
from .staleness import (
    StalenessParams,
    calibrated_defaults,
    calibrate_ghost_sigma,
    calibrate_beta_M,
    retention,
    kappa,
    emit_calibration_report,
)


# ---------------------------------------------------------------------- #
# Default calibrated parameters (paper baseline)
# ---------------------------------------------------------------------- #
def default_params(delta_ref: int = 60,
                   ghost_rmse_cells: float = 10.0,
                   map_retention: float = 0.60,
                   rng=None) -> StalenessParams:
  """Return calibrated degradation parameters for the reference interval."""
  return calibrated_defaults(delta_ref=delta_ref, ghost_rmse_cells=ghost_rmse_cells,
                             map_retention=map_retention, rng=rng)


@dataclass
class SweepResult:
  tau_mean: np.ndarray
  hpc_mean: np.ndarray
  hpc_lo: np.ndarray
  hpc_hi: np.ndarray
  coll_mean: np.ndarray
  coll_lo: np.ndarray
  coll_hi: np.ndarray
  near_mean: np.ndarray
  raw: dict


def _ci95(samples: np.ndarray) -> tuple[float, float, float]:
  """Return (mean, lo, hi) with a normal 95 % confidence band."""
  m = float(np.mean(samples))
  se = float(np.std(samples, ddof=1) / max(np.sqrt(len(samples)), 1.0))
  return m, m - 1.96 * se, m + 1.96 * se


def run_single_trial(seed: int,
                     link: str,
                     base_tau: float,
                     params: StalenessParams,
                     policy: SyncPolicy = SyncPolicy.PERIODIC,
                     adapt_age_steps: float | None = None,
                     horizon: int = 600) -> MissionResult:
  """One Monte-Carlo mission rollout."""
  field = build_priority_field(50, 50, seed=seed)
  channel = NTNChannel(link, rng=seed + 17, base_tau_override=base_tau)
  tau_ref = channel.expected_tau(3000)
  cfg = MissionConfig(
    num_uav=3,
    horizon=horizon,
    policy=policy,
    tau_ref=tau_ref,
    adapt_age_steps=adapt_age_steps if adapt_age_steps is not None else 30.0,
    adapt_retention_drop=1.1,  # age-only adaptive for fair cost matching
  )
  return run_mission(field, cfg, staleness_params=params,
                     channel=NTNChannel(link, rng=seed + 17, base_tau_override=base_tau),
                     rng=seed + 31)


def sweep_tau(link: str = "medium",
              tau_grid=None,
              n_mc: int = 50,
              params: StalenessParams | None = None,
              seed_base: int = 1000) -> SweepResult:
  """Sweep mean synchronization interval and aggregate HPC / collision metrics."""
  if tau_grid is None:
    tau_grid = np.array([5, 10, 20, 40, 80, 160, 320], dtype=float)
  if params is None:
    params = default_params()

  hpc_all = np.zeros((len(tau_grid), n_mc))
  coll_all = np.zeros((len(tau_grid), n_mc))
  near_all = np.zeros((len(tau_grid), n_mc))

  for i, bt in enumerate(tau_grid):
    for j in range(n_mc):
      seed = seed_base + i * 1000 + j
      res = run_single_trial(seed, link, float(bt), params)
      hpc_all[i, j] = res.hpc_pct
      coll_all[i, j] = res.collision_rate
      near_all[i, j] = res.near_miss_rate

  hpc_m, hpc_lo, hpc_hi = [], [], []
  coll_m, coll_lo, coll_hi = [], [], []
  near_m = []
  for i in range(len(tau_grid)):
    m, lo, hi = _ci95(hpc_all[i])
    hpc_m.append(m); hpc_lo.append(lo); hpc_hi.append(hi)
    m, lo, hi = _ci95(coll_all[i])
    coll_m.append(m); coll_lo.append(lo); coll_hi.append(hi)
    near_m.append(float(np.mean(near_all[i])))

  return SweepResult(
    tau_mean=np.asarray(tau_grid, dtype=float),
    hpc_mean=np.asarray(hpc_m),
    hpc_lo=np.asarray(hpc_lo),
    hpc_hi=np.asarray(hpc_hi),
    coll_mean=np.asarray(coll_m),
    coll_lo=np.asarray(coll_lo),
    coll_hi=np.asarray(coll_hi),
    near_mean=np.asarray(near_m),
    raw=dict(hpc=hpc_all, coll=coll_all, near=near_all, link=link, n_mc=n_mc),
  )


def compare_policies(link: str = "medium",
                     n_mc: int = 50,
                     params: StalenessParams | None = None,
                     seed_base: int = 5000) -> dict:
  """Compare periodic vs. adaptive at matched average uplink cost."""
  if params is None:
    params = default_params()

  base_tau = LINK_PRESETS[link].base_tau

  def mean_uplink(age_cap: float, n: int = 15) -> float:
    ups = []
    for j in range(n):
      field = build_priority_field(50, 50, seed=seed_base + j)
      channel = NTNChannel(link, rng=seed_base + j, base_tau_override=base_tau)
      tau_ref = channel.expected_tau(2000)
      cfg = MissionConfig(num_uav=3, horizon=600, policy=SyncPolicy.ADAPTIVE,
                          tau_ref=tau_ref, adapt_age_steps=age_cap,
                          adapt_retention_drop=1.1)  # age-only for fair cost match
      res = run_mission(field, cfg, staleness_params=params, channel=channel,
                        rng=seed_base + j + 7)
      ups.append(res.uplink_cost)
    return float(np.mean(ups))

  # Target uplink from periodic baseline.
  per_uplinks = []
  for j in range(15):
    res = run_single_trial(seed_base + j, link, base_tau, params,
                           policy=SyncPolicy.PERIODIC)
    per_uplinks.append(res.uplink_cost)
  target_up = float(np.mean(per_uplinks))

  # Binary-search age cap so adaptive uplink matches periodic.
  lo, hi = 5.0, 120.0
  for _ in range(14):
    mid = 0.5 * (lo + hi)
    up = mean_uplink(mid)
    if up > target_up:      # syncing too often -> raise cap
      lo = mid
    else:                   # syncing too rarely -> lower cap
      hi = mid
  age_cap = 0.5 * (lo + hi)

  rows = {"periodic": [], "adaptive": []}
  for j in range(n_mc):
    seed = seed_base + j
    rows["periodic"].append(
      run_single_trial(seed, link, base_tau, params, policy=SyncPolicy.PERIODIC))
    field = build_priority_field(50, 50, seed=seed)
    channel = NTNChannel(link, rng=seed + 17, base_tau_override=base_tau)
    tau_ref = channel.expected_tau(2000)
    cfg = MissionConfig(num_uav=3, horizon=600, policy=SyncPolicy.ADAPTIVE,
                        tau_ref=tau_ref, adapt_age_steps=age_cap,
                        adapt_retention_drop=1.1)
    rows["adaptive"].append(
      run_mission(field, cfg, staleness_params=params, channel=channel, rng=seed + 31))

  def summarise(lst):
    hpc = np.array([r.hpc_pct for r in lst])
    coll = np.array([r.collision_rate for r in lst])
    up = np.array([r.uplink_cost for r in lst])
    m, lo, hi = _ci95(hpc)
    mc, clo, chi = _ci95(coll)
    return dict(hpc_mean=m, hpc_lo=lo, hpc_hi=hi,
                coll_mean=mc, coll_lo=clo, coll_hi=chi,
                uplink_mean=float(np.mean(up)), age_cap=age_cap)

  return dict(link=link, periodic=summarise(rows["periodic"]),
              adaptive=summarise(rows["adaptive"]), n_mc=n_mc)


def sensitivity_analysis(n_mc: int = 30,
                         seed_base: int = 9000) -> dict:
  """One-at-a-time sensitivity on beta_M, sigma_g, and hotspot fraction."""
  base = default_params()
  field_seed = 42
  bt = LINK_PRESETS["medium"].base_tau

  def run_with(params, hotspot_frac=0.12):
    hpc, coll = [], []
    for j in range(n_mc):
      field = build_priority_field(50, 50, seed=field_seed + j, hotspot_frac=hotspot_frac)
      channel = NTNChannel("medium", rng=seed_base + j, base_tau_override=bt)
      tau_ref = channel.expected_tau(2000)
      cfg = MissionConfig(num_uav=3, horizon=600, policy=SyncPolicy.PERIODIC, tau_ref=tau_ref)
      res = run_mission(field, cfg, staleness_params=params, channel=channel, rng=seed_base + j + 7)
      hpc.append(res.hpc_pct)
      coll.append(res.collision_rate)
    return float(np.mean(hpc)), float(np.mean(coll))

  results = {}

  # beta_M sweep (map fade)
  beta_vals = np.linspace(0.005, 0.04, 6)
  bh, bc = [], []
  for b in beta_vals:
    p = StalenessParams(beta_M=float(b), sigma_M=base.sigma_M, sigma_g=base.sigma_g)
    h, c = run_with(p)
    bh.append(h); bc.append(c)
  results["beta_M"] = dict(values=beta_vals.tolist(), hpc=bh, coll=bc)

  # sigma_g sweep (ghost drift)
  sg_vals = np.linspace(1.5, 5.0, 6)
  sh, sc = [], []
  for sg in sg_vals:
    p = StalenessParams(beta_M=base.beta_M, sigma_M=base.sigma_M, sigma_g=float(sg))
    h, c = run_with(p)
    sh.append(h); sc.append(c)
  results["sigma_g"] = dict(values=sg_vals.tolist(), hpc=sh, coll=sc)

  # hotspot fraction
  hf_vals = [0.08, 0.10, 0.12, 0.15, 0.18]
  hh, hc = [], []
  for hf in hf_vals:
    h, c = run_with(base, hotspot_frac=hf)
    hh.append(h); hc.append(c)
  results["hotspot_frac"] = dict(values=hf_vals, hpc=hh, coll=hc)

  return results


def operating_bounds(sweep: SweepResult,
                     hpc_thr: float = 65.0,
                     coll_thr: float = 0.40) -> dict:
  """Extract the largest mean tau that satisfies both KPI thresholds."""
  ok = (sweep.hpc_mean >= hpc_thr) & (sweep.coll_mean <= coll_thr)
  if not ok.any():
    return dict(feasible=False, max_tau=None, hpc_thr=hpc_thr, coll_thr=coll_thr)
  idx = np.where(ok)[0][-1]
  return dict(
    feasible=True,
    max_tau=float(sweep.tau_mean[idx]),
    hpc_at_bound=float(sweep.hpc_mean[idx]),
    coll_at_bound=float(sweep.coll_mean[idx]),
    hpc_thr=hpc_thr,
    coll_thr=coll_thr,
  )


# ---------------------------------------------------------------------- #
# Plotting
# ---------------------------------------------------------------------- #
def _setup_mpl():
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  return plt


def plot_tau_sweep(sweeps: dict[str, SweepResult], out_dir: Path):
  """Figure: HPC and collision rate vs. mean tau with 95 % confidence bands."""
  plt = _setup_mpl()
  out_dir.mkdir(parents=True, exist_ok=True)
  colors = {"good": "#2ca02c", "medium": "#ff7f0e", "poor": "#d62728"}

  fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

  for link, sw in sweeps.items():
    c = colors.get(link, "C0")
    axes[0].plot(sw.tau_mean, sw.hpc_mean, "o-", color=c, label=link, lw=2)
    axes[0].fill_between(sw.tau_mean, sw.hpc_lo, sw.hpc_hi, color=c, alpha=0.15)
    axes[1].plot(sw.tau_mean, sw.coll_mean, "s-", color=c, label=link, lw=2)
    axes[1].fill_between(sw.tau_mean, sw.coll_lo, sw.coll_hi, color=c, alpha=0.15)

  axes[0].axhline(65, ls="--", color="gray", lw=1, label="HPC bound (65 %)")
  axes[1].axhline(0.4, ls="--", color="gray", lw=1, label="collision bound (0.4)")
  axes[0].set_xlabel(r"Mean synchronization interval $\bar{\tau}$ (steps)")
  axes[0].set_ylabel("High-priority coverage HPC (%)")
  axes[0].set_title("(a) HPC vs. mean $\\bar{\\tau}$")
  axes[0].legend(fontsize=9)
  axes[0].grid(True, alpha=0.3)

  axes[1].set_xlabel(r"Mean synchronization interval $\bar{\tau}$ (steps)")
  axes[1].set_ylabel("Collision rate (fraction of steps)")
  axes[1].set_title("(b) Collision rate vs. mean $\\bar{\\tau}$")
  axes[1].legend(fontsize=9)
  axes[1].grid(True, alpha=0.3)

  fig.tight_layout()
  for ext in ("png", "pdf"):
    fig.savefig(out_dir / f"Figure_Staleness_TauSweep.{ext}", dpi=200, bbox_inches="tight")
  plt.close(fig)


def plot_policy_comparison(comparisons: dict, out_dir: Path):
  """Figure: periodic vs. adaptive at matched uplink cost."""
  plt = _setup_mpl()
  out_dir.mkdir(parents=True, exist_ok=True)

  links = list(comparisons.keys())
  x = np.arange(len(links))
  w = 0.35

  fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

  for ax, prefix, ylab, title in [
    (axes[0], "hpc", "HPC (%)", "(a) HPC: periodic vs. adaptive"),
    (axes[1], "coll", "Collision rate", "(b) Collision: periodic vs. adaptive"),
  ]:
    per_m = [comparisons[l]["periodic"][f"{prefix}_mean"] for l in links]
    ada_m = [comparisons[l]["adaptive"][f"{prefix}_mean"] for l in links]
    per_lo = [comparisons[l]["periodic"][f"{prefix}_lo"] for l in links]
    per_hi = [comparisons[l]["periodic"][f"{prefix}_hi"] for l in links]
    ada_lo = [comparisons[l]["adaptive"][f"{prefix}_lo"] for l in links]
    ada_hi = [comparisons[l]["adaptive"][f"{prefix}_hi"] for l in links]

    ax.bar(x - w / 2, per_m, w, label="Periodic", color="#1f77b4", alpha=0.85)
    ax.bar(x + w / 2, ada_m, w, label="Adaptive (matched cost)", color="#ff7f0e", alpha=0.85)
    ax.errorbar(x - w / 2, per_m,
                yerr=[np.array(per_m) - np.array(per_lo), np.array(per_hi) - np.array(per_m)],
                fmt="none", color="k", capsize=3)
    ax.errorbar(x + w / 2, ada_m,
                yerr=[np.array(ada_m) - np.array(ada_lo), np.array(ada_hi) - np.array(ada_m)],
                fmt="none", color="k", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(links)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

  fig.tight_layout()
  for ext in ("png", "pdf"):
    fig.savefig(out_dir / f"Figure_Staleness_PolicyComparison.{ext}", dpi=200, bbox_inches="tight")
  plt.close(fig)


def plot_sensitivity(sens: dict, out_dir: Path):
  """Figure: one-at-a-time parameter sensitivity."""
  plt = _setup_mpl()
  out_dir.mkdir(parents=True, exist_ok=True)

  fig, axes = plt.subplots(2, 3, figsize=(12, 7))
  panels = [
    ("beta_M", r"$\beta_M$ (map fade)", axes[0, 0], axes[1, 0]),
    ("sigma_g", r"$\sigma_g$ (ghost drift)", axes[0, 1], axes[1, 1]),
    ("hotspot_frac", "Hotspot fraction", axes[0, 2], axes[1, 2]),
  ]
  for key, xlab, ax_h, ax_c in panels:
    d = sens[key]
    xs = d["values"]
    ax_h.plot(xs, d["hpc"], "o-", color="#2ca02c", lw=2)
    ax_h.set_xlabel(xlab)
    ax_h.set_ylabel("HPC (%)")
    ax_h.grid(True, alpha=0.3)
    ax_c.plot(xs, d["coll"], "s-", color="#d62728", lw=2)
    ax_c.set_xlabel(xlab)
    ax_c.set_ylabel("Collision rate")
    ax_c.grid(True, alpha=0.3)

  fig.suptitle("One-at-a-time sensitivity (medium link, N=30)", fontsize=12)
  fig.tight_layout()
  for ext in ("png", "pdf"):
    fig.savefig(out_dir / f"Figure_Staleness_Sensitivity.{ext}", dpi=200, bbox_inches="tight")
  plt.close(fig)


def plot_calibration_curve(params: StalenessParams, out_dir: Path):
  """Figure: ghost RMSE and map retention vs. tau (calibration curves)."""
  plt = _setup_mpl()
  out_dir.mkdir(parents=True, exist_ok=True)
  taus = np.arange(10, 141, 5, dtype=float)

  from .staleness import StalenessModel
  sm = StalenessModel(params)
  ghost_rmse = [sm.ghost_rmse(t) for t in taus]
  ret_curve = [retention(t, params.beta_M) for t in taus]
  kappas = [float(channel_kappa(t)) for t in taus]

  fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
  axes[0].plot(taus, ghost_rmse, "b-", lw=2)
  axes[0].axhline(10, ls="--", color="gray", label="target 10 cells")
  axes[0].axvline(60, ls=":", color="gray", label=r"$\tau=60$")
  axes[0].set_xlabel(r"$\tau$ (steps)")
  axes[0].set_ylabel("Ghost RMSE (cells)")
  axes[0].legend(fontsize=8)
  axes[0].grid(True, alpha=0.3)

  axes[1].plot(taus, ret_curve, "g-", lw=2)
  axes[1].axhline(0.6, ls="--", color="gray", label="target 0.6")
  axes[1].axvline(60, ls=":", color="gray")
  axes[1].set_xlabel(r"$\Delta$ (steps)")
  axes[1].set_ylabel(r"$R(\Delta)=\exp(-\beta_M \Delta)$")
  axes[1].legend(fontsize=8)
  axes[1].grid(True, alpha=0.3)

  axes[2].plot(taus, kappas, "m-", lw=2)
  axes[2].set_xlabel(r"$\tau$ (steps)")
  axes[2].set_ylabel(r"$\kappa(\tau) = (\tau-1)/2$")
  axes[2].grid(True, alpha=0.3)

  fig.suptitle("Staleness calibration curves", fontsize=12)
  fig.tight_layout()
  for ext in ("png", "pdf"):
    fig.savefig(out_dir / f"Figure_Staleness_Calibration.{ext}", dpi=200, bbox_inches="tight")
  plt.close(fig)


def run_full_study(out_dir: str | Path = "output",
                   n_mc: int = 50,
                   quick: bool = False) -> dict:
  """Execute the complete Monte-Carlo study and write figures + JSON summary."""
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  if quick:
    tau_grid = TAU_SWEEP_QUICK
  else:
    tau_grid = TAU_SWEEP_GRID

  params = default_params()
  summary = dict(
    params=dict(beta_M=params.beta_M, sigma_M=params.sigma_M, sigma_g=params.sigma_g),
    calibration_tau_ref=60,
    ghost_rmse_target_cells=10.0,
    map_retention_target=0.60,
    n_mc=n_mc,
  )

  # Calibration figure
  plot_calibration_curve(params, out_dir)

  # Tau sweeps per link quality
  sweeps = {}
  bounds = {}
  for link in ("good", "medium", "poor"):
    print(f"[sweep] link={link}  N={n_mc}")
    sw = sweep_tau(link=link, tau_grid=tau_grid, n_mc=n_mc, params=params)
    sweeps[link] = sw
    bounds[link] = operating_bounds(sw)
    print(f"  operating bound: {bounds[link]}")

  plot_tau_sweep(sweeps, out_dir)
  summary["operating_bounds"] = bounds

  # Policy comparison
  comparisons = {}
  for link in ("good", "medium", "poor"):
    print(f"[policy] link={link}  N={n_mc}")
    comparisons[link] = compare_policies(link=link, n_mc=n_mc, params=params)
  plot_policy_comparison(comparisons, out_dir)
  summary["policy_comparison"] = comparisons

  # Sensitivity (lighter MC for speed)
  print("[sensitivity]")
  sens = sensitivity_analysis(n_mc=min(20, n_mc))
  plot_sensitivity(sens, out_dir)
  summary["sensitivity"] = sens

  # Aggregate sweep statistics for JSON export
  summary["tau_sweeps"] = {
    link: dict(
      tau_mean=sw.tau_mean.tolist(),
      hpc_mean=sw.hpc_mean.tolist(),
      coll_mean=sw.coll_mean.tolist(),
    )
    for link, sw in sweeps.items()
  }

  with open(out_dir / "staleness_study_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

  print(f"\nStudy complete. Outputs in {out_dir.resolve()}")
  return summary
