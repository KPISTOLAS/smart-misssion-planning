"""Publication figures for the closed-loop EPCA-M simulator."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from epca_staleness.environment import build_priority_field
from epca_staleness.experiments import default_params
from epca_staleness.channel import NTNChannel

from .closed_loop import ClosedLoopConfig, SyncPolicy, run_closed_loop
from .experiments import SweepResult, compare_modes


def _save(fig, stem: Path):
    fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_tau_sweep(sweep_on: SweepResult,
                   sweep_off: SweepResult | None,
                   out: Path):
    """HPC and collision vs mean tau (with/without staleness)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(sweep_on.x, sweep_on.hpc_mean, "o-", label="Closed-loop + staleness", color="#c0392b")
    ax.fill_between(sweep_on.x, sweep_on.hpc_lo, sweep_on.hpc_hi, alpha=0.2, color="#c0392b")
    if sweep_off is not None:
        ax.plot(sweep_off.x, sweep_off.hpc_mean, "s--", label="Inference only (no staleness)", color="#2980b9")
        ax.fill_between(sweep_off.x, sweep_off.hpc_lo, sweep_off.hpc_hi, alpha=0.15, color="#2980b9")
    ax.set_xlabel("Mean synchronization interval $\\bar{\\tau}$ (steps)")
    ax.set_ylabel("HPC (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("(a) High-priority coverage")

    ax = axes[1]
    ax.plot(sweep_on.x, sweep_on.coll_mean, "o-", color="#c0392b")
    ax.fill_between(sweep_on.x, sweep_on.coll_lo, sweep_on.coll_hi, alpha=0.2, color="#c0392b")
    if sweep_off is not None:
        ax.plot(sweep_off.x, sweep_off.coll_mean, "s--", color="#2980b9")
        ax.fill_between(sweep_off.x, sweep_off.coll_lo, sweep_off.coll_hi, alpha=0.15, color="#2980b9")
    ax.set_xlabel("Mean synchronization interval $\\bar{\\tau}$ (steps)")
    ax.set_ylabel("Collision rate")
    ax.grid(True, alpha=0.3)
    ax.set_title("(b) Collision rate")
    fig.suptitle("Closed-loop EPCA-M: staleness impact on mission KPIs", fontsize=11)
    fig.tight_layout()
    _save(fig, out / "Figure_ClosedLoop_TauSweep")


def plot_mode_comparison(modes: dict, out: Path, title: str = "Closed-loop vs baselines"):
    """Bar chart: closed-loop, perfect-info, no-staleness."""
    names = list(modes.keys())
    hpc = [modes[n]["hpc_mean"] for n in names]
    hpc_err = [(modes[n]["hpc_hi"] - modes[n]["hpc_lo"]) / 2 for n in names]
    coll = [modes[n]["coll_mean"] for n in names]
    coll_err = [(modes[n]["coll_hi"] - modes[n]["coll_lo"]) / 2 for n in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(x, hpc, yerr=hpc_err, capsize=4, color=["#c0392b", "#27ae60", "#2980b9"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Closed-loop", "Perfect info", "No staleness"], rotation=15, ha="right")
    axes[0].set_ylabel("HPC (%)")
    axes[0].set_title("(a) Coverage")
    axes[1].bar(x, coll, yerr=coll_err, capsize=4, color=["#c0392b", "#27ae60", "#2980b9"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Closed-loop", "Perfect info", "No staleness"], rotation=15, ha="right")
    axes[1].set_ylabel("Collision rate")
    axes[1].set_title("(b) Safety")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _save(fig, out / "Figure_ClosedLoop_ModeComparison")


def plot_pipeline_evolution(out: Path, seed: int = 42):
    """Suggested Figure: priority field evolution + trajectories under staleness."""
    field = build_priority_field(50, 50, seed=seed)
    params = default_params(rng=seed)
    channel = NTNChannel("medium", rng=seed + 17, base_tau_override=35.0)
    cfg = ClosedLoopConfig(horizon=200, policy=SyncPolicy.PERIODIC, tau_ref=channel.expected_tau(1000))
    result = run_closed_loop(field, cfg, staleness_params=params, channel=channel,
                             rng=seed + 31, record_snapshots=True)

    if not result.W_snapshots or not result.traj_snapshots:
        return

    n_panels = min(3, len(result.W_snapshots))
    idx = np.linspace(0, len(result.W_snapshots) - 1, n_panels, dtype=int)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    for ax, k in zip(axes, idx):
        W = result.W_snapshots[k]
        pos = result.traj_snapshots[k]
        im = ax.imshow(W.T, origin="lower", cmap="YlOrRd", vmin=0, vmax=W.max())
        ax.contour(field.obstacle.T, levels=[0.5], colors="k", linewidths=0.5)
        for u in range(len(pos)):
            ax.plot(pos[u, 1], pos[u, 0], "o", color=["#2ecc71", "#3498db", "#9b59b6"][u % 3], ms=8)
        ax.set_title(f"Sync event {k+1}")
        ax.set_xlabel("col"); ax.set_ylabel("row")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, label="$\\hat{W}_i$ (degraded belief)")
    fig.suptitle("Closed-loop pipeline under staleness: belief evolution & UAV positions", fontsize=11)
    fig.subplots_adjust(right=0.88)
    _save(fig, out / "Figure_ClosedLoop_PipelineEvolution")


def generate_all_figures(summary: dict, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s_on = summary.get("tau_sweep_staleness", {})
    s_off = summary.get("tau_sweep_no_staleness", {})
    if s_on:
        sweep_on = SweepResult(
            x=np.array(s_on["x"]), hpc_mean=np.array(s_on["hpc_mean"]),
            hpc_lo=np.array(s_on["hpc_lo"]), hpc_hi=np.array(s_on["hpc_hi"]),
            coll_mean=np.array(s_on["coll_mean"]), coll_lo=np.array(s_on["coll_lo"]),
            coll_hi=np.array(s_on["coll_hi"]), raw=s_on.get("raw", {}),
        )
        sweep_off = None
        if s_off:
            sweep_off = SweepResult(
                x=np.array(s_off["x"]), hpc_mean=np.array(s_off["hpc_mean"]),
                hpc_lo=np.array(s_off["hpc_lo"]), hpc_hi=np.array(s_off["hpc_hi"]),
                coll_mean=np.array(s_off["coll_mean"]), coll_lo=np.array(s_off["coll_lo"]),
                coll_hi=np.array(s_off["coll_hi"]), raw=s_off.get("raw", {}),
            )
        plot_tau_sweep(sweep_on, sweep_off, out_dir)
    if summary.get("mode_comparison_medium"):
        plot_mode_comparison(summary["mode_comparison_medium"], out_dir)
    plot_pipeline_evolution(out_dir)
