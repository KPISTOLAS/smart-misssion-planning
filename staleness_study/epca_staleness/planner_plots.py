"""Publication figures for planner evaluation (Section V)."""

from __future__ import annotations

from pathlib import Path
import numpy as np


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_swarm_size_sweep(sweep: dict, out_dir: Path, metric: str = "hpc"):
    """Swarm size U vs HPC / collision / duration with error bars."""
    plt = _mpl()
    out_dir.mkdir(parents=True, exist_ok=True)
    U_vals = sorted(sweep.keys())
    planners = list(next(iter(sweep.values())).keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(planners)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    panels = [("hpc_mean", "hpc_lo", "hpc_hi", "HPC (%)", "(a)"),
              ("collision_mean", None, None, "Collision rate", "(b)"),
              ("duration_mean", None, None, "Mission duration (steps)", "(c)")]

    for ax, (key, lo, hi, ylab, title) in zip(axes, panels):
        for i, p in enumerate(planners):
            ys = [getattr(sweep[U][p], key) for U in U_vals]
            ax.plot(U_vals, ys, "o-", color=colors[i], label=p, lw=1.8, ms=5)
            if lo and hi:
                ylo = [getattr(sweep[U][p], lo) for U in U_vals]
                yhi = [getattr(sweep[U][p], hi) for U in U_vals]
                ax.fill_between(U_vals, ylo, yhi, color=colors[i], alpha=0.12)
        ax.set_xlabel("Fleet size U")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.legend(fontsize=7, ncol=2, loc="lower right")

    fig.suptitle("Swarm size sensitivity (N=50, 50×50 grid)", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"Figure_Planner_SwarmSize.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_bars(ablation: dict, out_dir: Path):
    """Ablation comparison bar chart."""
    plt = _mpl()
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(ablation.keys())
    labels = [n.replace("ablation_", "").replace("iuef_em", "Full IUEF-EM") for n in names]
    hpc = [ablation[n].hpc_mean for n in names]
    hpc_err = [ablation[n].hpc_std for n in names]
    coll = [ablation[n].collision_mean for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(names))
    axes[0].bar(x, hpc, yerr=hpc_err, capsize=4, color="#2ca02c", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    axes[0].set_ylabel("HPC (%)")
    axes[0].set_title("(a) High-priority coverage")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, coll, color="#d62728", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    axes[1].set_ylabel("Collision rate")
    axes[1].set_title("(b) Congestion / collision")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("IUEF-EM ablation study (U=3, N=50)", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"Figure_Planner_Ablation.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_bars(baselines: dict, out_dir: Path):
    """External baseline comparison."""
    plt = _mpl()
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(baselines.keys())
    hpc = [baselines[n].hpc_mean for n in names]
    coll = [baselines[n].collision_mean for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, hpc, w, label="HPC (%)", color="#2ca02c", alpha=0.85)
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, [c * 100 for c in coll], w, label="Collision (%)", color="#d62728", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("HPC (%)")
    ax2.set_ylabel("Collision rate (%)")
    ax.set_title("Planner comparison: IUEF-EM vs baselines (U=3, N=50)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"Figure_Planner_Baselines.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_staleness_planner(sweep: dict, out_dir: Path):
    """Planner HPC / collision vs mean tau under staleness."""
    plt = _mpl()
    out_dir.mkdir(parents=True, exist_ok=True)
    taus = sorted(sweep.keys())
    planners = list(next(iter(sweep.values())).keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(planners)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, p in enumerate(planners):
        hpc = [sweep[t][p].hpc_mean for t in taus]
        coll = [sweep[t][p].collision_mean for t in taus]
        axes[0].plot(taus, hpc, "o-", color=colors[i], label=p, lw=2)
        axes[1].plot(taus, coll, "s-", color=colors[i], label=p, lw=2)
    axes[0].set_xlabel(r"Mean synchronization interval $\bar{\tau}$ (steps)")
    axes[0].set_ylabel("HPC (%)")
    axes[0].set_title("(a) HPC vs staleness")
    axes[1].set_xlabel(r"Mean $\bar{\tau}$ (steps)")
    axes[1].set_ylabel("Collision rate")
    axes[1].set_title("(b) Congestion vs staleness")
    for ax in axes:
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Planner performance under calibrated staleness", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"Figure_Planner_Staleness.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_ablation_latex_table(ablation: dict, out_dir: Path):
    """LaTeX-ready ablation table."""
    rows = []
    for name, s in ablation.items():
        label = name.replace("_", r"\_")
        rows.append(
            f"{label} & {s.hpc_mean:.1f} $\\pm$ {s.hpc_std:.1f} & "
            f"{s.coverage_mean:.1f} & {s.duration_mean:.0f} & "
            f"{s.collision_mean:.3f} $\\pm$ {s.collision_std:.3f} \\\\"
        )
    tex = (
        "\\begin{table}[t]\n\\centering\n\\caption{IUEF-EM ablation study "
        "(50$\\times$50, $U{=}3$, $N{=}50$).}\n"
        "\\begin{tabular}{lcccc}\n\\hline\n"
        "Variant & HPC (\\%) & Coverage (\\%) & Duration & Collision \\\\\n\\hline\n"
        + "\n".join(rows) +
        "\n\\hline\n\\end{tabular}\n\\end{table}\n"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "Table_Ablation.tex").write_text(tex)
