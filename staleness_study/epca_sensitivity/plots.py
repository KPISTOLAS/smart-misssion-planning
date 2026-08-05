"""Sensitivity study figures and LaTeX export."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .analysis import latex_table_hotspot_u, latex_table_priority_quality, insight_sentences


def _save(fig, stem: Path):
    fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_tau_bound_heatmap(summary: dict, out: Path,
                           x_key: str = "d_safe",
                           title: str = "Operating τ bound"):
    """Heatmap of max τ vs two swept parameters (uses operating_bound from summary)."""
    data = summary.get(x_key, {})
    if not data:
        return
    labels = list(data.keys())
    taus = []
    for k in labels:
        b = data[k]["operating_bound"]
        taus.append(b.get("max_tau") if b.get("feasible") else np.nan)
    fig, ax = plt.subplots(figsize=(8, 2.5))
    arr = np.array(taus).reshape(1, -1)
    im = ax.imshow(arr, aspect="auto", cmap="YlGn", vmin=0, vmax=np.nanmax(arr) if np.any(~np.isnan(arr)) else 60)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks([0])
    ax.set_yticklabels([x_key])
    for j, v in enumerate(taus):
        txt = f"{v:.0f}" if not np.isnan(v) else "—"
        ax.text(j, 0, txt, ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im, ax=ax, label=r"$\bar{\tau}_{\max}$ (steps)")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, out / f"Figure_Sensitivity_TauBound_{x_key}")


def plot_hotspot_density_tau_bounds(summary: dict, out: Path):
    """Bar chart: operating τ bound vs hotspot density."""
    hd = summary.get("hotspot_density", {})
    if not hd:
        return
    labels, taus, hpc_at = [], [], []
    for k in ("low", "medium", "high"):
        if k not in hd:
            continue
        b = hd[k]["operating_bound"]
        labels.append(k)
        taus.append(b.get("max_tau") if b.get("feasible") else 0)
        hpc_at.append(b.get("hpc_at_bound", 0))
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(x - 0.2, taus, 0.4, label=r"$\bar{\tau}_{\max}$", color="#3498db")
    ax1.set_ylabel(r"Operating $\bar{\tau}$ bound (steps)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, hpc_at, 0.4, label="HPC@bound (%)", color="#e74c3c", alpha=0.7)
    ax2.set_ylabel("HPC at bound (%)")
    ax1.set_xlabel("Hotspot density")
    ax1.set_title("Operating τ bound vs. hotspot density")
    fig.tight_layout()
    _save(fig, out / "Figure_Sensitivity_HotspotTauBound")


def plot_fleet_size_tau_bounds(summary: dict, out: Path):
    fs = summary.get("fleet_size", {})
    if not fs:
        return
    us = sorted(int(k) for k in fs)
    taus = [fs[str(u)]["operating_bound"].get("max_tau", 0)
            if fs[str(u)]["operating_bound"].get("feasible") else 0 for u in us]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(us, taus, "o-", lw=2, color="#2ecc71")
    ax.set_xlabel("Fleet size $U$")
    ax.set_ylabel(r"Operating $\bar{\tau}_{\max}$ (steps)")
    ax.set_title("Operating τ bound vs. fleet size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out / "Figure_Sensitivity_FleetTauBound")


def plot_d_safe_tau_heatmap(summary: dict, out: Path):
    """τ bound vs d_safe — heatmap strip."""
    plot_tau_bound_heatmap(
        summary, out, x_key="d_safe",
        title=r"Operating $\bar{\tau}$ bound vs. safety distance $d_{\mathrm{safe}}$",
    )


def plot_priority_quality(summary: dict, out: Path):
    pq = summary.get("priority_quality", {})
    if not pq:
        return
    labels = list(pq.keys())
    hpc = [pq[k]["hpc_mean"] for k in labels]
    coll = [pq[k]["coll_mean"] for k in labels]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(x, hpc, color=["#27ae60", "#f39c12", "#c0392b"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Perfect", "Mild noise", "Severe noise"], rotation=15)
    axes[0].set_ylabel("HPC (%)")
    axes[0].set_title("(a) Coverage under imperfect $W_i$")
    axes[1].bar(x, [pq[k].get("targeting_error_mean", 0) for k in labels], color=["#27ae60", "#f39c12", "#c0392b"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Perfect", "Mild noise", "Severe noise"], rotation=15)
    axes[1].set_ylabel("Targeting error (%)")
    axes[1].set_title("(b) Mis-targeted visits (planner-high, not true hotspot)")
    fig.suptitle("Planner robustness: perfect vs. noisy priority fields", fontsize=11)
    fig.tight_layout()
    _save(fig, out / "Figure_Sensitivity_PriorityQuality")


def plot_tau_sweep_panel(summary: dict, out: Path, section: str = "hotspot_density"):
    """Multi-panel HPC vs τ for a sensitivity section."""
    data = summary.get(section, {})
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}
    for k, v in data.items():
        tau = v["tau_mean"]
        hpc = v["hpc_mean"]
        c = colors.get(k, None)
        ax.plot(tau, hpc, "o-", label=k, color=c, lw=2)
    ax.axhline(65, ls="--", color="gray", lw=1, label="HPC bound")
    ax.set_xlabel(r"Mean $\bar{\tau}$ (steps)")
    ax.set_ylabel("HPC (%)")
    ax.set_title(f"HPC vs. $\\bar{{\\tau}}$ ({section})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out / f"Figure_Sensitivity_TauSweep_{section}")


def generate_all_figures(summary: dict, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_hotspot_density_tau_bounds(summary, out_dir)
    plot_fleet_size_tau_bounds(summary, out_dir)
    plot_d_safe_tau_heatmap(summary, out_dir)
    plot_priority_quality(summary, out_dir)
    plot_tau_sweep_panel(summary, out_dir, "hotspot_density")

    # LaTeX tables
    tex = latex_table_hotspot_u(summary)
    tex += "\n\n" + latex_table_priority_quality(summary)
    (out_dir / "Table_Sensitivity.tex").write_text(tex)

    insights = insight_sentences(summary)
    (out_dir / "sensitivity_insights.txt").write_text("\n".join(f"• {s}" for s in insights))
