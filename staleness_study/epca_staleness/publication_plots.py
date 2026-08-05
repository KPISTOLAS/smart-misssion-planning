"""Publication figures: regime separation, staleness ablations, operating envelope."""

from __future__ import annotations

from pathlib import Path
import numpy as np


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_regime_comparison(summary: dict, out_dir: Path):
    """Three-regime summary with regime-appropriate primary metrics."""
    plt = _mpl()
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # R1: secondary KPIs (HPC saturates)
    if "R1" in summary and "baselines" in summary["R1"]:
        b = summary["R1"]["baselines"]
        names = list(b.keys())[:6]
        energy = [b[n]["energy_mean"] for n in names]
        dur = [b[n]["duration_mean"] for n in names]
        coll = [b[n]["collision_mean"] * 100 for n in names]
        x = np.arange(len(names))
        axes[0].bar(x - 0.2, energy, 0.2, label="Energy (J/UAV·h)", color="#ff7f0e")
        axes[0].bar(x, dur, 0.2, label="Duration (steps)", color="#1f77b4")
        axes[0].bar(x + 0.2, coll, 0.2, label="Coll×100", color="#d62728")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
        axes[0].set_title("R1: Planner-perfect\n(secondary KPIs)")
        axes[0].legend(fontsize=6)

    # R2 ablation at tau=50
    if "R2" in summary:
        tau_key = "50" if "50" in summary["R2"].get("tau_sweep", {}) else list(summary["R2"]["tau_sweep"].keys())[0]
        ab = summary["R2"]["tau_sweep"].get(tau_key, {})
        names = list(ab.keys())
        labels = [n.replace("ablation_", "").replace("iuef_em", "Full") for n in names]
        hpc = [ab[n]["hpc_mean"] for n in names]
        coll = [ab[n]["collision_mean"] for n in names]
        ret = [ab[n]["retained_mean"] for n in names]
        x = np.arange(len(names))
        axes[1].bar(x - 0.25, hpc, 0.25, label="HPC", color="#2ca02c")
        axes[1].bar(x, [c * 100 for c in coll], 0.25, label="Coll×100", color="#d62728")
        axes[1].bar(x + 0.25, [r * 100 for r in ret], 0.25, label="Retained×100", color="#1f77b4")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, fontsize=7, rotation=25, ha="right")
        axes[1].set_title(f"R2: Staleness (τ̄={tau_key})")
        axes[1].legend(fontsize=6)

    # R3 closed-loop: targeting error + collision (HPC often plateaus)
    if "R3" in summary:
        modes = summary["R3"].get("modes", {})
        names = list(modes.keys())
        mae = [modes[n].get("inference_mae_mean", 0) for n in names]
        coll = [modes[n]["coll_mean"] for n in names]
        hpc = [modes[n]["hpc_mean"] for n in names]
        x = np.arange(len(names))
        axes[2].bar(x - 0.2, mae, 0.2, label="Targeting MAE", color="#9467bd")
        axes[2].bar(x, [c * 100 for c in coll], 0.2, label="Coll×100", color="#d62728")
        axes[2].bar(x + 0.2, hpc, 0.2, label="HPC", color="#2ca02c", alpha=0.7)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(["closed", "perfect", "no_stal"], fontsize=8)
        axes[2].set_title("R3: Closed-loop\n(targeting error + safety)")
        axes[2].legend(fontsize=6)

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Three evaluation regimes (report separately)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"Figure_Regime_Comparison.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_staleness_secondary(summary: dict, out_dir: Path):
    """Secondary KPIs where HPC saturates: collision, retained, uplink."""
    plt = _mpl()
    out_dir = Path(out_dir)
    if "R2" not in summary:
        return
    tau_sweep = summary["R2"]["tau_sweep"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = [
        ("hpc_mean", "HPC (%)", axes[0, 0]),
        ("collision_mean", "Collision rate", axes[0, 1]),
        ("retained_mean", "Retained hotspot frac", axes[1, 0]),
        ("uplink_mean", "Uplink cost (syncs/step)", axes[1, 1]),
        ("duration_mean", "Mission duration (steps)", None),
    ]
    colors = plt.cm.Set2(np.linspace(0, 1, 5))
    for tau_str, ab in tau_sweep.items():
        names = list(ab.keys())
        short = [n.replace("ablation_", "").replace("iuef_em", "Full") for n in names]
        for key, ylab, ax in metrics:
            if ax is None:
                continue
            vals = [ab[n][key] for n in names]
            off = (float(tau_str) - 50) / 30 * 0.15
            x = np.arange(len(names)) + off
            ax.bar(x, vals, width=0.12, label=f"τ̄={tau_str}", alpha=0.85)
            ax.set_xticks(np.arange(len(names)))
            ax.set_xticklabels(short, fontsize=7, rotation=20, ha="right")
            ax.set_ylabel(ylab)
            ax.grid(True, axis="y", alpha=0.3)
    for ax in axes.flat:
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("IUEF-EM ablations under moderate staleness (secondary KPIs)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"Figure_Ablation_Staleness_Secondary.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_operating_envelope(summary: dict, out_dir: Path):
    """Operating envelope heatmap: factor × setting → τ_max."""
    plt = _mpl()
    out_dir = Path(out_dir)
    env = summary.get("operating_envelope", {})
    bounds = env.get("bounds", [])
    if not bounds:
        return

    factors = sorted(set(b["factor"] for b in bounds if b.get("tau_max_steps") is not None))
    fig, axes = plt.subplots(1, 2, figsize=(12, max(3, len(factors) * 0.8 + 1)))

    # Panel A: tau_max heatmap strip per factor
    ax = axes[0]
    rows, labels, vals = [], [], []
    for fac in factors:
        row = [b for b in bounds if b["factor"] == fac and b.get("feasible")]
        if not row:
            continue
        for b in row:
            rows.append(f"{fac}:{b['setting']}")
            vals.append(b.get("tau_max_steps") or 0)
            labels.append(str(b["setting"]))
    if vals:
        arr = np.array(vals).reshape(-1, 1)
        im = ax.imshow(arr, aspect="auto", cmap="YlGn", vmin=0, vmax=max(vals) if vals else 60)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=8)
        ax.set_xticks([0])
        ax.set_xticklabels([r"$\bar{\tau}_{\max}$ (steps)"])
        for i, v in enumerate(vals):
            ax.text(0, i, f"{v:.0f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.02)

    # Panel B: HPC vs collision at bound
    ax = axes[1]
    feas = [b for b in bounds if b.get("feasible") and b.get("hpc_at_bound")]
    if feas:
        hp = [b["hpc_at_bound"] for b in feas]
        cp = [b["coll_at_bound"] * 100 for b in feas]
        lbl = [f"{b['factor'][:4]}:{b['setting']}" for b in feas]
        sc = ax.scatter(cp, hp, s=80, c=[b.get("tau_max_steps", 0) for b in feas], cmap="viridis")
        for i, lb in enumerate(lbl):
            ax.annotate(lb, (cp[i], hp[i]), fontsize=6, alpha=0.8)
        ax.axhline(65, ls="--", color="gray", lw=1)
        ax.axvline(40, ls="--", color="gray", lw=1)
        ax.set_xlabel("Collision @ bound (%)")
        ax.set_ylabel("HPC @ bound (%)")
        ax.set_title("Operating envelope (feasible configs)")
        fig.colorbar(sc, ax=ax, label=r"$\bar{\tau}_{\max}$")

    fig.suptitle("Integrated operating envelope (HPC≥65%, coll<0.4)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"Figure_Operating_Envelope.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_operating_envelope_table(summary: dict, out_dir: Path) -> str:
    env = summary.get("operating_envelope", {})
    bounds = env.get("bounds", [])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Integrated operating envelope: largest $\bar{\tau}$ (steps) with HPC$\geq$65\% and collision$<$0.4. "
        r"Regimes R1--R3 must not be merged.}",
        r"\label{tab:operating_envelope}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Factor & Setting & $\bar{\tau}_{\max}$ & HPC@bound & Coll@bound & Regime & Feasible \\",
        r"\midrule",
    ]
    for b in bounds:
        tau = b.get("tau_max_steps")
        tau_s = f"{tau:.0f}" if tau is not None else "---"
        hpc = f"{b['hpc_at_bound']:.1f}" if b.get("hpc_at_bound") is not None else "---"
        coll = f"{b['coll_at_bound']:.3f}" if b.get("coll_at_bound") is not None else "---"
        feas = "Yes" if b.get("feasible") else "No"
        lines.append(
            f"{b['factor']} & {b['setting']} & {tau_s} & {hpc} & {coll} & {b.get('regime', '')} & {feas} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    tex = "\n".join(lines)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "Table_Operating_Envelope.tex").write_text(tex)
    return tex


def write_regime_ablation_table(summary: dict, out_dir: Path):
    """LaTeX: ablation under staleness τ=50 with secondary KPIs."""
    if "R2" not in summary:
        return
    tau_key = "50" if "50" in summary["R2"]["tau_sweep"] else list(summary["R2"]["tau_sweep"].keys())[0]
    ab = summary["R2"]["tau_sweep"][tau_key]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{IUEF-EM ablations under moderate staleness ($\bar{{\tau}}={tau_key}$, N=50). "
        r"Secondary KPIs discriminate where HPC does not.}}",
        r"\label{tab:ablation_staleness}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Variant & HPC (\%) & Coll. & Near-miss & Retained & Uplink & Syncs & Duration \\",
        r"\midrule",
    ]
    labels = {"iuef_em": "Full IUEF-EM", "ablation_no_balance": "No balance",
              "ablation_no_congestion": "No cong.", "ablation_no_priority": "No priority",
              "ablation_no_astar": "No A*"}
    for k, v in ab.items():
        lbl = labels.get(k, k)
        lines.append(
            f"{lbl} & {v['hpc_mean']:.1f} & {v['collision_mean']:.3f} & "
            f"{v['near_miss_mean']:.3f} & {v['retained_mean']:.2f} & "
            f"{v['uplink_mean']:.4f} & {v['n_syncs_mean']:.1f} & {v.get('duration_mean', 0):.0f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (Path(out_dir) / "Table_Ablation_Staleness.tex").write_text("\n".join(lines))


def generate_publication_figures(summary: dict, out_dir: Path):
    plot_regime_comparison(summary, out_dir)
    plot_ablation_staleness_secondary(summary, out_dir)
    plot_operating_envelope(summary, out_dir)
    write_operating_envelope_table(summary, out_dir)
    write_regime_ablation_table(summary, out_dir)
