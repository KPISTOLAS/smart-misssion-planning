"""NTN link-budget module: slant range, FSPL, SNR CDF, outage probability, implied τ̄.

Computes link statistics from first principles (TR 38.811 NTN-Suburban LMS with
shadowed/blocked states, canopy blockage). Monte Carlo over elevation angles for
each link class. SNR follows::

    SNR = EIRP − L_FSPL + G/T + k₀ − 10 log₁₀(B) + shadow + fading − L_impl − L_atm

where ``G/T`` is the receiver figure of merit (dB/K) and ``k₀ = −10 log₁₀(k_B)``
(+228.6 dBW/K/Hz). Noise is **not** subtracted again via ``kT_sys B`` and NF when
``G/T`` is already referenced to system noise temperature.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
import numpy as np


C_MPS = 299_792_458.0
K_B = 1.380649e-23
K0_DB = -10.0 * np.log10(K_B)  # +228.599 dBW/K/Hz (Boltzmann constant term)


@dataclass
class LinkBudgetInputs:
    f_c_hz: float = 2.0e9
    bandwidth_hz: float = 10e6
    h_sat_m: float = 600_000.0
    eirp_dbw: float = 40.0
    g_t_dbk: float = 7.0          # G/T figure of merit (dB/K), not G alone
    n_mc: int = 20_000
    seed: int = 42


@dataclass
class LMSShadowModel:
    """TR 38.811 suburban LMS mixture at a reference elevation."""

    los_frac: float
    shadow_frac: float
    blocked_frac: float
    k_los_db: float
    k_shadow_db: float
    shadow_std_db: float
    blocked_mean_db: float
    blocked_std_db: float
    canopy_prob: float = 0.0
    canopy_mean_db: float = 18.0
    canopy_std_db: float = 6.0


@dataclass
class LinkClassSpec:
    name: str
    theta_min_deg: float
    theta_max_deg: float
    gamma_th_db: float              # MCS SNR threshold (dB)
    impl_loss_db: float             # implementation / pointing loss
    atmos_scint_db: float           # atmospheric + scintillation loss (mean)
    lms: LMSShadowModel
    mcs_label: str = ""


def _lms_probs_vs_elevation(theta_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Elevation-dependent LOS / shadow / blocked fractions (TR 38.811 style)."""
    th = np.clip(theta_deg, 1.0, 90.0)
    # Smooth transition: high elevation → mostly LOS; below 10° → blocked-heavy.
    p_los = np.clip(1.0 / (1.0 + np.exp(-(th - 25.0) / 6.0)), 0.05, 0.95)
    p_blocked = np.clip(1.0 / (1.0 + np.exp((th - 12.0) / 4.0)) - 0.35, 0.0, 0.75)
    p_shadow = np.clip(1.0 - p_los - p_blocked, 0.0, 0.85)
    total = p_los + p_shadow + p_blocked
    return p_los / total, p_shadow / total, p_blocked / total


LINK_CLASSES: dict[str, LinkClassSpec] = {
    "good": LinkClassSpec(
        name="good",
        theta_min_deg=55.0,
        theta_max_deg=90.0,
        gamma_th_db=0.0,
        impl_loss_db=1.0,
        atmos_scint_db=0.5,
        mcs_label="QPSK (γ_th=0 dB)",
        lms=LMSShadowModel(
            los_frac=0.85, shadow_frac=0.12, blocked_frac=0.03,
            k_los_db=12.0, k_shadow_db=8.0, shadow_std_db=2.0,
            blocked_mean_db=18.0, blocked_std_db=4.0,
            canopy_prob=0.02, canopy_mean_db=12.0, canopy_std_db=4.0,
        ),
    ),
    "medium": LinkClassSpec(
        name="medium",
        theta_min_deg=20.0,
        theta_max_deg=70.0,
        gamma_th_db=6.0,
        impl_loss_db=2.0,
        atmos_scint_db=1.5,
        mcs_label="16-QAM (γ_th=6 dB)",
        lms=LMSShadowModel(
            los_frac=0.50, shadow_frac=0.35, blocked_frac=0.15,
            k_los_db=8.0, k_shadow_db=5.0, shadow_std_db=4.0,
            blocked_mean_db=25.0, blocked_std_db=6.0,
            canopy_prob=0.12, canopy_mean_db=18.0, canopy_std_db=5.0,
        ),
    ),
    "poor": LinkClassSpec(
        name="poor",
        theta_min_deg=5.0,
        theta_max_deg=40.0,
        gamma_th_db=12.0,
        impl_loss_db=3.5,
        atmos_scint_db=3.0,
        mcs_label="64-QAM (γ_th=12 dB)",
        lms=LMSShadowModel(
            los_frac=0.15, shadow_frac=0.40, blocked_frac=0.45,
            k_los_db=4.0, k_shadow_db=2.0, shadow_std_db=8.0,
            blocked_mean_db=32.0, blocked_std_db=8.0,
            canopy_prob=0.35, canopy_mean_db=22.0, canopy_std_db=7.0,
        ),
    ),
}


def slant_range_m(theta_deg: np.ndarray, h_sat_m: float) -> np.ndarray:
    """Geometric slant range for LEO at elevation θ (degrees)."""
    th = np.deg2rad(np.clip(theta_deg, 0.1, 89.9))
    Re = 6_371_000.0
    return np.sqrt((Re + h_sat_m) ** 2 - (Re * np.cos(th)) ** 2) - Re * np.sin(th)


def fspl_db(d_m: np.ndarray, f_hz: float) -> np.ndarray:
    return 20.0 * np.log10(4.0 * np.pi * d_m * f_hz / C_MPS)


def snr_db(
    eirp_dbw: float,
    g_over_t_dbk: float,
    fspl_dbw: np.ndarray,
    small_scale_db: np.ndarray,
    extra_loss_db: np.ndarray,
    impl_loss_db: float,
    atmos_loss_db: float,
    bandwidth_hz: float,
) -> np.ndarray:
    """Eq. link_budget_correct: EIRP − L_FSPL + G/T + k₀ − 10 log₁₀(B) − losses + fading."""
    return (
        eirp_dbw
        - fspl_dbw
        + g_over_t_dbk
        + K0_DB
        - 10.0 * np.log10(bandwidth_hz)
        + small_scale_db
        - extra_loss_db
        - impl_loss_db
        - atmos_loss_db
    )


def _rician_fading_db(n: int, k_db: float, rng: np.random.Generator) -> np.ndarray:
    k_lin = 10.0 ** (k_db / 10.0)
    s = np.sqrt(k_lin / (k_lin + 1.0))
    sigma = np.sqrt(1.0 / (2 * (k_lin + 1.0)))
    real = s + sigma * rng.standard_normal(n)
    imag = sigma * rng.standard_normal(n)
    power = real ** 2 + imag ** 2
    return 10.0 * np.log10(np.maximum(power, 1e-12))


def _sample_lms_loss_db(
    n: int,
    spec: LinkClassSpec,
    theta_deg: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw TR 38.811 LMS excess loss (positive dB) and small-scale fading."""
    lms = spec.lms
    p_los, p_shadow, p_blk = _lms_probs_vs_elevation(theta_deg)
    p_los = 0.5 * lms.los_frac + 0.5 * p_los
    p_shadow = 0.5 * lms.shadow_frac + 0.5 * p_shadow
    p_blk = np.clip(1.0 - p_los - p_shadow, 0.0, 0.95)
    total = p_los + p_shadow + p_blk
    p_los, p_shadow, p_blk = p_los / total, p_shadow / total, p_blk / total

    u = rng.random(n)
    state = np.zeros(n, dtype=int)
    state[u >= p_los] = 1
    state[u >= p_los + p_shadow] = 2

    extra_loss = np.zeros(n)
    fading = np.zeros(n)
    los_m = state == 0
    sh_m = state == 1
    blk_m = state == 2

    if los_m.any():
        fading[los_m] = _rician_fading_db(int(los_m.sum()), lms.k_los_db, rng)
    if sh_m.any():
        # Log-normal shadowing: positive excess loss relative to mean path.
        extra_loss[sh_m] = np.abs(rng.normal(0.0, lms.shadow_std_db, int(sh_m.sum())))
        fading[sh_m] = _rician_fading_db(int(sh_m.sum()), lms.k_shadow_db, rng)
    if blk_m.any():
        extra_loss[blk_m] = rng.normal(lms.blocked_mean_db, lms.blocked_std_db, int(blk_m.sum()))
        extra_loss[blk_m] = np.maximum(extra_loss[blk_m], 8.0)

    canopy = rng.random(n) < lms.canopy_prob
    if canopy.any():
        extra_loss[canopy] += rng.normal(lms.canopy_mean_db, lms.canopy_std_db, int(canopy.sum()))

    return extra_loss, fading


def outage_probability(snr_db_samples: np.ndarray, snr_thr_db: float) -> float:
    return float(np.mean(snr_db_samples < snr_thr_db))


def implied_mean_tau_steps(
    p_out: float,
    tau_nom: float = 45.0,
    tau_min: float = 5.0,
    tau_max: float = 320.0,
) -> float:
    """Map outage probability to mean inter-sync interval (steps).

    Higher ``p_out`` inflates effective interval via retransmissions; nominal
  ``tau_nom`` is the clear-sky base interval used by the NTN channel model.
    """
    p_out = float(np.clip(p_out, 0.0, 0.99))
    inflation = 1.0 + 2.5 * p_out / max(1.0 - p_out, 0.01)
    return float(np.clip(tau_nom * inflation, tau_min, tau_max))


def run_link_budget(inputs: LinkBudgetInputs | None = None) -> dict:
    """Monte Carlo link budget for all link classes."""
    inputs = inputs or LinkBudgetInputs()
    rng = np.random.default_rng(inputs.seed)
    results: dict = {
        "inputs": asdict(inputs),
        "k0_db": K0_DB,
        "snr_formula": (
            "EIRP - L_FSPL + G/T + k0 - 10*log10(B) + fading - L_LMS - L_impl - L_atm"
        ),
        "classes": {},
    }

    for name, spec in LINK_CLASSES.items():
        theta = rng.uniform(spec.theta_min_deg, spec.theta_max_deg, inputs.n_mc)
        d = slant_range_m(theta, inputs.h_sat_m)
        pl = fspl_db(d, inputs.f_c_hz)
        extra_loss, fading = _sample_lms_loss_db(inputs.n_mc, spec, theta, rng)
        snr = snr_db(
            inputs.eirp_dbw,
            inputs.g_t_dbk,
            pl,
            fading,
            extra_loss,
            spec.impl_loss_db,
            spec.atmos_scint_db,
            inputs.bandwidth_hz,
        )
        p_out = outage_probability(snr, spec.gamma_th_db)
        tau_nom = {"good": 20.0, "medium": 45.0, "poor": 80.0}[name]
        tau_bar = implied_mean_tau_steps(p_out, tau_nom=tau_nom)
        snr_sorted = np.sort(snr)
        cdf_stride = max(1, len(snr_sorted) // 200)
        results["classes"][name] = {
            "theta_range_deg": [spec.theta_min_deg, spec.theta_max_deg],
            "gamma_th_db": spec.gamma_th_db,
            "mcs_label": spec.mcs_label,
            "impl_loss_db": spec.impl_loss_db,
            "atmos_scint_db": spec.atmos_scint_db,
            "shadow_model": {
                "lms_los_frac": spec.lms.los_frac,
                "lms_shadow_frac": spec.lms.shadow_frac,
                "lms_blocked_frac": spec.lms.blocked_frac,
                "k_los_db": spec.lms.k_los_db,
                "k_shadow_db": spec.lms.k_shadow_db,
                "shadow_std_db": spec.lms.shadow_std_db,
                "blocked_mean_db": spec.lms.blocked_mean_db,
                "blocked_std_db": spec.lms.blocked_std_db,
                "canopy_prob": spec.lms.canopy_prob,
                "canopy_mean_db": spec.lms.canopy_mean_db,
            },
            "mean_snr_db": float(np.mean(snr)),
            "snr_p5_db": float(np.percentile(snr, 5)),
            "snr_p50_db": float(np.percentile(snr, 50)),
            "snr_p95_db": float(np.percentile(snr, 95)),
            "p_out": p_out,
            "tau_nom_steps": tau_nom,
            "implied_tau_bar_steps": tau_bar,
            "snr_cdf": {
                "snr_db": snr_sorted[::cdf_stride].tolist(),
                "cdf": (np.arange(1, len(snr_sorted) + 1) / len(snr_sorted))[::cdf_stride].tolist(),
            },
        }
    return results


def write_latex_link_table(budget: dict, out_path: Path) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{NTN link budget by link class (2\,GHz, 10\,MHz, $h_{\mathrm{sat}}=600$\,km). "
        r"SNR $=$ EIRP $-$ $L_{\mathrm{FSPL}}$ $+$ $G/T$ $+$ $k_0$ $-$ $10\log_{10}(B)$ "
        r"$+$ fading $-$ $L_{\mathrm{LMS}}$ $-$ $L_{\mathrm{impl}}$ $-$ $L_{\mathrm{atm}}$.}",
        r"\label{tab:link_budget}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Class & $\theta$ ($^\circ$) & MCS / $\gamma_{\mathrm{th}}$ (dB) & "
        r"$L_{\mathrm{atm+scin}}$ (dB) & $L_{\mathrm{impl}}$ (dB) & Mean SNR (dB) & $p_{\mathrm{out}}$ \\",
        r"\midrule",
    ]
    for name, row in budget["classes"].items():
        th = row["theta_range_deg"]
        lines.append(
            f"{name} & [{th[0]:.0f}, {th[1]:.0f}] & {row['mcs_label']} & "
            f"{row['atmos_scint_db']:.1f} & {row['impl_loss_db']:.1f} & "
            f"{row['mean_snr_db']:.1f} & {row['p_out']:.3f} \\\\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{7}{l}{\footnotesize Shadow: TR\,38.811 suburban LMS "
        r"(LOS/shadow/blocked) + canopy blockage; see \texttt{link\_budget.json} for mixture fractions.} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex = "\n".join(lines)
    out_path.write_text(tex)
    return tex


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_snr_cdf(budget: dict, out_dir: Path) -> None:
    """Fig. fig:snr_cdf — empirical SNR CDFs per link class."""
    plt = _setup_mpl()
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"good": "#2ca02c", "medium": "#ff7f0e", "poor": "#d62728"}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, row in budget["classes"].items():
        cdf = row["snr_cdf"]
        ax.plot(cdf["snr_db"], cdf["cdf"], lw=2, color=colors.get(name, "C0"), label=name)
        ax.axvline(row["gamma_th_db"], ls=":", color=colors.get(name, "C0"), alpha=0.6)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("CDF")
    ax.set_title("Empirical SNR CDF by link class (dotted: $\\gamma_{\\mathrm{th}}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"Figure_SNR_CDF.{ext}", dpi=200, bbox_inches="tight")
    # LaTeX label alias
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_snr_cdf.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_aoi_surface(budget: dict, out_dir: Path) -> dict:
    """Fig. fig:aoi_surface — mean κ(τ) vs p_out and τ_nom; reports scheduling dominance."""
    from epca_staleness.channel import NTNChannel, kappa, sync_presets_from_budget

    plt = _setup_mpl()
    out_dir.mkdir(parents=True, exist_ok=True)
    sync_presets_from_budget(budget)

    p_out_grid = np.linspace(0.0, 0.45, 16)
    tau_nom_grid = np.array([20.0, 45.0, 80.0])
    mean_kappa = np.zeros((len(tau_nom_grid), len(p_out_grid)))
    aoi_inflation = np.zeros_like(mean_kappa)

    for i, tau_nom in enumerate(tau_nom_grid):
        for j, p_out in enumerate(p_out_grid):
            ch = NTNChannel.from_budget_class(
                "medium" if tau_nom == 45.0 else ("good" if tau_nom == 20.0 else "poor"),
                p_out_override=p_out,
                tau_nom_override=tau_nom,
                rng=42,
            )
            samples = ch.sample_sequence(5000)
            mean_kappa[i, j] = float(np.mean(kappa(samples)))
            aoi_inflation[i, j] = mean_kappa[i, j] / max(kappa(tau_nom), 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    im0 = axes[0].contourf(p_out_grid, tau_nom_grid, mean_kappa, levels=20, cmap="viridis")
    axes[0].set_xlabel("$p_{\\mathrm{out}}$")
    axes[0].set_ylabel("$\\tau_{\\mathrm{nom}}$ (steps)")
    axes[0].set_title("(a) Mean $\\kappa(\\tau) = (\\tau{-}1)/2$")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(p_out_grid, tau_nom_grid, aoi_inflation, levels=20, cmap="magma")
    axes[1].set_xlabel("$p_{\\mathrm{out}}$")
    axes[1].set_ylabel("$\\tau_{\\mathrm{nom}}$ (steps)")
    axes[1].set_title("(b) AoI inflation vs clear-sky nominal")
    fig.colorbar(im1, ax=axes[1])

    # Mark calibrated operating points from budget.
    for name, row in budget["classes"].items():
        axes[0].plot(row["p_out"], row["tau_nom_steps"], "wo", ms=7, mew=1.5)
        axes[1].plot(row["p_out"], row["tau_nom_steps"], "wo", ms=7, mew=1.5)

    fig.tight_layout()
    for stem in ("Figure_AoI_Surface", "fig_aoi_surface"):
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Check whether mean AoI is scheduling-limited.
    max_inflation = float(np.max(aoi_inflation))
    finding = (
        "DT staleness is scheduling-limited rather than link-limited"
        if max_inflation < 1.15
        else "link outage materially inflates mean AoI"
    )
    return {"max_aoi_inflation": max_inflation, "finding": finding}


def emit_link_budget(out_dir: Path | str = "output") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = run_link_budget()
    (out_dir / "link_budget.json").write_text(json.dumps(budget, indent=2))
    write_latex_link_table(budget, out_dir / "Table_Link_Budget.tex")
    plot_snr_cdf(budget, out_dir)
    aoi_meta = plot_aoi_surface(budget, out_dir)
    budget["aoi_surface"] = aoi_meta

    from epca_staleness.channel import sync_presets_from_budget
    presets = sync_presets_from_budget(budget)
    (out_dir / "ntn_channel_presets.json").write_text(json.dumps(presets, indent=2))

    cal_path = out_dir / "calibration_report.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
        cal["p_out_per_class"] = {k: v["p_out"] for k, v in budget["classes"].items()}
        cal["gamma_th_per_class"] = {k: v["gamma_th_db"] for k, v in budget["classes"].items()}
        cal["aoi_surface_finding"] = aoi_meta["finding"]
        cal_path.write_text(json.dumps(cal, indent=2))

    (out_dir / "link_budget.json").write_text(json.dumps(budget, indent=2))
    return budget
