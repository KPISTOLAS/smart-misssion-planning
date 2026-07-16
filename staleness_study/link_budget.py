"""NTN link-budget module: slant range, FSPL, SNR CDF, outage probability, implied τ̄.

Computes link statistics from first principles (TR 38.811 NTN-Suburban, shadowed
Rician fading). Monte Carlo over elevation angles for each link class.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
from scipy import stats


C_MPS = 299_792_458.0
K_B = 1.380649e-23
T_SYS = 290.0  # K


@dataclass
class LinkBudgetInputs:
    f_c_hz: float = 2.0e9
    bandwidth_hz: float = 10e6
    h_sat_m: float = 600_000.0
    eirp_dbw: float = 40.0
    g_t_dbk: float = 7.0
    noise_figure_db: float = 5.0
    n_mc: int = 20_000
    seed: int = 42


@dataclass
class LinkClassSpec:
    name: str
    theta_min_deg: float
    theta_max_deg: float
    k_factor_db: float = 6.0          # Rician K (dB), suburban NTN
    shadow_std_db: float = 4.0        # log-normal shadowing σ


LINK_CLASSES = {
    "good": LinkClassSpec("good", 55.0, 90.0, k_factor_db=10.0, shadow_std_db=2.0),
    "medium": LinkClassSpec("medium", 30.0, 70.0, k_factor_db=6.0, shadow_std_db=4.0),
    "poor": LinkClassSpec("poor", 10.0, 40.0, k_factor_db=3.0, shadow_std_db=6.0),
}


def slant_range_m(theta_deg: np.ndarray, h_sat_m: float) -> np.ndarray:
    """Geometric slant range for LEO at elevation θ (degrees)."""
    th = np.deg2rad(np.clip(theta_deg, 0.1, 89.9))
    # Earth radius ~6371 km; satellite at h_sat above surface
    Re = 6_371_000.0
    return np.sqrt((Re + h_sat_m) ** 2 - (Re * np.cos(th)) ** 2) - Re * np.sin(th)


def fspl_db(d_m: np.ndarray, f_hz: float) -> np.ndarray:
    return 20.0 * np.log10(4.0 * np.pi * d_m * f_hz / C_MPS)


def thermal_noise_dbw(bandwidth_hz: float, nf_db: float) -> float:
    n0 = 10.0 * np.log10(K_B * T_SYS)
    return n0 + 10.0 * np.log10(bandwidth_hz) + nf_db


def snr_db(eirp_dbw: float, g_t_dbk: float, fspl_dbw: np.ndarray,
           shadow_db: np.ndarray, fading_db: np.ndarray) -> np.ndarray:
    return eirp_dbw + g_t_dbk - fspl_dbw + shadow_db + fading_db - thermal_noise_dbw(10e6, 5.0)


def shadowed_rician_fading_db(n: int, k_db: float, rng: np.random.Generator) -> np.ndarray:
    """Shadowed Rician in dB (simplified TR 38.811 suburban)."""
    k_lin = 10.0 ** (k_db / 10.0)
    s = np.sqrt(k_lin / (k_lin + 1.0))
    sigma = np.sqrt(1.0 / (2 * (k_lin + 1.0)))
    real = s + sigma * rng.standard_normal(n)
    imag = sigma * rng.standard_normal(n)
    power = real ** 2 + imag ** 2
    return 10.0 * np.log10(np.maximum(power, 1e-12))


def outage_probability(snr_db_samples: np.ndarray, snr_thr_db: float = 0.0) -> float:
    return float(np.mean(snr_db_samples < snr_thr_db))


def implied_mean_tau_steps(p_out: float, tau_min: float = 5.0, tau_max: float = 320.0,
                           ref_tau: float = 45.0) -> float:
    """Map outage probability to mean inter-sync interval (steps).

    Higher p_out → shorter τ̄. Linear interpolation anchored at ref link stats.
    """
    p_out = float(np.clip(p_out, 0.01, 0.99))
    # τ̄ ∝ (1 − p_out) / p_out scaled to ref_tau at p_out=0.15
    scale = ref_tau * 0.15 / 0.85
    tau = scale * (1.0 - p_out) / p_out
    return float(np.clip(tau, tau_min, tau_max))


def run_link_budget(inputs: LinkBudgetInputs | None = None) -> dict:
    """Monte Carlo link budget for all link classes."""
    inputs = inputs or LinkBudgetInputs()
    rng = np.random.default_rng(inputs.seed)
    results = {"inputs": asdict(inputs), "classes": {}}

    for name, spec in LINK_CLASSES.items():
        theta = rng.uniform(spec.theta_min_deg, spec.theta_max_deg, inputs.n_mc)
        d = slant_range_m(theta, inputs.h_sat_m)
        pl = fspl_db(d, inputs.f_c_hz)
        shadow = rng.normal(0.0, spec.shadow_std_db, inputs.n_mc)
        fading = shadowed_rician_fading_db(inputs.n_mc, spec.k_factor_db, rng)
        snr = snr_db(inputs.eirp_dbw, inputs.g_t_dbk, pl, shadow, fading)
        p_out = outage_probability(snr, snr_thr_db=0.0)
        tau_bar = implied_mean_tau_steps(p_out)
        snr_sorted = np.sort(snr)
        cdf_x = snr_sorted
        cdf_y = np.arange(1, len(snr_sorted) + 1) / len(snr_sorted)
        results["classes"][name] = {
            "theta_range_deg": [spec.theta_min_deg, spec.theta_max_deg],
            "mean_snr_db": float(np.mean(snr)),
            "snr_p5_db": float(np.percentile(snr, 5)),
            "snr_p50_db": float(np.percentile(snr, 50)),
            "snr_p95_db": float(np.percentile(snr, 95)),
            "p_out": p_out,
            "implied_tau_bar_steps": tau_bar,
            "snr_cdf": {"snr_db": cdf_x[::max(1, len(cdf_x)//200)].tolist(),
                        "cdf": cdf_y[::max(1, len(cdf_y)//200)].tolist()},
        }
    return results


def write_latex_link_table(budget: dict, out_path: Path) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{NTN link budget by link class (2\,GHz, 10\,MHz, $h_{\mathrm{sat}}=600$\,km).}",
        r"\label{tab:link_budget}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Class & $\theta$ range ($^\circ$) & Mean SNR (dB) & $p_{\mathrm{out}}$ & Implied $\bar{\tau}$ (steps) \\",
        r"\midrule",
    ]
    for name, row in budget["classes"].items():
        th = row["theta_range_deg"]
        lines.append(
            f"{name} & [{th[0]:.0f}, {th[1]:.0f}] & {row['mean_snr_db']:.1f} & "
            f"{row['p_out']:.3f} & {row['implied_tau_bar_steps']:.0f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    out_path.write_text(tex)
    return tex


def emit_link_budget(out_dir: Path | str = "output") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = run_link_budget()
    (out_dir / "link_budget.json").write_text(json.dumps(budget, indent=2))
    write_latex_link_table(budget, out_dir / "Table_Link_Budget.tex")
    # Patch calibration report p_out if present
    cal_path = out_dir / "calibration_report.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
        cal["p_out_per_class"] = {k: v["p_out"] for k, v in budget["classes"].items()}
        cal_path.write_text(json.dumps(cal, indent=2))
    return budget
