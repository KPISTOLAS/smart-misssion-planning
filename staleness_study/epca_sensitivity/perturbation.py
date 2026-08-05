"""±30% parameter perturbation sensitivity for operating τ bounds."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from epca_staleness.experiments import default_params, SweepResult
from epca_staleness.staleness import StalenessParams, calibrated_defaults

from .experiments import _run_tau_sweep
from .analysis import extract_operating_bound
from .map_generator import SyntheticMapConfig
from .analysis import _ci95


@dataclass
class PerturbationResult:
    parameter: str
    nominal: float
    minus_30: float
    plus_30: float
    tau_bound_nominal: float | None
    tau_bound_minus: float | None
    tau_bound_plus: float | None
    delta_minus_pct: float | None
    delta_plus_pct: float | None


def sweep_parameter_perturbation(
    param_name: str,
    nominal_value: float,
    perturb_frac: float = 0.30,
    n_mc: int = 10,
    tau_grid=None,
    seed_base: int = 30000,
) -> PerturbationResult:
    """Sweep τ bound at nominal, -30%, +30% for one staleness parameter."""
    if tau_grid is None:
        tau_grid = np.array([10, 15, 20, 25, 30, 40, 50, 60], dtype=float)
    vals = {
        "nominal": nominal_value,
        "minus": nominal_value * (1 - perturb_frac),
        "plus": nominal_value * (1 + perturb_frac),
    }
    bounds = {}
    base = calibrated_defaults()
    map_cfg = SyntheticMapConfig()

    for label, v in vals.items():
        if param_name == "beta_M":
            p = StalenessParams(beta_M=v, sigma_M=base.sigma_M, sigma_g=base.sigma_g)
        elif param_name == "sigma_M":
            p = StalenessParams(beta_M=base.beta_M, sigma_M=v, sigma_g=base.sigma_g)
        elif param_name == "sigma_g":
            p = StalenessParams(beta_M=base.beta_M, sigma_M=base.sigma_M, sigma_g=v)
        else:
            raise ValueError(param_name)
        sw = _run_tau_sweep(map_cfg, dict(num_uav=3), tau_grid, n_mc, p,
                            seed_base + int(v * 1000), "medium")
        b = extract_operating_bound(sw)
        bounds[label] = b.get("max_tau") if b.get("feasible") else None

    def _delta(new, ref):
        if new is None or ref is None or ref == 0:
            return None
        return 100.0 * (new - ref) / ref

    return PerturbationResult(
        parameter=param_name,
        nominal=nominal_value,
        minus_30=vals["minus"],
        plus_30=vals["plus"],
        tau_bound_nominal=bounds["nominal"],
        tau_bound_minus=bounds["minus"],
        tau_bound_plus=bounds["plus"],
        delta_minus_pct=_delta(bounds["minus"], bounds["nominal"]),
        delta_plus_pct=_delta(bounds["plus"], bounds["nominal"]),
    )


def run_full_perturbation_study(n_mc: int = 10, quick: bool = False) -> dict:
    """±30% perturbation on β_M, σ_M, σ_g with multi-seed τ-bound comparison."""
    if quick:
        n_mc = 5
    base = calibrated_defaults()
    results = {}
    for pname, val in [("beta_M", base.beta_M), ("sigma_M", base.sigma_M), ("sigma_g", base.sigma_g)]:
        results[pname] = sweep_parameter_perturbation(pname, val, n_mc=n_mc).__dict__
    return results


def latex_table_perturbation(results: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sensitivity of operating $\bar{\tau}$ bound to $\pm30\%$ staleness parameters}",
        r"\label{tab:perturbation_30}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Parameter & Nominal & $\bar{\tau}_{-30\%}$ & $\bar{\tau}_{+30\%}$ & $\Delta\tau$ (\%) \\",
        r"\midrule",
    ]
    labels = {"beta_M": r"$\beta_M$", "sigma_M": r"$\sigma_M$", "sigma_g": r"$\sigma_g$"}
    for k, r in results.items():
        tb = r.get("tau_bound_nominal")
        tm = r.get("tau_bound_minus")
        tp = r.get("tau_bound_plus")
        dm = r.get("delta_minus_pct")
        dp = r.get("delta_plus_pct")
        delta_str = f"{dm:+.0f}/{dp:+.0f}" if dm is not None and dp is not None else "--"
        lines.append(
            f"{labels.get(k, k)} & {tb if tb is not None else '--'} & "
            f"{tm if tm is not None else '--'} & {tp if tp is not None else '--'} & {delta_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)
