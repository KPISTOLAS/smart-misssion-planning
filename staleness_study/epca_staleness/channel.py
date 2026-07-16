"""NTN-like stochastic channel model for the synchronization interval ``tau``.

The synchronization interval is driven by a nominal clear-sky interval
``tau_nom`` and an outage probability ``p_out`` derived from the link budget
(``link_budget.py``). Both the reported ``p_out`` and the simulated outage
process share the same ``gamma_th`` threshold and budget-derived ``p_outage``.

One realisation::

    tau = max(1, round( base_tau * (1 + jitter * randn())  +  outage_penalty ))

where ``outage_penalty`` is nonzero with probability ``p_outage`` (equals budget
``p_out``) and models retransmission / handover inflation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class LinkQuality:
    """Parameters describing one NTN link-quality regime."""

    name: str
    base_tau: float
    jitter: float
    p_outage: float
    outage_scale: float
    gamma_th_db: float = 0.0
    tau_max: int = 400


# Defaults before link-budget sync; overwritten by ``sync_presets_from_budget``.
LINK_PRESETS: dict[str, LinkQuality] = {
    "good": LinkQuality(name="good", base_tau=20.0, jitter=0.30, p_outage=0.05,
                        outage_scale=0.8, gamma_th_db=0.0),
    "medium": LinkQuality(name="medium", base_tau=45.0, jitter=0.30, p_outage=0.10,
                          outage_scale=1.2, gamma_th_db=6.0),
    "poor": LinkQuality(name="poor", base_tau=80.0, jitter=0.30, p_outage=0.15,
                        outage_scale=1.8, gamma_th_db=12.0),
}

_BUDGET_SYNCED = False


def sync_presets_from_budget(budget: dict[str, Any]) -> dict[str, dict]:
    """Couple ``LINK_PRESETS`` to link-budget ``p_out``, ``tau_nom``, ``gamma_th``."""
    global LINK_PRESETS, _BUDGET_SYNCED
    out: dict[str, dict] = {}
    for name, row in budget.get("classes", {}).items():
        if name not in LINK_PRESETS:
            continue
        old = LINK_PRESETS[name]
        p_out = float(row.get("p_out", old.p_outage))
        tau_nom = float(row.get("tau_nom_steps", old.base_tau))
        gamma_th = float(row.get("gamma_th_db", old.gamma_th_db))
        # Heavier tails when outage probability is high (blocked / canopy regimes).
        outage_scale = old.outage_scale * (1.0 + 2.0 * p_out)
        updated = replace(
            old,
            base_tau=tau_nom,
            p_outage=p_out,
            gamma_th_db=gamma_th,
            outage_scale=outage_scale,
        )
        LINK_PRESETS[name] = updated
        out[name] = {
            "base_tau": updated.base_tau,
            "p_outage": updated.p_outage,
            "gamma_th_db": updated.gamma_th_db,
            "outage_scale": updated.outage_scale,
            "implied_tau_bar_steps": row.get("implied_tau_bar_steps"),
        }
    _BUDGET_SYNCED = True
    return out


def load_budget_and_sync(path: Path | str) -> dict[str, dict]:
    """Load ``link_budget.json`` and sync channel presets."""
    data = json.loads(Path(path).read_text())
    return sync_presets_from_budget(data)


class NTNChannel:
    """Draws stochastic synchronization intervals ``tau`` from an NTN link model."""

    def __init__(self, link="medium", rng=None, base_tau_override: float | None = None,
                 p_outage_override: float | None = None):
        if isinstance(link, str):
            if link not in LINK_PRESETS:
                raise ValueError(f"Unknown link preset {link!r}; choose from {list(LINK_PRESETS)}")
            link = LINK_PRESETS[link]
        if base_tau_override is not None:
            link = replace(link, base_tau=float(base_tau_override))
        if p_outage_override is not None:
            link = replace(link, p_outage=float(p_outage_override))
        self.link = link
        self.rng = np.random.default_rng(rng)

    @classmethod
    def from_budget_class(
        cls,
        name: str,
        p_out_override: float | None = None,
        tau_nom_override: float | None = None,
        rng=None,
    ) -> "NTNChannel":
        """Construct channel from budget-synced preset with optional overrides."""
        return cls(
            name,
            rng=rng,
            base_tau_override=tau_nom_override,
            p_outage_override=p_out_override,
        )

    # ------------------------------------------------------------------ #
    # Core sampling
    # ------------------------------------------------------------------ #
    def sample_tau(self) -> int:
        """Return a single realisation of the synchronization interval (steps)."""
        lk = self.link
        tau = lk.base_tau * (1.0 + lk.jitter * self.rng.standard_normal())
        if self.rng.random() < lk.p_outage:
            tau += lk.base_tau * lk.outage_scale * self.rng.exponential(1.0)
        tau = int(round(tau))
        return int(min(lk.tau_max, max(1, tau)))

    def sample_sequence(self, n_events: int) -> np.ndarray:
        """Return ``n_events`` independent interval draws (steps)."""
        return np.array([self.sample_tau() for _ in range(n_events)], dtype=int)

    # ------------------------------------------------------------------ #
    # Analytics helpers (used for calibration / reporting)
    # ------------------------------------------------------------------ #
    def expected_tau(self, n_mc: int = 20000) -> float:
        """Monte-Carlo estimate of E[tau] for the current link regime."""
        return float(np.mean(self.sample_sequence(n_mc)))

    def describe(self) -> dict:
        """Return a JSON-serialisable summary of the channel statistics."""
        samples = self.sample_sequence(20000)
        return {
            "link": self.link.name,
            "base_tau": self.link.base_tau,
            "jitter": self.link.jitter,
            "p_outage": self.link.p_outage,
            "gamma_th_db": self.link.gamma_th_db,
            "outage_scale": self.link.outage_scale,
            "budget_synced": _BUDGET_SYNCED,
            "mean_tau": float(np.mean(samples)),
            "std_tau": float(np.std(samples)),
            "p95_tau": float(np.percentile(samples, 95)),
            "kappa_mean": float(np.mean((samples - 1) / 2.0)),
        }


def kappa(tau) -> np.ndarray:
    """Mean AoI over one interval when age is absolute: ``(τ−1)/2``."""
    tau = np.asarray(tau, dtype=float)
    return (tau - 1.0) / 2.0
