"""NTN-like stochastic channel model for the synchronization interval ``tau``.

The original EPCA-M model uses a *fixed* periodic synchronization interval
``tau`` (every ``tau`` steps, with ``Delta t = 1.0`` s).  In a realistic
Non-Terrestrial-Network (NTN) uplink (e.g. LEO relay / satellite backhaul for a
digital twin) the effective synchronization interval is a random variable driven
by:

  * a deterministic *base latency* (propagation + fixed processing), and
  * a *variable delay* term from queueing, packet loss and retransmissions
    (ARQ/HARQ), plus occasional deep-fade *outages*.

We model one realisation of the synchronization interval (in simulation steps)
as::

    tau = max(1, round( base_tau * (1 + jitter * randn())  +  outage_penalty ))

where ``outage_penalty`` is nonzero with probability ``p_outage`` and models a
retransmission storm / handover gap that inflates the interval by a
multiplicative factor drawn from an exponential-like tail.

Three configurable *link-quality* presets are provided (good / medium / poor).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class LinkQuality:
    """Parameters describing one NTN link-quality regime.

    Attributes
    ----------
    name:
        Human-readable label (``good`` / ``medium`` / ``poor``).
    base_tau:
        Nominal (mean) synchronization interval in simulation steps.
    jitter:
        Relative standard deviation of the Gaussian delay term
        (``0.3`` reproduces the ``1 + 0.3*randn()`` model in the brief).
    p_outage:
        Probability that a given synchronization event suffers an outage /
        retransmission storm (paper suggests 5-15 %).
    outage_scale:
        Mean *additional* multiplicative inflation during an outage, expressed
        as a fraction of ``base_tau`` (drawn from an exponential distribution so
        the tail is heavy but finite in expectation).
    tau_max:
        Hard cap on any single interval (guards against pathological draws and
        keeps the mission horizon bounded).
    """

    name: str
    base_tau: float
    jitter: float
    p_outage: float
    outage_scale: float
    tau_max: int = 400


# Link-quality presets.  base_tau grows and outages become more frequent /
# heavier as the link degrades.  These are the values swept in the paper.
LINK_PRESETS = {
    "good": LinkQuality(name="good", base_tau=20.0, jitter=0.30, p_outage=0.05, outage_scale=0.8),
    "medium": LinkQuality(name="medium", base_tau=45.0, jitter=0.30, p_outage=0.10, outage_scale=1.2),
    "poor": LinkQuality(name="poor", base_tau=80.0, jitter=0.30, p_outage=0.15, outage_scale=1.8),
}


class NTNChannel:
    """Draws stochastic synchronization intervals ``tau`` from an NTN link model.

    Parameters
    ----------
    link:
        Either a preset key (``"good"`` / ``"medium"`` / ``"poor"``) or a
        :class:`LinkQuality` instance.
    rng:
        A ``numpy.random.Generator`` (or seed) for reproducibility.
    base_tau_override:
        Optional explicit ``base_tau`` (steps).  Useful for the "HPC vs average
        tau" sweep where we vary the mean interval while holding the outage
        statistics of a link regime fixed.
    """

    def __init__(self, link="medium", rng=None, base_tau_override: float | None = None):
        if isinstance(link, str):
            if link not in LINK_PRESETS:
                raise ValueError(f"Unknown link preset {link!r}; choose from {list(LINK_PRESETS)}")
            link = LINK_PRESETS[link]
        if base_tau_override is not None:
            link = replace(link, base_tau=float(base_tau_override))
        self.link = link
        self.rng = np.random.default_rng(rng)

    # ------------------------------------------------------------------ #
    # Core sampling
    # ------------------------------------------------------------------ #
    def sample_tau(self) -> int:
        """Return a single realisation of the synchronization interval (steps)."""
        lk = self.link
        # Gaussian jitter around the base latency.
        tau = lk.base_tau * (1.0 + lk.jitter * self.rng.standard_normal())
        # Occasional outage / retransmission storm (heavy but integrable tail).
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
            "outage_scale": self.link.outage_scale,
            "mean_tau": float(np.mean(samples)),
            "std_tau": float(np.std(samples)),
            "p95_tau": float(np.percentile(samples, 95)),
            "kappa_mean": float(np.mean((samples - 1) / (2.0 * samples))),
        }


def kappa(tau) -> np.ndarray:
    """Normalized average Age-of-Information ``kappa(tau) = (tau-1)/(2 tau)``.

    Works on scalars or arrays.  This is the closed-form time-average of the
    saw-tooth AoI ``age(t) = (t mod tau)/tau`` over one interval.
    """
    tau = np.asarray(tau, dtype=float)
    return (tau - 1.0) / (2.0 * tau)
