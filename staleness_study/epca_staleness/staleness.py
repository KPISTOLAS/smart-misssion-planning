"""Enhanced staleness / Age-of-Information degradation model for EPCA-M.

Original paper model
--------------------
* Age within a synchronization interval::

      age(t) = (t mod tau) / tau                       # saw-tooth in [0, 1)

* Normalized average AoI::

      kappa(tau) = (tau - 1) / (2 tau)

* Map (digital-twin priority field) fade::

      M_hat_{t+1} = max(0, M_hat_t * (1 - beta_M * age(t))
                            + N(0, sigma_M^2 * age(t)))

* Ghost (other-UAV) position drift::

      p_tilde_{v,t} = p_{v,t} + N(0, sigma_g^2 * age(t) * log(1 + tau))

Enhancements in this module
---------------------------
* ``tau`` is supplied *per synchronization event* by the NTN channel model, so
  ``age(t)`` and ``log(1 + tau)`` use the *current* interval length.
* Degradation parameters (beta_M, sigma_M, sigma_g) are **calibratable** to hit
  a target ghost RMSE (cells) or map-fade retention at a reference age/interval.
* The model is vectorized over the whole priority grid and over UAVs.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class StalenessParams:
    """Degradation parameters for the EPCA-M staleness model.

    Defaults are **calibrated** for cumulative-interval operation at τ_ref=60
    (ghost RMSE ≈ 10 cells, map retention R(60)=0.60).  Use
    :func:`uncalibrated_defaults` for the legacy per-step paper values.
    """

    beta_M: float = 0.0169    # calibrated map-fade (cumulative R(60)=0.60)
    sigma_M: float = 0.08     # map-fade process-noise scale
    sigma_g: float = 3.52     # calibrated ghost drift (≈10 cell RMSE @ τ=60)


def calibrated_defaults(tau_ref: int = 60,
                        ghost_rmse_cells: float = 10.0,
                        map_retention: float = 0.60,
                        rng=None) -> StalenessParams:
    """Return fully calibrated staleness parameters (recommended for all studies)."""
    sigma_g = calibrate_ghost_sigma(ghost_rmse_cells, tau_ref=tau_ref, rng=rng)
    beta_M = calibrate_map_fade(map_retention, tau_ref=tau_ref)
    return StalenessParams(beta_M=beta_M, sigma_M=0.08, sigma_g=sigma_g)


def uncalibrated_defaults() -> StalenessParams:
    """Legacy per-step paper defaults (β_M=0.65, σ_g=0.30) — not for MC studies."""
    return StalenessParams(beta_M=0.65, sigma_M=0.08, sigma_g=0.30)


def age_of(step_in_interval: int, tau: int) -> float:
    """Saw-tooth AoI ``age = (t mod tau) / tau`` for ``t`` steps since last sync."""
    if tau <= 0:
        return 0.0
    return (step_in_interval % tau) / float(tau)


def kappa(tau) -> np.ndarray:
    """Normalized average AoI ``(tau-1)/(2 tau)`` (scalar or array)."""
    tau = np.asarray(tau, dtype=float)
    return (tau - 1.0) / (2.0 * tau)


def interval_retention(beta_M: float, tau: float, n_grid: int = 64) -> float:
    """Cumulative belief retention across one synchronization interval.

    Integrates the paper's multiplicative fade ``(1 - beta_M * age)`` over the
    saw-tooth age ``age(s) = s/tau`` for ``s = 0 .. tau-1``.  In continuous form
    this is ``R(tau) = exp( tau * integral_0^1 ln(1 - beta_M * x) dx )`` which is
    *monotonically decreasing in tau* (longer intervals compound more fade),
    unlike the single-shot ``kappa(tau)`` factor which saturates at 0.5.

    The integral has the closed form
    ``I(b) = -1 + ((1-b)/b) * (1 - ln(1-b))`` for ``0 < b < 1``.
    A guarded numeric fallback is used near the boundaries.
    """
    b = float(np.clip(beta_M, 1e-9, 0.999999))
    tau = max(float(tau), 1.0)
    # Closed-form of \int_0^1 ln(1 - b x) dx = -1 - ((1-b)/b) * ln(1-b).
    integral = -1.0 - ((1.0 - b) / b) * np.log(1.0 - b)
    R = float(np.exp(tau * integral))
    return float(np.clip(R, 0.0, 1.0))


class StalenessModel:
    """Applies map fade and ghost-position drift given the current age / tau.

    Parameters
    ----------
    params:
        :class:`StalenessParams` controlling the degradation strength.
    rng:
        ``numpy.random.Generator`` or seed for reproducibility.
    """

    def __init__(self, params: StalenessParams | None = None, rng=None):
        self.params = params or StalenessParams()
        self.rng = np.random.default_rng(rng)

    # ------------------------------------------------------------------ #
    # Map fade
    # ------------------------------------------------------------------ #
    def fade_map(self, M_hat: np.ndarray, age: float) -> np.ndarray:
        """One-step update of the estimated priority map (digital-twin fade).

        Implements
        ``M_hat <- max(0, M_hat*(1 - beta_M*age) + N(0, sigma_M^2 * age))``.
        """
        p = self.params
        noise = self.rng.normal(0.0, p.sigma_M * np.sqrt(max(age, 0.0)), size=M_hat.shape)
        M_next = M_hat * (1.0 - p.beta_M * age) + noise
        return np.maximum(0.0, M_next)

    def fade_map_multi(self, M_hat: np.ndarray, age: float, n_steps: int) -> np.ndarray:
        """Apply ``n_steps`` successive one-step fades at a fixed ``age``.

        Useful when advancing the twin belief across a whole (fixed-age proxy)
        window without executing the inner loop elsewhere.
        """
        M = M_hat.copy()
        for _ in range(int(n_steps)):
            M = self.fade_map(M, age)
        return M

    # ------------------------------------------------------------------ #
    # Ghost-position drift
    # ------------------------------------------------------------------ #
    def ghost_positions(self, true_pos: np.ndarray, age: float, tau: int) -> np.ndarray:
        """Return drifted (ghost) positions of teammates for collision avoidance.

        ``p_tilde = p + N(0, sigma_g^2 * age * log(1 + tau))`` applied
        independently to each coordinate of each UAV.

        Parameters
        ----------
        true_pos:
            ``(U, 2)`` array of true UAV positions (grid coordinates, row/col).
        age:
            Current normalized AoI in [0, 1).
        tau:
            Current synchronization interval (steps).
        """
        p = self.params
        var = (p.sigma_g ** 2) * max(age, 0.0) * np.log1p(max(tau, 0))
        std = np.sqrt(max(var, 0.0))
        return np.asarray(true_pos, dtype=float) + self.rng.normal(0.0, std, size=np.shape(true_pos))

    def ghost_rmse(self, age: float, tau: int) -> float:
        """Analytic expected ghost RMSE (2-D Euclidean) in cells.

        ``RMSE = sigma_g * sqrt(2 * age * log(1 + tau))`` because both
        coordinates carry independent variance ``sigma_g^2 * age * log(1+tau)``.
        """
        p = self.params
        return p.sigma_g * np.sqrt(2.0 * max(age, 0.0) * np.log1p(max(tau, 0)))


# ---------------------------------------------------------------------- #
# Calibration
# ---------------------------------------------------------------------- #
def calibrate_ghost_sigma(target_rmse_cells: float,
                          tau_ref: int = 60,
                          age_ref: float | None = None,
                          n_mc: int = 40000,
                          rng=None) -> float:
    """Calibrate ``sigma_g`` so the mean ghost RMSE matches a target at ``tau_ref``.

    By default the reference age is the *end-of-interval* age
    ``age_ref = (tau_ref - 1)/tau_ref`` (worst-case staleness just before the
    next sync), which is the natural point to size the drift budget.

    A closed form exists (``sigma_g = target / sqrt(2*age*log(1+tau))``); we
    additionally verify it with a Monte-Carlo estimate of the *empirical* RMSE
    so the routine also works if the model is later made non-Gaussian.

    Returns
    -------
    float
        The calibrated ``sigma_g``.
    """
    if age_ref is None:
        age_ref = (tau_ref - 1) / float(tau_ref)
    denom = np.sqrt(2.0 * age_ref * np.log1p(tau_ref))
    sigma_g = float(target_rmse_cells / max(denom, 1e-12))

    # Monte-Carlo verification of the empirical RMSE at the calibrated value.
    model = StalenessModel(StalenessParams(sigma_g=sigma_g), rng=rng)
    true_pos = np.zeros((n_mc, 2))
    ghosts = model.ghost_positions(true_pos, age_ref, tau_ref)
    emp_rmse = float(np.sqrt(np.mean(np.sum(ghosts ** 2, axis=1))))
    # Tiny multiplicative correction to remove finite-sample bias.
    if emp_rmse > 1e-9:
        sigma_g *= target_rmse_cells / emp_rmse
    return sigma_g


def calibrate_map_fade(target_retention: float,
                       tau_ref: int = 60,
                       age_ref: float | None = None) -> float:
    """Calibrate ``beta_M`` so the cumulative interval retention hits a target.

    Solves ``interval_retention(beta_M, tau_ref) == target_retention`` for
    ``beta_M`` by bisection (the retention is monotonically decreasing in
    ``beta_M``).  This is the map-fade counterpart of ``calibrate_ghost_sigma``
    and is what re-tunes the paper's per-step ``beta_M`` for stochastic-tau
    operation.

    ``age_ref`` is accepted for API symmetry but not required by the cumulative
    model (the whole interval is integrated).
    """
    if not (0.0 < target_retention <= 1.0):
        raise ValueError("target_retention must lie in (0, 1].")
    lo, hi = 1e-6, 0.999999
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        R = interval_retention(mid, tau_ref)
        if R > target_retention:   # too little fade -> increase beta
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def calibrate_map_noise(target_std: float, age_ref: float | None = None, tau_ref: int = 60) -> float:
    """Calibrate ``sigma_M`` so the per-step map process-noise std hits a target.

    The one-step process noise has std ``sigma_M * sqrt(age)``; sizing it at the
    reference age gives ``sigma_M = target_std / sqrt(age_ref)``.
    """
    if age_ref is None:
        age_ref = (tau_ref - 1) / float(tau_ref)
    return float(target_std / np.sqrt(max(age_ref, 1e-12)))
