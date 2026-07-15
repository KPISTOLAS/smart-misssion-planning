"""EPCA-M enhanced staleness / Age-of-Information (AoI) simulation package.

This package augments the EPCA-M (Energy-Priority Coverage with Aging - Monitoring)
digital-twin staleness model described in the paper with:

  1. A stochastic per-synchronization interval ``tau`` drawn from a lightweight
     NTN-like (Non-Terrestrial-Network) channel model (:mod:`.channel`).
  2. A calibratable staleness model for map fade and ghost-position drift
     (:mod:`.staleness`).
  3. An event-triggered / threshold-based *adaptive* synchronization policy
     compared against the classic *periodic* policy at equal average uplink cost
     (:mod:`.mission`).
  4. A faithful Python re-implementation of the IUEF-EM priority planner
     (Algorithm 1) coupled to the staleness-degraded digital twin
     (:mod:`.planner`, :mod:`.environment`, :mod:`.mission`).
  5. Monte-Carlo sweep / calibration / sensitivity drivers that reproduce the
     figures and operating bounds for the paper (:mod:`.experiments`).

The code is intentionally dependency-light (numpy + matplotlib) and fully
deterministic given a seed, for reproducibility.
"""

from .channel import NTNChannel, LINK_PRESETS
from .staleness import StalenessModel, StalenessParams, calibrate_ghost_sigma, calibrate_map_fade
from .environment import PriorityField, build_priority_field
from .planner import IUEFEMPlanner, PlannerOptions
from .mission import MissionConfig, run_mission, SyncPolicy

__all__ = [
    "NTNChannel",
    "LINK_PRESETS",
    "StalenessModel",
    "StalenessParams",
    "calibrate_ghost_sigma",
    "calibrate_map_fade",
    "PriorityField",
    "build_priority_field",
    "IUEFEMPlanner",
    "PlannerOptions",
    "MissionConfig",
    "run_mission",
    "SyncPolicy",
]

__version__ = "1.0.0"
