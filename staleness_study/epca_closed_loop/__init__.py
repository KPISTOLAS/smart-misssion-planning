"""End-to-end closed-loop EPCA-M simulation (sensing -> inference -> planning).

This package closes the loop missing in the original paper by coupling:
  * Tier-2 computer vision (EPCA-Det-s / YOLOv8-style detector),
  * IoT time-series forecasting (MLP on Herbal Plant features),
  * priority-field fusion (Eq. 3),
  * digital-twin staleness degradation,
  * IUEF-EM replanning under periodic / adaptive synchronization.
"""

from .closed_loop import ClosedLoopConfig, ClosedLoopResult, run_closed_loop
from .priority_field import PriorityFusionConfig, fuse_priority_field

__all__ = [
    "ClosedLoopConfig",
    "ClosedLoopResult",
    "run_closed_loop",
    "PriorityFusionConfig",
    "fuse_priority_field",
]
