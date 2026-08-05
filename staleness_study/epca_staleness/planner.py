"""Backward-compatible IUEF-EM planner facade (delegates to iuef_em module)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .iuef_em import IUEFEMOptions, build_iuef_em_plan
from .planning_utils import astar_grid, pick_depots


@dataclass
class PlannerOptions:
    plan_mode: str = "blend"
    blend_gamma: float = 0.45
    lloyd_iterations: int = 40
    max_targets: int = 500


class IUEFEMPlanner:
    """Legacy wrapper used by mission.py staleness simulator."""

    def __init__(self, options: PlannerOptions | None = None):
        self.opts = options or PlannerOptions()

    def pick_depots(self, trav, num_uav):
        return pick_depots(trav, num_uav)

    def build_plan(self, field, W_est, num_uav, starts=None):
        iopts = IUEFEMOptions(
            plan_mode=self.opts.plan_mode,
            blend_gamma=self.opts.blend_gamma,
            max_targets=self.opts.max_targets,
        )
        return build_iuef_em_plan(field, W_est, num_uav, starts, opts=iopts)
