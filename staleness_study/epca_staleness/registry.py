"""Planner registry — unified interface for all methods under fair comparison."""

from __future__ import annotations

from typing import Callable

from .iuef_em import AblationMode, IUEFEMOptions, build_iuef_em_plan
from .baselines import (
    build_darp_plan,
    build_priority_tsp_plan,
    build_lawnmower_plan,
    build_potential_field_plan,
    build_greedy_plan,
    build_decentralized_greedy_plan,
)

PlanBuilder = Callable[..., dict]

# All planners share signature: (field, W_est, num_uav, starts=None) -> plan dict


def _iuef_full(field, W_est, num_uav, starts=None):
    return build_iuef_em_plan(field, W_est, num_uav, starts, name="iuef_em")


def _ablation(mode: AblationMode):
    def builder(field, W_est, num_uav, starts=None):
        opts = IUEFEMOptions.from_ablation(mode)
        return build_iuef_em_plan(field, W_est, num_uav, starts, opts=opts, name=mode.value)
    return builder


PLANNER_REGISTRY: dict[str, PlanBuilder] = {
    # Proposed + ablations
    "iuef_em": _iuef_full,
    "ablation_no_balance": _ablation(AblationMode.NO_BALANCE),
    "ablation_no_congestion": _ablation(AblationMode.NO_CONGESTION),
    "ablation_no_priority": _ablation(AblationMode.NO_PRIORITY),
    "ablation_no_astar": _ablation(AblationMode.NO_ASTAR),
    # External baselines
    "darp": build_darp_plan,
    "priority_tsp": build_priority_tsp_plan,
    "lawnmower": build_lawnmower_plan,
    "potential_field": build_potential_field_plan,
    # Internal baselines
    "greedy": build_greedy_plan,
    "decentralized_greedy": build_decentralized_greedy_plan,
}

ABLATION_PLANNERS = [
    "iuef_em",
    "ablation_no_balance",
    "ablation_no_congestion",
    "ablation_no_priority",
    "ablation_no_astar",
]

BASELINE_PLANNERS = [
    "iuef_em",
    "darp",
    "priority_tsp",
    "lawnmower",
    "potential_field",
    "greedy",
    "decentralized_greedy",
]


def build_plan(planner_name: str, field, W_est, num_uav: int, starts=None) -> dict:
    """Dispatch to a registered planner (same inputs for all methods)."""
    if planner_name not in PLANNER_REGISTRY:
        raise KeyError(f"Unknown planner {planner_name!r}. Choose from {list(PLANNER_REGISTRY)}")
    return PLANNER_REGISTRY[planner_name](field, W_est, num_uav, starts)
