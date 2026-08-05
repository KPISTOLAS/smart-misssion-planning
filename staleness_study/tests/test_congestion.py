"""Congestion term and planner wiring tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epca_staleness.environment import build_priority_field
from epca_staleness.iuef_em import IUEFEMOptions, build_iuef_em_plan
from epca_staleness.planning_utils import pick_depots


def _forced_overlap_field(seed: int = 42):
    """Map with a narrow choke-point forcing path overlap (high O in corridor)."""
    field = build_priority_field(50, 50, seed=seed, hotspot_frac=0.15)
    trav = field.traversable
    # Elevate congestion along central vertical corridor
    O = field.O.copy()
    col = trav.shape[1] // 2
    for r in range(5, trav.shape[0] - 5):
        if trav[r, col]:
            O[r, col] = 1.0
            if col + 1 < trav.shape[1] and trav[r, col + 1]:
                O[r, col + 1] = 0.9
    field.O = O
    return field


def test_congestion_changes_plan_on_forced_overlap_map():
    field = _forced_overlap_field()
    starts = pick_depots(field.traversable, 3)
    W = field.W
    opts_off = IUEFEMOptions(lambda_cong=0.0, use_congestion=False)
    opts_on = IUEFEMOptions(lambda_cong=1.2, use_congestion=True)
    p0 = build_iuef_em_plan(field, W, 3, starts, opts=opts_off)
    p1 = build_iuef_em_plan(field, W, 3, starts, opts=opts_on)
    segs0 = [tuple(map(tuple, s)) for s in p0["segments"]]
    segs1 = [tuple(map(tuple, s)) for s in p1["segments"]]
    assert segs0 != segs1, "λ_cong=0 vs λ_cong=1.2 produced identical plans on overlap map"
