"""Link-budget formula, channel coupling, and planner cap tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import link_budget
from epca_staleness.channel import LINK_PRESETS, NTNChannel, sync_presets_from_budget
from epca_staleness.environment import build_priority_field, priority_field_W_stats
from epca_staleness.iuef_em import IUEFEMOptions, _partition_hotspots, build_iuef_em_plan
from epca_staleness.planning_utils import pick_depots
from epca_staleness.run_manifest import build_manifest


def test_snr_formula_uses_gt_not_double_noise():
    """Corrected budget should report ~50 dB mean SNR at 600 km, not ~21 dB."""
    budget = link_budget.run_link_budget(
        link_budget.LinkBudgetInputs(n_mc=5000, seed=0)
    )
    good = budget["classes"]["good"]
    assert good["mean_snr_db"] > 45.0
    assert good["mean_snr_db"] < 58.0


def test_channel_p_outage_matches_budget():
    budget = link_budget.run_link_budget(link_budget.LinkBudgetInputs(n_mc=2000, seed=1))
    presets = sync_presets_from_budget(budget)
    for name, row in budget["classes"].items():
        assert LINK_PRESETS[name].p_outage == pytest.approx(row["p_out"], abs=1e-6)
        assert LINK_PRESETS[name].gamma_th_db == pytest.approx(row["gamma_th_db"])
        assert presets[name]["p_outage"] == pytest.approx(row["p_out"], abs=1e-6)


def test_ntn_outage_rate_tracks_budget_p_out():
    budget = link_budget.run_link_budget(link_budget.LinkBudgetInputs(n_mc=3000, seed=2))
    sync_presets_from_budget(budget)
    for name, row in budget["classes"].items():
        ch = NTNChannel(name, rng=123)
        assert ch.link.p_outage == pytest.approx(row["p_out"], abs=1e-6)
        assert ch.link.gamma_th_db == pytest.approx(row["gamma_th_db"])


def test_max_targets_cap_enforced():
    field = build_priority_field(50, 50, seed=42, hotspot_frac=0.25)
    starts = pick_depots(field.traversable, 3)
    opts = IUEFEMOptions(max_targets=12)
    idx, _ = _partition_hotspots(field, field.W, starts, 3, opts)
    assert idx.shape[0] <= opts.max_targets


def test_W_range_not_unit_cube():
    stats = priority_field_W_stats(seed=42)
    assert stats["W_max"] > 1.0
    assert stats["sigma_max"] <= 1.0 + 1e-9
    assert stats["H_norm_max"] <= 1.0 + 1e-9


def test_manifest_insertion_score_ratio_form():
    m = build_manifest()
    assert m["kappa_w"] == 1.0
    assert m["kappa_e"] == 0.0
    assert m["kappa_d"] == m["lambda_cong"]
    assert "W_g / Delta_L" in m["insertion_score_formula"]
    assert m["N_tgt_max"] == 500
