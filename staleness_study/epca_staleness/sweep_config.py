"""Shared τ̄ sweep grids (must bracket operating envelope on both sides)."""

from __future__ import annotations

import numpy as np

# Extended grid per reviewer: spans sub-linear forcing regime under absolute age
TAU_SWEEP_GRID = np.array([5, 10, 20, 40, 80, 160, 320], dtype=float)
TAU_SWEEP_QUICK = np.array([5, 20, 80, 320], dtype=float)
TAU_SWEEP_N_MC = 200   # production MC for τ and β_M sweeps
TAU_SWEEP_N_MC_SMOKE = 20

HOTSPOT_QUANTILES = [0.08, 0.12, 0.20]
LAMBDA_CONG_SWEEP = [0.0, 0.5, 1.2, 3.0, 10.0]
