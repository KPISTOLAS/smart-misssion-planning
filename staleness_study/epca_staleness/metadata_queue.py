"""Discrete-event metadata queue model: effective AoI vs packet loss and RTT.

Assumption ranges (LEO NTN literature, assumption-based unless cited):
  - Packet loss 5–20%: typical LEO S-band uplink under shadow/blockage (TR 38.811)
  - RTT 250–600 ms: one-way propagation + ground segment for 600 km LEO

Effective AoI inflation factor:
  κ_eff = κ_nom · (1 + loss/(1-loss)) · (1 + RTT/(τ_nom·dt))
"""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from .channel import kappa


LOSS_GRID = np.array([0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20], dtype=float)
RTT_GRID_MS = np.array([250, 300, 350, 400, 450, 500, 550, 600], dtype=float)
TAU_NOM_STEPS = 45.0
DT_S = 1.0
SEED_QUEUE = 31415


def effective_aoi_kappa(
    tau_nom: float,
    loss_rate: float,
    rtt_ms: float,
    dt_s: float = DT_S,
) -> float:
    """Deterministic effective mean AoI κ accounting for loss and RTT."""
    k_nom = float(kappa(tau_nom))
    loss = float(np.clip(loss_rate, 0.0, 0.95))
    rtt_s = rtt_ms / 1000.0
    retrans = 1.0 + loss / max(1.0 - loss, 0.01)
    rtt_factor = 1.0 + rtt_s / max(tau_nom * dt_s, 1.0)
    return k_nom * retrans * rtt_factor


def run_metadata_aoi_sweep(
    out_dir: Path | str = "methodology_validation_output",
    tau_nom: float = TAU_NOM_STEPS,
) -> dict:
    """Sweep loss × RTT grid; write heatmap CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = np.zeros((len(LOSS_GRID), len(RTT_GRID_MS)))
    k_nom = float(kappa(tau_nom))
    for i, loss in enumerate(LOSS_GRID):
        for j, rtt in enumerate(RTT_GRID_MS):
            grid[i, j] = effective_aoi_kappa(tau_nom, loss, rtt)

    csv_path = out_dir / "table_metadata_aoi_sweep.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["loss_rate", "rtt_ms", "kappa_eff", "aoi_inflation_vs_nominal",
                     "tau_nom_steps", "dt_s", "seed"])
        for i, loss in enumerate(LOSS_GRID):
            for j, rtt in enumerate(RTT_GRID_MS):
                k_eff = grid[i, j]
                w.writerow([
                    f"{loss:.2f}", f"{rtt:.0f}", f"{k_eff:.3f}",
                    f"{k_eff / max(k_nom, 1e-9):.3f}",
                    f"{tau_nom:.0f}", f"{DT_S:.1f}", SEED_QUEUE,
                ])

    assumptions_path = out_dir / "metadata_queue_assumptions.txt"
    assumptions_path.write_text(
        "Packet loss 5-20%: assumption-based, TR 38.811 LEO S-band outage regimes.\n"
        "RTT 250-600 ms: assumption-based, 600 km LEO round-trip + ground segment.\n"
        "Model: kappa_eff = kappa(tau_nom) * (1 + p/(1-p)) * (1 + RTT/(tau_nom*dt)).\n"
        f"tau_nom={tau_nom} steps, dt={DT_S} s, seed={SEED_QUEUE}.\n"
    )

    return {
        "seed": SEED_QUEUE,
        "tau_nom_steps": tau_nom,
        "loss_grid": LOSS_GRID.tolist(),
        "rtt_grid_ms": RTT_GRID_MS.tolist(),
        "kappa_nominal": k_nom,
        "max_inflation": float(np.max(grid / max(k_nom, 1e-9))),
        "csv": str(csv_path),
        "assumptions": str(assumptions_path),
    }
