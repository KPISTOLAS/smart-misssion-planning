# Publication Study (90–95% Quality Checklist)

Unified driver for the three **explicitly separated** evaluation regimes required for the paper. Do **not** merge regimes in a single table or figure.

```bash
cd staleness_study
python3 run_publication_study.py --quick       # smoke test (~3–5 min)
python3 run_publication_study.py --n-mc 50     # paper run (R1–R3 @ N=50)
```

Outputs: `publication_output/`

| Artifact | Content |
|----------|---------|
| `publication_summary.json` | Full numeric results per regime |
| `Figure_Regime_Comparison.{png,pdf}` | R1 secondary KPIs / R2 ablation / R3 targeting error |
| `Figure_Ablation_Staleness_Secondary.{png,pdf}` | IUEF-EM ablations at τ̄∈{40,50,60} |
| `Figure_Operating_Envelope.{png,pdf}` | Integrated τ bounds (link, hotspot, fleet) |
| `Table_Operating_Envelope.tex` | LaTeX operating envelope |
| `Table_Ablation_Staleness.tex` | LaTeX ablation under τ̄=50 |

## Three Regimes (report separately)

### R1 — Planner-perfect (planning upper bound)

- **Setup:** Ground-truth priority field W, no staleness, no closed-loop inference.
- **Purpose:** Isolate planner and ablation effects.
- **Primary caveat:** HPC often **saturates at ~100%**; discrimination requires **secondary KPIs**: energy (J/UAV·h), mission duration, mission score, collision rate.
- **Driver:** `ablation_study()` + `baseline_comparison()` from `epca_staleness.planner_evaluation`.

### R2 — Staleness-moderate (mission-loop discrimination)

- **Setup:** Full staleness-coupled mission loop (map fade + ghost drift + NTN τ), IUEF-EM ablations.
- **τ̄ grid:** {40, 50, 60} steps — moderate staleness where ablations **discriminate** (unlike R1).
- **Secondary KPIs:** collision, near-miss, retained hotspot fraction, uplink cost, sync count, duration.
- **Driver:** `ablation_under_staleness()` / `ablation_staleness_sweep()`.

### R3 — Closed-loop (end-to-end)

- **Setup:** Tier-2 detector → forecaster → fused Ŵ → staleness → IUEF-EM replanning.
- **Modes:** `closed_loop` vs `perfect_info` vs `no_staleness` at medium link, τ̄≈45.
- **Primary caveat:** HPC can plateau; lead with **inference MAE (targeting error)**, collision, energy, retained fraction.
- **Driver:** `_compare_modes_extended()` in `epca_staleness.publication`.

## Operating Envelope

Integrates largest feasible mean τ (steps) subject to **HPC ≥ 65%** and **collision < 0.4** across:

| Factor | Settings |
|--------|----------|
| Link quality | good / medium / poor |
| Hotspot density | low / medium / high |
| Fleet size U | 1–8 |

Envelope sweeps use `envelope_n_mc = min(20, N)` to keep runtime tractable when R1–R3 run at N=50.

## Paper-Ready Text Snippets

**Regime separation (Methods):**
> We report three non-interchangeable regimes: (R1) planner-perfect ground-truth maps without staleness, establishing a planning upper bound; (R2) full staleness-coupled missions at moderate mean AoI τ̄∈{40,50,60}; and (R3) closed-loop inference with periodic NTN synchronization. Metrics are not pooled across regimes.

**R1 ablations:**
> Under R1, high-priority coverage saturates near 100% for all IUEF-EM variants; component contributions are therefore quantified via energy per UAV-hour, mission duration, and collision rate.

**R2 ablations:**
> At moderate staleness (τ̄=50), removing congestion awareness or load balancing increases collision rate and reduces retained hotspot fraction while HPC remains within a narrow band, confirming that safety and belief quality — not coverage alone — differentiate the architecture.

**R3 closed-loop:**
> End-to-end closed-loop operation degrades targeting accuracy (inference MAE) relative to perfect information while maintaining feasible collision rates; the gap quantifies the cost of on-board perception and forecasting under NTN latency.

**Operating envelope:**
> Table X summarizes the largest mean synchronization interval τ̄_max satisfying HPC≥65% and collision<0.4 for each environmental factor. Tighter bounds under high hotspot density and large fleets define the deployable operating envelope for EPCA-M.

## Related Drivers (also run at N=50)

```bash
python3 run_closed_loop.py --n-mc 50
python3 run_sensitivity_study.py --n-mc 50
python3 run_critical_fixes.py --n-mc 50
python3 run_planner_evaluation.py --n-mc 50
```
