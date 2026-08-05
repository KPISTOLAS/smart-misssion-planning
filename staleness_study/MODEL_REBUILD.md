# Staleness Model Rebuild (Reviewer Fix)

Replaces the normalized-age staleness model with **absolute-age** formulation.

## Model changes (Changes 1–3)

| Quantity | Old (broken) | New (correct) |
|----------|--------------|---------------|
| Age | `Δ/τ ∈ [0,1]` | `Δ` integer steps since sync |
| κ(τ) | `(τ−1)/(2τ)` → saturates | `(τ−1)/2` → grows linearly |
| Map fade | compounding `M*(1−β·age_frac)` | `M_sync · exp(−β_M·Δ)` |
| β_M | bisection on integral | **β_M = −ln(0.6)/60 = 0.00851** |
| Ghost | `σ_g² · age_frac · log(1+τ)` | `σ_g² · Δ` (Brownian) |

**Unit test:** `kappa(80)/kappa(40) ≈ 2.0` (was ~1.008).

## Run

```bash
cd staleness_study
python3 -m pytest tests/ -q
python3 run_model_rebuild.py --smoke          # N=20 sanity
python3 run_model_rebuild.py --n-mc 200       # production τ/β sweeps
```

## Outputs

| File | Content |
|------|---------|
| `calibration_report.json` | β_M derivation, R(20/60/160), ghost RMSE cells+m |
| `run_manifest.json` | Full parameter manifest + git SHA |
| `link_budget.json` + `Table_Link_Budget.tex` | NTN SNR CDF, p_out, implied τ̄ |
| `model_rebuild_output/` | Staleness sweeps + publication regimes |

## Extended τ̄ grid (Change 4)

`{5, 10, 20, 40, 80, 160, 320}` — brackets operating envelope on both sides.

## Safety (Change 6)

Mission loop reports **pre-deconfliction** and **post-deconfliction** collision rates via space-time cell reservation.

## Status

| Change | Status |
|--------|--------|
| 1–3 Model | Done + tests |
| 4 τ grid | Done |
| 5 Congestion wired into insertion score | Done + test |
| 6 ST deconfliction | Done (mission loop) |
| 7 R3 A/B/C sub-modes | Pending |
| 8 Budget metrics / quantile sweep | Done (SNR CDF + AoI surface) |
| 9 Stats harness on all tables | Module done; wire to exports |
| 10 Link budget | Done (corrected SNR, LMS sweep, NTN coupling) |
| 11 AoI threshold baseline | Pending |
| 12 Manifest | Done |
| 13 YOLO re-profile | Pending |
