# Critical Fixes — Review Checklist Implementation

Addresses the top-priority review items for EPCA-M simulation credibility.

## Run everything

```bash
cd staleness_study
pip install -r requirements.txt
python3 run_critical_fixes.py --quick      # smoke test
python3 run_critical_fixes.py --n-mc 50    # publication run
```

Outputs: `critical_fixes_output/`

| File | Content |
|------|---------|
| `Table_II_YOLO.tex` | Params (M) + GFLOPs for EPCA-Det-n/s/m/l/x @ 640² |
| `Table_Perturbation_30.tex` | ±30% staleness parameter sensitivity on τ bound |
| `critical_fixes_summary.json` | All metrics (multi-seed MC, baselines, closed-loop) |

## Checklist mapping

### 1. Table II — YOLO Params/GFLOPs ✅
- `epca_models/yolo_profile.py` profiles all EPCA-Det variants via **thop** on Ultralytics YOLOv8 weights
- Measured at 640×640 input (forward-pass MAC count)

### 2. Calibrated staleness (not heuristic) ✅
- `StalenessParams` defaults now **calibrated**: β_M≈0.017, σ_M=0.08, σ_g≈3.52
- `calibrated_defaults()` recomputes from targets: R(60)=0.60, ghost RMSE=10 cells
- Legacy values: `uncalibrated_defaults()` (β_M=0.65)

### 3. NTN channel model ✅
- `epca_staleness/channel.py` — stochastic τ from base latency + jitter + outages
- τ derived from simulated link in all mission runners

### 4. Multi-seed statistics ✅
- All MC drivers use N≥50 (configurable); report **mean ± 95% CI** or std
- Detector k-fold: 5 folds × 10 seeds with mean±std

### 5. ±30% sensitivity on τ bound ✅
- `epca_sensitivity/perturbation.py` — quantitative Δτ bound for β_M, σ_M, σ_g

### 6. Closed-loop simulation ✅
- `epca_closed_loop/` — rasterize detector + forecaster → W_i → staleness → IUEF-EM
- End-to-end HPC/collision in `run_critical_fixes.py` §4

### 7. Forecaster baselines ✅
- `forecaster_baselines.py`: **persistence**, **AR(1)**, **ARIMA** vs MLP

### 8. Planner baselines ✅
- **DARP**, **priority-TSP** (priority-aware CPP), lawnmower, potential-field in `baselines.py`
- Compared in `run_critical_fixes.py` §7

### 9. Detector multi-seed k-fold ✅
- `detector_eval.py` — precision/recall/F1/MAE on synthetic Plant Health corpus

## Methodology sentence

> Model complexity (Table II) is reported from thop profiling of Ultralytics YOLOv8 backbones at 640×640. Staleness parameters are calibrated so cumulative map retention R(60)=0.60 and end-of-interval ghost RMSE equals 10 grid cells; synchronization intervals τ are drawn from an NTN channel model (base latency, Gaussian jitter, outage inflation). Monte Carlo experiments (N=50, 10 RNG seeds) report mean±std or 95% CI. Operating τ bounds are extracted at HPC≥65% and collision<0.4; ±30% perturbations of β_M, σ_M, and σ_g quantify sensitivity. The closed-loop simulator rasterizes EPCA-Det stress detections and MLP/ARIMA IoT forecasts into W_i=αH_i+βσ_i−γO_i before staleness-coupled IUEF-EM replanning.
