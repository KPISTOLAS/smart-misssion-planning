# EPCA-M Synthetic Map Generation & Sensitivity Analysis

Parametric map families, Monte Carlo operating-bound extraction, and imperfect-priority robustness testing — integrated with the calibrated staleness model and IUEF-EM planner.

## Quick start

```bash
cd staleness_study
pip install -r requirements.txt
python3 run_sensitivity_study.py              # full study (N=50)
python3 run_sensitivity_study.py --quick      # smoke test
```

Outputs in `sensitivity_output/`:
| File | Content |
|------|---------|
| `sensitivity_summary.json` | All sweep results + operating τ bounds |
| `Table_Sensitivity.tex` | LaTeX tables (τ bound vs hotspot density, U, priority quality) |
| `Figure_Sensitivity_HotspotTauBound` | τ bound vs hotspot density |
| `Figure_Sensitivity_FleetTauBound` | τ bound vs fleet size U |
| `Figure_Sensitivity_TauBound_d_safe` | τ bound vs safety distance |
| `Figure_Sensitivity_PriorityQuality` | Perfect vs noisy W_i |
| `sensitivity_insights.txt` | Deployment insight bullets |

## Package structure

```
staleness_study/epca_sensitivity/
├── map_generator.py      # SyntheticMapGenerator (50×50, 54×72)
├── imperfect_priority.py # Detector/forecaster error injection
├── mission_runner.py     # Staleness-coupled mission with imperfect W
├── experiments.py        # MC sweeps (density, U, d_safe, terrain, grid)
├── analysis.py           # Operating-bound extraction, LaTeX tables
└── plots.py              # Heatmaps and bar charts
```

## Synthetic map parameters

| Parameter | Options | Effect |
|-----------|---------|--------|
| `hotspot_density` | low (3–5), medium (8–12), high (15–20) | Number of Gaussian hotspot peaks |
| `hotspot_strength` | (3, 12) default | Peak amplitude range added to W |
| `terrain_roughness` | flat / mild / rough | Elevation variance (2 / 8 / 18 m RMS) |
| `obstacle_density` | low / medium / high | Obstacle coverage fraction |
| `n_ecological_zones` | 4 default | Regional base-health offsets |
| Grid size | 50×50, 54×72 | Mission area resolution |

```python
from epca_sensitivity import SyntheticMapConfig, SyntheticMapGenerator

gen = SyntheticMapGenerator()
field = gen.generate(SyntheticMapConfig(N=54, M=72, hotspot_density="high", seed=42))
family = gen.generate_family(50, base_seed=1000)  # 50 reproducible maps
```

## Imperfect priority fields

Detector errors (10–20% FP/FN, confidence noise, spatial blur) and forecaster errors (Gaussian IoT noise, 5% anomaly spikes) are injected before Eq. (3) fusion:

```python
from epca_sensitivity import ImperfectPriorityConfig, corrupt_priority_field

cfg = ImperfectPriorityConfig(false_positive_rate=0.15, false_negative_rate=0.15)
W_hat, high_hat, diag = corrupt_priority_field(field, cfg, rng=42)
```

## Operating τ bound

For each configuration, the framework sweeps mean synchronization interval τ̄ and finds the **largest τ** satisfying:
- HPC ≥ 65%
- Collision rate < 0.4

This bound indicates how stale the digital twin can be before mission KPIs fail.

## Swept dimensions (full study)

| Sweep | Values |
|-------|--------|
| Hotspot density | low, medium, high |
| Fleet size U | 1, 2, 3, 4, 5, 6, 8 |
| Safety distance d_safe | 15–40 m |
| Terrain roughness | flat, mild, rough |
| Grid resolution | 50×50, 54×72 |
| Priority quality | perfect, mild noise (10%), severe noise (20%) |

## Methodology (ready to paste)

> Synthetic mission maps are generated parametrically on 50×50 and 54×72 grids with seed-controlled hotspot clusters (3–20 Gaussian peaks), ecological zone offsets, obstacle fields, and elevation roughness classes (flat/mild/rough). For each map family we run N=50 Monte Carlo trials per configuration, sweeping mean synchronization interval τ̄, fleet size U∈{1,…,8}, safety distance d_safe, and hotspot density. The calibrated staleness model (β_M≈0.017, σ_M=0.08, σ_g≈3.52) with stochastic NTN-like τ is applied at every replanning cycle. Operating bounds are defined as the largest τ̄ satisfying HPC>65% and collision rate<0.4. Upstream sensing imperfection is modelled via 10–20% detector false positives/negatives, confidence noise, and forecaster IoT spikes fused into the priority field W_i=αH_i+βσ_i−γO_i.

## Results (smoke test N=8; run N=50 for publication)

> Operating τ bounds at HPC≥65%, collision<0.4 saturate at τ̄=60 steps for low/medium/high hotspot density in the smoke grid (τ∈{20,40,60}), with HPC@bound 78–88% but rising collision rates at longer intervals (0.05–0.17). Fleet size U=6 tightens the bound to **τ̄≤20** steps (collision 0.33 at τ=20) due to inter-UAV congestion under ghost-position staleness, while U=2–4 tolerate τ̄≤60.

> Imperfect priority fields increase **targeting error** (visits to planner-high cells that are not true hotspots) and inference MAE (~12–18 on hotspot cells) relative to perfect W_i; interpret HPC jointly with WHPC and targeting error because staleness–congestion coupling modulates raw coverage.

## Suggested new figures

1. **Heatmap**: τ_bound vs (hotspot density × d_safe)
2. **Line plot**: τ_bound vs fleet size U (three link-quality colours)
3. **Grouped bars**: HPC and collision for perfect vs mild vs severe priority noise
