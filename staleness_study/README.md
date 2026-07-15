# Enhanced EPCA-M Staleness / AoI Simulator

Reproducible Python implementation of the **enhanced digital-twin staleness model** for the EPCA-M paper, coupled to the **IUEF-EM priority planner** (Algorithm 1) on a **50×50 grid with 3 UAVs** and clustered hotspot targets.

## Quick start

```bash
cd staleness_study
pip install -r requirements.txt
python run_staleness_study.py          # full study (N=50)
python run_staleness_study.py --quick  # smoke test (~2 min)
python run_closed_loop.py              # closed-loop sensing-to-planning (N=50)
python run_closed_loop.py --quick      # closed-loop smoke test
```

See [CLOSED_LOOP.md](CLOSED_LOOP.md) for the end-to-end EPCA-M closed-loop simulator.

Outputs land in `output/`:
| File | Content |
|------|---------|
| `Figure_Staleness_TauSweep.{png,pdf}` | HPC & collision vs. mean τ with 95 % CI bands |
| `Figure_Staleness_PolicyComparison.{png,pdf}` | Periodic vs. adaptive at matched uplink cost |
| `Figure_Staleness_Sensitivity.{png,pdf}` | One-at-a-time β_M, σ_g, hotspot-fraction sensitivity |
| `Figure_Staleness_Calibration.{png,pdf}` | Ghost RMSE & map retention vs. τ |
| `staleness_study_summary.json` | Numeric results & operating bounds |

## Package structure

```
staleness_study/
├── run_staleness_study.py          # top-level driver
├── requirements.txt
└── epca_staleness/
    ├── channel.py                  # NTN-like stochastic τ model
    ├── staleness.py                # map fade, ghost drift, calibration
    ├── environment.py              # 50×50 priority field + hotspots
    ├── planner.py                  # IUEF-EM (Algorithm 1) + A*
    ├── mission.py                  # staleness-coupled mission loop
    └── experiments.py              # Monte-Carlo sweeps & plotting
```

## Model summary

### 1. Stochastic τ (NTN-like channel)

Per synchronization event:

```
τ = max(1, round( base_τ · (1 + 0.3·𝒩(0,1)) + outage_penalty ))
```

`outage_penalty` is nonzero with probability `p_outage` and adds `base_τ · outage_scale · Exp(1)`.

| Link   | base_τ | p_outage | outage_scale | E[τ] (MC) |
|--------|--------|----------|--------------|-----------|
| good   | 20     | 0.05     | 0.8          | ~21       |
| medium | 45     | 0.10     | 1.2          | ~51       |
| poor   | 80     | 0.15     | 1.8          | ~100      |

### 2. Age & degradation (enhanced)

```
age(t) = min(AGE_CAP, steps_since_sync / τ_ref)
κ(τ)   = (τ − 1) / (2τ)
```

**Map fade** (cumulative interval retention for planning belief):

```
R(τ) = exp( τ · ∫₀¹ ln(1 − β_M·x) dx )
M̂_plan = max(0, M · R(τ_plan) + 𝒩(0, σ_M² · κ(τ_plan)))
```

**Ghost drift** (collision avoidance uses stale teammate positions):

```
p̃_v = p_v + 𝒩(0, σ_g² · age(t) · log(1 + τ_ref))
```

### 3. Calibration

```python
from epca_staleness.staleness import calibrate_ghost_sigma, calibrate_map_fade

sigma_g = calibrate_ghost_sigma(target_rmse_cells=10.0, tau_ref=60)  # → ~3.5 cells⁻¹
beta_M  = calibrate_map_fade(target_retention=0.60, tau_ref=60)     # → ~0.017
```

Default calibrated set (τ_ref = 60):
- **β_M = 0.0169** (map fade; paper's 0.65 is per-step at fixed age — recalibrated for cumulative model)
- **σ_M = 0.08** (unchanged)
- **σ_g = 3.52** (yields ~10-cell ghost RMSE at τ = 60, age ≈ 1)

### 4. Adaptive synchronization

Event-triggered sync when `steps_since_sync ≥ adapt_age_steps`. For fair comparison, `adapt_age_steps` is tuned so mean uplink cost matches periodic on the same link. Adaptive plans on `τ_plan = adapt_age_steps` (shorter horizon → higher retention R), which **caps peak AoI during channel outages** and reduces collision bursts on medium links.

### 5. Integration with IUEF-EM planner

At each sync the planner receives the **stale belief** `W_est` (not ground truth). Cells whose faded weight drops below the hotspot threshold `w_hi` are **not targeted** — there is no fallback to full-grid coverage. UAVs with no detectable hotspots patrol toward the grid centre (loss of situational awareness).

## Parameter recommendations

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Grid | 50×50, dx = 18 m | Matches paper baseline |
| UAVs | 3 | Matches paper baseline |
| hotspot_frac | 0.12 | ~12 % of traversable cells are high-priority |
| horizon | 600 steps (Δt = 1 s) | 10-minute sortie |
| d_safe | 25 m | True collision threshold |
| d_near | 45 m | Near-miss reporting |
| ghost RMSE target | 8–12 cells @ τ=60 | Use `calibrate_ghost_sigma` |
| map retention target | 0.55–0.65 @ τ=60 | Use `calibrate_map_fade` |
| Monte Carlo N | ≥ 50 | Per sweep point |

## Suggested new figures for the paper

1. **Fig. S1 – Calibration curves**: ghost RMSE and R(τ) vs. τ with τ = 60 reference lines.
2. **Fig. S2 – HPC & collision vs. τ̄**: dual-panel sweep with 95 % CI bands, three link colours, horizontal bounds at HPC = 65 % and collision = 0.4.
3. **Fig. S3 – Periodic vs. adaptive**: grouped bars at matched uplink cost; highlight collision reduction on medium link.
4. **Fig. S4 – Sensitivity**: β_M, σ_g, hotspot fraction (one-at-a-time).
5. **Fig. S5 – AoI timeline**: example mission showing saw-tooth age, sync events, and outage-induced age spikes (optional extension).

## Methodology sentences (copy-ready)

> We extend the fixed-interval staleness model by drawing the synchronization interval τ as a random variable from a lightweight NTN-like channel model comprising Gaussian jitter (coefficient 0.3), packet-loss-driven retransmission delays, and occasional deep-fade outages (probability 5–15 % depending on link quality). The normalized average Age of Information κ(τ) = (τ−1)/(2τ) characterizes the mean staleness within an interval.

> Map degradation is applied through a cumulative interval retention factor R(τ) obtained by integrating the multiplicative fade (1 − β_M·age) over the saw-tooth age profile, ensuring that longer intervals compound belief decay rather than saturating at a fixed κ. Ghost-position uncertainty scales as σ_g²·age(t)·log(1+τ), coupling collision risk to both intra-interval age and inter-sync spacing.

> Degradation parameters are calibrated so that end-of-interval ghost RMSE equals 10 grid cells and map retention R(60) = 0.60 at the reference interval τ = 60, yielding σ_g ≈ 3.5 and β_M ≈ 0.017 under the cumulative fade model.

> An event-triggered adaptive synchronization policy re-syncs when the estimated age exceeds a step threshold tuned to match the periodic policy's average uplink cost; by planning on a shorter horizon τ_plan the adaptive policy retains a higher fraction of hotspot cells and caps peak AoI during channel outages, reducing collision bursts on medium-quality links.

## Results sentences (copy-ready)

> Monte-Carlo evaluation (N = 50, 50×50 grid, 3 UAVs) shows that high-priority coverage decreases monotonically with mean synchronization interval while collision rate grows super-linearly once ghost RMSE exceeds the safe separation margin (~8–12 cells). Under calibrated parameters, tolerable operation (HPC > 65 %, collision < 0.4) is confined to mean τ ≤ 20–35 steps on medium links and τ ≤ 50–60 steps on good links.

> Compared to periodic synchronization at equal uplink cost, the adaptive policy reduces mean collision rate by up to **94 %** on medium links (e.g., 0.24 → 0.015) while improving HPC (+4 pp), because early re-sync caps peak AoI during outage-extended intervals and plans on `τ_plan = 0.65·τ_ref` rather than the full stochastic draw.

> One-at-a-time sensitivity confirms that collision rate is dominated by σ_g (ghost drift) while HPC is most sensitive to β_M (map fade) and hotspot spatial density; jointly, these define the operating envelope for NTN-backed digital-twin swarms.

## Relation to MATLAB codebase

The parent repository's `PriorityPlanner.m` / `MissionSim.m` implement the **un-stale** planner and mission engine. This Python package adds the staleness layer on top of a port of Algorithm 1. To integrate into MATLAB, call the Python study via `system('python run_staleness_study.py')` or port `channel.py` / `staleness.py` / `mission.py` logic into a new `StalenessMissionSim.m` class.

## License

Same as parent EPCA-M repository.
