# IEEE Experiment Plan (Swarm Size Optimization)

## 1) Study Objective
- Quantify the optimal UAV fleet size for adaptive PHM-CPP under safety, energy, and coverage constraints.
- Primary claim: an intermediate fleet size is often better than both very small and very large fleets.

## 2) Controlled Design
- Map grid: `N=54`, `M=72`, `dx=18`.
- Scenario matrix: `terrain x treeDensity = {flat,hilly,ridge} x {low,medium,high}`.
- Seeds per scenario: default `20` (increase to `30+` for final paper).
- Fleet sweep: default `1:12`.
- Shared mission target: `HPC >= 85%` via `BatteryAwareOrchestrator`.

## 3) Reproducible Pipeline
- Run raw experiments:
  - `runIEEEStudy`
  - Produces `ieeeStudyRaw.mat` + `ieeeStudyRaw.csv`.
- Analyze and summarize:
  - `analyzeIEEEStudy`
  - Produces `ieeeStudySummary.mat`, `ieeeSummaryByFleet.csv`, `ieeeOptimalCounts.csv`, and figures.

## 4) Metrics to Report (Main Paper)
- Mission time to target (`mission_time_s`).
- Fleet energy (`fleet_energy_MJ`).
- Energy per bit (`energy_per_bit_J`).
- HPC achieved (`HPC_pct`) and target-hit rate.
- Collision rate (`any_collision`).
- Composite score from `SwarmSizeOptimizer` for decision support.

## 5) Statistical Reporting
- For each fleet size, report:
  - mean, std, 95% CI (already in `ieeeSummaryByFleet.csv` for composite).
- Add pairwise significance tests in final manuscript:
  - preferred: Wilcoxon signed-rank (robust), or paired t-test if normality is supported.
- Include effect sizes (Cohen's d already approximated in summary).

## 6) Figures for IEEE Paper
- Fleet ranking curve: mean composite score with 95% CI (`ieeeFigure_FleetScore.fig`).
- Histogram of optimal fleet-size frequency (`ieeeFigure_OptimalHistogram.fig`).
- Optional additional plots from raw table:
  - collision rate vs fleet size,
  - HPC target-hit rate vs fleet size,
  - mission time vs fleet size.

## 7) Suggested Paper Section Mapping
- **Methods / Experimental Setup**: scenario matrix, seeds, fixed parameters, fairness protocol.
- **Results**: `ieeeSummaryByFleet.csv` + two figures.
- **Ablation / Sensitivity**: vary `scoreOpts` weights and show recommendation stability.
- **Limitations**: simulator assumptions (communication delay, wind, sensing noise model).

## 8) Checklist Before Submission
- Increase seeds to final value and rerun.
- Freeze all config values in scripts and cite them in paper text.
- Export final figures to vector format (PDF/EPS) with publication fonts.
- Add a short reproducibility appendix with exact script names and command order.

## 9) Comparative Analysis (Gold Standard)
- **Algorithm X (proposed)**: `BatteryAwareOrchestrator` with `plannerMode = 'priority'` (hybrid PriorityPlanner).
- **Baseline A**: `plannerMode = 'greedy'` — standard myopic multi-UAV greedy CPP.
- **Baseline B**: `plannerMode = 'decentralized_greedy'` — nearest-depot Voronoi assignment of high-priority targets plus per-UAV greedy sequencing and local A* stitching only (engineering surrogate for decentralized task allocation; cite as such in text).
- Scripts:
  - `runIEEEComparativeStudy` → `ieeeComparativeRaw.mat` / `.csv`
  - `analyzeIEEEComparativeStudy` → `ieeeComparativeByPlannerFleet.csv`, `ieeeComparativeSaturation.csv`, figures `ieeeFigure_ComparativeComposite.fig`, `ieeeFigure_CollisionCapacity.fig`, `ieeeFigure_OperabilitySurface.fig`

## 10) Scalability and Capacity Framing
- Report **collision rate vs fleet size** per planner (`ieeeFigure_CollisionCapacity.fig`).
- Use `ieeeComparativeSaturation.csv` (estimated first fleet size where mean collision rate exceeds a threshold) as a **saturation / capacity boundary** narrative when full safety to 12 UAVs is not guaranteed by the simulator.
- Separation margin scales with fleet size and `mapData.obsFrac` inside `SwarmSizeOptimizer.buildOrchCfgForFleet` (obstacle-fraction-aware `dSafe`).

## 11) Sensitivity / Operability Map
- Raw tables include `obsFrac` (fraction of obstacle cells) for continuous clutter reporting alongside discrete `treeDensity` tags.
- `ieeeFigure_OperabilitySurface.fig`: mean composite score over obstacle-fraction bins × fleet size for the **priority** planner (Algorithm X).

## 12) Algorithmic and Communication Discussion
- See `IEEE_ALGORITHMIC_NOTES.md` for asymptotic complexity sketch and how to separate **sensing information bits** (`energy_per_bit_J`) from **inter-agent communication** (not modeled in `MissionSim`; add explicit comm budget if you extend the simulator).
