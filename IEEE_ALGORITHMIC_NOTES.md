# Algorithmic Depth and Communication Notes ( Draft)

## 1) Coordination Complexity (Sketch)

Let `n` be the number of high-priority target cells considered in a sortie, `U` the fleet size, `I` the number of capacitated Voronoi iterations, and `N,M` the grid dimensions.

| Module | Dominant scaling (order of magnitude) | Notes |
|--------|----------------------------------------|--------|
| Capacitated Voronoi (`SpatialDecomposition.capacitatedVoronoi`) | `O(n U I)` | Per-iteration assignment scans targets against all seeds; `I` is small (tens). |
| Weighted greedy ordering (`PriorityPlanner.weightedGreedyOrder`) | `O(n^2)` worst case | NN-style insertion over remaining goals; typical `n` is far below `N*M`. |
| Grid A* stitching (`astarGrid` per leg) | Roughly `O(N M log(N M))` per long segment | Depends on traversable area and path length; bounded by planner horizon shortcuts. |
| **Decentralized baseline** (`BaselinePlanners.decentralizedVoronoiGreedy`) | `O(n U) + sum_u O(n_u^2)` for greedy orders | No global capacitated refinement; local A* only between consecutive goals per UAV. |

Overall, the hybrid planner is **polynomial in grid and target set size** with **no explicit communication rounds** in code (centralized map knowledge is assumed unless you add a comm model).

## 2) What Is *Not* Modeled Today

- **Inter-UAV communication latency, packet loss, or bandwidth limits** are not simulated.
- `energy_per_bit_J` in `Analytics` / logs refers to **sensing / information bits** (EIG proxy), **not** radio link bits.

## 3) How to Discuss Communication Overhead in the Paper (Two Honest Options)

**Option A — Minimal / honesty paragraph (fastest):**  
State that coordination is computed offline or on a ground station with perfect map knowledge; report planner CPU time as scalability evidence; defer explicit comm to future work.

**Option B — Additive comm model (recommended if reviewers push):**  
Define a per-sortie uplink payload, for example:

- Each UAV sends `B_state` bits per replanning cycle (pose, battery, local summary).
- Ground station broadcasts `B_map` bits when the fused IoT map updates.

Then total comm energy can be approximated as `E_comm ≈ P_radio * (B_up + B_down) / R_link` and added to mission energy as a sensitivity term. This is **independent** of `energy_per_bit_J` unless you deliberately unify budgets (not required).

## 4) Suggested One-Paragraph Paper Wording (Template)

> The simulator assumes centralized fusion of the IoT-informed map and therefore does not count multi-hop radio energy. The metric `energy_per_bit_J` quantifies **sensing efficiency** (joules per expected information bit from the GP surrogate), distinct from **coordination traffic**. We report planner runtime scaling with grid size and fleet size, and we bound safety via pairwise separation checks in `MissionSim`.

## 5) Optional Extension Hooks in Code

- `BatteryAwareOrchestrator` now supports `orchCfg.plannerMode` (`priority`, `greedy`, `decentralized_greedy`) for fair comparative studies.
- `mapData.obsFrac` supports clutter-sensitive `dSafe` scaling and operability plots.

