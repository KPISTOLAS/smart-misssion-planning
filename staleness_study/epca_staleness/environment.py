"""Priority-field / map generation for the EPCA-M staleness study.

This is a dependency-light Python port of the essential fields produced by the
MATLAB ``MapGenerator.m``:

  * discrete plant-health ``H in {0,...,4}``,
  * sensing uncertainty ``sigma in (0, 1]``,
  * binary obstacle mask,
  * obstacle-proximity penalty ``O in [0, 1]``,
  * composite IoT priority weight ``W = alpha*H + beta*sigma - gamma*O + hotspot``,
    where ``H in {0,...,4}`` (not unit-normalized), ``sigma,O in [0,1]``, and
    clustered Gaussian hotspots add 3--9 to ``W`` on peak cells,
  * high-priority (hotspot) mask ``H >= high_health_thr``.

Defaults reproduce the paper's baseline: a 50x50 grid with 3 UAVs and a
clustered hotspot structure (drought / pest pockets).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


def _smooth_field(n: int, m: int, w: int, rng) -> np.ndarray:
    """Box-blurred white noise -> a smooth random scalar field in [0, 1]."""
    raw = rng.random((n, m))
    kernel = np.ones((w, w)) / (w * w)
    # Separable convolution with 'same' semantics via padding + cumulative sums.
    pad = w // 2
    padded = np.pad(raw, pad, mode="reflect")
    out = np.zeros_like(raw)
    for i in range(n):
        for j in range(m):
            out[i, j] = padded[i:i + w, j:j + w].mean()
    out -= out.min()
    denom = out.max() if out.max() > 0 else 1.0
    return out / denom


def _percentile_thresholds(x: np.ndarray, ps) -> np.ndarray:
    return np.percentile(x.ravel(), ps)


def _distance_to_obstacles(obstacle: np.ndarray) -> np.ndarray:
    """8-connected multi-source BFS distance transform to nearest obstacle."""
    n, m = obstacle.shape
    D = np.full((n, m), np.inf)
    from collections import deque
    q = deque()
    for i in range(n):
        for j in range(m):
            if obstacle[i, j]:
                D[i, j] = 0.0
                q.append((i, j))
    if not q:
        D[:] = max(n, m)
        return D
    nbr = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while q:
        r, c = q.popleft()
        base = D[r, c]
        for dr, dc in nbr:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < m and base + 1 < D[nr, nc]:
                D[nr, nc] = base + 1
                q.append((nr, nc))
    return D


@dataclass
class PriorityField:
    """Container for a generated priority field / map instance."""

    N: int
    M: int
    dx: float                         # meters per cell edge
    H: np.ndarray                     # discrete health in {0..4}
    sigma: np.ndarray                 # sensing uncertainty in (0,1]
    obstacle: np.ndarray             # bool obstacle mask
    O: np.ndarray                     # obstacle proximity in [0,1]
    W: np.ndarray                     # composite priority weight
    high_mask: np.ndarray            # bool hotspot mask (H >= thr) & traversable
    Z_m: np.ndarray = None           # optional elevation (m) for A* slope costs
    meta: dict = field(default_factory=dict)

    @property
    def traversable(self) -> np.ndarray:
        return ~self.obstacle

    @property
    def n_high(self) -> int:
        return int(np.count_nonzero(self.high_mask))


def build_priority_field(N: int = 50,
                         M: int = 50,
                         dx: float = 18.0,
                         alpha: float = 1.0,
                         beta: float = 0.55,
                         gamma: float = 0.32,
                         high_health_thr: int = 3,
                         tree_density: str = "medium",
                         hotspot_frac: float = 0.12,
                         seed: int | None = 42) -> PriorityField:
    """Build a 50x50 (default) EPCA-M priority field with clustered hotspots.

    Parameters mirror ``MapGenerator.build`` in the MATLAB codebase, with an
    extra ``hotspot_frac`` that fixes the fraction of traversable cells flagged
    high-priority (the top ``hotspot_frac`` by composite weight ``W``).  Keeping
    hotspots sparse and clustered makes high-priority coverage (HPC) sensitive
    to staleness rather than raw fleet capacity.
    """
    rng = np.random.default_rng(seed)

    # --- obstacles (trees / no-fly patches) -----------------------------------
    # Paint circular tree clusters until a target coverage fraction is reached
    # (keeps the traversable field open enough for a meaningful 50x50 study).
    target_frac = {"low": 0.06, "medium": 0.12, "high": 0.20}[tree_density]
    rmin = {"low": 2, "medium": 2, "high": 3}[tree_density]
    obstacle = np.zeros((N, M), dtype=bool)
    max_paint = int(target_frac * N * M)
    guard = 0
    while obstacle.sum() < max_paint and guard < 5000:
        guard += 1
        r0 = int(rng.integers(0, N))
        c0 = int(rng.integers(0, M))
        rad = int(rng.integers(rmin, rmin + 2))
        r1, r2 = max(0, r0 - rad), min(N, r0 + rad + 1)
        c1, c2 = max(0, c0 - rad), min(M, c0 + rad + 1)
        for r in range(r1, r2):
            for c in range(c1, c2):
                if (r - r0) ** 2 + (c - c0) ** 2 <= rad * rad:
                    obstacle[r, c] = True
    # Keep border strip free (depot-style starts).
    obstacle[:2, :] = obstacle[-2:, :] = obstacle[:, :2] = obstacle[:, -2:] = False

    # --- health H and uncertainty sigma --------------------------------------
    base = _smooth_field(N, M, 8, rng)
    stress = _smooth_field(N, M, 5, rng)
    H = np.zeros((N, M))
    thr = _percentile_thresholds(base, [20, 40, 60, 80])
    H[base <= thr[0]] = 0
    H[(base > thr[0]) & (base <= thr[1])] = 1
    H[(base > thr[1]) & (base <= thr[2])] = 2
    H[(base > thr[2]) & (base <= thr[3])] = 3
    H[base > thr[3]] = 4
    # Clustered stress pockets raise health -> concentrated hotspots.
    pocket = stress > np.percentile(stress.ravel(), 75)
    H[pocket] = np.minimum(4, H[pocket] + 1)

    sigma = 0.15 + 0.85 * (stress - stress.min()) / max(np.ptp(stress), 1e-9)
    # Obstacles obscure sensing -> raise nearby uncertainty.
    dil = _distance_to_obstacles(obstacle) <= 2
    sigma[dil] = np.minimum(1.0, sigma[dil] + 0.25)
    H[obstacle] = 0
    sigma[obstacle] = 1.0

    # --- obstacle proximity O and composite weight W -------------------------
    if obstacle.any():
        dist = _distance_to_obstacles(obstacle)
        O = np.exp(-dist ** 2 / (2 * 2.5 ** 2))
    else:
        O = np.zeros((N, M))

    # --- clustered hotspot intensity (pest / drought centres) ----------------
    # A handful of Gaussian peaks of *varying amplitude* create a wide dynamic
    # range of priority within the hotspot set.  Under staleness the weaker
    # hotspot cells (peak skirts) fade below the planner threshold first, so
    # HPC degrades gradually with tau instead of collapsing all at once.
    n_peaks = 8
    hotspot = np.zeros((N, M))
    ii, jj = np.meshgrid(np.arange(N), np.arange(M), indexing="ij")
    for _ in range(n_peaks):
        pr, pc = rng.integers(3, N - 3), rng.integers(3, M - 3)
        amp = rng.uniform(3.0, 9.0)
        wid = rng.uniform(3.0, 6.0)
        hotspot += amp * np.exp(-((ii - pr) ** 2 + (jj - pc) ** 2) / (2 * wid ** 2))
    hotspot[obstacle] = 0.0

    W_base = alpha * H + beta * sigma - gamma * O
    W = W_base + hotspot

    # Sparse, clustered hotspots: top `hotspot_frac` of traversable cells by W.
    trav = ~obstacle
    Wtrav = W[trav]
    if Wtrav.size > 0:
        w_hi = float(np.quantile(Wtrav, 1.0 - hotspot_frac))
    else:
        w_hi = float(high_health_thr)
    high_mask = (W >= w_hi) & trav

    # Mild hilly terrain for elevation-aware A* (optional but enabled by default).
    ctr = np.array([N / 2, M / 2])
    rr, cc = np.meshgrid(np.arange(N), np.arange(M), indexing="ij")
    Z_m = 12.0 * np.exp(-((rr - ctr[0]) ** 2 + (cc - ctr[1]) ** 2) / (0.15 * (N + M)))
    Z_m += 6.0 * np.exp(-((rr - 0.35 * N) ** 2 + (cc - 0.62 * M) ** 2) / (0.08 * (N + M)))
    Z_m[obstacle] = Z_m[~obstacle].mean() if trav.any() else 0.0

    return PriorityField(
        N=N, M=M, dx=dx, H=H, sigma=sigma, obstacle=obstacle, O=O, W=W,
        high_mask=high_mask, Z_m=Z_m,
        meta=dict(
            alpha=alpha, beta=beta, gamma=gamma, high_health_thr=high_health_thr,
            tree_density=tree_density, seed=seed, hotspot_frac=hotspot_frac,
            w_hi=w_hi,
            W_min=float(W[trav].min()) if trav.any() else 0.0,
            W_max=float(W[trav].max()) if trav.any() else 0.0,
            W_base_min=float(W_base[trav].min()) if trav.any() else 0.0,
            W_base_max=float(W_base[trav].max()) if trav.any() else 0.0,
            H_norm_range=[0.0, 1.0],
            sigma_range=[0.0, 1.0],
            O_range=[0.0, 1.0],
            normalization_note=(
                "W uses H in {0..4} and hotspot bumps; only sigma,O are in [0,1]. "
                "Planner thresholds use quantile w_hi on raw W, not unit cube."
            ),
        ),
    )


def priority_field_W_stats(seed: int = 42) -> dict:
    """Report observed W range at a fixed seed (for manifest / hyperparam tables)."""
    field = build_priority_field(seed=seed)
    trav = field.traversable
    W = field.W[trav]
    Hn = field.H[trav] / 4.0
    return {
        "seed": seed,
        "W_min": float(W.min()),
        "W_max": float(W.max()),
        "w_hi": float(field.meta.get("w_hi", 0.0)),
        "H_norm_min": float(Hn.min()),
        "H_norm_max": float(Hn.max()),
        "sigma_min": float(field.sigma[trav].min()),
        "sigma_max": float(field.sigma[trav].max()),
        "O_min": float(field.O[trav].min()),
        "O_max": float(field.O[trav].max()),
    }
