"""Parametric synthetic priority-map generator for EPCA-M sensitivity studies.

Generates seed-controlled families of realistic 50×50 and 54×72 grids with
configurable hotspot density/strength, terrain roughness, obstacles, and
ecological zone structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from epca_staleness.environment import PriorityField, _distance_to_obstacles, _smooth_field


class HotspotDensity(str, Enum):
    LOW = "low"          # 3–5 Gaussian hotspot centres
    MEDIUM = "medium"    # 8–12 centres
    HIGH = "high"        # 15–20 centres


class TerrainRoughness(str, Enum):
    FLAT = "flat"
    MILD = "mild"
    ROUGH = "rough"


# Peak-count ranges per density class (inclusive).
HOTSPOT_COUNT = {
    HotspotDensity.LOW: (3, 5),
    HotspotDensity.MEDIUM: (8, 12),
    HotspotDensity.HIGH: (15, 20),
}

# Elevation variance scaling (m RMS) per roughness class.
TERRAIN_VARIANCE = {
    TerrainRoughness.FLAT: 2.0,
    TerrainRoughness.MILD: 8.0,
    TerrainRoughness.ROUGH: 18.0,
}

# Obstacle coverage fraction per tree-density label.
OBSTACLE_FRAC = {"low": 0.06, "medium": 0.12, "high": 0.20}


@dataclass
class SyntheticMapConfig:
    """Parameters for one synthetic priority map."""

    N: int = 50
    M: int = 50
    dx: float = 18.0
    seed: int | None = 42
    # Hotspot structure
    hotspot_density: HotspotDensity | str = HotspotDensity.MEDIUM
    hotspot_strength: tuple[float, float] = (3.0, 12.0)   # peak amplitude range
    hotspot_width: tuple[float, float] = (2.5, 7.0)       # Gaussian σ (cells)
    # Terrain & obstacles
    terrain_roughness: TerrainRoughness | str = TerrainRoughness.MILD
    obstacle_density: str = "medium"   # low / medium / high traversability loss
    # Ecological zones (number of Voronoi-like regions with distinct base health)
    n_ecological_zones: int = 4
    zone_health_spread: float = 1.5     # max offset added to base H per zone
    # Composite weights (Eq. 3)
    alpha: float = 1.0
    beta: float = 0.55
    gamma: float = 0.32
    hotspot_frac: float = 0.12          # fraction of traversable cells flagged high


@dataclass
class SyntheticMapGenerator:
    """Factory for reproducible synthetic EPCA-M priority fields."""

    default_config: SyntheticMapConfig = field(default_factory=SyntheticMapConfig)

    def generate(self, config: SyntheticMapConfig | None = None) -> PriorityField:
        cfg = config or self.default_config
        return _build_map(cfg)

    def generate_family(self, n_maps: int, base_seed: int = 0,
                        config: SyntheticMapConfig | None = None) -> list[PriorityField]:
        """Return ``n_maps`` independent maps with seeds ``base_seed + k``."""
        cfg_template = config or self.default_config
        out = []
        for k in range(n_maps):
            cfg = SyntheticMapConfig(**{**cfg_template.__dict__, "seed": base_seed + k})
            out.append(_build_map(cfg))
        return out


def generate_map_family(n_maps: int, **kwargs) -> list[PriorityField]:
    """Convenience wrapper around :class:`SyntheticMapGenerator`."""
    cfg = SyntheticMapConfig(**kwargs)
    return SyntheticMapGenerator(cfg).generate_family(n_maps, base_seed=cfg.seed or 0, config=cfg)


def _resolve_enum(val, enum_cls):
    if isinstance(val, enum_cls):
        return val
    return enum_cls(str(val).lower())


def _ecological_zones(n: int, m: int, n_zones: int, spread: float, rng) -> np.ndarray:
    """Piecewise-constant zone map with distinct base health offsets."""
    zone_id = np.zeros((n, m), dtype=int)
    centres = np.column_stack([
        rng.integers(2, n - 2, size=n_zones),
        rng.integers(2, m - 2, size=n_zones),
    ])
    rr, cc = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    for i in range(n):
        for j in range(m):
            d = np.min((centres[:, 0] - i) ** 2 + (centres[:, 1] - j) ** 2)
            zone_id[i, j] = int(np.argmin((centres[:, 0] - i) ** 2 + (centres[:, 1] - j) ** 2))
    offsets = rng.uniform(-spread, spread, size=n_zones)
    return offsets[zone_id]


def _build_map(cfg: SyntheticMapConfig) -> PriorityField:
    rng = np.random.default_rng(cfg.seed)
    N, M = cfg.N, cfg.M
    density = _resolve_enum(cfg.hotspot_density, HotspotDensity)
    rough = _resolve_enum(cfg.terrain_roughness, TerrainRoughness)
    n_lo, n_hi = HOTSPOT_COUNT[density]
    n_peaks = int(rng.integers(n_lo, n_hi + 1))

    # --- obstacles ------------------------------------------------------------
    target_frac = OBSTACLE_FRAC.get(cfg.obstacle_density, 0.12)
    rmin = {"low": 2, "medium": 2, "high": 3}.get(cfg.obstacle_density, 2)
    obstacle = np.zeros((N, M), dtype=bool)
    max_paint = int(target_frac * N * M)
    guard = 0
    while obstacle.sum() < max_paint and guard < 5000:
        guard += 1
        r0, c0 = int(rng.integers(0, N)), int(rng.integers(0, M))
        rad = int(rng.integers(rmin, rmin + 2))
        for r in range(max(0, r0 - rad), min(N, r0 + rad + 1)):
            for c in range(max(0, c0 - rad), min(M, c0 + rad + 1)):
                if (r - r0) ** 2 + (c - c0) ** 2 <= rad * rad:
                    obstacle[r, c] = True
    obstacle[:2, :] = obstacle[-2:, :] = obstacle[:, :2] = obstacle[:, -2:] = False

    # --- base health with ecological zones ------------------------------------
    base = _smooth_field(N, M, 8, rng)
    stress = _smooth_field(N, M, 5, rng)
    zone_off = _ecological_zones(N, M, cfg.n_ecological_zones, cfg.zone_health_spread, rng)
    H = np.zeros((N, M))
    thr = np.percentile(base.ravel(), [20, 40, 60, 80])
    H[base <= thr[0]] = 0
    H[(base > thr[0]) & (base <= thr[1])] = 1
    H[(base > thr[1]) & (base <= thr[2])] = 2
    H[(base > thr[2]) & (base <= thr[3])] = 3
    H[base > thr[3]] = 4
    H = np.clip(H + zone_off, 0, 4)
    pocket = stress > np.percentile(stress.ravel(), 75)
    H[pocket] = np.minimum(4, H[pocket] + 1)

    sigma = 0.15 + 0.85 * (stress - stress.min()) / max(np.ptp(stress), 1e-9)
    dil = _distance_to_obstacles(obstacle) <= 2
    sigma[dil] = np.minimum(1.0, sigma[dil] + 0.25)
    H[obstacle] = 0
    sigma[obstacle] = 1.0

    if obstacle.any():
        dist = _distance_to_obstacles(obstacle)
        O = np.exp(-dist ** 2 / (2 * 2.5 ** 2))
    else:
        O = np.zeros((N, M))

    # --- Gaussian hotspot peaks -----------------------------------------------
    hotspot = np.zeros((N, M))
    ii, jj = np.meshgrid(np.arange(N), np.arange(M), indexing="ij")
    amp_lo, amp_hi = cfg.hotspot_strength
    wid_lo, wid_hi = cfg.hotspot_width
    peak_coords = []
    for _ in range(n_peaks):
        pr, pc = int(rng.integers(3, N - 3)), int(rng.integers(3, M - 3))
        if obstacle[pr, pc]:
            continue
        amp = rng.uniform(amp_lo, amp_hi)
        wid = rng.uniform(wid_lo, wid_hi)
        hotspot += amp * np.exp(-((ii - pr) ** 2 + (jj - pc) ** 2) / (2 * wid ** 2))
        peak_coords.append((pr, pc, amp))
    hotspot[obstacle] = 0.0

    W = cfg.alpha * H + cfg.beta * sigma - cfg.gamma * O + hotspot
    trav = ~obstacle
    Wtrav = W[trav]
    w_hi = float(np.quantile(Wtrav, 1.0 - cfg.hotspot_frac)) if Wtrav.size else 3.0
    high_mask = (W >= w_hi) & trav

    # --- terrain elevation Z_m ------------------------------------------------
    z_scale = TERRAIN_VARIANCE[rough]
    rough_field = _smooth_field(N, M, 6, rng)
    ctr = np.array([N / 2, M / 2])
    rr, cc = np.meshgrid(np.arange(N), np.arange(M), indexing="ij")
    Z_m = z_scale * (0.6 * rough_field + 0.4 * np.sin(rr / 4) * np.cos(cc / 5))
    Z_m += 0.3 * z_scale * np.exp(-((rr - 0.35 * N) ** 2 + (cc - 0.62 * M) ** 2) / (0.1 * (N + M)))
    Z_m[obstacle] = Z_m[~obstacle].mean() if trav.any() else 0.0

    return PriorityField(
        N=N, M=M, dx=cfg.dx, H=H, sigma=sigma, obstacle=obstacle, O=O, W=W,
        high_mask=high_mask, Z_m=Z_m,
        meta=dict(
            alpha=cfg.alpha, beta=cfg.beta, gamma=cfg.gamma,
            seed=cfg.seed, w_hi=w_hi, hotspot_frac=cfg.hotspot_frac,
            hotspot_density=density.value, n_hotspot_peaks=n_peaks,
            terrain_roughness=rough.value, obstacle_density=cfg.obstacle_density,
            n_ecological_zones=cfg.n_ecological_zones, z_rms=float(np.std(Z_m[trav])),
            peak_coords=peak_coords,
        ),
    )
