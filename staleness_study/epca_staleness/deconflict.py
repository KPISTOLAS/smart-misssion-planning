"""Space-time cell reservation for multi-UAV deconfliction (Eq. 3 hard constraint).

Each UAV reserves ``(row, col, t)`` tuples along its intended path. A* treats
occupied space-time cells as blocked for other UAVs. Reports pre- and
post-deconfliction collision rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SpaceTimeReservation:
    """Centralized space-time reservation table (Tier-3 planner)."""

    horizon: int
    n_rows: int
    n_cols: int
    # reservations[u][t] = (r, c) or None if idle
    reservations: dict[int, dict[int, tuple[int, int]]] = field(default_factory=dict)

    def is_blocked(self, u: int, t: int, r: int, c: int) -> bool:
        """True if another UAV has reserved (r,c) at time t."""
        for other, schedule in self.reservations.items():
            if other == u:
                continue
            pos = schedule.get(t)
            if pos is not None and pos == (r, c):
                return True
        return False

    def reserve_path(self, u: int, path: list, start_t: int = 0):
        """Reserve cells along ``path`` starting at simulation time ``start_t``."""
        sched = self.reservations.setdefault(u, {})
        for i, cell in enumerate(path):
            r, c = int(cell[0]), int(cell[1])
            sched[start_t + i] = (r, c)

    def clear_uav(self, u: int):
        self.reservations.pop(u, None)


def build_reservations(segments: list, horizon: int, n_rows: int, n_cols: int,
                       start_offsets: list[int] | None = None) -> SpaceTimeReservation:
    """Pre-reserve full planned paths for all UAVs."""
    st = SpaceTimeReservation(horizon=horizon, n_rows=n_rows, n_cols=n_cols)
    if start_offsets is None:
        start_offsets = [0] * len(segments)
    for u, seg in enumerate(segments):
        st.reserve_path(u, seg, start_t=start_offsets[u])
    return st


def apply_spacetime_hold(u: int, t: int, nxt: tuple, ghosts_or_peers: np.ndarray,
                         reservation: SpaceTimeReservation | None,
                         d_avoid_cells: float) -> bool:
    """Return True if UAV u must hold at t (ghost avoidance or ST reservation)."""
    if reservation is not None:
        r, c = int(round(nxt[0])), int(round(nxt[1]))
        if reservation.is_blocked(u, t, r, c):
            return True
    for v in range(ghosts_or_peers.shape[0]):
        if v == u:
            continue
        dgc = np.hypot(nxt[0] - ghosts_or_peers[v, 0], nxt[1] - ghosts_or_peers[v, 1])
        if dgc < d_avoid_cells:
            return True
    return False


def pairwise_collision_rate(positions: np.ndarray, U: int,
                            d_safe_cells: float, d_near_cells: float) -> tuple[bool, bool]:
    coll = near = False
    for a in range(U):
        for b in range(a + 1, U):
            d = np.hypot(positions[a, 0] - positions[b, 0],
                         positions[a, 1] - positions[b, 1])
            if d < d_safe_cells:
                coll = True
            if d < d_near_cells:
                near = True
    return coll, near
