"""Direction-aware Theta* with segment-integrated costs and fail-closed limits."""

from functools import lru_cache
import heapq
import math
import time
import numpy as np
from scipy.ndimage import binary_dilation, maximum_filter, minimum_filter
from .terrain import TerrainGrid, TraversabilityLimits


class PlanningBudgetExceeded(RuntimeError):
    pass


def trace_segment(a, b):
    """Exact grid traversal, including both side cells at a corner crossing.

    Coordinates are fractional (row, column), with integer cell centers. Yield
    (row, column, fraction of segment length inside cell); corner-only contacts
    have zero length but still require a feasibility check.
    """
    r, c = (math.floor(v + 0.5) for v in a)
    end_r, end_c = (math.floor(v + 0.5) for v in b)
    dr, dc = b[0] - a[0], b[1] - a[1]
    sr, sc = (1 if dr > 0 else -1), (1 if dc > 0 else -1)
    tr = ((r + 0.5 * sr - a[0]) / dr) if dr else math.inf
    tc = ((c + 0.5 * sc - a[1]) / dc) if dc else math.inf
    dtr, dtc = (1 / abs(dr) if dr else math.inf), (1 / abs(dc) if dc else math.inf)
    t = 0.0
    while True:
        next_t = min(tr, tc, 1.0)
        yield r, c, max(0.0, next_t - t)
        if next_t >= 1.0:
            if (r, c) != (end_r, end_c):
                yield end_r, end_c, 0.0
            return
        if abs(tr - tc) < 1e-12:
            yield r + sr, c, 0.0
            yield r, c + sc, 0.0
            r += sr
            c += sc
            tr += dtr
            tc += dtc
        elif tr < tc:
            r += sr
            tr += dtr
        else:
            c += sc
            tc += dtc
        t = next_t


class DirectionalThetaStar:
    """Costs depend on movement direction, not on arrival heading/turn radius.

    This is an any-angle heuristic search, not a proof of global optimality.
    Local control must additionally validate kinematics and dynamic obstacles.
    """

    def __init__(
        self,
        terrain: TerrainGrid,
        limits=None,
        robot_radius_m=0.8,
        dynamic_blocked=None,
    ):
        self.terrain = terrain
        self.limits = limits or TraversabilityLimits()
        if not math.isfinite(robot_radius_m) or robot_radius_m < 0:
            raise ValueError("Robot radius must be finite and nonnegative")
        gx0, gx1, gy0, gy1 = terrain.gradients
        valid = (
            np.isfinite(gx0) & np.isfinite(gx1) & np.isfinite(gy0) & np.isfinite(gy1)
        )
        valid &= (gx0 <= gx1) & (gy0 <= gy1)
        steepness = np.hypot(
            np.maximum(abs(gx0), abs(gx1)), np.maximum(abs(gy0), abs(gy1))
        )
        blocked = (
            np.asarray(terrain.blocked, dtype=bool)
            | ~valid
            | (steepness > math.tan(math.radians(self.limits.cliff_deg)) + 1e-7)
        )
        if dynamic_blocked is not None:
            if dynamic_blocked.shape != blocked.shape:
                raise ValueError("Dynamic overlay shape does not match prior")
            blocked |= dynamic_blocked
        # Conservative circular footprint: cells touched by the body are hazards.
        padding = robot_radius_m + (
            math.hypot(terrain.dx, terrain.dy) / 2 if robot_radius_m else 0
        )
        nr, nc = math.ceil(padding / abs(terrain.dy)), math.ceil(
            padding / abs(terrain.dx)
        )
        rr, cc = np.mgrid[-nr : nr + 1, -nc : nc + 1]
        footprint = (rr * terrain.dy) ** 2 + (
            cc * terrain.dx
        ) ** 2 <= padding**2 + 1e-12
        self.blocked = binary_dilation(blocked, structure=footprint, border_value=1)
        # Bound gradients across the body as well as across each coarse DEM cell.
        self.gx_min = minimum_filter(
            gx0, footprint=footprint, mode="constant", cval=np.nan
        )
        self.gx_max = maximum_filter(
            gx1, footprint=footprint, mode="constant", cval=np.nan
        )
        self.gy_min = minimum_filter(
            gy0, footprint=footprint, mode="constant", cval=np.nan
        )
        self.gy_max = maximum_filter(
            gy1, footprint=footprint, mode="constant", cval=np.nan
        )
        self._cached_edge = lru_cache(maxsize=100_000)(self._cell_edge)
        self.expanded = 0

    def _rate(self, r, c, vx, vy):
        if (
            r < 0
            or c < 0
            or r >= self.blocked.shape[0]
            or c >= self.blocked.shape[1]
            or self.blocked[r, c]
        ):
            return math.inf
        gx0, gx1, gy0, gy1 = (
            self.gx_min[r, c],
            self.gx_max[r, c],
            self.gy_min[r, c],
            self.gy_max[r, c],
        )
        if not all(math.isfinite(v) for v in (gx0, gx1, gy0, gy1)):
            return math.inf

        def interval(x, y):
            return (
                min(gx0 * x, gx1 * x) + min(gy0 * y, gy1 * y),
                max(gx0 * x, gx1 * x) + max(gy0 * y, gy1 * y),
            )

        pitch_min, pitch_max = interval(vx, vy)
        roll_min, roll_max = interval(-vy, vx)
        up, down, roll = (
            max(0, pitch_max),
            max(0, -pitch_min),
            max(abs(roll_min), abs(roll_max)),
        )
        lim = self.limits
        up_limit = math.tan(math.radians(lim.climb_deg))
        down_limit = math.tan(math.radians(lim.descent_deg))
        roll_limit = math.tan(math.radians(lim.roll_deg))
        if up > up_limit + 1e-7 or down > down_limit + 1e-7 or roll > roll_limit + 1e-7:
            return math.inf
        return (
            1
            + lim.uphill_weight * (up / up_limit) ** 2
            + lim.downhill_weight * (down / down_limit) ** 2
            + lim.roll_weight * (roll / roll_limit) ** 2
        )

    def segment_cost(self, a, b):
        """Metric cost, inf if any touched cell or footprint violates the model."""
        try:
            self.terrain.cell(a)
            self.terrain.cell(b)
        except ValueError:
            return math.inf
        distance = math.dist(a, b)
        if distance == 0:
            r, c = self.terrain.cell(a)
            return 0.0 if not self.blocked[r, c] else math.inf
        vx, vy = (b[0] - a[0]) / distance, (b[1] - a[1]) / distance
        cost = 0.0
        for r, c, fraction in trace_segment(
            self.terrain.fractional_cell(a), self.terrain.fractional_cell(b)
        ):
            rate = self._rate(r, c, vx, vy)
            if not math.isfinite(rate):
                return math.inf
            cost += fraction * distance * rate
        return cost

    def _cell_edge(self, a, b):
        return self.segment_cost(self.terrain.xy(a), self.terrain.xy(b))

    def plan(self, start_xy, goal_xy, max_expansions=200_000, timeout_s=30.0):
        if max_expansions < 1 or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Planning budgets must be positive and finite")
        start = self.terrain.cell(start_xy)
        goal = self.terrain.cell(goal_xy)
        if self.blocked[start] or self.blocked[goal]:
            return []
        if start == goal:
            return (
                [start_xy, goal_xy]
                if math.isfinite(self.segment_cost(start_xy, goal_xy))
                else []
            )
        initial = self.segment_cost(start_xy, self.terrain.xy(start))
        if not math.isfinite(initial):
            return []
        queue = [(math.dist(start_xy, goal_xy), initial, start)]
        scores = {start: initial}
        parents = {start: start}
        self.expanded = 0
        deadline = time.monotonic() + timeout_s
        while queue:
            _, cost, current = heapq.heappop(queue)
            if cost != scores.get(current):
                continue
            if self.expanded >= max_expansions or time.monotonic() > deadline:
                raise PlanningBudgetExceeded(
                    "Planning budget exceeded; reduce search window or use a coarser conservative prior"
                )
            self.expanded += 1
            if current == goal:
                cells = [current]
                while parents[current] != current:
                    current = parents[current]
                    cells.append(current)
                points = [self.terrain.xy(c) for c in reversed(cells)]
                points[0], points[-1] = start_xy, goal_xy
                # Check exact endpoints too; never erase hazards to attach a goal.
                return (
                    points
                    if all(
                        math.isfinite(self.segment_cost(a, b))
                        for a, b in zip(points, points[1:])
                    )
                    else []
                )
            r, c = current
            for dr, dc in (
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ):
                neighbor = (r + dr, c + dc)
                if (
                    not (
                        0 <= neighbor[0] < self.blocked.shape[0]
                        and 0 <= neighbor[1] < self.blocked.shape[1]
                    )
                    or self.blocked[neighbor]
                ):
                    continue
                # In heterogeneous terrain the visible parent shortcut can cost
                # MORE than the path via current, so compare both alternatives.
                candidate, via = cost + self._cached_edge(current, neighbor), current
                parent = parents[current]
                if parent != current:
                    shortcut = scores[parent] + self._cached_edge(parent, neighbor)
                    if shortcut <= candidate + 1e-10:
                        candidate, via = shortcut, parent
                if math.isfinite(candidate) and candidate + 1e-10 < scores.get(
                    neighbor, math.inf
                ):
                    scores[neighbor] = candidate
                    parents[neighbor] = via
                    heapq.heappush(
                        queue,
                        (
                            candidate + math.dist(self.terrain.xy(neighbor), goal_xy),
                            candidate,
                            neighbor,
                        ),
                    )
        return []


def sparse_waypoints(points, max_spacing_m=25.0):
    """Preserve ALL route bends; split long legs without shortcutting hazards."""
    if not math.isfinite(max_spacing_m) or max_spacing_m <= 0:
        raise ValueError("Waypoint spacing must be finite and positive")
    if not points:
        return []
    result = [tuple(points[0])]
    for a, b in zip(points, points[1:]):
        pieces = max(1, math.ceil(math.dist(a, b) / max_spacing_m))
        result.extend(
            (a[0] + (b[0] - a[0]) * i / pieces, a[1] + (b[1] - a[1]) * i / pieces)
            for i in range(1, pieces + 1)
        )
    return result
