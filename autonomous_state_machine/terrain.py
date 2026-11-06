"""ROS-independent elevation prior in projected metric coordinates (row, column)."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class TraversabilityLimits:
    cliff_deg: float = 22.0
    climb_deg: float = 22.0
    descent_deg: float = 22.0
    roll_deg: float = 15.0
    uphill_weight: float = 1.0
    downhill_weight: float = 1.5
    roll_weight: float = 3.0

    def __post_init__(self):
        for key in ("cliff_deg", "climb_deg", "descent_deg", "roll_deg"):
            if not 0 < getattr(self, key) < 90:
                raise ValueError(f"{key} must be between 0 and 90 degrees")
        for key in ("uphill_weight", "downhill_weight", "roll_weight"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")


@dataclass
class TerrainGrid:
    """x0/y0 locate the center of cell [0,0]; signed dy preserves north-up rasters."""

    blocked: np.ndarray
    gx_min: np.ndarray
    gx_max: np.ndarray
    gy_min: np.ndarray
    gy_max: np.ndarray
    x0: float
    y0: float
    dx: float
    dy: float
    metadata: dict | None = None

    def __post_init__(self):
        if self.blocked.ndim != 2 or min(self.blocked.shape) < 2:
            raise ValueError("Terrain must have at least two rows and columns")
        if any(a.shape != self.blocked.shape for a in self.gradients):
            raise ValueError("Terrain layers must have matching shapes")
        if (
            not all(math.isfinite(v) for v in (self.x0, self.y0, self.dx, self.dy))
            or self.dx == 0
            or self.dy == 0
        ):
            raise ValueError(
                "Terrain coordinates and resolution must be finite and nonzero"
            )

    @property
    def gradients(self):
        return self.gx_min, self.gx_max, self.gy_min, self.gy_max

    def xy(self, cell):
        row, col = cell
        return self.x0 + col * self.dx, self.y0 + row * self.dy

    def fractional_cell(self, xy):
        x, y = xy
        return (y - self.y0) / self.dy, (x - self.x0) / self.dx

    def cell(self, xy):
        row, col = self.fractional_cell(xy)
        if not all(math.isfinite(v) for v in (row, col)):
            raise ValueError("Coordinates must be finite")
        # Never clamp an out-of-map request into a traversable edge cell.
        if not (
            -0.5 <= row < self.blocked.shape[0] - 0.5
            and -0.5 <= col < self.blocked.shape[1] - 0.5
        ):
            raise ValueError("Requested coordinate lies outside the terrain map")
        return math.floor(row + 0.5), math.floor(col + 0.5)

    def window(self, start, goal, margin_m=100.0):
        if not math.isfinite(margin_m) or margin_m < 0:
            raise ValueError("Window margin must be finite and nonnegative")
        a, b = self.cell(start), self.cell(goal)
        mr, mc = math.ceil(margin_m / abs(self.dy)), math.ceil(margin_m / abs(self.dx))
        r0, r1 = max(0, min(a[0], b[0]) - mr), min(
            self.blocked.shape[0], max(a[0], b[0]) + mr + 1
        )
        c0, c1 = max(0, min(a[1], b[1]) - mc), min(
            self.blocked.shape[1], max(a[1], b[1]) + mc + 1
        )
        if r1 - r0 < 2 or c1 - c0 < 2:
            raise ValueError("Planning window is too small")
        sl = np.s_[r0:r1, c0:c1]
        return TerrainGrid(
            self.blocked[sl],
            *(a[sl] for a in self.gradients),
            *self.xy((r0, c0)),
            self.dx,
            self.dy,
            self.metadata,
        )

    @classmethod
    def open(cls, directory):
        path = Path(directory)
        meta = json.loads((path / "metadata.json").read_text())
        if meta.get("schema") != "yonder.terrain.v1":
            raise ValueError("Unsupported terrain artifact schema")
        arrays = [
            np.load(path / (name + ".npy"), mmap_mode="r", allow_pickle=False)
            for name in ("blocked", "gx_min", "gx_max", "gy_min", "gy_max")
        ]
        return cls(
            *arrays, **{k: meta[k] for k in ("x0", "y0", "dx", "dy")}, metadata=meta
        )
