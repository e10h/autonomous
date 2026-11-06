import os
import numpy as np
from controller import Node as WbNode, Supervisor

from . import config


class MarkerDrawer:
    def __init__(self, robot: Supervisor) -> None:
        self._robot = robot
        try:
            root = self._robot.getRoot()
            self._children = root.getField('children') if root else None
        except Exception:
            self._children = None
        self._relative: WbNode | None = None
        self._absolute: WbNode | None = None
        self._oa_ribbon: WbNode | None = None
        self._goal: WbNode | None = None
        self._costmap_data = None
        self._elev_grid_data = None

    def get_ground_z(self, x: float, y: float) -> float:
        """Accurately query the ground surface Z elevation at Webots (x, y)."""
        # 1. Fast, exact lookup from static_slope_costmap.npy if present
        if self._costmap_data is None:
            for path in [os.getenv("YONDER_STATIC_SLOPE_MAP", ""), "static_slope_costmap.npy"]:
                if os.path.isfile(path):
                    try:
                        d = np.load(path, allow_pickle=True).item()
                        self._costmap_data = (
                            float(d["x_min"]), float(d["y_min"]), float(d["resolution"]),
                            d["elevation"]
                        )
                        break
                    except Exception:
                        pass

        if self._costmap_data is not None:
            x_min, y_min, res, elev = self._costmap_data
            nr, nc = elev.shape
            i = int(round((x - x_min) / res))
            j = int(round((y - y_min) / res))
            if 0 <= j < nr and 0 <= i < nc:
                return float(elev[j, i])

        # 2. Supervisor lookup via ELEV_GRID node
        try:
            if self._elev_grid_data is None:
                elev_node = self._robot.getFromDef("ELEV_GRID")
                terrain_node = self._robot.getFromDef("GAUNTLET_TERRAIN")
                if elev_node and terrain_node:
                    tpos = terrain_node.getPosition()
                    x_dim = elev_node.getField("xDimension").getSFInt32()
                    y_dim = elev_node.getField("yDimension").getSFInt32()
                    dx = elev_node.getField("xSpacing").getSFFloat()
                    dy = elev_node.getField("ySpacing").getSFFloat()
                    h_field = elev_node.getField("height")
                    heights = [h_field.getMFFloat(idx) for idx in range(h_field.getCount())]
                    self._elev_grid_data = (tpos, x_dim, y_dim, dx, dy, heights)
            if self._elev_grid_data is not None:
                tpos, x_dim, y_dim, dx, dy, heights = self._elev_grid_data
                i = int(round((x - tpos[0]) / dx))
                j = int(round((y - tpos[1]) / dy))
                i = max(0, min(x_dim - 1, i))
                j = max(0, min(y_dim - 1, j))
                return tpos[2] + heights[j * x_dim + i]
        except Exception:
            pass

        # 3. Fallback to chassis position
        try:
            return self._robot.getSelf().getPosition()[2] - 0.40
        except Exception:
            return 0.50

    def draw_absolute_path(self, points: list[tuple[float, float]]) -> None:
        """Draw 3D glowing AR waypoint ribbon floating safely above terrain."""
        if not points:
            return
        # Hover neatly 0.45m above the ground surface so it NEVER intersects the dirt
        bases = [(x, y, self.get_ground_z(x, y) + 0.45) for x, y in points]
        self._absolute = self._replace(self._absolute, bases, '0 1 1')

    def draw_oa_path(self, points: list[tuple[float, float]]) -> None:
        """Draw local obstacle avoidance replanned correction ribbon in glowing tan."""
        if not points:
            if self._oa_ribbon is not None:
                try:
                    self._oa_ribbon.remove()
                except Exception:
                    pass
                self._oa_ribbon = None
            return
        # Hover neatly +0.50m above ground (layered over the +0.45m cyan global path)
        # Using glowing warm tan (RGB 0.85 0.70 0.42)
        bases = [(x, y, self.get_ground_z(x, y) + 0.50) for x, y in points]
        self._oa_ribbon = self._replace(self._oa_ribbon, bases, '0.85 0.70 0.42')

    def draw_relative_path(self, points: list[tuple[float, float]]) -> None:
        self.draw_oa_path(points)

    def draw_goal(self, x: float, y: float) -> None:
        z = self.get_ground_z(x, y) + 0.50
        self._goal = self._replace(self._goal, [(x, y, z)], '1 0 1')

    def _replace(
        self,
        old: WbNode | None,
        bases: list[tuple[float, float, float]],
        rgb: str,
    ) -> WbNode | None:
        if old is not None:
            try:
                old.remove()
            except Exception:
                pass

        if self._children is None:
            try:
                root = self._robot.getRoot()
                if root:
                    self._children = root.getField('children')
            except Exception:
                pass

        if not bases or self._children is None:
            return None

        # Build clean 3D glowing waypoint line:
        # Sleek double-sided ribbon (width 0.18m) hovering above ground
        w = 0.18
        verts = []
        quad_indices = []

        for idx, (x, y, z) in enumerate(bases):
            if idx < len(bases) - 1:
                dx = bases[idx + 1][0] - x
                dy = bases[idx + 1][1] - y
            elif idx > 0:
                dx = x - bases[idx - 1][0]
                dy = y - bases[idx - 1][1]
            else:
                dx, dy = 1.0, 0.0
            L = (dx**2 + dy**2)**0.5
            nx = -dy / L if L > 1e-4 else 0.0
            ny = dx / L if L > 1e-4 else 1.0

            lx = x + nx * (w / 2.0)
            ly = y + ny * (w / 2.0)
            rx = x - nx * (w / 2.0)
            ry = y - ny * (w / 2.0)

            verts.append(f'{lx:.3f} {ly:.3f} {z:.3f}')
            verts.append(f'{rx:.3f} {ry:.3f} {z:.3f}')

            if idx < len(bases) - 1:
                p0 = 2 * idx
                p1 = 2 * idx + 1
                p2 = 2 * idx + 2
                p3 = 2 * idx + 3
                quad_indices.append(f'{p0} {p1} {p3} {p2} -1')

        vrml = f"""
            Group {{
              children [
                Shape {{
                  appearance Appearance {{
                    material Material {{ diffuseColor {rgb} emissiveColor {rgb} }}
                  }}
                  geometry IndexedFaceSet {{
                    coord Coordinate {{ point [ {' '.join(verts)} ] }}
                    coordIndex [ {' '.join(quad_indices)} ]
                  }}
                }}
              ]
            }}
        """

        try:
            self._children.importMFNodeFromString(-1, vrml)
            return self._children.getMFNode(-1)
        except Exception:
            return None
