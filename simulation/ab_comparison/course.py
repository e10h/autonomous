"""Deterministic A/B course: a loose-scree side-slope above a chasm, with a level bypass.

Mechanism test, not a sampled Utah DEM. Both planners use the same elevation,
obstacle mask, and footprint.

Terrain (Webots frame, per lane):
  * Start (-14, 0) and goal (14, 0) sit on level ground.
  * Between them a 21 deg cross-slope runs along the direct line and falls away
    toward -y into a chasm (~7 m drop). Slope angle is below the 22 deg scalar
    cliff limit, so a scalar-slope planner treats it as drivable.
  * Level ground at y >= 7 (the bypass) is ~7 m longer.
  * The ground is loose scree (friction 0.30). Lateral slip starts when
    tan(slope) > mu, i.e. above atan(0.30) = 16.7 deg, so a 21 deg cross-slope
    slides the rover downhill into the chasm. The direction-aware planner caps
    cross-slope roll at 12 deg, below the slip angle.
"""
import json
import math
from pathlib import Path
import time
import numpy as np
from autonomous_state_machine.terrain import TerrainGrid, TraversabilityLimits
from autonomous_state_machine.directional_planner import DirectionalThetaStar

RESOLUTION = 0.2
XMIN, YMIN = -16.0, -11.0
NX, NY = 161, 111
START, GOAL = (-14.0, 0.0), (14.0, 0.0)
RADIUS = 0.8
SLOPE_DEG = 21.0
SCREE_FRICTION = 0.30
CHASM_Y = -3.0
SLOPE_TOP_Y = 7.0
CHASM_DROP = 7.0
LIMITS = TraversabilityLimits(cliff_deg=22, climb_deg=22, descent_deg=22, roll_deg=12)
LANE_OFFSETS = {'isotropic': -19, 'anisotropic': 19}


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def elevation(x, y, slope_deg=SLOPE_DEG):
    """Height at (x, y). Slope region fades in over |x| in [3, 11] so the
    transition itself is gentle (no artificial wall at the ends)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lat = 1.0 - _smoothstep((np.abs(x) - 3.0) / 8.0)
    t = math.tan(math.radians(slope_deg))
    w = 0.45
    slope = -t * w * np.logaddexp(0, (SLOPE_TOP_Y - y) / w)
    wall = -CHASM_DROP / (1.0 + np.exp(-(CHASM_Y - y) / 0.15))
    return lat * (slope + wall)


def make_course():
    y = np.arange(NY) * RESOLUTION + YMIN
    x = np.arange(NX) * RESOLUTION + XMIN
    xx, yy = np.meshgrid(x, y)
    z = elevation(xx, yy).astype('float32')
    gy, gx = np.gradient(z, RESOLUTION, RESOLUTION)
    blocked = np.zeros(z.shape, dtype=bool)
    return TerrainGrid(blocked, gx, gx.copy(), gy, gy.copy(), XMIN, YMIN, RESOLUTION, RESOLUTION), z


class IsotropicThetaStar(DirectionalThetaStar):
    """Baseline: same Theta*, cliff limit, footprint, and scalar slope cost.

    Heading never changes feasibility/cost; roll and pitch are not distinguished.
    """
    def _rate(self, r, c, vx, vy):
        if r < 0 or c < 0 or r >= self.blocked.shape[0] or c >= self.blocked.shape[1] or self.blocked[r, c]:
            return math.inf
        values = [self.gx_min[r, c], self.gx_max[r, c], self.gy_min[r, c], self.gy_max[r, c]]
        if not all(math.isfinite(v) for v in values):
            return math.inf
        slope = math.hypot(max(abs(values[0]), abs(values[1])), max(abs(values[2]), abs(values[3])))
        return 1 + 0.2 * (slope / math.tan(math.radians(self.limits.cliff_deg))) ** 2


def evaluate_route(terrain, points):
    """Independent dense centerline pitch/roll audit (not used to select paths)."""
    samples = []
    for a, b in zip(points, points[1:]):
        length = math.dist(a, b)
        if not length:
            continue
        vx, vy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        n = max(1, math.ceil(length / .05))
        for i in range(n):
            x = a[0] + (b[0] - a[0]) * (i + .5) / n
            y = a[1] + (b[1] - a[1]) * (i + .5) / n
            r, c = terrain.cell((x, y))
            gx = float(terrain.gx_min[r, c])
            gy = float(terrain.gy_min[r, c])
            pitch = math.degrees(math.atan(gx * vx + gy * vy))
            roll = math.degrees(math.atan(abs(-gx * vy + gy * vx)))
            samples.append((length / n, pitch, roll))
    slip = math.degrees(math.atan(SCREE_FRICTION))
    return {'length_m': sum(v[0] for v in samples),
            'max_centerline_pitch_deg': max(abs(v[1]) for v in samples),
            'max_centerline_roll_deg': max(v[2] for v in samples),
            'distance_over_slip_angle_m': sum(d for d, p, r in samples if r > slip)}


def _orientation_looking_at(position, target, fov):
    """Webots Viewpoint (ENU, z up) axis-angle for a camera at `position` looking at `target`."""
    f = np.array(target, float) - np.array(position, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    # Webots camera frame: x = forward, y = left, z = up.
    R = np.column_stack([f, -right, up])
    angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis = axis / np.linalg.norm(axis)
    return f"orientation {axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f} {angle:.6f} position {position[0]} {position[1]} {position[2]} fieldOfView {fov}"


def build(output):
    output = Path(output)
    (output / 'worlds').mkdir(parents=True, exist_ok=True)
    terrain, z = make_course()
    results = {}
    for name, kind in [('isotropic', IsotropicThetaStar), ('anisotropic', DirectionalThetaStar)]:
        planner = kind(terrain, LIMITS, RADIUS)
        t = time.perf_counter()
        points = planner.plan(START, GOAL, timeout_s=90)
        if not points:
            raise RuntimeError(f'{name} found no route')
        results[name] = {'points': points, 'planning_seconds': time.perf_counter() - t,
                         'expanded_nodes': planner.expanded, **evaluate_route(terrain, points)}
    spec = {'course': 'loose-scree 21-degree side-slope above a chasm, with a level bypass',
            'start': START, 'goal': GOAL, 'resolution_m': RESOLUTION, 'robot_radius_m': RADIUS,
            'slope_deg': SLOPE_DEG, 'scree_friction': SCREE_FRICTION,
            'slip_angle_deg': math.degrees(math.atan(SCREE_FRICTION)),
            'chasm_y': CHASM_Y, 'chasm_drop_m': CHASM_DROP,
            'planner_roll_limit_deg': LIMITS.roll_deg, 'scalar_cliff_limit_deg': LIMITS.cliff_deg,
            'baseline_scalar_slope_weight': 0.2, 'fall_height_m': 2.5,
            'controller': 'identical waypoint follower in rover_trial.py',
            'source': 'procedural mechanism test; no real-Utah terrain claim', 'results': results}
    (output / 'plans.json').write_text(json.dumps(spec, indent=2) + '\n')

    heights = ' '.join(f'{v:.5f}' for v in z.ravel())
    shape = f'''Shape {{ appearance PBRAppearance {{ baseColor 0.64 0.53 0.36 roughness 1 metalness 0 }}
      geometry DEF GROUND ElevationGrid {{ xDimension {NX} yDimension {NY}
      xSpacing {RESOLUTION} ySpacing {RESOLUTION} height [ {heights} ] thickness 3 }} }}'''
    ground_z = float(elevation(START[0], START[1]))
    view = _orientation_looking_at((0, -60, 40), (0, 0, 0), 1.18)
    pieces = [f'''#VRML_SIM R2025a utf8
EXTERNPROTO "../../canyon_gauntlet/protos/rover_terrain.proto"
WorldInfo {{ basicTimeStep 16 randomSeed 7 title "Isotropic vs Anisotropic - chasm comparison"
  contactProperties [
    ContactProperties {{ material1 "default" material2 "default" coulombFriction [ 1.2 ] bounce 0 maxContactJoints 8 }}
    ContactProperties {{ material1 "default" material2 "scree" coulombFriction [ {SCREE_FRICTION} ] bounce 0 maxContactJoints 8 }}
  ] }}
Viewpoint {{ {view} }}
Background {{ skyColor [ 0.08 0.10 0.14 ] }}
DirectionalLight {{ direction -0.3 0.4 -1 intensity 1.2 ambientIntensity 0.65 }}
''']
    for idx, (mode, color) in enumerate([('isotropic', '1 0.45 0.12'), ('anisotropic', '0.1 0.85 1')]):
        offset = LANE_OFFSETS[mode]
        geometry = shape if idx == 0 else 'Shape { appearance PBRAppearance { baseColor 0.64 0.53 0.36 roughness 1 metalness 0 } geometry USE GROUND }'
        pieces.append(f'Solid {{ translation {offset + XMIN} {YMIN} 0 children [ {geometry} ] boundingObject USE GROUND '
                      f'contactMaterial "scree" locked TRUE name "{mode}_terrain" }}')
        # Chasm floor so a fallen rover lands (and stays in view) instead of leaving the map.
        pieces.append(f'Solid {{ translation {offset} -17.5 -9.5 children [ Shape {{ appearance PBRAppearance {{ baseColor 0.18 0.16 0.14 roughness 1 metalness 0 }} '
                      f'geometry Box {{ size 34 28 1 }} }} ] boundingObject Box {{ size 34 28 1 }} contactMaterial "scree" locked TRUE name "{mode}_chasm_floor" }}')
        p0, p1 = results[mode]['points'][:2]
        init_yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        pieces.append(f'''DEF {mode.upper()} rover_terrain {{ translation {START[0] + offset} {START[1]} {ground_z + .45}
          rotation 0 0 1 {init_yaw:.5f} name "{mode}" controller "<extern>" supervisor TRUE }}''')
        for a, b in zip(results[mode]['points'], results[mode]['points'][1:]):
            n = max(1, math.ceil(math.dist(a, b) / .15))
            verts = []
            faces = []
            vx, vy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(vx, vy)
            nx, ny = -vy / length * .07, vx / length * .07
            for i in range(n + 1):
                x = a[0] + vx * i / n
                y = a[1] + vy * i / n
                for sign in [-1, 1]:
                    px, py = x + sign * nx, y + sign * ny
                    verts.append(f'{px + offset:.4f} {py:.4f} {float(elevation(px, py)) + .07:.4f}')
                if i < n:
                    faces.append(f'{2 * i} {2 * i + 2} {2 * i + 3} {2 * i + 1} -1')
            pieces.append(f'''Shape {{ appearance PBRAppearance {{ baseColor {color} emissiveColor {color} roughness 1 metalness 0 }}
              geometry IndexedFaceSet {{ coord Coordinate {{ point [ {', '.join(verts)} ] }} coordIndex [ {' '.join(faces)} ] }} }}''')
        # Red tape along the chasm lip.
        for x0 in np.arange(-10, 10.1, 1.0):
            pieces.append(f'Transform {{ translation {offset + x0:.2f} {CHASM_Y + 0.25} {float(elevation(x0, CHASM_Y + 0.25)) + .05:.4f} '
                          f'children [ Shape {{ appearance PBRAppearance {{ baseColor 1 0.1 0.1 emissiveColor 0.8 0.05 0.05 }} geometry Box {{ size 0.5 0.12 0.04 }} }} ] }}')
        for point, point_color in [(START, '0.2 1 0.2'), (GOAL, '1 0.2 0.1')]:
            pieces.append(f'''Transform {{ translation {offset + point[0]} {point[1]} {float(elevation(*point)) + .04}
              children [ Shape {{ appearance PBRAppearance {{ baseColor {point_color} emissiveColor {point_color} }} geometry Cylinder {{ height .03 radius .4 }} }} ] rotation 1 0 0 1.57079632679 }}''')
    (output / 'worlds/comparison.wbt').write_text('\n'.join(pieces))
    return spec


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('output')
    args = p.parse_args()
    print(json.dumps(build(args.output), indent=2))
