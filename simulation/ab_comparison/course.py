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
import os
from pathlib import Path
import sys
import time

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

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
ROCK_POS = (2.0, 9.0)
ROCK_RADIUS = 0.70

ROCK_BASE_VERTICES = [
    (0.0282555, -0.0123062, 0.0493622), (0.0514526, 0.0516891, 0.0471497),
    (0.0213318, 0.0449829, 0.0468445), (0.0608444, 0.0574036, 0.032486),
    (-0.01614, 0.0463638, 0.0209618), (-0.0175323, 0.0003681, 0.0294952),
    (0.0659485, 0.0076895, -0.0014772), (0.029644, -0.0428085, 0.0261192),
    (-0.0213318, -0.040596, 0.0248642), (0.0575638, -0.0246735, 0.0051146),
    (0.0346298, 0.055397, 0.0064392), (-0.0625458, -0.0035353, 0.0255203),
    (0.004591, -0.0539703, -0.0393677), (-0.0445328, -0.031601, 0.0104427),
    (-0.0316048, -0.0461426, -0.0520782), (0.0381699, -0.018795, -0.0489807),
    (-0.0555496, -0.0468445, -0.0379105), (-0.0018876, 0.0425644, -0.0396271),
    (-0.066803, -0.0030928, -0.029541), (-0.0359573, -0.0154018, -0.0562286),
    (-0.0384674, 0.0333023, -0.0195274)
]
ROCK_COORD_INDEX = (
    '2 0 1 -1, 1 0 3 -1, 4 0 2 -1, 0 4 5 -1, 3 0 6 -1, 7 0 5 -1, 6 0 7 -1, 8 7 5 -1, '
    '6 7 9 -1, 10 3 6 -1, 11 5 4 -1, 11 8 5 -1, 12 7 8 -1, 12 9 7 -1, 6 9 12 -1, '
    '13 8 11 -1, 14 12 8 -1, 15 6 12 -1, 15 10 6 -1, 13 11 16 -1, 16 8 13 -1, '
    '14 8 16 -1, 15 12 14 -1, 10 15 17 -1, 18 11 4 -1, 16 11 18 -1, 14 16 19 -1, '
    '15 14 19 -1, 19 16 18 -1, 19 17 15 -1, 18 20 19 -1, 17 19 20 -1, 4 20 18 -1, '
    '4 17 20 -1, 4 10 17 -1, 2 10 4 -1, 1 3 10 -1, 1 10 2 -1'
)
ROCK_TEX_COORD = (
    '0.6519888 0.9328683, 0.7415154 0.7278224, 0.7583726 0.9886748, 0.7586426 0.7597129, '
    '0.8404567 0.5771889, 0.7916614 0.8227384, 0.4865306 0.8964634, 0.7415154 0.7278224, '
    '0.6519888 0.9328683, 0.7415154 0.7278224, 0.4865306 0.8964634, 0.5380667 0.7254828, '
    '0.7916614 0.8227384, 0.8404567 0.5771889, 0.9466503 0.8113729, 0.9085732 1.0, '
    '0.758906 0.9799206, 0.7586426 0.8227384, 0.9466503 0.8113729, 0.8404567 0.5771889, '
    '0.9469442 0.5691234, 0.563445 0.5691234, 0.7586426 0.613364, 0.5380667 0.7254828, '
    '0.1250233 0.861748, 0.124784 0.633634, 0.1660463 0.739833, 0.1450739 0.0904869, '
    '0.2137036 0.0, 0.3691042 0.1395276, 0.3694084 0.6643548, 0.5380667 0.7254828, '
    '0.4865306 0.8964634, 0.3694084 0.6643548, 0.563445 0.5691234, 0.5380667 0.7254828, '
    '0.6877791 0.289174, 0.8347366 0.0551657, 0.9323464 0.1935238, 0.6877791 0.289174, '
    '0.6879001 0.0, 0.8347366 0.0551657, 0.9323465 0.0864366, 0.9999999 0.1926008, '
    '0.9326506 0.4389925, 0.3031699 0.8246872, 0.3694084 0.8976216, 0.1660463 0.8671747, '
    '0.7113631 0.4050269, 0.6877791 0.289174, 0.9323464 0.1935238, 0.0 0.8388469, '
    '0.1122519 0.633634, 0.1122419 0.9866771, 0.3654387 0.3123293, 0.1450739 0.0904869, '
    '0.3691042 0.1395276, 0.3031699 0.8246872, 0.1660463 0.8671747, 0.3569988 0.633634, '
    '0.3569988 0.633634, 0.3694084 0.8976216, 0.3031699 0.8246872, 0.7113631 0.4050269, '
    '0.9323464 0.1935238, 0.807003 0.4389925, 0.3654387 0.3123293, 0.3694084 0.4920835, '
    '0.2444831 0.5753585, 0.1450739 0.0904869, 0.3654387 0.3123293, 0.0779419 0.2491205, '
    '0.4751778 0.0, 0.6872136 0.0613657, 0.6173913 0.26805, 0.3569988 0.633634, '
    '0.1660463 0.8671747, 0.1764471 0.656506, 0.2444831 0.5753585, 0.178291 0.633634, '
    '0.1451926 0.5064259, 0.3654387 0.3123293, 0.2444831 0.5753585, 0.1451926 0.5064259, '
    '0.1451926 0.5064259, 0.178291 0.633634, 0.022482 0.5410983, 0.1451926 0.5064259, '
    '0.0779419 0.2491205, 0.3654387 0.3123293, 0.022482 0.5410983, 0.0 0.3611155, '
    '0.1451926 0.5064259, 0.0779419 0.2491205, 0.1451926 0.5064259, 0.0 0.3611155, '
    '0.6173913 0.26805, 0.4810471 0.1416896, 0.4751778 0.0, 0.6173913 0.26805, '
    '0.3694084 0.2665703, 0.4810471 0.1416896, 0.6173913 0.26805, 0.5157984 0.4494928, '
    '0.3694084 0.2665703, 0.6877791 0.4261287, 0.5157984 0.4494928, 0.6173913 0.26805, '
    '0.6620587 0.5423295, 0.5953204 0.5691234, 0.5157984 0.4494928, 0.6620587 0.5423295, '
    '0.5157984 0.4494928, 0.6877791 0.4261287'
)
ROCK_TEX_COORD_INDEX = (
    '0 1 2 -1 3 4 5 -1 6 7 8 -1 9 10 11 -1 12 13 14 -1 15 16 17 -1 18 19 20 -1 '
    '21 22 23 -1 24 25 26 -1 27 28 29 -1 30 31 32 -1 33 34 35 -1 36 37 38 -1 39 40 41 -1 '
    '42 43 44 -1 45 46 47 -1 48 49 50 -1 51 52 53 -1 54 55 56 -1 57 58 59 -1 60 61 62 -1 '
    '63 64 65 -1 66 67 68 -1 69 70 71 -1 72 73 74 -1 75 76 77 -1 78 79 80 -1 81 82 83 -1 '
    '84 85 86 -1 87 88 89 -1 90 91 92 -1 93 94 95 -1 96 97 98 -1 99 100 101 -1 102 103 104 -1 '
    '105 106 107 -1 108 109 110 -1 111 112 113 -1'
)


def make_boulder_vrml(name: str, x: float, y: float, z: float, scale: float = 11.5, mass: float = 450.0) -> str:
    scaled_pts = ' '.join(f'{p[0]*scale:.4f} {p[1]*scale:.4f} {p[2]*scale:.4f}' for p in ROCK_BASE_VERTICES)
    radius = 0.065 * scale  # ~0.75m
    return f'''DEF {name.upper()} Solid {{
  translation {x:.2f} {y:.2f} {z:.3f}
  name "{name}"
  locked FALSE
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 0.72 0.65 0.55
        baseColorMap ImageTexture {{
          url [ "textures/rock.png", "../textures/rock.png" ]
        }}
        roughness 0.95
        metalness 0.05
      }}
      geometry IndexedFaceSet {{
        coord Coordinate {{
          point [ {scaled_pts} ]
        }}
        coordIndex [ {ROCK_COORD_INDEX} ]
        texCoord TextureCoordinate {{
          point [ {ROCK_TEX_COORD} ]
        }}
        texCoordIndex [ {ROCK_TEX_COORD_INDEX} ]
      }}
    }}
  ]
  boundingObject Sphere {{
    radius {radius:.2f}
  }}
  physics Physics {{
    density -1
    mass {mass}
    damping Damping {{
      linear 0.4
      angular 0.4
    }}
  }}
}}'''


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
        
        offset = LANE_OFFSETS[name]
        active_color = '1 0.45 0.12' if name == 'isotropic' else '0.1 0.85 1'
        active_emissive = '0.9 0.4 0.1' if name == 'isotropic' else '0.2 0.85 1'
        ghost_color = '0.4 0.25 0.15' if name == 'isotropic' else '0.08 0.32 0.40'
        ghost_emissive = '0.15 0.08 0.04' if name == 'isotropic' else '0.04 0.16 0.20'
        
        segments_vrml = []
        ghost_vrml = []
        segment_stats = []
        for a, b in zip(points, points[1:]):
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
            
            mid_x, mid_y = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            r, c = terrain.cell((mid_x, mid_y))
            gx = float(terrain.gx_min[r, c])
            gy = float(terrain.gy_min[r, c])
            pitch = math.degrees(math.atan(gx * (vx/length) + gy * (vy/length)))
            roll = math.degrees(math.atan(abs(-gx * (vy/length) + gy * (vx/length))))
            slope = math.degrees(math.atan(math.hypot(gx, gy)))
            segment_stats.append({'pitch': round(pitch, 1), 'roll': round(roll, 1), 'slope': round(slope, 1)})
            
            pts_str = ', '.join(verts)
            faces_str = ' '.join(faces)
            act_s = f'Shape {{ appearance PBRAppearance {{ baseColor {active_color} emissiveColor {active_emissive} roughness 0.2 metalness 0 }} geometry IndexedFaceSet {{ coord Coordinate {{ point [ {pts_str} ] }} coordIndex [ {faces_str} ] }} }}'
            gho_s = f'Shape {{ appearance PBRAppearance {{ baseColor {ghost_color} emissiveColor {ghost_emissive} transparency 0.72 roughness 1 metalness 0 }} geometry IndexedFaceSet {{ coord Coordinate {{ point [ {pts_str} ] }} coordIndex [ {faces_str} ] }} }}'
            segments_vrml.append(act_s)
            ghost_vrml.append(gho_s)
            
        results[name] = {'points': points, 'planning_seconds': time.perf_counter() - t,
                         'expanded_nodes': planner.expanded,
                         'segments_vrml': segments_vrml,
                         'ghost_vrml': ghost_vrml,
                         'segment_stats': segment_stats,
                         **evaluate_route(terrain, points)}
    spec = {'course': 'loose-scree 21-degree side-slope above a chasm, with a level bypass',
            'start': START, 'goal': GOAL, 'resolution_m': RESOLUTION, 'robot_radius_m': RADIUS,
            'slope_deg': SLOPE_DEG, 'scree_friction': SCREE_FRICTION,
            'slip_angle_deg': math.degrees(math.atan(SCREE_FRICTION)),
            'chasm_y': CHASM_Y, 'chasm_drop_m': CHASM_DROP,
            'planner_roll_limit_deg': LIMITS.roll_deg, 'scalar_cliff_limit_deg': LIMITS.cliff_deg,
            'baseline_scalar_slope_weight': 0.2, 'fall_height_m': 2.5,
            'rock': {'x': ROCK_POS[0], 'y': ROCK_POS[1], 'radius': ROCK_RADIUS},
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
        # Ghost ribbon for pre-computed planned path (stays visible as reference corridor)
        for gho_shape in results[mode]['ghost_vrml']:
            pieces.append(gho_shape)
        # Group for dynamic growing path animation
        pieces.append(f'DEF {mode.upper()}_ACTIVE_TRAIL Group {{ children [ ] }}')
        # Boulder obstacle (authentic 3D textured rock mesh with real 450kg dynamic ODE gravity physics)
        rock_z = float(elevation(ROCK_POS[0], ROCK_POS[1])) + 0.65
        pieces.append(make_boulder_vrml(f"{mode}_rock", offset + ROCK_POS[0], ROCK_POS[1], rock_z, scale=11.5, mass=450.0))
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
