"""Trio Comparison Benchmark: No UGIS (2D Unaware) vs Isotropic vs Anisotropic (UGIS).

Three rovers race simultaneously in 3 parallel lanes across a challenging terrain:
- Trail runs from Start (-14.0, 0.0) to Goal (14.0, 0.0).
- A dynamic boulder sits at (0.0, 0.0) on the centerline.
- To the right (y < -2.2m): A sheer 7-meter cliff into a dark canyon ravine.
- In the middle (y in [-2.0, 6.0]): A steep 20.5-degree scree slope (friction 0.28, slip angle 15.6 deg).
- Inland to the left (y > 6.5m): A flat, safe ridge plateau (z = 0, roll < 1 deg).

Contenders:
1. No UGIS (2D Flat Costmap): Blind to elevation and cliffs. Chooses the fatal right detour (y = -1.8m),
   drives off the cliff lip, and plummets 7m into the ravine.
2. Isotropic (UGIS Elevation Aware, Scalar Slope Only): Knows the cliff exists so it avoids the ravine,
   but evaluates only scalar slope gradient without heading-dependent roll. Takes the shorter side-slope
   route (y in [3.4, 3.6]), traversing horizontally across a 20.5 deg scree slope. Wheels slip down,
   lose lateral traction, and get caught / stuck on the steep slope.
3. Anisotropic (Directional Theta* + UGIS): Recognizes that horizontal traversal across the 20.5 deg slope
   violates the 12 deg lateral roll limit. Routes cleanly up onto the safe inland ridge plateau (y >= 7.5m),
   bypasses both the boulder and the slope hazard on level ground, and reaches the goal safely to win.
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
START = (-14.0, 0.0)
GOAL = (14.0, 0.0)
GOAL_UNAWARE = (14.0, -0.2)
RADIUS = 0.8
SLOPE_DEG = 20.5
SCREE_FRICTION = 0.28
CHASM_Y = -2.5
SLOPE_TOP_Y = 6.0
CHASM_DROP = 7.0
LIMITS = TraversabilityLimits(cliff_deg=22, climb_deg=22, descent_deg=22, roll_deg=12)

LANE_OFFSETS = {
    'unaware': -30.0,
    'isotropic': 0.0,
    'anisotropic': 30.0
}
ROCK_POS = (0.0, 0.0)
ROCK_RADIUS = 0.72

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
    radius = 0.065 * scale
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
  boundingObject Sphere {{ radius {radius:.3f} }}
  physics Physics {{
    density -1
    mass {mass:.1f}
    centerOfMass [ 0 0 0 ]
  }}
}}'''


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def elevation(x, y):
    """Terrain elevation: flat start/goal, 20.5 deg side slope in middle, sheer cliff at y < -2.2m."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lat = 1.0 - _smoothstep((np.abs(x) - 3.0) / 8.0)
    t = math.tan(math.radians(SLOPE_DEG))
    w = 0.45
    slope = -t * w * np.logaddexp(0.0, (SLOPE_TOP_Y - y) / w)
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
    """Evaluates scalar slope only: ignores heading-dependent roll."""
    def _rate(self, r, c, vx, vy):
        gx0, gx1 = self.gx_min[r, c], self.gx_max[r, c]
        gy0, gy1 = self.gy_min[r, c], self.gy_max[r, c]
        gx = max(abs(gx0), abs(gx1))
        gy = max(abs(gy0), abs(gy1))
        steepness = math.hypot(gx, gy)
        deg = math.degrees(math.atan(steepness))
        if deg > self.limits.cliff_deg:
            return float('inf')
        return 1.0 + 0.15 * math.tan(math.radians(deg))


def evaluate_route(terrain, points):
    samples = []
    total_len = 0.0
    max_pitch = max_roll = max_slope = 0.0
    for a, b in zip(points, points[1:]):
        n = max(1, math.ceil(math.dist(a, b) / 0.15))
        vx, vy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(vx, vy)
        total_len += length
        if length < 1e-9:
            continue
        hvx, hvy = vx / length, vy / length
        for i in range(n):
            x = a[0] + vx * (i + 0.5) / n
            y = a[1] + vy * (i + 0.5) / n
            r, c = terrain.cell((x, y))
            gx = float(terrain.gx_min[r, c])
            gy = float(terrain.gy_min[r, c])
            pitch = math.degrees(math.atan(gx * hvx + gy * hvy))
            roll = math.degrees(math.atan(abs(-gx * hvy + gy * hvx)))
            slope = math.degrees(math.atan(math.hypot(gx, gy)))
            max_pitch = max(max_pitch, abs(pitch))
            max_roll = max(max_roll, roll)
            max_slope = max(max_slope, slope)
            samples.append((length / n, pitch, roll, slope))
    return {
        'length_m': total_len,
        'max_centerline_pitch_deg': max_pitch,
        'max_centerline_roll_deg': max_roll,
        'max_slope_deg': max_slope
    }


def _orientation_looking_at(position, target, fov):
    f = np.array(target, float) - np.array(position, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    R = np.column_stack([f, -right, up])
    angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis = axis / np.linalg.norm(axis)
    return f"orientation {axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f} {angle:.6f} position {position[0]} {position[1]} {position[2]} fieldOfView {fov}"


def build(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    terrain_3d, z = make_course()

    # Flat terrain for 2D Unaware planner
    gx_flat = np.zeros_like(terrain_3d.gx_min)
    gy_flat = np.zeros_like(terrain_3d.gy_min)
    terrain_2d = TerrainGrid(np.zeros_like(terrain_3d.blocked),
                             gx_flat, gx_flat.copy(), gy_flat, gy_flat.copy(),
                             XMIN, YMIN, RESOLUTION, RESOLUTION)

    # Physical boulder at (0, 0)
    r_rock, c_rock = terrain_3d.cell(ROCK_POS)
    rad_cells = int(np.ceil(ROCK_RADIUS / RESOLUTION))
    for dr in range(-rad_cells, rad_cells + 1):
        for dc in range(-rad_cells, rad_cells + 1):
            if dr*dr + dc*dc <= rad_cells*rad_cells:
                terrain_3d.blocked[r_rock + dr, c_rock + dc] = True
                terrain_2d.blocked[r_rock + dr, c_rock + dc] = True

    # 1. Contender 1: Unaware (2D flat costmap)
    planner_unaware = DirectionalThetaStar(terrain_2d, LIMITS, robot_radius_m=RADIUS)
    t0 = time.perf_counter()
    points_unaware = planner_unaware.plan(START, GOAL_UNAWARE)
    time_unaware = time.perf_counter() - t0

    # 2. Contender 2: Isotropic (UGIS elevation, scalar slope only)
    planner_iso = IsotropicThetaStar(terrain_3d, LIMITS, robot_radius_m=RADIUS)
    t0 = time.perf_counter()
    points_iso = planner_iso.plan(START, GOAL)
    time_iso = time.perf_counter() - t0

    # 3. Contender 3: Anisotropic (Directional Theta* + UGIS)
    planner_aniso = DirectionalThetaStar(terrain_3d, LIMITS, robot_radius_m=RADIUS)
    t0 = time.perf_counter()
    points_aniso = planner_aniso.plan(START, GOAL)
    time_aniso = time.perf_counter() - t0

    results = {}
    config = [
        ('unaware', points_unaware, time_unaware, planner_unaware, '1 0.35 0.1', '0.4 0.15 0.05'),
        ('isotropic', points_iso, time_iso, planner_iso, '1 0.75 0.1', '0.4 0.30 0.05'),
        ('anisotropic', points_aniso, time_aniso, planner_aniso, '0.1 0.85 1', '0.08 0.32 0.40')
    ]

    for name, points, p_time, planner, active_color, ghost_color in config:
        segments_vrml = []
        ghost_vrml = []
        segment_stats = []
        offset = LANE_OFFSETS[name]
        for a, b in zip(points, points[1:]):
            n = max(1, math.ceil(math.dist(a, b) / 0.15))
            verts = []
            faces = []
            vx, vy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(vx, vy)
            nx, ny = -vy / length * 0.08, vx / length * 0.08
            for i in range(n + 1):
                px = a[0] + vx * i / n
                py = a[1] + vy * i / n
                for sign in [-1, 1]:
                    sx, sy = px + sign * nx, py + sign * ny
                    verts.append(f'{sx + offset:.4f} {sy:.4f} {float(elevation(sx, sy)) + 0.07:.4f}')
                if i < n:
                    faces.append(f'{2 * i} {2 * i + 2} {2 * i + 3} {2 * i + 1} -1')
            
            mid_x, mid_y = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            r, c = terrain_3d.cell((mid_x, mid_y))
            gx = float(terrain_3d.gx_min[r, c])
            gy = float(terrain_3d.gy_min[r, c])
            pitch = math.degrees(math.atan(gx * (vx/length) + gy * (vy/length)))
            roll = math.degrees(math.atan(abs(-gx * (vy/length) + gy * (vx/length))))
            slope = math.degrees(math.atan(math.hypot(gx, gy)))
            segment_stats.append({'pitch': round(pitch, 1), 'roll': round(roll, 1), 'slope': round(slope, 1)})

            pts_str = ', '.join(verts)
            faces_str = ' '.join(faces)
            act_s = f'Shape {{ appearance PBRAppearance {{ baseColor {active_color} emissiveColor {active_color} roughness 0.2 metalness 0 }} geometry IndexedFaceSet {{ coord Coordinate {{ point [ {pts_str} ] }} coordIndex [ {faces_str} ] }} }}'
            gho_s = f'Shape {{ appearance PBRAppearance {{ baseColor {ghost_color} emissiveColor {ghost_color} transparency 0.72 roughness 1 metalness 0 }} geometry IndexedFaceSet {{ coord Coordinate {{ point [ {pts_str} ] }} coordIndex [ {faces_str} ] }} }}'
            segments_vrml.append(act_s)
            ghost_vrml.append(gho_s)
            
        results[name] = {
            'points': points,
            'planning_seconds': p_time,
            'expanded_nodes': planner.expanded,
            'segments_vrml': segments_vrml,
            'ghost_vrml': ghost_vrml,
            'segment_stats': segment_stats,
            **evaluate_route(terrain_3d, points)
        }

    spec = {
        'course': 'Trio Benchmark: No UGIS vs Isotropic vs Anisotropic (UGIS)',
        'start': START, 'goal': GOAL, 'goal_unaware': GOAL_UNAWARE,
        'resolution_m': RESOLUTION, 'robot_radius_m': RADIUS,
        'slope_deg': SLOPE_DEG, 'scree_friction': SCREE_FRICTION,
        'chasm_y': CHASM_Y, 'chasm_drop_m': CHASM_DROP,
        'planner_roll_limit_deg': LIMITS.roll_deg, 'scalar_cliff_limit_deg': LIMITS.cliff_deg,
        'fall_height_m': 2.5,
        'rock': {'x': ROCK_POS[0], 'y': ROCK_POS[1], 'radius': ROCK_RADIUS},
        'results': results
    }
    (output / 'plans.json').write_text(json.dumps(spec, indent=2) + '\n')

    # Webots World Generation
    heights = ' '.join(f'{v:.5f}' for v in z.ravel())
    ground_shape = f'''Shape {{ appearance PBRAppearance {{ baseColor 0.64 0.53 0.36 roughness 1 metalness 0 }}
      geometry DEF GROUND ElevationGrid {{ xDimension {NX} yDimension {NY}
      xSpacing {RESOLUTION} ySpacing {RESOLUTION} height [ {heights} ] thickness 3 }} }}'''
    ground_z = float(elevation(START[0], START[1]))
    view = _orientation_looking_at((0, -60, 42), (0, 0, 0), 1.25)

    pieces = [f'''#VRML_SIM R2025a utf8
EXTERNPROTO "../../canyon_gauntlet/protos/rover_terrain.proto"
WorldInfo {{ basicTimeStep 16 randomSeed 7 title "Trio Benchmark - No UGIS vs Isotropic vs Anisotropic"
  contactProperties [
    ContactProperties {{ material1 "default" material2 "default" coulombFriction [ 1.2 ] bounce 0 maxContactJoints 8 }}
    ContactProperties {{ material1 "default" material2 "scree" coulombFriction [ {SCREE_FRICTION} ] bounce 0 maxContactJoints 8 }}
  ] }}
Viewpoint {{ {view} }}
Background {{ skyColor [ 0.08 0.10 0.14 ] }}
DirectionalLight {{ direction -0.3 0.4 -1 intensity 1.2 ambientIntensity 0.65 }}
''']

    for idx, (mode, points, _, _, _, _) in enumerate(config):
        offset = LANE_OFFSETS[mode]
        geometry = ground_shape if idx == 0 else 'Shape { appearance PBRAppearance { baseColor 0.64 0.53 0.36 roughness 1 metalness 0 } geometry USE GROUND }'
        pieces.append(f'Solid {{ translation {offset + XMIN} {YMIN} 0 children [ {geometry} ] boundingObject USE GROUND '
                      f'contactMaterial "scree" locked TRUE name "{mode}_terrain" }}')
        
        # Canyon ravine floor
        pieces.append(f'Solid {{ translation {offset} {CHASM_Y - 5.0} {-CHASM_DROP - 0.2} children [ Shape {{ appearance PBRAppearance {{ baseColor 0.18 0.16 0.14 roughness 1 metalness 0 }} '
                      f'geometry Box {{ size 40 12 0.4 }} }} ] boundingObject Box {{ size 40 12 0.4 }} contactMaterial "scree" locked TRUE name "{mode}_chasm_floor" }}')
        
        p0, p1 = points[:2]
        init_yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        pieces.append(f'''DEF {mode.upper()} rover_terrain {{ translation {START[0] + offset} {START[1]} {ground_z + 0.45}
          rotation 0 0 1 {init_yaw:.5f} name "{mode}" controller "<extern>" supervisor TRUE }}''')
        
        # Ghost ribbon
        for gho_shape in results[mode]['ghost_vrml']:
            pieces.append(gho_shape)
            
        # Active trail group
        pieces.append(f'DEF {mode.upper()}_ACTIVE_TRAIL Group {{ children [ ] }}')
        
        # Dynamic Boulder
        rock_z = float(elevation(ROCK_POS[0], ROCK_POS[1])) + 0.65
        pieces.append(make_boulder_vrml(f"{mode}_rock", offset + ROCK_POS[0], ROCK_POS[1], rock_z, scale=11.5, mass=450.0))
        
        # Red warning tape along cliff lip
        for x0 in np.arange(-14, 14.1, 1.0):
            pieces.append(f'Transform {{ translation {offset + x0:.2f} {CHASM_Y + 0.3} {float(elevation(x0, CHASM_Y + 0.3)) + 0.05:.4f} '
                          f'children [ Shape {{ appearance PBRAppearance {{ baseColor 1 0.1 0.1 emissiveColor 0.8 0 0 roughness 0.3 }} '
                          f'geometry Box {{ size 0.7 0.1 0.03 }} }} ] }}')

        # Start and Goal pads
        pieces.append(f'''Transform {{ translation {offset + START[0]} {START[1]} {ground_z + 0.04} children [
          Shape {{ appearance PBRAppearance {{ baseColor 0.2 0.9 0.3 roughness 1 }} geometry Cylinder {{ height 0.08 radius 0.9 }} }} ] }}''')
        
        m_goal = GOAL_UNAWARE if mode == 'unaware' else GOAL
        goal_z = float(elevation(m_goal[0], m_goal[1]))
        pieces.append(f'''Transform {{ translation {offset + m_goal[0]} {m_goal[1]} {goal_z + 0.04} children [
          Shape {{ appearance PBRAppearance {{ baseColor 0.95 0.3 0.2 roughness 1 }} geometry Cylinder {{ height 0.08 radius 0.9 }} }} ] }}''')

    world_dir = output / 'worlds'
    world_dir.mkdir(parents=True, exist_ok=True)
    (world_dir / 'trio_comparison.wbt').write_text('\n'.join(pieces) + '\n')
    print(f'Wrote {world_dir / "trio_comparison.wbt"}')
    return spec


if __name__ == '__main__':
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    build(out_dir)

