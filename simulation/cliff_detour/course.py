"""Cliff Detour Benchmark: Terrain-Unaware (2D) vs Terrain-Aware (3D GIS) Avoidance.

Demonstrates the fatal danger of 2D flat-costmap obstacle avoidance in rough terrain:
- A trail runs along y = 0 from start (-10, 0) to goal (10, 0).
- To the left (y > 0): Safe, level inland terrain (z = 0, pitch = 0, roll < 1 deg).
- To the right (y < -1.3): A sheer 6-meter cliff drop / canyon chasm.
- A large dynamic boulder sits directly on the trail at (0.0, 0.0).

Comparison:
- Unaware (2D Flat Costmap): Blind to elevation drops. Sees free 2D space on both sides.
  Because the goal has a slight rightward bias, it chooses the right detour (y = -1.8),
  driving straight off the cliff lip and plummeting 6m into the canyon.
- Terrain-Aware (3D Directional Theta* + UGIS DEM):
  Evaluates 3D elevation gradient & roll stability from UGIS.
  Detects that dodging right causes lethal roll (> 35 deg) and cliff drop.
  Fail-closed policy rejects the right detour and routes left (y = +1.8), safely reaching the goal.
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
XMIN, YMIN = -16.0, -10.0
NX, NY = 161, 101
START = (-10.0, 0.0)
GOAL_3D = (10.0, 0.0)
GOAL_2D = (10.0, -0.05)
RADIUS = 0.8
CLIFF_Y = -1.8
CLIFF_DROP = 6.0
LIMITS = TraversabilityLimits(cliff_deg=22, climb_deg=22, descent_deg=22, roll_deg=12)
LANE_OFFSETS = {'unaware': -16, 'terrain_aware': 16}
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


def elevation(x, y):
    """Height at (x, y). Flat inland ground for y >= -1.0; sheer cliff drop starting at y = -1.3 down to -6m."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    drop = -CLIFF_DROP / (1.0 + np.exp(-(CLIFF_Y - y) / 0.12))
    return drop


def make_course():
    y = np.arange(NY) * RESOLUTION + YMIN
    x = np.arange(NX) * RESOLUTION + XMIN
    xx, yy = np.meshgrid(x, y)
    z = elevation(xx, yy).astype('float32')
    gy, gx = np.gradient(z, RESOLUTION, RESOLUTION)
    blocked = np.zeros(z.shape, dtype=bool)
    return TerrainGrid(blocked, gx, gx.copy(), gy, gy.copy(), XMIN, YMIN, RESOLUTION, RESOLUTION), z


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
    
    # 2D flat costmap terrain (zero elevation/gradients)
    gx_flat = np.zeros_like(terrain_3d.gx_min)
    gy_flat = np.zeros_like(terrain_3d.gy_min)
    terrain_2d = TerrainGrid(np.zeros_like(terrain_3d.blocked),
                             gx_flat, gx_flat.copy(), gy_flat, gy_flat.copy(),
                             XMIN, YMIN, RESOLUTION, RESOLUTION)

    # Mark physical boulder in both
    r_rock, c_rock = terrain_3d.cell(ROCK_POS)
    rad_cells = int(np.ceil(ROCK_RADIUS / RESOLUTION))
    for dr in range(-rad_cells, rad_cells + 1):
        for dc in range(-rad_cells, rad_cells + 1):
            if dr*dr + dc*dc <= rad_cells*rad_cells:
                terrain_3d.blocked[r_rock + dr, c_rock + dc] = True
                terrain_2d.blocked[r_rock + dr, c_rock + dc] = True

    # Compute Plans
    # 1. 2D Unaware Planner: blind to cliff elevation drop, goes right
    planner_2d = DirectionalThetaStar(terrain_2d, LIMITS, robot_radius_m=RADIUS)
    t0 = time.perf_counter()
    points_2d = planner_2d.plan(START, GOAL_2D)
    time_2d = time.perf_counter() - t0

    # 2. 3D Terrain-Aware Planner: detects cliff drop & roll limit, goes left inland
    planner_3d = DirectionalThetaStar(terrain_3d, LIMITS, robot_radius_m=RADIUS)
    t0 = time.perf_counter()
    points_3d = planner_3d.plan(START, GOAL_3D)
    time_3d = time.perf_counter() - t0

    print("Points 2D (Unaware):", points_2d)
    print("Points 3D (Terrain-Aware):", points_3d)

    results = {}
    for name, points, p_time, planner, active_color, ghost_color in [
        ('unaware', points_2d, time_2d, planner_2d, '1 0.45 0.12', '0.4 0.25 0.15'),
        ('terrain_aware', points_3d, time_3d, planner_3d, '0.1 0.85 1', '0.08 0.32 0.40')
    ]:
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
        'course': 'Cliff Detour Dilemma: 2D Unaware vs 3D Terrain-Aware Avoidance',
        'start': START, 'goal_3d': GOAL_3D, 'goal_2d': GOAL_2D,
        'resolution_m': RESOLUTION, 'robot_radius_m': RADIUS,
        'cliff_y': CLIFF_Y, 'cliff_drop_m': CLIFF_DROP,
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
    view = _orientation_looking_at((0, -45, 30), (0, 0, 0), 1.15)

    pieces = [f'''#VRML_SIM R2025a utf8
EXTERNPROTO "../../canyon_gauntlet/protos/rover_terrain.proto"
WorldInfo {{ basicTimeStep 16 randomSeed 7 title "Cliff Detour Dilemma - 2D Unaware vs 3D Terrain-Aware"
  contactProperties [
    ContactProperties {{ material1 "default" material2 "default" coulombFriction [ 1.2 ] bounce 0 maxContactJoints 8 }}
    ContactProperties {{ material1 "default" material2 "scree" coulombFriction [ 0.3 ] bounce 0 maxContactJoints 8 }}
  ] }}
Viewpoint {{ {view} }}
Background {{ skyColor [ 0.08 0.10 0.14 ] }}
DirectionalLight {{ direction -0.3 0.4 -1 intensity 1.2 ambientIntensity 0.65 }}
''']

    for idx, (mode, color) in enumerate([('unaware', '1 0.45 0.12'), ('terrain_aware', '0.1 0.85 1')]):
        offset = LANE_OFFSETS[mode]
        geometry = ground_shape if idx == 0 else 'Shape { appearance PBRAppearance { baseColor 0.64 0.53 0.36 roughness 1 metalness 0 } geometry USE GROUND }'
        pieces.append(f'Solid {{ translation {offset + XMIN} {YMIN} 0 children [ {geometry} ] boundingObject USE GROUND '
                      f'contactMaterial "default" locked TRUE name "{mode}_terrain" }}')
        
        # Canyon chasm floor
        pieces.append(f'Solid {{ translation {offset} {CLIFF_Y - 4.5} {-CLIFF_DROP - 0.2} children [ Shape {{ appearance PBRAppearance {{ baseColor 0.18 0.16 0.14 roughness 1 metalness 0 }} '
                      f'geometry Box {{ size 36 10 0.4 }} }} ] boundingObject Box {{ size 36 10 0.4 }} contactMaterial "scree" locked TRUE name "{mode}_chasm_floor" }}')
        
        p0, p1 = results[mode]['points'][:2]
        init_yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        pieces.append(f'''DEF {mode.upper()} rover_terrain {{ translation {START[0] + offset} {START[1]} {ground_z + 0.45}
          rotation 0 0 1 {init_yaw:.5f} name "{mode}" controller "<extern>" supervisor TRUE }}''')
        
        # Ghost ribbon
        for gho_shape in results[mode]['ghost_vrml']:
            pieces.append(gho_shape)
            
        # Active trail group
        pieces.append(f'DEF {mode.upper()}_ACTIVE_TRAIL Group {{ children [ ] }}')
        
        # Dynamic ODE Boulder
        rock_z = float(elevation(ROCK_POS[0], ROCK_POS[1])) + 0.65
        pieces.append(make_boulder_vrml(f"{mode}_rock", offset + ROCK_POS[0], ROCK_POS[1], rock_z, scale=11.5, mass=450.0))
        
        # Red warning tape along cliff lip
        for x0 in np.arange(-14, 14.1, 1.0):
            pieces.append(f'Transform {{ translation {offset + x0:.2f} {CLIFF_Y + 0.3} {float(elevation(x0, CLIFF_Y + 0.3)) + 0.05:.4f} '
                          f'children [ Shape {{ appearance PBRAppearance {{ baseColor 1 0.1 0.1 emissiveColor 0.8 0 0 roughness 0.3 }} '
                          f'geometry Box {{ size 0.7 0.1 0.03 }} }} ] }}')

        # Start and Goal pads
        pieces.append(f'''Transform {{ translation {offset + START[0]} {START[1]} {ground_z + 0.04} children [
          Shape {{ appearance PBRAppearance {{ baseColor 0.2 0.9 0.3 roughness 1 }} geometry Cylinder {{ height 0.08 radius 0.9 }} }} ] }}''')
        
        mode_goal = GOAL_2D if mode == 'unaware' else GOAL_3D
        goal_z = float(elevation(mode_goal[0], mode_goal[1]))
        pieces.append(f'''Transform {{ translation {offset + mode_goal[0]} {mode_goal[1]} {goal_z + 0.04} children [
          Shape {{ appearance PBRAppearance {{ baseColor 0.95 0.3 0.2 roughness 1 }} geometry Cylinder {{ height 0.08 radius 0.9 }} }} ] }}''')

    world_dir = output / 'worlds'
    world_dir.mkdir(parents=True, exist_ok=True)
    (world_dir / 'cliff_detour.wbt').write_text('\n'.join(pieces) + '\n')
    print(f'Wrote {world_dir / "cliff_detour.wbt"}')
    return spec


if __name__ == '__main__':
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    build(out_dir)
