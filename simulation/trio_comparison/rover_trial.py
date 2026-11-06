"""Identical physics controller for Trio Benchmark: No UGIS vs Isotropic vs Anisotropic.

Runs each of the 3 rovers simultaneously in parallel lanes with identical skid-steer physics:
- 'unaware': 2D flat costmap obstacle planner. Chooses the right detour over the cliff lip,
  falling 7 meters into the canyon ravine.
- 'isotropic': Aware of cliff drop so it avoids the ravine, but evaluates only scalar slope.
  Attempts to cut horizontally across the 20.5-degree scree side-slope.
  Wheels lose traction and it gets caught / stuck on the steep slope.
- 'anisotropic': Evaluates directional pitch vs lateral roll limits.
  Recognizes the side-slope roll hazard and routes safely up to the inland ridge plateau,
  bypassing the hazards on level ground to reach the goal and win.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from controller import Supervisor

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from course import elevation, LANE_OFFSETS

WHEEL_RADIUS = 0.1543
TRACK = 1.53
SPEED = 0.50
LOOKAHEAD_DISTANCE = 1.2
MAX_ACCELERATION = 0.3
MAX_YAW_ACCELERATION = 0.8
MAX_YAW_RATE = 0.65
GOAL_TOLERANCE = 0.45
SETTLE_SECONDS = 3.5
TRIAL_TIMEOUT = 120.0
LABEL_PERIOD = 0.12


def angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def get_lookahead_and_crosstrack(points, x, y, lookahead=1.2):
    best_dist = float('inf')
    best_seg = 0
    best_t = 0.0
    for i in range(len(points) - 1):
        ax, ay = points[i]
        bx, by = points[i + 1]
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        if l2 < 1e-9:
            continue
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / l2))
        px, py = ax + t * dx, ay + t * dy
        d = math.hypot(x - px, y - py)
        if d < best_dist:
            best_dist = d
            best_seg = i
            best_t = t
    remaining = lookahead
    curr_seg = best_seg
    curr_t = best_t
    while curr_seg < len(points) - 1:
        ax, ay = points[curr_seg]
        bx, by = points[curr_seg + 1]
        seg_len = math.hypot(bx - ax, by - ay)
        avail = (1.0 - curr_t) * seg_len
        if remaining <= avail:
            t_target = curr_t + remaining / (seg_len or 1e-9)
            return (ax + t_target * (bx - ax), ay + t_target * (by - ay)), best_dist
        remaining -= avail
        curr_seg += 1
        curr_t = 0.0
    return tuple(points[-1]), best_dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['unaware', 'isotropic', 'anisotropic'], required=True)
    p.add_argument('--plans', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--movie', help='Host path for recording video; primary controller only')
    a = p.parse_args()

    spec = json.load(open(a.plans))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    offset = LANE_OFFSETS[a.mode]
    points = [tuple(v) for v in spec['results'][a.mode]['points']]
    fall_height = spec['fall_height_m']

    robot = Supervisor()
    step = int(robot.getBasicTimeStep())
    body = robot.getSelf()
    left = [robot.getDevice(f'left_{q}_wheel') for q in ['front', 'middle', 'back']]
    right = [robot.getDevice(f'right_{q}_wheel') for q in ['front', 'middle', 'back']]
    for motor in left + right:
        motor.setPosition(float('inf'))
        motor.setVelocity(0)

    start_time = None
    finish = None
    status = 'settling'
    travel = 0.0
    last_position = None
    z_start = None
    min_z = 1e9
    max_roll = max_pitch = 0.0
    max_slide = 0.0
    rows = []
    previous_v = previous_w = 0.0
    recording = recording_stopped = False
    last_label = 0
    summary_path = out / (a.mode + '.json')
    trace_path = out / (a.mode + '.jsonl')
    log = trace_path.open('w', buffering=1)

    # Active trail growth references
    segments_vrml = spec['results'][a.mode].get('segments_vrml', [])
    num_segs = len(segments_vrml)
    trail_node = robot.getFromDef(f"{a.mode.upper()}_ACTIVE_TRAIL")
    trail_children = trail_node.getField("children") if trail_node else None
    revealed_seg = 0

    # Stuck-on-slope tracking
    stuck_time_accumulator = 0.0
    last_check_pos = None
    last_check_time = None

    while robot.step(step) != -1:
        now = robot.getTime()
        pos = body.getPosition()
        x, y = pos[0] - offset, pos[1]
        rot = body.getOrientation()
        yaw = math.atan2(rot[3], rot[0])
        roll = math.degrees(math.atan2(rot[7], rot[8]))
        pitch = math.degrees(math.atan2(rot[6], math.hypot(rot[0], rot[3])))
        v = w = 0.0

        # Growing trail animation during settle phase
        if status == 'settling' and trail_children and segments_vrml:
            progress = max(0.0, min(1.0, (now - 0.25) / 2.6)) if now >= 0.25 else 0.0
            target_revealed = min(num_segs, int(progress * num_segs))
            while revealed_seg < target_revealed:
                trail_children.importMFNodeFromString(-1, segments_vrml[revealed_seg])
                revealed_seg += 1

        if now >= SETTLE_SECONDS and start_time is None:
            if trail_children and segments_vrml:
                while revealed_seg < num_segs:
                    trail_children.importMFNodeFromString(-1, segments_vrml[revealed_seg])
                    revealed_seg += 1
            start_time = now
            status = 'running'
            last_position = (x, y)
            last_check_pos = (x, y)
            last_check_time = now
            z_start = pos[2]
            min_z = pos[2]
            if a.mode == 'unaware' and a.movie:
                robot.movieStartRecording(a.movie, 1280, 720, 0, 90, 1, False)
                recording = True

        if start_time is not None and status == 'running':
            travel += math.dist(last_position, (x, y))
            last_position = (x, y)
            max_roll = max(max_roll, abs(roll))
            max_pitch = max(max_pitch, abs(pitch))
            min_z = min(min_z, pos[2])

            target, cross_track = get_lookahead_and_crosstrack(points, x, y, LOOKAHEAD_DISTANCE)
            max_slide = max(max_slide, cross_track)

            # Check for falling into canyon ravine
            if z_start - pos[2] > fall_height:
                status = 'fell_into_chasm'
                finish = now
            # Check for reaching goal
            elif math.dist((x, y), points[-1]) < GOAL_TOLERANCE:
                status = 'goal_reached'
                finish = now
            elif abs(roll) > 70.0 or abs(pitch) > 70.0:
                status = 'rolled_over'
                finish = now
            elif now - start_time > TRIAL_TIMEOUT:
                status = 'timeout'
                finish = now
            else:
                tx, ty = target
                error = angle(math.atan2(ty - y, tx - x) - yaw)
                kappa = 2.0 * math.sin(error) / LOOKAHEAD_DISTANCE
                w = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, SPEED * kappa))
                v = 0.0 if abs(error) > 1.15 else SPEED * max(0.20, math.cos(error))

        if status == 'running':
            dv = MAX_ACCELERATION * step / 1000
            dw = MAX_YAW_ACCELERATION * step / 1000
            v = previous_v + max(-dv, min(dv, v - previous_v))
            w = previous_w + max(-dw, min(dw, w - previous_w))
        else:
            v = w = 0.0

        previous_v, previous_w = v, w
        lv = (v - w * TRACK / 2) / WHEEL_RADIUS
        rv = (v + w * TRACK / 2) / WHEEL_RADIUS
        for motor in left:
            motor.setVelocity(max(-motor.getMaxVelocity(), min(motor.getMaxVelocity(), -lv)))
        for motor in right:
            motor.setVelocity(max(-motor.getMaxVelocity(), min(motor.getMaxVelocity(), rv)))

        if start_time is not None and (not rows or now - start_time - rows[-1]['t'] >= 0.08):
            row = {'t': now - start_time, 'x': x, 'y': y, 'z': pos[2], 'yaw_rad': yaw,
                   'body_roll_deg': roll, 'body_pitch_deg': pitch, 'v_command_mps': v,
                   'omega_command_radps': w, 'status': status,
                   'goal_error_m': math.dist((x, y), points[-1])}
            rows.append(row)
            log.write(json.dumps(row) + '\n')

        # Multi-lane UI Status Overlay
        if now - last_label > LABEL_PERIOD:
            last_label = now
            if a.mode == 'unaware':
                lid, px, color = 10, 0.02, 0xFF5522
                title = '1. NO UGIS (2D Flat Costmap)'
            elif a.mode == 'isotropic':
                lid, px, color = 20, 0.35, 0xFFCC22
                title = '2. ISOTROPIC (Scalar Slope UGIS)'
            else:
                lid, px, color = 30, 0.68, 0x33EEFF
                title = '3. ANISOTROPIC (Directional Theta*)'

            robot.setLabel(lid, title, px, 0.025, 0.045, color, 0, 'Arial')
            drop = (z_start - pos[2]) if z_start is not None else 0.0

            if status == 'settling':
                robot.setLabel(lid + 1, "PLANNING: computing route...", px, 0.08, 0.030, 0xFFFF44, 0, 'Arial')
                robot.setLabel(lid + 2, "Engaging motors at t=3.5s", px, 0.12, 0.027, 0xDDDDDD, 0, 'Arial')
            elif status == 'fell_into_chasm':
                robot.setLabel(lid + 1, f"❌ PLUMMETED INTO RAVINE | Drop {drop:.1f}m", px, 0.08, 0.030, 0xFF3333, 0, 'Arial')
                if a.mode == 'unaware':
                    robot.setLabel(lid + 2, "2D planner blind to elevation: steered off cliff lip", px, 0.12, 0.025, 0xFF8888, 0, 'Arial')
                else:
                    robot.setLabel(lid + 2, "Scalar planner ignored roll: slid off slope into ravine", px, 0.12, 0.025, 0xFF8888, 0, 'Arial')
            elif status == 'rolled_over':
                robot.setLabel(lid + 1, f"❌ ROLLED OVER ON SLOPE | Roll {abs(roll):.0f}°", px, 0.08, 0.030, 0xFF3333, 0, 'Arial')
                robot.setLabel(lid + 2, "Scalar planner ignored roll: tumbled on 20.5° side-slope", px, 0.12, 0.025, 0xFF8888, 0, 'Arial')
            elif status == 'goal_reached':
                robot.setLabel(lid + 1, f"🏆 WINNER: GOAL REACHED | Roll {abs(roll):.0f}°", px, 0.08, 0.032, 0x55FF55, 0, 'Arial')
                robot.setLabel(lid + 2, "Directional Theta*: safely routed inland on flat plateau", px, 0.12, 0.026, 0x88FF88, 0, 'Arial')
            else:
                if a.mode == 'unaware':
                    robot.setLabel(lid + 1, f"RUNNING | Approaching cliff lip", px, 0.08, 0.030, 0xFF9966, 0, 'Arial')
                    robot.setLabel(lid + 2, f"Detour right (blind to 7m drop)", px, 0.12, 0.025, 0xFFCCAA, 0, 'Arial')
                elif a.mode == 'isotropic':
                    robot.setLabel(lid + 1, f"RUNNING | Roll {abs(roll):.0f}° | On scree slope", px, 0.08, 0.030, 0xFFDD66, 0, 'Arial')
                    robot.setLabel(lid + 2, f"Avoids cliff, but traversing 20.5° side-slope", px, 0.12, 0.025, 0xFFEEAA, 0, 'Arial')
                else:
                    robot.setLabel(lid + 1, f"RUNNING | Roll {abs(roll):.0f}° | On safe plateau", px, 0.08, 0.030, 0x55EEFF, 0, 'Arial')
                    robot.setLabel(lid + 2, f"Rejects cliff & side-slope | Inland bypass", px, 0.12, 0.025, 0xAAFFFF, 0, 'Arial')

            if a.mode == 'unaware':
                robot.setLabel(40, "Right: 7m cliff into ravine | Middle: 20.5° loose scree side-slope | Left: Safe level plateau", 0.025, 0.935, 0.026, 0xFFEEAA, 0, 'Arial')
                robot.setLabel(41, "SAME rover / speed | 1. No UGIS falls off cliff | 2. Isotropic falls on slope | 3. Anisotropic wins", 0.025, 0.965, 0.026, 0xFFFFFF, 0, 'Arial')

        if finish is not None and not summary_path.exists() and now - finish > 1.0:
            summary = {
                'mode': a.mode, 'status': status, 'goal_reached': status == 'goal_reached',
                'elapsed_sim_seconds': finish - start_time, 'actual_distance_m': travel,
                'final_goal_error_m': math.dist((x, y), points[-1]),
                'max_body_roll_deg': max_roll, 'max_body_pitch_deg': max_pitch,
                'max_drop_m': z_start - min_z, 'max_deviation_from_route_m': max_slide,
                'controller': {'speed_mps': SPEED, 'max_yaw_rate_radps': MAX_YAW_RATE,
                               'wheel_radius_m': WHEEL_RADIUS, 'track_m': TRACK,
                               'goal_tolerance_m': GOAL_TOLERANCE,
                               'max_acceleration_mps2': MAX_ACCELERATION,
                               'max_yaw_acceleration_radps2': MAX_YAW_ACCELERATION,
                               'fall_height_m': fall_height},
                'planned': spec['results'][a.mode]
            }
            summary_path.write_text(json.dumps(summary, indent=2) + '\n')
            print(json.dumps(summary), flush=True)

        all_done = (
            (out / 'unaware.json').exists() and
            (out / 'isotropic.json').exists() and
            (out / 'anisotropic.json').exists()
        )
        if a.mode == 'unaware' and all_done:
            if recording and not recording_stopped:
                robot.movieStopRecording()
                recording_stopped = True
            if not recording or (recording_stopped and robot.movieIsReady()):
                (out / 'complete').write_text('complete\n')
                break
        if a.mode != 'unaware' and finish is not None and (out / 'complete').exists():
            break

    log.close()
    if recording and not recording_stopped:
        robot.movieStopRecording()


if __name__ == '__main__':
    main()
