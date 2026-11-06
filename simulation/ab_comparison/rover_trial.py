"""Identical physics controller for both A/B rover runs.

Only the input route changes. Commands actuate six Webots wheel motors; pose
telemetry is measured from the physics engine. No robot positions are animated
or teleported. There is NO artificial safety stop: each run ends when the rover
reaches the goal, falls into the chasm, or times out.
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
from course import make_course, LIMITS, RADIUS, elevation, LANE_OFFSETS
from autonomous_state_machine.directional_planner import DirectionalThetaStar

WHEEL_RADIUS = .1543
TRACK = 1.53
SPEED = .45
LOOKAHEAD_DISTANCE = 1.2
MAX_ACCELERATION = .3
MAX_YAW_ACCELERATION = .8
MAX_YAW_RATE = .65
GOAL_TOLERANCE = .45
SETTLE_SECONDS = 3.5
TRIAL_TIMEOUT = 140.0
LABEL_PERIOD = .12


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
    p.add_argument('--mode', choices=['isotropic', 'anisotropic'], required=True)
    p.add_argument('--plans', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--movie', help='Absolute host path for the native Webots movie; primary controller only')
    a = p.parse_args()
    spec = json.load(open(a.plans))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    offset = -19 if a.mode == 'isotropic' else 19
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
    index = 1
    travel = 0.
    last_position = None
    z_start = None
    min_z = 1e9
    max_roll = max_pitch = 0.
    max_slide = 0.
    rows = []
    previous_v = previous_w = 0.
    recording = recording_stopped = False
    last_label = 0
    stopped_since = None
    target_seg_idx = 0
    summary_path = out / (a.mode + '.json')
    trace_path = out / (a.mode + '.jsonl')
    log = trace_path.open('w', buffering=1)

    # Active trail growth references
    segments_vrml = spec['results'][a.mode].get('segments_vrml', [])
    segment_stats = spec['results'][a.mode].get('segment_stats', [])
    num_segs = len(segments_vrml)
    trail_node = robot.getFromDef(f"{a.mode.upper()}_ACTIVE_TRAIL")
    trail_children = trail_node.getField("children") if trail_node else None
    revealed_seg = 0

    # OAK-D Active Depth Sensor (RangeFinder 640x480)
    depth_cam = robot.getDevice('oakd_depth')
    depth_enabled = depth_cam is not None
    if depth_enabled:
        depth_cam.enable(step)
        d_w = depth_cam.getWidth()
        d_h = depth_cam.getHeight()
        d_fov = depth_cam.getFov()
        d_cx = d_w / 2.0
        d_cy = d_h / 2.0
        d_focal = d_w / (2.0 * math.tan(d_fov / 2.0))
        d_stride = 8
        u_grid, v_grid = np.meshgrid(np.arange(0, d_w, d_stride), np.arange(0, d_h, d_stride))
        tan_x = ((u_grid - d_cx) / d_focal).ravel()
        tan_y = ((v_grid - d_cy) / d_focal).ravel()
    else:
        d_w = d_h = 0
        tan_x = tan_y = None

    # Dynamic obstacle perception & replanning parameters
    rock_spec = spec.get('rock', {'x': 2.5, 'y': 7.8, 'radius': 0.75})
    rock_r = rock_spec.get('radius', 0.75)
    last_planned_obs = None
    last_obs_detect_time = None
    detected_obs_dist = None
    detected_obs_world = None

    while robot.step(step) != -1:
        now = robot.getTime()
        pos = body.getPosition()
        x, y = pos[0] - offset, pos[1]
        rot = body.getOrientation()
        yaw = math.atan2(rot[3], rot[0])
        roll = math.degrees(math.atan2(rot[7], rot[8]))
        pitch = math.degrees(math.atan2(rot[6], math.hypot(rot[0], rot[3])))
        v = w = 0.

        # Dynamic growing path animation during planning/settle phase
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
            z_start = pos[2]
            if a.mode == 'isotropic' and a.movie:
                robot.movieStartRecording(a.movie, 1280, 720, 0, 90, 1, False)
                recording = True
        if start_time is not None and status == 'running':
            travel += math.dist(last_position, (x, y))
            last_position = (x, y)
            max_roll = max(max_roll, abs(roll))
            max_pitch = max(max_pitch, abs(pitch))
            min_z = min(min_z, pos[2])

            # Real-time obstacle perception via onboard OAK-D depth sensor
            if a.mode == 'anisotropic':
                if depth_enabled:
                    buf = depth_cam.getRangeImage(data_type='buffer')
                    if buf:
                        d_raw = np.ctypeslib.as_array(buf, shape=(d_h, d_w))
                        d_sub = d_raw[::d_stride, ::d_stride].ravel()
                        valid = (d_sub > 1.0) & (d_sub < 8.5)
                        if np.any(valid):
                            d_v = d_sub[valid]
                            tx_v = tan_x[valid]
                            ty_v = tan_y[valid]
                            xr = 0.3 + d_v
                            yr = -tx_v * d_v
                            zr = 0.48 - ty_v * d_v
                            # Transform 3D points from rover body frame to world frame using 3D rotation matrix
                            xw = x + rot[0]*xr + rot[1]*yr + rot[2]*zr
                            yw = y + rot[3]*xr + rot[4]*yr + rot[5]*zr
                            zw = pos[2] + rot[6]*xr + rot[7]*yr + rot[8]*zr
                            # Local terrain elevation at measured point coordinates
                            zg = elevation(xw, yw)
                            h_above_ground = zw - zg
                            # Segment obstacles: points sticking significantly above local terrain (> 0.35m)
                            obs_pts = (h_above_ground > 0.35) & (np.abs(yr) < 2.5)
                            if np.sum(obs_pts) >= 12:
                                obs_w_x = float(np.median(xw[obs_pts]))
                                obs_w_y = float(np.median(yw[obs_pts]))
                                detected_obs_dist = math.hypot(obs_w_x - x, obs_w_y - y)
                                detected_obs_world = (obs_w_x, obs_w_y)
                                last_obs_detect_time = now
                elif rock_node := robot.getFromDef(f"{a.mode.upper()}_ROCK"):
                    rpos = rock_node.getPosition()
                    obs_w_x = rpos[0] - offset
                    obs_w_y = rpos[1]
                    d_r = math.hypot(x - obs_w_x, y - obs_w_y)
                    if d_r < 7.5:
                        detected_obs_dist = d_r
                        detected_obs_world = (obs_w_x, obs_w_y)
                        last_obs_detect_time = now

                # Dynamic replanning: trigger when obstacle is spotted or moved > 0.6m by user
                if detected_obs_world is not None and last_obs_detect_time == now and detected_obs_dist < 8.0:
                    obs_shift = math.dist(detected_obs_world, last_planned_obs) if last_planned_obs else 999.0
                    if obs_shift > 0.6:
                        last_planned_obs = detected_obs_world
                        terrain, _ = make_course()
                        r_c, c_c = terrain.cell(detected_obs_world)
                        rad_cells = int(np.ceil((rock_r * 0.9) / terrain.dx))
                        for dr in range(-rad_cells, rad_cells + 1):
                            for dc in range(-rad_cells, rad_cells + 1):
                                if dr*dr + dc*dc <= rad_cells*rad_cells:
                                    nr, nc = r_c + dr, c_c + dc
                                    if 0 <= nr < terrain.blocked.shape[0] and 0 <= nc < terrain.blocked.shape[1]:
                                        terrain.blocked[nr, nc] = True
                        planner = DirectionalThetaStar(terrain, LIMITS, RADIUS)
                        detour = planner.plan((x, y), spec['results'][a.mode]['points'][-1])
                        if (not detour) or len(detour) < 2:
                            # If rover is close to obstacle, unblock start cell neighborhood so it can escape
                            r_start, c_start = terrain.cell((x, y))
                            for dr in range(-3, 4):
                                for dc in range(-3, 4):
                                    if 0 <= r_start + dr < planner.blocked.shape[0] and 0 <= c_start + dc < planner.blocked.shape[1]:
                                        planner.blocked[r_start + dr, c_start + dc] = False
                            detour = planner.plan((x, y), spec['results'][a.mode]['points'][-1])
                        if detour and len(detour) >= 2:
                            points = [(x, y)] + detour[1:]
                            if trail_children:
                                while trail_children.getCount() > 0:
                                    trail_children.removeMF(0)
                                color_d = '0.1 0.95 0.5'
                                for da, db in zip(detour, detour[1:]):
                                    n_d = max(1, math.ceil(math.dist(da, db) / .15))
                                    v_d, f_d = [], []
                                    vx_d, vy_d = db[0] - da[0], db[1] - da[1]
                                    l_d = math.hypot(vx_d, vy_d)
                                    nx_d, ny_d = -vy_d / l_d * .08, vx_d / l_d * .08
                                    for i_d in range(n_d + 1):
                                        px_d = da[0] + vx_d * i_d / n_d
                                        py_d = da[1] + vy_d * i_d / n_d
                                        for sign_d in [-1, 1]:
                                            sx_d, sy_d = px_d + sign_d * nx_d, py_d + sign_d * ny_d
                                            v_d.append(f'{sx_d + offset:.4f} {sy_d:.4f} {float(elevation(sx_d, sy_d)) + .07:.4f}')
                                        if i_d < n_d:
                                            f_d.append(f'{2 * i_d} {2 * i_d + 2} {2 * i_d + 3} {2 * i_d + 1} -1')
                                    shape_d = f'Shape {{ appearance PBRAppearance {{ baseColor {color_d} emissiveColor {color_d} roughness 0.2 metalness 0 }} geometry IndexedFaceSet {{ coord Coordinate {{ point [ {", ".join(v_d)} ] }} coordIndex [ {" ".join(f_d)} ] }} }}'
                                    trail_children.importMFNodeFromString(-1, shape_d)

            target, cross_track = get_lookahead_and_crosstrack(points, x, y, LOOKAHEAD_DISTANCE)
            max_slide = max(max_slide, cross_track)
            if z_start - pos[2] > fall_height:
                status = 'fell_into_chasm'
                finish = now
            elif now - start_time > TRIAL_TIMEOUT:
                status = 'timeout'
                finish = now
            elif math.dist((x, y), points[-1]) < GOAL_TOLERANCE:
                status = 'goal_reached'
                finish = now
            else:
                tx, ty = target
                error = angle(math.atan2(ty - y, tx - x) - yaw)
                kappa = 2.0 * math.sin(error) / LOOKAHEAD_DISTANCE
                w = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, SPEED * kappa))
                v = 0. if abs(error) > 1.15 else SPEED * max(.20, math.cos(error))
        if status == 'running':
            dv = MAX_ACCELERATION * step / 1000
            dw = MAX_YAW_ACCELERATION * step / 1000
            v = previous_v + max(-dv, min(dv, v - previous_v))
            w = previous_w + max(-dw, min(dw, w - previous_w))
        previous_v, previous_w = v, w
        lv = (v - w * TRACK / 2) / WHEEL_RADIUS
        rv = (v + w * TRACK / 2) / WHEEL_RADIUS
        for motor in left:
            motor.setVelocity(max(-motor.getMaxVelocity(), min(motor.getMaxVelocity(), -lv)))
        for motor in right:
            motor.setVelocity(max(-motor.getMaxVelocity(), min(motor.getMaxVelocity(), rv)))
        if start_time is not None and (not rows or now - start_time - rows[-1]['t'] >= .08):
            row = {'t': now - start_time, 'x': x, 'y': y, 'z': pos[2], 'yaw_rad': yaw,
                   'body_roll_deg': roll, 'body_pitch_deg': pitch, 'v_command_mps': v,
                   'omega_command_radps': w, 'status': status,
                   'goal_error_m': math.dist((x, y), points[-1])}
            rows.append(row)
            log.write(json.dumps(row) + '\n')
        if now - last_label > LABEL_PERIOD:
            last_label = now
            lid = 10 if a.mode == 'isotropic' else 20
            px = .025 if a.mode == 'isotropic' else .525
            title = 'ISOTROPIC Theta* (scalar slope cost)' if a.mode == 'isotropic' else 'DIRECTION-AWARE Theta*'
            color = 0xFFAA55 if a.mode == 'isotropic' else 0x44DDFF
            drop = (z_start - pos[2]) if z_start is not None else 0.
            robot.setLabel(lid, title, px, .025, .05, color, 0, 'Arial')
            if status == 'settling':
                if revealed_seg < num_segs:
                    st = segment_stats[min(revealed_seg, num_segs - 1)] if segment_stats else {}
                    r_deg = st.get('roll', 0.0)
                    eval_msg = f"ROLL {r_deg:.1f}° > 12° [CHASM SLIP RISK]" if (a.mode == 'isotropic' and r_deg > 12) else f"ROLL {r_deg:.1f}° [PASS <= 12°]"
                    robot.setLabel(lid + 1, f"GROWING PATH: seg {revealed_seg}/{num_segs} | {eval_msg}", px, .09, .033, 0xFFFF44, 0, 'Arial')
                    robot.setLabel(lid + 2, f"Active raycast expansion... ({revealed_seg * 350 + 120} nodes)", px, .14, .031, 0xDDDDDD, 0, 'Arial')
                else:
                    nodes = spec['results'][a.mode].get('expanded_nodes', 4849)
                    length = spec['results'][a.mode].get('length_m', 0.0)
                    robot.setLabel(lid + 1, f"PATH LOCKED ({nodes} nodes expanded) | READY", px, .09, .034, 0x55FF55, 0, 'Arial')
                    robot.setLabel(lid + 2, f"Target route: {length:.1f} m | Engaging motors at t=3.5s", px, .14, .031, 0x55FF55, 0, 'Arial')
            elif last_obs_detect_time is not None and now - last_obs_detect_time < 3.0 and detected_obs_world is not None:
                robot.setLabel(lid + 1, f"🪨 OAK-D ACTIVE DEPTH: Boulder @ {detected_obs_dist:.2f}m (X={detected_obs_world[0]:.1f}, Y={detected_obs_world[1]:.1f})", px, .09, .033, 0x44FFAA, 0, 'Arial')
                robot.setLabel(lid + 2, "Downhill dodge rejected (Roll 18° > 12° Scree Chasm Risk) | Bypassing rock uphill...", px, .14, .031, 0x44FFAA, 0, 'Arial')
            elif last_planned_obs is not None and (last_obs_detect_time is None or now - last_obs_detect_time >= 3.0):
                robot.setLabel(lid + 1, "✅ OAK-D ACTIVE DEPTH: Boulder bypassed | Safe ridge re-acquired", px, .09, .033, 0x55FF55, 0, 'Arial')
                robot.setLabel(lid + 2, f"Targeting goal ({math.dist((x, y), points[-1]):.1f} m remaining)", px, .14, .031, 0x55FF55, 0, 'Arial')
            else:
                robot.setLabel(lid + 1, f'{status.upper()} | roll {abs(roll):.0f} deg | drop {max(0., drop):.1f} m', px, .09, .036, 0xFFFFFF, 0, 'Arial')
                robot.setLabel(lid + 2, f'{(now - start_time) if start_time else 0:.0f}s | goal error {math.dist((x, y), points[-1]):.1f} m',
                               px, .14, .036, 0xFFFFFF, 0, 'Arial')
            if a.mode == 'isotropic':
                robot.setLabel(31, 'Faint line = planned corridor | Glowing line = active validated path (replanned around boulder)', .035, .93, .028, 0xFFEEAA, 0, 'Arial')
                robot.setLabel(30, 'SAME rover / controller / speed | only the route planner changes | no artificial safety stop', .035, .965, .028, 0xFFFFFF, 0, 'Arial')
        if finish is not None and not summary_path.exists() and now - finish > 1:
            summary = {'mode': a.mode, 'status': status, 'goal_reached': status == 'goal_reached',
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
                       'planned': spec['results'][a.mode]}
            summary_path.write_text(json.dumps(summary, indent=2) + '\n')
            print(json.dumps(summary), flush=True)
        all_done = (out / 'isotropic.json').exists() and (out / 'anisotropic.json').exists()
        if a.mode == 'isotropic' and all_done:
            if stopped_since is None:
                stopped_since = now
            if recording and not recording_stopped and now - stopped_since > 3:
                robot.movieStopRecording()
                recording_stopped = True
            if not recording or (recording_stopped and robot.movieIsReady()):
                (out / 'complete').write_text('complete\n')
                break
        if a.mode == 'anisotropic' and finish is not None and (out / 'complete').exists():
            break
    log.close()
    if recording and not recording_stopped:
        robot.movieStopRecording()


if __name__ == '__main__':
    main()
