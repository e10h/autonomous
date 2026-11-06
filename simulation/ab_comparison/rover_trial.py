"""Identical physics controller for both A/B rover runs.

Only the input route changes. Commands actuate six Webots wheel motors; pose
telemetry is measured from the physics engine. No robot positions are animated
or teleported. There is NO artificial safety stop: each run ends when the rover
reaches the goal, falls into the chasm, or times out.
"""
import argparse
import json
import math
from pathlib import Path
from controller import Supervisor

WHEEL_RADIUS = .1543
TRACK = 1.53
SPEED = .45
LOOKAHEAD_DISTANCE = 2.2
MAX_ACCELERATION = .3
MAX_YAW_ACCELERATION = .8
MAX_YAW_RATE = .65
GOAL_TOLERANCE = .45
SETTLE_SECONDS = 3.0
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
    summary_path = out / (a.mode + '.json')
    trace_path = out / (a.mode + '.jsonl')
    log = trace_path.open('w', buffering=1)
    while robot.step(step) != -1:
        now = robot.getTime()
        pos = body.getPosition()
        x, y = pos[0] - offset, pos[1]
        rot = body.getOrientation()
        yaw = math.atan2(rot[3], rot[0])
        roll = math.degrees(math.atan2(rot[7], rot[8]))
        pitch = math.degrees(math.atan2(rot[6], math.hypot(rot[0], rot[3])))
        v = w = 0.
        if now >= SETTLE_SECONDS and start_time is None:
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
                v = 0. if abs(error) > .45 else SPEED * max(.25, math.cos(error))
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
            robot.setLabel(lid + 1, f'{status.upper()} | roll {abs(roll):.0f} deg | drop {max(0., drop):.1f} m', px, .09, .036, 0xFFFFFF, 0, 'Arial')
            robot.setLabel(lid + 2, f'{(now - start_time) if start_time else 0:.0f}s | goal error {math.dist((x, y), points[-1]):.1f} m',
                           px, .14, .036, 0xFFFFFF, 0, 'Arial')
            if a.mode == 'isotropic':
                robot.setLabel(31, '21 deg loose-scree side-slope above a chasm (friction 0.30) | red tape = chasm lip', .035, .93, .028, 0xFFFFFF, 0, 'Arial')
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
        if a.mode == 'anisotropic' and (out / 'complete').exists():
            break
    log.close()
    if recording and not recording_stopped:
        robot.movieStopRecording()


if __name__ == '__main__':
    main()
