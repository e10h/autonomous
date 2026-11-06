#!/usr/bin/env python3
"""Run the bundled Canyon Gauntlet without modifying the ROS workspace."""
import argparse
import os
import re
import math
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / 'simulation/canyon_gauntlet'
REMOTE = '/tmp/autonomous-canyon-demo'
LOG = '/tmp/canyon-gauntlet.log'


def docker(container, *args, **kwargs):
    return subprocess.run(['docker', 'exec', container, *args], check=True, **kwargs)


def wait_for(predicate, timeout, message):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    raise RuntimeError(message)


def port_ready():
    try:
        with socket.create_connection(('127.0.0.1', 1234), timeout=0.5):
            return True
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', default='docker_environment-rover-1')
    parser.add_argument('--workspace', default='/home/yonder/ros2_ws', help='ROS workspace inside container')
    parser.add_argument('--webots', default=os.getenv('WEBOTS_BIN') or shutil.which('webots') or str(Path.home() / 'Applications/Webots.app/Contents/MacOS/webots'))
    parser.add_argument('--no-webots', action='store_true', help='Use an already open, freshly reset bundled world')
    parser.add_argument('--restart-container', action='store_true', help='Stop all processes in this simulation container before launch')
    parser.add_argument('--start-only', action='store_true', help='Await operator input instead of sending the demo goal')
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    if not args.no_webots and port_ready():
        parser.error('Webots controller port 1234 is busy. Close the previous world, or reset the bundled world and use --no-webots.')
    subprocess.run(['docker', 'restart' if args.restart_container else 'start', args.container], check=True)
    if not args.restart_container:
        running = subprocess.run(['docker', 'exec', args.container, 'pgrep', '-f', '[r]os2 launch|[a]utonomous_state_machine'], capture_output=True)
        if running.returncode == 0:
            parser.error('ROS processes already run in this container. Use a separate container or explicitly pass --restart-container.')
    with tempfile.TemporaryDirectory(prefix='canyon-stage-') as tmp:
        stage = Path(tmp) / 'autonomous-canyon-demo'
        ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
        shutil.copytree(REPO / 'autonomous_state_machine', stage / 'autonomous_state_machine', ignore=ignore)
        shutil.copytree(DEMO, stage / 'demo', ignore=ignore)
        subprocess.run(['docker', 'cp', str(stage), f'{args.container}:/tmp/'], check=True)
    setup = f'source /opt/ros/jazzy/setup.bash && source {shlex.quote(args.workspace + "/install/setup.bash")} && '
    env = (f'export PYTHONPATH={REMOTE}/demo/runtime:{REMOTE}:$PYTHONPATH '
           f'YONDER_STATIC_SLOPE_MAP={REMOTE}/demo/static_slope_costmap.npy '
           'YONDER_SIMULATION=1 YONDER_PUBLISH_COSTMAP=1 YONDER_ENABLE_DEPTH=0 '
           'YONDER_SIM_ROBOT_RADIUS_M=0.50 YONDER_SIM_BACKUP_DURATION=0.0 && ')
    if not args.no_webots:
        subprocess.Popen([args.webots, '--mode=realtime', str(DEMO / 'worlds/terrain_world.wbt')])
    wait_for(port_ready, 40, 'Webots controller port did not open.')
    docker(args.container, 'bash', '-lc', setup + env +
           f'python3 -c "import autonomous_state_machine.global_map as m; print(m.__file__)"')
    docker(args.container, 'bash', '-lc', f'test -f {shlex.quote(args.workspace + "/install/setup.bash")}')
    docker(args.container, 'bash', '-c', f': > {LOG}')
    subprocess.run(['docker', 'exec', '-d', args.container, 'bash', '-lc', setup + env +
                    f'exec ros2 launch {REMOTE}/demo/demo.launch.py > {LOG} 2>&1'], check=True)
    def logs():
        return docker(args.container, 'bash', '-c', f'if [ -f {LOG} ]; then cat {LOG}; fi', capture_output=True, text=True).stdout
    def ready():
        current = logs()
        if 'process has died' in current or 'Traceback (most recent call last)' in current:
            raise RuntimeError(f'Simulation process failed. Inspect {LOG}; reset the Webots world before retrying.')
        return ('FromLL service is available' in current and
                'Corrected for magnetic declination' in current)
    wait_for(ready, 60,
             f'Navigation initialization failed. Inspect docker exec {args.container} cat {LOG}')
    print('Canyon Gauntlet connected. Webots stays open after this launcher exits.', flush=True)
    if args.start_only:
        return
    # Give navsat datum calibration and DDS discovery time to settle.
    time.sleep(4)
    docker(args.container, 'bash', '-lc', setup + env +
           'python3 -c "from webots_bridge.send_waypoint import main; main()" '
           '--east -3.6 --north 9.0 --speed 0.6', timeout=30)
    wait_for(lambda: '[TF] map→odom goal:' in logs(), 15, 'Waypoint was not transformed into odometry coordinates.')
    target = re.search(r'map→odom goal: \(([-\d.]+), ([-\d.]+)\)', logs())
    if target is None or math.hypot(float(target[1]) + 3.6, float(target[2]) - 9.0) > 1.0:
        raise RuntimeError('Waypoint coordinates do not match the requested route; refusing to count this run as a success.')
    wait_for(lambda: 'goal confirmed' in logs(), args.timeout,
             f'Goal was not confirmed within {args.timeout}s. Inspect {LOG}; simulation remains open.')
    print('\n'.join(logs().splitlines()[-8:]), flush=True)


if __name__ == '__main__':
    main()
