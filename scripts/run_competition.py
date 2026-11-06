#!/usr/bin/env python3
"""Programmatic end-to-end autonomous competition runner for Yonder Dynamics.

Starts Webots in realtime, launches the ROS 2 autonomy stack inside the
Docker container, verifies transforms and services, dispatches competition
GPS waypoints, and monitors navigation, obstacle avoidance, and goal arrival.
"""

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time

import shutil

CONTAINER_NAME = "docker_environment-rover-1"
def _find_webots() -> str:
    candidates = [
        os.getenv("WEBOTS_BIN"),
        shutil.which("webots"),
        os.path.expanduser("~/Applications/Webots.app/Contents/MacOS/webots"),
        "/Applications/Webots.app/Contents/MacOS/webots",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return "/Applications/Webots.app/Contents/MacOS/webots"

WEBBOTS_BIN_FOUND = _find_webots()
WEBOTS_APP = WEBBOTS_BIN_FOUND

# Resolve default world file relative to repository workspace if possible
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../../.."))
_DEFAULT_WORLD_CANDIDATES = [
    os.path.join(_WORKSPACE_ROOT, "sim_environment", "simulation", "worlds", "terrain_world.wbt"),
    "/Users/ethan/Code/yonder/sim_environment/simulation/worlds/terrain_world.wbt",
    "/Users/ethan/Code/yonderit/sim_environment/simulation/worlds/terrain_world.wbt",
]
_DEFAULT_WORLD = next((w for w in _DEFAULT_WORLD_CANDIDATES if os.path.exists(w)), _DEFAULT_WORLD_CANDIDATES[0])
WORLD_FILE = os.getenv("WEBOTS_WORLD_PATH", _DEFAULT_WORLD)
LOG_FILE = "/tmp/competition-run.log"


def log(msg: str):
    print(f"\033[1;36m[AUTON-RUNNER]\033[0m {msg}", flush=True)


def check_port(host: str, port: int, timeout: float = 1.0) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        s.close()
        return True
    except (socket.error, ConnectionRefusedError):
        return False


def run_cmd(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def clean_environment():
    log("Cleaning up any existing Webots or ROS container processes...")
    subprocess.run(["pkill", "-9", "-f", "webots"], capture_output=True)
    run_cmd([
        "docker", "exec", CONTAINER_NAME, "bash", "-c",
        "pkill -9 -f 'ros2|competition_sim|autonomous_state_machine|sim_localization|navsat_transform|static_transform|webots_bridge|Supervisor' || true"
    ])
    time.sleep(1.0)


def start_webots(world_file: str = WORLD_FILE):
    proc = subprocess.Popen([
        WEBOTS_APP, "--mode=realtime", world_file
    ])

    log("Waiting for Webots controller TCP server on port 1234...")
    deadline = time.time() + 30.0
    while time.time() < deadline:
        if check_port("127.0.0.1", 1234):
            time.sleep(3.0)
            log("Webots TCP port 1234 is LISTENING and ready!")
            subprocess.run(["osascript", "-e", 'tell application "Webots" to activate'], capture_output=True)
            return proc
        time.sleep(0.5)

    raise TimeoutError("Webots port 1234 did not open within 30 seconds.")


def start_ros_stack(anisotropic: bool = True):
    mode = "ANISOTROPIC (static slope layer ON)" if anisotropic else "BASELINE (isotropic, slope layer OFF)"
    log(f"Launching ROS 2 competition stack inside Docker container: {mode}")
    slope_export = "export YONDER_STATIC_SLOPE_MAP=1 && " if anisotropic else "unset YONDER_STATIC_SLOPE_MAP && "
    run_cmd([
        "docker", "exec", CONTAINER_NAME, "bash", "-c",
        f"rm -f {LOG_FILE}"
    ])
    run_cmd([
        "docker", "exec", "-d", CONTAINER_NAME, "bash", "-c",
        f"source /ros_entrypoint.sh && source /home/yonder/ros2_ws/install/setup.bash && "
        f"export YONDER_SIMULATION=1 && export YONDER_PUBLISH_COSTMAP=1 && "
        f"export YONDER_SIM_ROBOT_RADIUS_M=0.50 && export YONDER_SIM_BACKUP_DURATION=0.0 && "
        f"export YONDER_ENABLE_DEPTH=0 && {slope_export}"
        f"ros2 launch webots_bridge competition_sim.launch.py > {LOG_FILE} 2>&1"
    ])

    log("Waiting for FromLL service and Datum UTM calibration...")
    deadline = time.time() + 50.0
    datum_ready = False
    fromll_ready = False

    while time.time() < deadline:
        res = run_cmd(["docker", "exec", CONTAINER_NAME, "cat", LOG_FILE])
        output = res.stdout or ""
        if "FromLL service is available" in output:
            fromll_ready = True
        if "Corrected for magnetic declination" in output or "Datum UTM coordinate" in output:
            datum_ready = True

        if fromll_ready and datum_ready:
            log("ROS 2 navigation stack and datum are fully initialized!")
            return
        time.sleep(1.0)

    raise TimeoutError("ROS 2 navigation stack failed to initialize within 35 seconds.")


def dispatch_waypoint(east: float, north: float, speed: float, max_attempts: int = 5):
    log(f"Dispatching waypoint: east={east:+.2f} m, north={north:+.2f} m, speed={speed:.2f} m/s...")
    for attempt in range(1, max_attempts + 1):
        cmd = (
            f"source /ros_entrypoint.sh && source /home/yonder/ros2_ws/install/setup.bash && "
            f"python3 -c '\n"
            f"import rclpy\n"
            f"from webots_bridge.send_waypoint import WaypointSender\n"
            f"rclpy.init()\n"
            f"node = WaypointSender({east}, {north}, {speed}, 0)\n"
            f"for _ in range(50):\n"
            f"    rclpy.spin_once(node, timeout_sec=0.1)\n"
            f"    if node.done.is_set():\n"
            f"        for _ in range(15):\n"
            f"            rclpy.spin_once(node, timeout_sec=0.05)\n"
            f"        break\n"
            f"rclpy.shutdown()\n"
            f"'"
        )
        run_cmd(["docker", "exec", CONTAINER_NAME, "bash", "-c", cmd])
        time.sleep(1.0)
        res = run_cmd(["docker", "exec", CONTAINER_NAME, "cat", LOG_FILE])
        if "Received Autonomous Input" in (res.stdout or ""):
            log("Waypoint successfully ingested by autonomous state machine.")
            return
        log(f"Waiting for state machine to ingest waypoint (attempt {attempt})...")
        time.sleep(1.5)


ARTIFACT_DIR = os.getenv("AUTON_ARTIFACT_DIR", "/tmp/artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


TRAJ_FILE = "/tmp/trajectory.csv"
_RECORDER = (
    "import rclpy,time\n"
    "from rclpy.node import Node\n"
    "from nav_msgs.msg import Odometry\n"
    "rclpy.init(); n=Node('traj_rec'); t0=time.time(); f=open('%s','w'); f.write('t,x,y\\n')\n"
    "def cb(m):\n"
    "    f.write('%%.2f,%%.3f,%%.3f\\n'%%(time.time()-t0,m.pose.pose.position.x,m.pose.pose.position.y)); f.flush()\n"
    "n.create_subscription(Odometry,'/autonomous/localization/odometry/local',cb,10)\n"
    "rclpy.spin(n)\n"
) % TRAJ_FILE


def start_trajectory_recorder():
    run_cmd(["docker", "exec", CONTAINER_NAME, "bash", "-c", f"rm -f {TRAJ_FILE}"])
    subprocess.run([
        "docker", "exec", "-d", CONTAINER_NAME, "bash", "-c",
        "source /ros_entrypoint.sh && source /home/yonder/ros2_ws/install/setup.bash && "
        f"python3 -c \"{_RECORDER}\" "
    ], capture_output=True)


def fetch_trajectory() -> list[tuple[float, float, float]]:
    res = run_cmd(["docker", "exec", CONTAINER_NAME, "cat", TRAJ_FILE])
    pts = []
    for line in (res.stdout or "").splitlines()[1:]:
        try:
            t, x, y = (float(v) for v in line.split(","))
            pts.append((t, x, y))
        except ValueError:
            pass
    return pts


def monitor_navigation(target_east: float, target_north: float, timeout_sec: float = 90.0, tag: str = "anisotropic_ground"):
    log(f"Monitoring autonomous navigation (timeout: {timeout_sec}s)...")
    start_time = time.time()
    reached = False
    elapsed = None
    seen_lines = set()
    last_snap_time = 0.0
    snap_idx = 1

    # Bring Webots to front so the user can watch the 3D simulation
    subprocess.run(["osascript", "-e", 'tell application "Webots" to activate'], capture_output=True)

    while time.time() - start_time < timeout_sec:
        # Check log for state machine events
        res = run_cmd(["docker", "exec", CONTAINER_NAME, "tail", "-n", "15", LOG_FILE])
        lines = (res.stdout or "").strip().split("\n")
        for line in lines:
            if any(k in line for k in ["Transition", "Target reached", "ALIGNED", "Obstacle detected", "GoalState", "ValidatingState"]):
                if line not in seen_lines:
                    seen_lines.add(line)
                    clean = re.sub(r"\[.*?\]\s*", "", line, count=2)
                    print(f"  \033[33m→\033[0m {clean}")

        # Capture periodic visual snapshots of Webots during obstacle avoidance
        now = time.time()
        if now - last_snap_time >= 4.0:
            last_snap_time = now
            subprocess.run(["osascript", "-e", 'tell application "Webots" to activate'], capture_output=True)
            subprocess.run(["screencapture", "-x", f"{ARTIFACT_DIR}/{tag}_webots_live.png"], capture_output=True)
            snap_idx += 1

        if "GoalState" in res.stdout or "goal confirmed" in res.stdout:
            elapsed = time.time() - start_time
            reached = True
            log(f"\033[1;32mGOAL REACHED in {elapsed:.1f} seconds!\033[0m")
            subprocess.run(["screencapture", "-x", f"{ARTIFACT_DIR}/{tag}_webots_goal.png"], capture_output=True)
            break

        time.sleep(1.0)
    else:
        log("\033[1;31mMonitoring timed out before GoalState was achieved.\033[0m")

    # Fetch final telemetry
    odom_res = run_cmd([
        "docker", "exec", CONTAINER_NAME, "bash", "-c",
        "source /ros_entrypoint.sh && source /home/yonder/ros2_ws/install/setup.bash && "
        "ros2 topic echo /autonomous/localization/odometry/local --once --timeout 5"
    ])
    contact_res = run_cmd([
        "docker", "exec", CONTAINER_NAME, "bash", "-c",
        "source /ros_entrypoint.sh && source /home/yonder/ros2_ws/install/setup.bash && "
        "ros2 topic echo /simulation/contacts --once --timeout 5"
    ])

    print("\n" + "=" * 60)
    print("FINAL MISSION TELEMETRY")
    print("=" * 60)

    # Parse position
    x_match = re.search(r"x:\s*([-\d.]+)", odom_res.stdout or "")
    y_match = re.search(r"y:\s*([-\d.]+)", odom_res.stdout or "")
    final_x = float(x_match.group(1)) if x_match else None
    final_y = float(y_match.group(1)) if y_match else None
    err = None
    steps = "0"

    if final_x is not None and final_y is not None:
        err = ((final_x - target_east) ** 2 + (final_y - target_north) ** 2) ** 0.5
        print(f"Final Position (odom):  x = {final_x:+.3f} m, y = {final_y:+.3f} m")
        print(f"Target Waypoint:        x = {target_east:+.3f} m, y = {target_north:+.3f} m")
        print(f"Goal Position Error:    {err:.3f} m (threshold: 0.80 m)")
    
    # Parse rock contacts
    if "rock_contact_steps" in (contact_res.stdout or ""):
        steps_match = re.search(r'"rock_contact_steps":\s*(\d+)', contact_res.stdout)
        steps = steps_match.group(1) if steps_match else "0"
        print(f"Rock Contact Steps:     {steps} steps (0 = flawless obstacle avoidance)")

    print("=" * 60 + "\n")
    return {
        "reached": reached,
        "time_s": elapsed if reached else None,
        "final_error_m": err,
        "rock_contact_steps": int(steps),
    }


def main():
    parser = argparse.ArgumentParser(description="End-to-end programmatic competition runner")
    parser.add_argument("--world", type=str, default=WORLD_FILE, help="world file to load")
    parser.add_argument("--east", type=float, default=-3.6, help="target east meters relative to start")
    parser.add_argument("--north", type=float, default=9.0, help="target north meters relative to start")
    parser.add_argument("--speed", type=float, default=0.6, help="drive speed (m/s)")
    parser.add_argument("--timeout", type=float, default=90.0, help="mission timeout seconds")
    parser.add_argument("--baseline", action="store_true", help="disable the anisotropic static slope layer")
    parser.add_argument("--tag", type=str, default=None, help="artifact filename prefix")
    parser.add_argument("--keep-running", action="store_true", help="keep Webots and ROS running after mission")
    args = parser.parse_args()

    clean_environment()
    webots_proc = None
    try:
        webots_proc = start_webots(args.world)
        start_ros_stack(anisotropic=not args.baseline)
        tag = args.tag or ("baseline" if args.baseline else "anisotropic")
        # Give 4 seconds for autonomous_state_machine to finish setup and subscribe
        time.sleep(4.0)
        subprocess.run(["osascript", "-e", 'tell application "Webots" to activate'], capture_output=True)
        time.sleep(0.5)
        subprocess.run(["screencapture", "-x", f"{ARTIFACT_DIR}/{tag}_webots_clean.png"], capture_output=True)
        start_trajectory_recorder()
        dispatch_waypoint(args.east, args.north, args.speed)
        result = monitor_navigation(args.east, args.north, timeout_sec=args.timeout, tag=tag)
        result.update(tag=tag, mode="baseline" if args.baseline else "anisotropic",
                      target=[args.east, args.north], trajectory=fetch_trajectory())
        with open(f"{ARTIFACT_DIR}/{tag}_result.json", "w") as f:
            json.dump(result, f)
        log(f"Saved result to {ARTIFACT_DIR}/{tag}_result.json")
        if args.keep_running and webots_proc:
            log("\033[1;32m[AUTON-RUNNER] Mission complete! Rover is at goal. Keeping Webots open for inspection...\033[0m")
            try:
                webots_proc.wait()
            except KeyboardInterrupt:
                pass
    finally:
        if not args.keep_running:
            log("Shutting down Webots...")
            subprocess.run(["pkill", "-9", "-f", "webots"], capture_output=True)


if __name__ == "__main__":
    main()
