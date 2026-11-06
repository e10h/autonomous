#!/usr/bin/env python3
"""Run the Cliff Detour Dilemma Demo: 2D Unaware vs 3D Terrain-Aware Avoidance.

Launches Webots with both rovers running simultaneously in parallel lanes.
- Lane 1 ('unaware'): 2D flat costmap planner. Blind to elevation drops,
  it chooses the right detour around the boulder and plummets 6 meters into the canyon.
- Lane 2 ('terrain_aware'): 3D Directional Theta* planner backed by UGIS elevation.
  Detects the right-hand drop as an unsafe hazard, routes safely inland to the left,
  and reaches the goal.
"""
import os
import sys
import time
import socket
import subprocess
import shutil
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_DIR / "simulation" / "cliff_detour"
WORLD_FILE = DEMO_DIR / "worlds" / "cliff_detour.wbt"
PLANS_FILE = DEMO_DIR / "plans.json"
OUTPUT_DIR = REPO_DIR / "outputs" / "cliff-detour"
WEBOTS_BIN = os.getenv("WEBOTS_BIN") or shutil.which("webots") or str(Path.home() / "Applications/Webots.app/Contents/MacOS/webots")
WEBOTS_HOME = str(Path.home() / "Applications/Webots.app")
PYTHON_LIB = str(Path.home() / "Applications/Webots.app/Contents/lib/controller/python")

def port_ready(host="127.0.0.1", port=1234, timeout=0.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def main():
    print("=" * 70)
    print("STARTING CLIFF DETOUR DILEMMA DEMO: 2D UNAWARE VS 3D TERRAIN-AWARE")
    print("=" * 70)
    
    # 1. Clean previous runs
    subprocess.run(["pkill", "-9", "-f", "webots"], capture_output=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in OUTPUT_DIR.glob("*.json*"):
        f.unlink()
    if (OUTPUT_DIR / "complete").exists():
        (OUTPUT_DIR / "complete").unlink()
    time.sleep(1.0)

    # 2. Build course
    print(f"[DEMO] Generating Cliff Detour course world and specifications...")
    subprocess.run([sys.executable, str(DEMO_DIR / "course.py"), str(DEMO_DIR)], check=True)

    # 3. Start Webots
    print(f"[DEMO] Launching Webots: {WORLD_FILE}")
    webots_proc = subprocess.Popen([WEBOTS_BIN, "--mode=realtime", str(WORLD_FILE)])

    deadline = time.time() + 45.0
    while time.time() < deadline:
        if port_ready():
            print("[DEMO] Webots controller port 1234 is ready!")
            break
        time.sleep(0.5)
    else:
        raise TimeoutError("Webots port 1234 did not open within 45s.")

    subprocess.run(["osascript", "-e", 'tell application "Webots" to activate'], capture_output=True)

    # 4. Launch both controllers
    env_base = os.environ.copy()
    env_base["WEBOTS_HOME"] = WEBOTS_HOME
    env_base["PYTHONPATH"] = f"{PYTHON_LIB}:{REPO_DIR}"
    
    env_unaware = env_base.copy()
    env_unaware["WEBOTS_CONTROLLER_URL"] = "unaware"

    env_aware = env_base.copy()
    env_aware["WEBOTS_CONTROLLER_URL"] = "terrain_aware"

    movie_path = str(OUTPUT_DIR / "cliff_detour_run.mp4")
    cmd_unaware = [
        sys.executable, str(DEMO_DIR / "rover_trial.py"),
        "--mode", "unaware",
        "--plans", str(PLANS_FILE),
        "--output", str(OUTPUT_DIR),
        "--movie", movie_path,
    ]

    cmd_aware = [
        sys.executable, str(DEMO_DIR / "rover_trial.py"),
        "--mode", "terrain_aware",
        "--plans", str(PLANS_FILE),
        "--output", str(OUTPUT_DIR),
    ]

    print("[DEMO] Starting 2D Terrain-Unaware controller...")
    proc_unaware = subprocess.Popen(cmd_unaware, env=env_unaware)

    print("[DEMO] Starting 3D Terrain-Aware controller...")
    proc_aware = subprocess.Popen(cmd_aware, env=env_aware)

    print("[DEMO] Both rovers active in Webots simulation. Monitoring progress...")
    
    start_t = time.time()
    next_snap = start_t + 5.0
    snap_idx = 1
    
    while True:
        unaware_done = (OUTPUT_DIR / "unaware.json").exists()
        aware_done = (OUTPUT_DIR / "terrain_aware.json").exists()
        complete = (OUTPUT_DIR / "complete").exists()

        now = time.time()
        if now >= next_snap and not complete:
            snap_path = OUTPUT_DIR / f"snap_{snap_idx:02d}.png"
            subprocess.run(["screencapture", "-x", str(snap_path)], capture_output=True)
            snap_idx += 1
            next_snap = now + 5.0

        if unaware_done and aware_done:
            print("[DEMO] Both controllers finished! Waiting for video to finalize...")
            t_wait = time.time()
            while not (OUTPUT_DIR / "complete").exists() and time.time() - t_wait < 30:
                time.sleep(0.5)
            break
        
        if now - start_t > 150.0:
            print("[DEMO] Timeout waiting for trial completion.")
            break
            
        time.sleep(1.0)

    subprocess.run(["screencapture", "-x", str(OUTPUT_DIR / "final_state.png")], capture_output=True)
    
    for p in (proc_unaware, proc_aware):
        if p.poll() is None:
            p.terminate()

    print("=" * 70)
    print("DEMO COMPLETE! Results in:", OUTPUT_DIR)
    print("=" * 70)

if __name__ == "__main__":
    main()

