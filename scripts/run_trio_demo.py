#!/usr/bin/env python3
"""Run the Trio Benchmark Demo: No UGIS vs Isotropic vs Anisotropic.

Launches Webots with 3 rovers running simultaneously in parallel lanes:
- Lane 1 ('unaware'): No UGIS (2D flat costmap). Steers off the cliff and falls 7m into the ravine.
- Lane 2 ('isotropic'): Isotropic Theta* with UGIS elevation. Avoids the cliff, but traverses
  horizontally across a 20.5-degree scree side-slope. Wheels slip down and it gets caught on the slope.
- Lane 3 ('anisotropic'): Directional Theta* with UGIS elevation. Rejects the side-slope roll hazard,
  routes safely inland on the flat plateau, and reaches the goal to win.
"""
import os
import sys
import time
import socket
import subprocess
import shutil
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_DIR / "simulation" / "trio_comparison"
WORLD_FILE = DEMO_DIR / "worlds" / "trio_comparison.wbt"
PLANS_FILE = DEMO_DIR / "plans.json"
OUTPUT_DIR = REPO_DIR / "outputs" / "trio-comparison"
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
    print("=" * 75)
    print("STARTING TRIO BENCHMARK: NO UGIS vs ISOTROPIC vs ANISOTROPIC")
    print("=" * 75)

    # 1. Clean previous runs
    subprocess.run(["pkill", "-9", "-f", "webots"], capture_output=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in OUTPUT_DIR.glob("*.json*"):
        f.unlink()
    if (OUTPUT_DIR / "complete").exists():
        (OUTPUT_DIR / "complete").unlink()
    time.sleep(1.0)

    # 2. Build course & world
    print("[TRIO] Generating course terrain and Webots world...")
    subprocess.run([sys.executable, str(DEMO_DIR / "course.py"), str(DEMO_DIR)], check=True)

    # 3. Start Webots
    print(f"[TRIO] Launching Webots: {WORLD_FILE}")
    webots_proc = subprocess.Popen([WEBOTS_BIN, "--mode=realtime", str(WORLD_FILE)])

    deadline = time.time() + 45.0
    while time.time() < deadline:
        if port_ready():
            print("[TRIO] Webots controller port 1234 is ready!")
            break
        time.sleep(0.5)
    else:
        raise TimeoutError("Webots port 1234 did not open within 45s.")

    subprocess.run(["osascript", "-e", 'tell application "Webots" to activate'], capture_output=True)

    # 4. Launch all 3 controllers
    env_base = os.environ.copy()
    env_base["WEBOTS_HOME"] = WEBOTS_HOME
    env_base["PYTHONPATH"] = f"{PYTHON_LIB}:{REPO_DIR}"

    movie_path = str(OUTPUT_DIR / "trio_comparison_run.mp4")

    # Unaware (Lane 1)
    env_unaware = env_base.copy()
    env_unaware["WEBOTS_CONTROLLER_URL"] = "unaware"
    cmd_unaware = [
        sys.executable, str(DEMO_DIR / "rover_trial.py"),
        "--mode", "unaware",
        "--plans", str(PLANS_FILE),
        "--output", str(OUTPUT_DIR),
        "--movie", movie_path,
    ]

    # Isotropic (Lane 2)
    env_iso = env_base.copy()
    env_iso["WEBOTS_CONTROLLER_URL"] = "isotropic"
    cmd_iso = [
        sys.executable, str(DEMO_DIR / "rover_trial.py"),
        "--mode", "isotropic",
        "--plans", str(PLANS_FILE),
        "--output", str(OUTPUT_DIR),
    ]

    # Anisotropic (Lane 3)
    env_aniso = env_base.copy()
    env_aniso["WEBOTS_CONTROLLER_URL"] = "anisotropic"
    cmd_aniso = [
        sys.executable, str(DEMO_DIR / "rover_trial.py"),
        "--mode", "anisotropic",
        "--plans", str(PLANS_FILE),
        "--output", str(OUTPUT_DIR),
    ]

    print("[TRIO] Starting 1. No UGIS controller...")
    p_unaware = subprocess.Popen(cmd_unaware, env=env_unaware)

    print("[TRIO] Starting 2. Isotropic controller...")
    p_iso = subprocess.Popen(cmd_iso, env=env_iso)

    print("[TRIO] Starting 3. Anisotropic controller...")
    p_aniso = subprocess.Popen(cmd_aniso, env=env_aniso)

    print("[TRIO] All 3 rovers active. Monitoring execution...")

    start_t = time.time()
    next_snap = start_t + 5.0
    snap_idx = 1

    while True:
        unaware_done = (OUTPUT_DIR / "unaware.json").exists()
        iso_done = (OUTPUT_DIR / "isotropic.json").exists()
        aniso_done = (OUTPUT_DIR / "anisotropic.json").exists()
        complete = (OUTPUT_DIR / "complete").exists()

        now = time.time()
        if now >= next_snap and not complete:
            snap_path = OUTPUT_DIR / f"snap_{snap_idx:02d}.png"
            subprocess.run(["screencapture", "-x", str(snap_path)], capture_output=True)
            snap_idx += 1
            next_snap = now + 5.0

        if unaware_done and iso_done and aniso_done:
            print("[TRIO] All 3 controllers finished! Finalizing video recording...")
            t_wait = time.time()
            while not (OUTPUT_DIR / "complete").exists() and time.time() - t_wait < 30.0:
                time.sleep(0.5)
            break

        if now - start_t > 300.0:
            print("[TRIO] Timeout waiting for trials to complete.")
            break

        time.sleep(1.0)

    subprocess.run(["screencapture", "-x", str(OUTPUT_DIR / "final_state.png")], capture_output=True)

    for p in (p_unaware, p_iso, p_aniso):
        if p.poll() is None:
            p.terminate()

    print("=" * 75)
    print("TRIO BENCHMARK COMPLETE! Results in:", OUTPUT_DIR)
    print("=" * 75)


if __name__ == "__main__":
    main()
