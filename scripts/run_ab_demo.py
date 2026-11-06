#!/usr/bin/env python3
"""Run the controlled side-by-side Isotropic vs Anisotropic Webots comparison demo.

Launches Webots with both rovers running simultaneously in parallel lanes.
Isotropic cuts across the 18° cross-slope and halts on safety stop (roll > 15°).
Anisotropic (direction-aware) uses the level bypass ridge and reaches the goal.
"""
import os
import sys
import time
import socket
import subprocess
import shutil
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_DIR / "simulation" / "ab_comparison"
WORLD_FILE = DEMO_DIR / "worlds" / "comparison.wbt"
PLANS_FILE = DEMO_DIR / "plans.json"
OUTPUT_DIR = REPO_DIR / "outputs" / "ab-comparison"
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
    print("STARTING ISOTROPIC VS ANISOTROPIC SIDE-BY-SIDE WEBOTS DEMO")
    print("=" * 70)
    
    # 1. Clean previous runs
    subprocess.run(["pkill", "-9", "-f", "webots"], capture_output=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in OUTPUT_DIR.glob("*.json*"):
        f.unlink()
    if (OUTPUT_DIR / "complete").exists():
        (OUTPUT_DIR / "complete").unlink()
    time.sleep(1.0)

    # 2. Start Webots
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

    # Bring Webots to foreground so the user sees it
    subprocess.run(["osascript", "-e", 'tell application "Webots" to activate'], capture_output=True)

    # 3. Launch both controllers
    env_base = os.environ.copy()
    env_base["WEBOTS_HOME"] = WEBOTS_HOME
    env_base["PYTHONPATH"] = f"{PYTHON_LIB}:{REPO_DIR}"
    
    env_iso = env_base.copy()
    env_iso["WEBOTS_CONTROLLER_URL"] = "isotropic"

    env_aniso = env_base.copy()
    env_aniso["WEBOTS_CONTROLLER_URL"] = "anisotropic"

    movie_path = str(OUTPUT_DIR / "ab_comparison_run.mp4")
    cmd_iso = [
        sys.executable, str(DEMO_DIR / "rover_trial.py"),
        "--mode", "isotropic",
        "--plans", str(PLANS_FILE),
        "--output", str(OUTPUT_DIR),
        "--movie", movie_path,
    ]

    cmd_aniso = [
        sys.executable, str(DEMO_DIR / "rover_trial.py"),
        "--mode", "anisotropic",
        "--plans", str(PLANS_FILE),
        "--output", str(OUTPUT_DIR),
    ]

    print("[DEMO] Starting Isotropic controller...")
    proc_iso = subprocess.Popen(cmd_iso, env=env_iso)

    print("[DEMO] Starting Anisotropic controller...")
    proc_aniso = subprocess.Popen(cmd_aniso, env=env_aniso)

    print("[DEMO] Both rovers active in Webots simulation. Monitoring progress...")
    
    # Periodic screenshot capture
    start_t = time.time()
    next_snap = start_t + 5.0
    snap_idx = 1
    
    while True:
        iso_done = (OUTPUT_DIR / "isotropic.json").exists()
        aniso_done = (OUTPUT_DIR / "anisotropic.json").exists()
        complete = (OUTPUT_DIR / "complete").exists()

        now = time.time()
        if now >= next_snap and not complete:
            snap_path = OUTPUT_DIR / f"snap_{snap_idx:02d}.png"
            subprocess.run(["screencapture", "-x", str(snap_path)], capture_output=True)
            snap_idx += 1
            next_snap = now + 6.0

        if iso_done and aniso_done:
            print("[DEMO] Both controllers finished! Waiting for video to finalize...")
            t_wait = time.time()
            while not (OUTPUT_DIR / "complete").exists() and time.time() - t_wait < 40:
                time.sleep(0.5)
            break
        
        if now - start_t > 260.0:
            print("[DEMO] Timeout waiting for trial completion.")
            break
            
        time.sleep(1.0)

    # Final screenshot
    subprocess.run(["screencapture", "-x", str(OUTPUT_DIR / "final_state.png")], capture_output=True)
    
    # Terminate controller processes if still alive
    for p in (proc_iso, proc_aniso):
        if p.poll() is None:
            p.terminate()

    print("=" * 70)
    print("DEMO COMPLETE! Results in:", OUTPUT_DIR)
    print("=" * 70)

if __name__ == "__main__":
    main()
