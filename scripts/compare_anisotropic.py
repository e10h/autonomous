#!/usr/bin/env python3
"""A/B demo: run the same Canyon Gauntlet mission with and without anisotropic
(static slope layer) pathfinding, then build a side-by-side HTML report.

Usage:
    python3 scripts/compare_anisotropic.py [--east -3.6 --north 9.0 --timeout 120]
    python3 scripts/compare_anisotropic.py --report-only   # re-render from saved JSON

Output: $AUTON_ARTIFACT_DIR (default /tmp/artifacts)/anisotropic_comparison.html
"""
import argparse
import html
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.getenv("AUTON_ARTIFACT_DIR", "/tmp/artifacts")
# Canyon Gauntlet boulders (Webots frame, from build_canyon_gauntlet_sim.py).
# Odom frame is assumed to be Webots rotated 180 deg (goal 3.6,-9.0 <-> 9.0 north / -3.6 east).
ROCKS = [(-0.6, -3.5), (0.6, -3.7), (1.8, -4.8), (5.3, -5.5), (1.9, -7.6), (5.3, -7.5), (1.8, -9.6), (5.4, -9.6)]
ROCKS_ODOM = [(-x, -y) for x, y in ROCKS]


def run(mode, args):
    cmd = [sys.executable, os.path.join(HERE, "run_competition.py"),
           "--east", str(args.east), "--north", str(args.north),
           "--speed", str(args.speed), "--timeout", str(args.timeout), "--tag", mode]
    if mode == "baseline":
        cmd.append("--baseline")
    subprocess.run(cmd, check=False)


def load(mode):
    with open(f"{ART}/{mode}_result.json") as f:
        return json.load(f)


def svg(res, color, size=420, lo=-6, hi=12):
    sc = lambda x, y: ((x - lo) / (hi - lo) * size, size - (y - lo) / (hi - lo) * size)
    out = [f'<svg width="{size}" height="{size}" style="background:#fafafa;border:1px solid #ccc">']
    for x, y in ROCKS_ODOM:
        px, py = sc(x, y)
        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{0.4/(hi-lo)*size:.1f}" fill="#777"/>')
    pts = " ".join("%.1f,%.1f" % sc(x, y) for _, x, y in res["trajectory"])
    if pts:
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
    out.append('<circle cx="%.1f" cy="%.1f" r="6" fill="green"/>' % sc(0, 0))
    out.append('<circle cx="%.1f" cy="%.1f" r="7" fill="red"/>' % sc(*res["target"]))
    out.append("</svg>")
    return "".join(out)


def panel(title, res, color):
    t = f'{res["time_s"]:.1f} s' if res["time_s"] is not None else "n/a"
    e = f'{res["final_error_m"]:.2f} m' if res["final_error_m"] is not None else "n/a"
    ok = "&#9989; reached" if res["reached"] else "&#10060; did NOT reach goal"
    return (f"<div><h2>{html.escape(title)}</h2>{svg(res, color)}<ul>"
            f"<li>Result: {ok}</li><li>Time to goal: {t}</li><li>Final error: {e}</li>"
            f'<li>Rock contact steps: {res["rock_contact_steps"]}</li></ul></div>')


def report():
    b, a = load("baseline"), load("anisotropic")
    body = panel("Without anisotropic pathfinding", b, "#d33") + panel("With anisotropic pathfinding", a, "#27a")
    path = f"{ART}/anisotropic_comparison.html"
    with open(path, "w") as f:
        f.write("<html><body style='font-family:sans-serif'><h1>Canyon Gauntlet: anisotropic A/B</h1>"
                "<p>Green = start, red = goal, grey = boulders.</p>"
                f"<div style='display:flex;gap:40px'>{body}</div></body></html>")
    print(f"Report written: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--east", type=float, default=-3.6)
    p.add_argument("--north", type=float, default=9.0)
    p.add_argument("--speed", type=float, default=0.6)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    if not args.report_only:
        for mode in ("baseline", "anisotropic"):
            run(mode, args)
    report()


if __name__ == "__main__":
    main()
