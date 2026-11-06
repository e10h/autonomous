#!/usr/bin/env python3
"""Offline LiDAR prior preprocessing and unreviewed direction-aware macro-routes."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autonomous_state_machine.dem import prepare_geotiff
from autonomous_state_machine.terrain import TerrainGrid, TraversabilityLimits
from autonomous_state_machine.directional_planner import (
    DirectionalThetaStar,
    sparse_waypoints,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    ingest = subs.add_parser("prepare")
    ingest.add_argument("dem")
    ingest.add_argument("output")
    ingest.add_argument(
        "--resolution",
        type=float,
        help="Conservative planning cell size in metres; never finer than source",
    )
    ingest.add_argument(
        "--vertical-scale",
        type=float,
        required=True,
        help="Explicit input elevation unit multiplier to metres (1 for metres)",
    )
    ingest.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="Optional crop in source CRS units",
    )
    plan = subs.add_parser("plan")
    plan.add_argument("terrain")
    plan.add_argument("output")
    plan.add_argument(
        "--start", type=float, nargs=2, required=True, metavar=("EASTING", "NORTHING")
    )
    plan.add_argument(
        "--goal", type=float, nargs=2, required=True, metavar=("EASTING", "NORTHING")
    )
    plan.add_argument("--radius", type=float, default=0.8)
    plan.add_argument("--margin", type=float, default=100)
    plan.add_argument("--spacing", type=float, default=25)
    plan.add_argument("--timeout", type=float, default=30)
    plan.add_argument("--max-expansions", type=int, default=200000)
    plan.add_argument("--max-roll", type=float, default=15)
    plan.add_argument("--max-climb", type=float, default=22)
    plan.add_argument("--max-descent", type=float, default=22)
    args = parser.parse_args()
    if args.command == "prepare":
        print(
            json.dumps(
                prepare_geotiff(
                    args.dem,
                    args.output,
                    args.resolution,
                    args.vertical_scale,
                    bounds=args.bounds,
                ),
                indent=2,
            )
        )
        return
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(
            f"Refusing to replace an existing reviewed or unreviewed route: {out}"
        )
    terrain = TerrainGrid.open(args.terrain)
    window = terrain.window(args.start, args.goal, args.margin)
    limits = TraversabilityLimits(
        cliff_deg=terrain.metadata["limits"]["cliff_deg"],
        climb_deg=args.max_climb,
        descent_deg=args.max_descent,
        roll_deg=args.max_roll,
    )
    started = time.monotonic()
    planner = DirectionalThetaStar(window, limits, args.radius)
    points = planner.plan(
        tuple(args.start), tuple(args.goal), args.max_expansions, args.timeout
    )
    if not points:
        raise RuntimeError(
            "No safe route in this planning window; increase --margin or review terrain/limits. No fallback route was produced."
        )
    sparse = sparse_waypoints(points, args.spacing)
    route = {
        "schema": "yonder.macro-route.v1",
        "review_status": "unreviewed",
        "map_sha256": terrain.metadata["source_sha256"],
        "terrain_id": terrain.metadata.get("terrain_id") or hashlib.sha256(
            json.dumps(terrain.metadata, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "crs": terrain.metadata["crs"],
        "coordinate_unit": "metre",
        "limits": asdict(limits),
        "robot_radius_m": args.radius,
        "resolution_m": terrain.metadata["resolution_m"],
        "max_spacing_m": args.spacing,
        "waypoints": sparse,
        "length_m": sum(math.dist(a, b) for a, b in zip(sparse, sparse[1:])),
        "directional_cost": sum(
            planner.segment_cost(a, b) for a, b in zip(sparse, sparse[1:])
        ),
        "expanded_nodes": planner.expanded,
        "wall_seconds": time.monotonic() - started,
        "planning_window": {
            "shape": list(window.blocked.shape),
            "x0": window.x0,
            "y0": window.y0,
            "margin_m": args.margin,
        },
        "radio_transmitted": False,
    }
    route["route_sha256"] = hashlib.sha256(
        json.dumps(route, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(route, indent=2, allow_nan=False) + "\n")
    print(
        f'Unreviewed route: {route["length_m"]:.1f} m; {len(sparse)} waypoints; {route["wall_seconds"]:.2f}s. Saved {out}'
    )


if __name__ == "__main__":
    main()
