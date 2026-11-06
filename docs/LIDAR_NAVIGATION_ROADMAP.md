# LiDAR-to-local-navigation roadmap

## Objective

An operator loads a 10–20 cm global LiDAR elevation map (100 MB–1 GB), reviews a
100–500 m direction-aware macro-route, and approves sparse macro-waypoints for
radio handoff. The rover follows that route using a separate local OAK-D terrain
and obstacle overlay and a local trajectory controller.

Pipeline:

1. Ingest georeferenced GeoTIFF or ground-classified PLY without loading the whole
   point cloud or raster into RAM. Preserve CRS, units, nodata, and map identity.
2. Precompute elevation gradients and hard hazards. Slopes above 22° are a hard
   exclusion under the current policy. Below that cap, traversability depends on
   direction: pitch = atan(gradient dot heading); roll = atan(gradient dot lateral).
   Uphill, downhill, and lateral limits remain independently configurable.
3. Plan with direction-aware Theta*: integrate edge costs over the entire segment,
   check directional limits and the rover footprint, and retain safe detours.
   Theta* is an any-angle heuristic planner; do not claim global optimality.
4. Operator previews, edits, and explicitly approves the route in the UI. Every
   edited leg must be validated again. No planner output is automatically approved.
5. Export sparse waypoints, normally 15–30 m apart, retaining turns needed for
   clearance. Packetize to the radio MTU: a 500 m route may need multiple ~100-byte
   packets. Include route/map identity, coordinate origin, sequence, and integrity.
6. On rover: maintain a roughly 10 m × 10 m rolling local terrain/obstacle envelope
   from `/autonomous/obstacle_avoidance/oakd/pointcloud`. Keep the static prior and
   dynamic observations separate. New free-space observations must never erase
   hard cliffs from the global prior. Track registration uncertainty and enlarge
   clearance or stop when map alignment is uncertain.
7. A local trajectory controller selects safe gaps and avoids new rocks while
   progressing along the reviewed macro-route. Stop and request a new route when
   no safe local continuation exists; never silently fall back to a weaker plan.

## Implementation stages

- [x] Preserve the Canyon Gauntlet example and reproducible simulation assets.
- [x] Shared directional edge model and fail-closed Theta* used by static planning.
- [x] Tiled GeoTIFF preprocessing, conservative macro-resolution prior, map metadata.
- [x] Offline macro-route planning and safe waypoint spacing with measurable tests.
- [ ] Ground-classified PLY ingestion and coordinate registration workflow.
- [ ] Frontend upload/job progress/map preview/route review and approval record.
- [ ] Compact radio codec, reassembly, CRC, acknowledgement, and mission handoff.
- [ ] Local directional terrain estimates, dynamic overlay, and drift-aware fusion.
- [ ] Local controller integration and narrow-gate/dynamic obstacle acceptance runs.

## Acceptance evidence

Test climb versus cross-slope feasibility on the same hillside, uphill/downhill
asymmetry, narrow cliffs between samples, nodata and out-of-map refusal, blocked
endpoints, footprint clearance, no corner-cutting, and detours around a barrier.
For ingestion, compare tiled results across tile seams and ensure downsampling
cannot average away a hard cliff or unknown cell. Benchmark 100–500 m routes and
100 MB–1 GB inputs separately; a small fixture does not establish that scale.

Keep projected map coordinates distinct from ROS odom and Webots' sign-flipped
simulation frame. Real-map registration and rover stability/traction parameters
must be supplied explicitly before hardware deployment.
