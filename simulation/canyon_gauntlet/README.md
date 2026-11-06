# Canyon Gauntlet: anisotropic terrain navigation

This bundle preserves the Canyon Gauntlet demonstration: a box-canyon blockage,
a narrow ridge, a tilted boulder gate, and a goal basin. The planner uses terrain
gradients to distinguish climb, descent, and cross-slope costs, rather than
assuming the terrain has the same cost in every direction.

## Run

Prerequisites:

- Webots R2025a on the host (macOS was tested).
- Docker running (on this Mac: `colima start`).
- An existing ROS 2 Jazzy simulation container with a built Yonder workspace,
  `custom_msg_types`, `robot_localization`, `tf2_ros`, `webots_ros2_driver`, NumPy,
  and SciPy. This repository is an autonomy package, not a complete ROS distro or
  container image. The tested container is `docker_environment-rover-1` and its
  workspace is `/home/yonder/ros2_ws`.

Close any previous Webots simulation, then run from the repository root:

```sh
python3 scripts/run_canyon_gauntlet.py --restart-container
```

`--restart-container` restarts the selected simulation container, stopping all its
processes. Omit it when using a clean container. Override `--container`,
`--workspace` (path inside the container), or `--webots` as needed. On Linux,
`WEBOTS_CONTROLLER_URL` may need to be adapted to your host networking in the
launch file or container configuration.

The launcher stages the checked-out planner and bundled simulation runtime under
`/tmp/autonomous-canyon-demo` **inside the container**, starts the bundled world,
waits for navigation and GPS datum calibration to initialize, and sends the relative GPS goal
`east=-3.6 m, north=9 m` at `0.6 m/s`. It waits up to 120 seconds for goal
confirmation and leaves Webots and ROS running afterward. A timeout raises an
error; it is not reported as a success. The launcher also verifies the converted
odometry goal matches the requested route, rejecting an uncalibrated origin goal.

For an already open, freshly reset copy of this bundled world:

```sh
python3 scripts/run_canyon_gauntlet.py --no-webots --restart-container
```

Use `--start-only` to initialize without dispatching a goal. Runtime logs:

```sh
docker exec docker_environment-rover-1 cat /tmp/canyon-gauntlet.log
```

The ROS workspace source and installed files are not overwritten. No machine
specific source paths or unpublished `/tmp` assets are required.

## What is included

- `worlds/terrain_world.wbt`, its traversability texture, and the matching
  `static_slope_costmap.npy` (all generated together).
- `real_usgs.tif`: cached elevation input used by the terrain builder; terrain
  hazards are synthesized from this input, not an unmodified real-world course.
- `protos/rover_terrain.proto` and only the mesh files it references.
- `runtime/webots_bridge` and `runtime/rover_localization`: simulation support
  code taken from the working local ROS workspace, including clock, localization,
  waypoint submission, and route visualization.
- Local launch files, so the simulation uses the bundled Python code and config
  instead of whichever bridge or localization version is installed elsewhere.

## Physics and rendering fixes

The original CAD mesh collision groups were unstable against the detailed
terrain. The bundled rover uses six cylindrical wheel collision shapes and five
box collision shapes derived from the original mesh bounds; visual meshes,
joints, and inertia parameters remain present. The sensor mount is a fixed
`Pose` under the robot, so its gyro and accelerometer use robot physics. The
world uses an 8 ms physics step and at most eight contacts per collision pair.
The route marker omits the `solid FALSE` field rejected by the tested Webots
runtime.

These changes alter the simulation collision approximation. This demo is a
navigation example, not a validated model of hardware dynamics or a guarantee
of collision-free behavior on arbitrary terrain. Depth processing is disabled
for this static terrain example; the matching precomputed slope map is enabled.

## Verification

The five anisotropic planner regression tests passed in the ROS container.
The packaged launcher was tested with a reset copy of the bundled world and
completed the intended nonzero goal with a reported 0.60 m error. No physics or
marker warnings recurred during that run. See `validation.log` and
`validation.json`. Goal
confirmation alone does not establish zero obstacle contacts.

The source terrain builder is `scripts/build_canyon_gauntlet_sim.py`. If changing
the terrain, regenerate the world, texture, and costmap together; do not mix a
map from another demo with this world.
