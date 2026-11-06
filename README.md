# Autonomous State Machine

This ROS 2 package contains the main state machine that orchestrates the autonomous behavior of the rover. It handles global path planning, obstacle tracking, obstacle avoidance, AR tag detection/alignment, and smooth transitions between various operational states.

## Configuration

High-level parameters for the node, including the global map manager and obstacle avoidance, can be configured via `autonomous_state_machine/config.py` (no rebuild required for Python changes, but restart the node).

### Key Configurable Parameters
- **`PERSISTENT_GRID_SIZE`**: Total continuous size of the square map canvas in meters.
- **`OVERLAP_SIZE` / `SHIFT_SIZE`**: Manage the sliding window behavior when the rover approaches the edge of the persistent grid, keeping the rover centered.
- **`GRID_CELL_SIZE`**: The physical resolution of a single grid cell in meters (e.g., 0.10m).
- **`WRITE_ASCII_MAP`** / **`ASCII_MAP_PATH`**: Set `WRITE_ASCII_MAP = True` to write an ASCII representation of the global grid and path to `ASCII_MAP_PATH` (default `/tmp/rover_map.txt`) once per second (useful for debugging).
Use `watch -n 0.5 cat /tmp/rover_map.txt` to see this map.
- **`BACKUP_DURATION`**: Time in seconds the rover back-drives when encountering an immediate obstacle before executing an avoidance maneuver (the cooldown for recalculating paths dynamically scales with this).
- **`SIMULATION`**: Flag to toggle simulation-specific behavior (e.g., automatically turning off obstacle degradation).
- **`CONFIDENCE_*`**: Settings related to probabilistic obstacle tracking (thresholds, increase amounts for hits, decay amounts for free space).

## Coordinate Frames and TF Tree

### Expected TF Tree

```
map
 └── odom              (static identity in simulation; EKF output on hardware)
      └── base_link    (robot body; published by sim_localization or the real EKF)
           └── oakd_depth_camera   (static offset; published by rover_localization.launch.py)
```

The `map → odom` transform is a static identity in simulation (published by `rover_localization.launch.py`). On hardware, it is produced by the global EKF (`robot_localization`) via a Navsat transform node. The `odom → base_link` transform is published by `sim_localization.py` in simulation, or by the local EKF on hardware.

### Frame Usage by Component

| Data | Frame | Source / Notes |
|------|-------|----------------|
| `local_odom_position` | **odom** | `/autonomous/localization/odometry/local`; used for all navigation and map work |
| `global_map` origin + updates | **odom** | `GlobalMapManager` is initialized from `local_odom_position` on first pointcloud |
| Theta\* start + goal | **odom** | `get_path()` inputs are `local_odom_position` and `goal_odom` |
| OA waypoints (Theta\* output) | **odom** | Passed to `AlignAndMove`; compared with `local_odom_position` in `align.py` |
| `goal_odom` | **odom** | GPS goal transformed via TF2 `map → odom`; stored in `SharedData.goal_odom` |
| `goal_map_xy` | **map** (GPS/NED) | Output of the `robot_localization` `/fromLL` service |
| `cur_position` | **odom** | Same value as `local_odom_position` (set by the same odom callback), later overwritten with an odom-frame average of `position_history` by `ValidatingState`; used as the spiral search center in `forward_spiral_state.py`/`backward_spiral_state.py` and as a readiness check in `replan_to_goal`/`on_receive_new_auton_input` |
| ASCII map robot position | **odom** | `write_ascii_map` called with `local_odom_position` so robot marker is correct |
| Every-tick goal check (`_tick`) | **odom** | `dist(local_odom_position, goal_odom) < 2 m` → `validating_state` |
| OA completion goal check | **odom** | Same distance check in `NewObstacleAvoidanceControllerState` |
| PP steering (`local_odom_position`) | **odom** | `new_waypoints_odom` (TF2-converted) passed to PP; `drive_state.py` uses `local_odom_position` as `cur_pose` |
| `new_waypoints_odom` | **odom** | `[local_odom_position]` + TF2 map→odom conversion of `new_waypoints` (goal point(s) only); set by `_convert_waypoints_to_odom()` in `_tick` |

### Goal Transform Pipeline

```
GPS lat/lon
   │  /fromLL service call            (on_receive_new_auton_input in utils.py)
   ▼
goal_map_xy  (map / GPS-NED frame)
   │  TF2 lookup_transform('odom','map')  (_transform_goal_to_odom in autonomous_state_machine.py)
   │  triggered once per waypoint update via goal_dirty flag
   ▼
goal_odom  (odom frame)
   │  used by: _tick() goal check, _on_pointcloud() Theta*, OA completion check,
   │           _rebuild_main_controller_from_current_path()
```

In simulation `map ≡ odom` (identity static TF), so `goal_odom == goal_map_xy`. On hardware the EKF may introduce a small drift offset between the two frames over time.

### Subscribing to Odom

The node subscribes to `LOCAL_ODOM_TOPIC = "/autonomous/localization/odometry/local"` (defined in `shared_data.py`). This topic must be published before the rover can navigate — confirm with:

```bash
ros2 topic echo /autonomous/localization/odometry/local
```

---

## Heartbeat / Comms-Loss Recovery

The state machine monitors a heartbeat topic published by the basestation. If the signal is absent for longer than the configured timeout, the rover immediately stops, sets its LED to red, and autonomously navigates back to the last position where a heartbeat was confirmed.

### How it works

1. The basestation publishes a `std_msgs/msg/Bool` message (value `true`) at ~1 Hz on `/autonomous/heartbeat` (`SharedData.HEARTBEAT_TOPIC_NAME`).
2. Each received message updates `SharedData.last_heartbeat_time` and `SharedData.last_safe_position` (current odom position).
3. On the first odom message, `last_safe_position` is seeded to the rover's starting odom position as a fallback.
4. Every tick `_tick()` checks: if at least one heartbeat was ever received **and** `time() - last_heartbeat_time > HEARTBEAT_TIMEOUT`, the rover enters `HeartbeatState` and begins returning to `last_safe_position`.
5. `heartbeat_recovery_active` is set to `True` to prevent re-triggering while the rover is already returning. It is cleared when the operator sends a new `AutonInput` (implying comms have been restored).
6. Obstacle avoidance remains fully active during the return journey.

### Config flags (`config.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `HEARTBEAT_ENABLED` | `True` | Set to `False` when running without a basestation (e.g., local sim testing with no heartbeat publisher). Disables the subscriber and the timeout check entirely. |
| `HEARTBEAT_TIMEOUT` | `5.0` | Seconds of silence before the rover declares comms lost and begins returning. |

The topic name is `SharedData.HEARTBEAT_TOPIC_NAME = "/autonomous/heartbeat"` (follows the same convention as all other topic names in this package).

### Simulation testing

**1. Launch the node normally** (with `HEARTBEAT_ENABLED = True` in `config.py`).

**2. Start publishing a heartbeat** at 1 Hz to simulate an active basestation:
```bash
ros2 topic pub /autonomous/heartbeat std_msgs/msg/Bool "data: true" -r 1
```

**3. Verify the node is receiving it:**
```bash
ros2 topic echo /autonomous/heartbeat
```

**4. Simulate comms loss** by stopping the publisher (Ctrl-C in that terminal). After `HEARTBEAT_TIMEOUT` seconds (default 5 s) you should see:
```
[WARN] [autonomous_state_machine]: [Heartbeat] Signal lost — forcing return to last safe position
[WARN] [HeartbeatState]: ENTER — heartbeat lost, returning to last safe position
```
and the `/set_state` topic will show the transition:
```bash
ros2 topic echo /set_state
```

**5. Confirm the LED command** goes red (`[255, 0, 0]`) on `/led`:
```bash
ros2 topic echo /led
```

**6. Restore comms and send a new mission** to re-engage normal operation:
```bash
# restart the heartbeat publisher
ros2 topic pub /autonomous/heartbeat std_msgs/msg/Bool "data: true" -r 1
# then send a new AutonInput — this clears heartbeat_recovery_active
```

---

## Known Issues

- **Field of View Degradation Issue**: The local occupancy grid currently provides "free space" grid readings that are actually outside the physical field of view of the sensors. When integrated into the global map, this causes the probability mapping to falsely degrade (clear) tracked obstacle cells that it cannot actually see. 


## Reproducible Canyon Gauntlet demo

See [simulation/canyon_gauntlet/README.md](simulation/canyon_gauntlet/README.md) for the bundled terrain, anisotropic slope map, corrected rover physics, and launcher.

## Global LiDAR macro-route development

See [the implementation guide](docs/GLOBAL_PLANNER.md) and
[the end-to-end roadmap](docs/LIDAR_NAVIGATION_ROADMAP.md) for tiled Utah DEM
preprocessing, direction-aware Theta*, and unreviewed sparse route export.
