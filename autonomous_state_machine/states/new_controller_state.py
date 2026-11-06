from autonomous_state_machine.states.state_class import State
from autonomous_state_machine.pp import PurePursuit
from autonomous_state_machine.align import AlignAndMove
import autonomous_state_machine.config as config
import math
import numpy as np
from geometry_msgs.msg import Twist

class NewControllerState(State):
    def __init__(self, shared_data, state_instance):
        super().__init__()
        self.shared_data = shared_data
        self.state_instance = state_instance

    def on_enter(self):
        with self.shared_data.mutex:
            # Retire the previous controller before considering a new mission.
            # A rejected route must not leave stale motion commands running.
            self.shared_data.controller = None
            self.shared_data.navigation_blocked = True
            self.shared_data.auton_publisher.publish(Twist())
            waypoints = self.shared_data.new_waypoints_odom
            if len(waypoints) < 2:
                self.log("odom waypoints not ready — cannot init controller", logtype="error")
                self.shared_data.new_waypoints_flag = False
                return

            # Check if global terrain/rock obstacles require global Theta* path generation
            gm = getattr(self.shared_data, "global_map", None)
            if gm is not None and getattr(gm, "static_slope_data", None) is not None:
                start_pt = waypoints[0]
                goal_pt = waypoints[-1]
                planned = gm.get_path(
                    start_pt[0], start_pt[1],
                    goal_pt[0], goal_pt[1],
                    robot_radius_m=config.OA_ROBOT_RADIUS_M
                )
                if len(planned) < 2:
                    self.log("No terrain-safe route at configured rover clearance — holding stopped", logtype="warn")
                    self.shared_data.new_waypoints_flag = False
                    return
                self.log(f"[GlobalMap] Injected directional Theta* route ({len(planned)} waypoints)")
                waypoints = planned

            # Publish planned path for Webots and UI visualization before movement starts
            if getattr(self.shared_data, "oa_path_publisher", None) is not None and len(waypoints) >= 2:
                try:
                    from nav_msgs.msg import Path
                    from geometry_msgs.msg import PoseStamped
                    path_msg = Path()
                    path_msg.header.frame_id = 'odom'
                    for px, py in waypoints:
                        ps = PoseStamped()
                        ps.header.frame_id = 'odom'
                        ps.pose.position.x = float(px)
                        ps.pose.position.y = float(py)
                        path_msg.poses.append(ps)
                    self.shared_data.oa_path_publisher.publish(path_msg)
                    self.log(f"[Viz] Broadcasted planned route ({len(waypoints)} waypoints) to /autonomous/viz/oa_path")
                except Exception as e:
                    self.log(f"[Viz] Could not publish planned path: {e}", logtype="warn")

            if config.CONTROLLER_TYPE == "align":
                self.shared_data.controller = AlignAndMove(waypoints, logger=self.shared_data.logger)
                self.shared_data.controller.start_next_target()
                self.log(f"ENTER — AlignAndMove initialized, {len(waypoints)} odom waypoints")
            else:
                diff = np.array(waypoints[-1]) - np.array(waypoints[-2])
                next_orientation = math.atan2(diff[1], diff[0])
                self.shared_data.controller = PurePursuit(
                    waypoints,
                    self.shared_data.cur_heading,
                    next_orientation,
                    tolerance=config.GOAL_REACHED_RADIUS_M
                )
                self.log(f"ENTER — PurePursuit initialized, {len(waypoints)} odom waypoints, goal heading {math.degrees(next_orientation):.1f}°")
            self.shared_data.new_waypoints_flag = False
            self.shared_data.navigation_blocked = False

    def step(self):
        with self.shared_data.mutex:
            if self.shared_data.controller is None:
                return self.state_instance.new_controller_state
            self.log(f"{type(self.shared_data.controller).__name__} ready — → DriveState")
            return self.state_instance.drive_state
