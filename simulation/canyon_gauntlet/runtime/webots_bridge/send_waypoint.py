"""Send a local offset through the competition GPS waypoint interface."""

import argparse
import math
import threading

import rclpy
from custom_msg_types.msg import AutonInput
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix


EARTH_RADIUS_M = 6_378_137.0


class WaypointSender(Node):
    def __init__(self, east_m: float, north_m: float, speed: float, object_id: int):
        super().__init__("competition_waypoint_sender")
        self.east_m = east_m
        self.north_m = north_m
        self.speed = speed
        self.object_id = object_id
        self.done = threading.Event()
        self.publisher = self.create_publisher(AutonInput, "/autonomous/input", 10)
        self.subscription = self.create_subscription(
            NavSatFix,
            "/autonomous/localization/gps/fix",
            self.on_fix,
            10,
        )

    def on_fix(self, fix: NavSatFix) -> None:
        if self.done.is_set() or not math.isfinite(fix.latitude) or not math.isfinite(fix.longitude):
            return

        latitude = fix.latitude + math.degrees(self.north_m / EARTH_RADIUS_M)
        longitude = fix.longitude + math.degrees(
            self.east_m / (EARTH_RADIUS_M * math.cos(math.radians(fix.latitude)))
        )

        msg = AutonInput()
        msg.latitude = latitude
        msg.longitude = longitude
        msg.speed = self.speed
        msg.object_id = self.object_id
        self.publisher.publish(msg)
        self.get_logger().info(
            f"Sent competition waypoint: east={self.east_m:.2f} m, "
            f"north={self.north_m:.2f} m -> ({latitude:.8f}, {longitude:.8f})"
        )
        self.done.set()


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Send a waypoint relative to the rover through /autonomous/input"
    )
    parser.add_argument("--east", type=float, required=True, help="meters east of current GPS fix")
    parser.add_argument("--north", type=float, required=True, help="meters north of current GPS fix")
    parser.add_argument("--speed", type=float, default=0.6)
    parser.add_argument("--object-id", type=int, default=AutonInput.NONE, choices=range(0, 9))
    parsed, ros_args = parser.parse_known_args(args=args)

    rclpy.init(args=ros_args)
    node = WaypointSender(parsed.east, parsed.north, parsed.speed, parsed.object_id)
    while rclpy.ok() and not node.done.wait(0.05):
        rclpy.spin_once(node, timeout_sec=0.05)
    # Give DDS a moment to deliver the one-shot message before destruction.
    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()
