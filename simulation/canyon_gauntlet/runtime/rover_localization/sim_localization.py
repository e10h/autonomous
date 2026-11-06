import math

import rclpy
import tf2_ros
from geometry_msgs.msg import TransformStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix


LAT0 = 32.8822983464544
LON0 = -117.23971214571563
EARTH_RADIUS = 6378137.0


class SimLocalization(Node):
    """Translate Webots GPS/heading into the competition odom + TF interface."""

    def __init__(self):
        super().__init__("sim_localization")
        self._tf = tf2_ros.TransformBroadcaster(self)
        self._odom_pub = self.create_publisher(
            Odometry, "/autonomous/localization/odometry/local", 10)
        self._x = self._y = self._yaw = self._vx = self._vy = 0.0

        self.create_subscription(
            NavSatFix, "/autonomous/localization/gps/fix", self._on_position, 10)
        self.create_subscription(
            Imu, "/autonomous/localization/gps/heading", self._on_heading, 10)
        self.create_subscription(
            TwistStamped, "/autonomous/localization/gps/velocity", self._on_velocity, 10)
        self.create_timer(0.1, self._publish)

    def _on_position(self, msg):
        dlat = math.radians(msg.latitude - LAT0)
        dlon = math.radians(msg.longitude - LON0)
        self._y = dlat * EARTH_RADIUS
        self._x = dlon * EARTH_RADIUS * math.cos(math.radians(LAT0))

    def _on_heading(self, msg):
        q = msg.orientation
        self._yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def _on_velocity(self, msg):
        self._vx = msg.twist.linear.x
        self._vy = msg.twist.linear.y

    def _publish(self):
        now = self.get_clock().now().to_msg()
        cy, sy = math.cos(self._yaw / 2.0), math.sin(self._yaw / 2.0)

        transform = TransformStamped()
        transform.header.stamp = now
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = self._x
        transform.transform.translation.y = self._y
        transform.transform.rotation.w = cy
        transform.transform.rotation.z = sy
        self._tf.sendTransform(transform)

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = self._x
        odom.pose.pose.position.y = self._y
        odom.pose.pose.orientation.w = cy
        odom.pose.pose.orientation.z = sy
        odom.twist.twist.linear.x = self._vx
        odom.twist.twist.linear.y = self._vy
        self._odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = SimLocalization()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
