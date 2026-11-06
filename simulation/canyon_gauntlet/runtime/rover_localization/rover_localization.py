#!/usr/bin/env python3

import math
import serial
import pynmea2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data

from std_msgs.msg import Header
from sensor_msgs.msg import NavSatFix, NavSatStatus, Imu
from geometry_msgs.msg import TwistWithCovarianceStamped
from rtcm_msgs.msg import Message


KNOT_TO_MPS = 0.514444


# ----------------------------
# Baked rover localization config
# ----------------------------

GPS_PORT = "/dev/serial/by-id/usb-FTDI_FT230X_Basic_UART_D30HGWX3-if00-port0"
GPS_BAUD = 115200

GPS_FRAME_ID = "gps_right_link"
HEADING_FRAME_ID = "base_link"
VELOCITY_FRAME_ID = "map"

RAW_GPS_TOPIC = "/autonomous/localization/gps/fix"
HEADING_IMU_TOPIC = "/autonomous/localization/gps/heading"
GPS_VEL_TOPIC = "/autonomous/localization/gps/vel"

# Dual-GNSS heading correction.
# Your antenna baseline appears to be left/right, so this stays -90 deg.
HEADING_OFFSET_DEG = -90.0

# Rough covariance values
DEFAULT_POSITION_VARIANCE = 4.0    # m^2
DEFAULT_ALTITUDE_VARIANCE = 16.0   # m^2
HEADING_VARIANCE = 0.01            # rad^2
VELOCITY_VARIANCE = 0.25           # (m/s)^2


def safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_angle_rad(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_to_quaternion(yaw):
    half = 0.5 * yaw
    return {
        "x": 0.0,
        "y": 0.0,
        "z": math.sin(half),
        "w": math.cos(half),
    }


class SimpleNmeaGpsDriver(Node):
    def __init__(self):
        super().__init__("simple_nmea_gps_driver")

        self.port = GPS_PORT
        self.baud = GPS_BAUD

        self.gps_frame_id = GPS_FRAME_ID
        self.heading_frame_id = HEADING_FRAME_ID
        self.velocity_frame_id = VELOCITY_FRAME_ID

        self.heading_offset_rad = math.radians(HEADING_OFFSET_DEG)

        self.default_position_variance = DEFAULT_POSITION_VARIANCE
        self.default_altitude_variance = DEFAULT_ALTITUDE_VARIANCE
        self.heading_variance = HEADING_VARIANCE
        self.velocity_variance = VELOCITY_VARIANCE

        self.fix_pub = self.create_publisher(
            NavSatFix,
            RAW_GPS_TOPIC,
            qos_profile_sensor_data,
        )

        self.heading_pub = self.create_publisher(
            Imu,
            HEADING_IMU_TOPIC,
            qos_profile_sensor_data,
        )

        self.vel_pub = self.create_publisher(
            TwistWithCovarianceStamped,
            GPS_VEL_TOPIC,
            qos_profile_sensor_data,
        )

        self.ser = serial.Serial(self.port, self.baud, timeout=0.01)

        qos_be = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)
        self.create_subscription(Message, "/autonomous/localization/gps/rtcm", self._inject_rtcm, qos_be)

        self.get_logger().info(
            f"Reading NMEA from {self.port}@{self.baud}. "
            f"Publishing {RAW_GPS_TOPIC}, {HEADING_IMU_TOPIC}, {GPS_VEL_TOPIC}."
        )

        self.timer = self.create_timer(0.01, self.read_once)

    def _inject_rtcm(self, msg: Message) -> None:
        try:
            self.ser.write(msg.message)
            self.get_logger().info(f"Wrote RTCM Correction: {msg.message}")
        except serial.SerialException as exc:
            self.get_logger().error(f"RTCM write error: {exc}")

    def read_once(self):
        try:
            raw = self.ser.readline()
            if not raw:
                return

            line = raw.decode("ascii", errors="ignore").strip()
            if not line or not line.startswith("$"):
                return

            try:
                msg = pynmea2.parse(line, check=True)
            except pynmea2.ParseError as exc:
                self.get_logger().warn(
                    f"NMEA parse error: {exc}; line={line[:100]}",
                    throttle_duration_sec=2.0,
                )
                return

            sentence_type = getattr(msg, "sentence_type", "")

            if sentence_type == "GGA":
                self.handle_gga(msg)
            elif sentence_type == "HDT":
                self.handle_hdt(msg)
            elif sentence_type == "RMC":
                self.handle_rmc(msg)

        except serial.SerialException as exc:
            self.get_logger().error(f"Serial error: {exc}")
        except Exception as exc:
            self.get_logger().error(f"Unexpected GPS driver error: {exc}")

    def handle_gga(self, msg):
        """
        GGA gives position fix data:
          latitude, longitude, altitude, fix quality, satellite count, HDOP.
        """
        stamp = self.get_clock().now().to_msg()

        fix_quality = safe_int(getattr(msg, "gps_qual", 0), default=0)

        fix = NavSatFix()
        fix.header = Header(stamp=stamp, frame_id=self.gps_frame_id)

        fix.status.service = NavSatStatus.SERVICE_GPS

        if fix_quality == 0:
            fix.status.status = NavSatStatus.STATUS_NO_FIX
            fix.latitude = float("nan")
            fix.longitude = float("nan")
            fix.altitude = float("nan")
            self.fix_pub.publish(fix)
            return

        # Common NMEA fix qualities:
        # 1 = GPS fix
        # 2 = DGPS fix
        # 4 = RTK fixed
        # 5 = RTK float
        if fix_quality in (2, 4, 5):
            fix.status.status = NavSatStatus.STATUS_GBAS_FIX
        else:
            fix.status.status = NavSatStatus.STATUS_FIX

        fix.latitude = safe_float(getattr(msg, "latitude", None), float("nan"))
        fix.longitude = safe_float(getattr(msg, "longitude", None), float("nan"))
        fix.altitude = safe_float(getattr(msg, "altitude", None), float("nan"))

        hdop = safe_float(getattr(msg, "horizontal_dil", None), None)

        if hdop is not None and hdop > 0.0:
            xy_var = max((hdop * 1.5) ** 2, 0.01)
        else:
            xy_var = self.default_position_variance

        z_var = self.default_altitude_variance

        fix.position_covariance = [
            xy_var, 0.0,    0.0,
            0.0,    xy_var, 0.0,
            0.0,    0.0,    z_var,
        ]
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN

        self.fix_pub.publish(fix)

    def handle_hdt(self, msg):
        """
        HDT gives true heading in degrees clockwise from true north.

        ROS ENU yaw:
          0 rad = east
          +pi/2 rad = north

        NMEA heading:
          0 deg = north
          90 deg = east

        Therefore:
          yaw_enu = radians(90 - heading_deg)
        """
        stamp = self.get_clock().now().to_msg()

        heading_deg = safe_float(getattr(msg, "heading", None), None)
        if heading_deg is None:
            return

        yaw = math.radians(90.0 - heading_deg)
        yaw = normalize_angle_rad(yaw + self.heading_offset_rad)

        q = yaw_to_quaternion(yaw)

        imu = Imu()
        imu.header = Header(stamp=stamp, frame_id=self.heading_frame_id)

        imu.orientation.x = q["x"]
        imu.orientation.y = q["y"]
        imu.orientation.z = q["z"]
        imu.orientation.w = q["w"]

        # Only yaw is trusted from the dual-GNSS baseline.
        # Roll and pitch are not measured, so give them huge covariance.
        imu.orientation_covariance = [
            999999.0, 0.0,      0.0,
            0.0,      999999.0, 0.0,
            0.0,      0.0,      self.heading_variance,
        ]

        # This virtual IMU does not provide angular velocity.
        imu.angular_velocity_covariance[0] = -1.0

        # This virtual IMU does not provide linear acceleration.
        imu.linear_acceleration_covariance[0] = -1.0

        self.heading_pub.publish(imu)

    def handle_rmc(self, msg):
        """
        RMC gives speed over ground and course over ground.

        Speed is in knots.
        Course is degrees clockwise from true north.

        This publishes velocity in an ENU/world-like frame:
          +X = east
          +Y = north
        """
        stamp = self.get_clock().now().to_msg()

        status = getattr(msg, "status", "")
        if status != "A":
            return

        speed_knots = safe_float(getattr(msg, "spd_over_grnd", None), None)
        course_deg = safe_float(getattr(msg, "true_course", None), None)

        if speed_knots is None or course_deg is None:
            return

        speed_mps = speed_knots * KNOT_TO_MPS
        course_rad = math.radians(course_deg)

        # Convert NMEA course to ENU velocity.
        vx_east = speed_mps * math.sin(course_rad)
        vy_north = speed_mps * math.cos(course_rad)

        vel = TwistWithCovarianceStamped()
        vel.header = Header(stamp=stamp, frame_id=self.velocity_frame_id)

        vel.twist.twist.linear.x = vx_east
        vel.twist.twist.linear.y = vy_north
        vel.twist.twist.linear.z = 0.0

        vel.twist.covariance = [
            self.velocity_variance, 0.0,                    0.0,      0.0,      0.0,      0.0,
            0.0,                    self.velocity_variance, 0.0,      0.0,      0.0,      0.0,
            0.0,                    0.0,                    999999.0, 0.0,      0.0,      0.0,
            0.0,                    0.0,                    0.0,      999999.0, 0.0,      0.0,
            0.0,                    0.0,                    0.0,      0.0,      999999.0, 0.0,
            0.0,                    0.0,                    0.0,      0.0,      0.0,      999999.0,
        ]

        self.vel_pub.publish(vel)

    def destroy_node(self):
        if hasattr(self, "ser") and self.ser and self.ser.is_open:
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SimpleNmeaGpsDriver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()