import array
import math
import json
import os
from collections.abc import Sequence

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from controller import GPS, Accelerometer, Camera, Compass, Gyro, Motor, RangeFinder, Supervisor
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Path
from rosgraph_msgs.msg import Clock
from rclpy.node import Node
from sensor_msgs.msg import (
    CameraInfo, CompressedImage, Image, Imu, NavSatFix, NavSatStatus, PointCloud2,
)
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64, Float64MultiArray, Header, MultiArrayDimension, String

from . import config
from . import transforms
from .markers import MarkerDrawer
from .simulation_speed import SimulationSpeed


def _pairs(data: Sequence[float]) -> list[tuple[float, float]]:
    return list(zip(data[0::2], data[1::2]))


class WebotsBridge(Node):
    def __init__(self, robot: Supervisor) -> None:
        super().__init__('webots_bridge')
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.simulation_speed = SimulationSpeed(robot)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.create_subscription(Float64, '/simulation/speed', self._on_simulation_speed, 10)
        self.last_depth_time = -1.0
        self.markers = MarkerDrawer(robot)

        self.left_wheels: list[Motor] = [
            robot.getDevice(f'left_{p}_wheel') for p in ('front', 'middle', 'back')]
        self.right_wheels: list[Motor] = [
            robot.getDevice(f'right_{p}_wheel') for p in ('front', 'middle', 'back')]
        for wheel in self.left_wheels + self.right_wheels:
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0.0)
        self.left_speed = 0.0
        self.right_speed = 0.0
        self.rocks = None
        self.rock_contact_steps = 0
        self.last_contact_report = -1.0
        self.contact_pub = self.create_publisher(String, '/simulation/contacts', 10)

        self.gps: GPS = robot.getDevice('GPS')
        self.compass: Compass = robot.getDevice('compass')
        self.accelerometer: Accelerometer = robot.getDevice('accelerometer')
        self.gyro: Gyro = robot.getDevice('gyro')
        self.depth_cam: RangeFinder = robot.getDevice('oakd_depth')
        self.depth_node = None
        # Optional: not every rover proto has a front camera.
        self.front_camera: Camera | None = robot.getDevice('front_camera')
        if self.front_camera is None:
            self.get_logger().warn('front_camera device not found; skipping camera publishing')
        self.depth_enabled = os.getenv("YONDER_ENABLE_DEPTH", "1") != "0"
        for sensor in (self.gps, self.compass, self.accelerometer, self.gyro,
                       self.front_camera):
            if sensor is not None:
                sensor.enable(self.timestep)
        if self.depth_cam is not None and self.depth_enabled:
            self.depth_cam.enable(self.timestep)

        self.depth_width = self.depth_cam.getWidth() if self.depth_cam else 640
        self.depth_height = self.depth_cam.getHeight() if self.depth_cam else 480
        self.focal_px = self.depth_width / (2 * math.tan(self.depth_cam.getFov() / 2)) if self.depth_cam else 500.0
        self.cx = self.depth_width / 2
        self.cy = self.depth_height / 2

        self._reset_position_done = False

        # Set from the first GPS reading so waypoints line up with the state machine's origin.
        self.origin_latlon: tuple[float, float] | None = None
        self.origin_world: tuple[float, float] | None = None

        self.create_subscription(Twist, '/autonomous/drivetrain/command', self._on_drive, 10)
        self.create_subscription(String, '/set_state', self._on_state, 10)
        self.create_subscription(
            Float64MultiArray, '/transformed_waypoints', self._on_transformed_waypoints, 10)
        self.create_subscription(
            Float64MultiArray, '/obstacle_avoidance/path', self._on_obstacle_avoidance_path, 10)
        self.create_subscription(Path, '/autonomous/viz/oa_path', self._on_competition_oa_path, 10)
        self.create_subscription(Float64MultiArray, '/auton/transformed/goal', self._on_goal, 10)

        self.gps_pub = self.create_publisher(NavSatFix, '/autonomous/localization/gps/fix', 10)
        self.heading_pub = self.create_publisher(Imu, '/autonomous/localization/gps/heading', 10)
        self.velocity_pub = self.create_publisher(
            TwistStamped, '/autonomous/localization/gps/velocity', 10)
        # The competition autonomy stack consumes the shorter /vel name.
        self.autonomy_velocity_pub = self.create_publisher(
            TwistStamped, '/autonomous/localization/gps/vel', 10)
        self.depth_image_pub = self.create_publisher(Image, config.OAKD_DEPTH_IMAGE_TOPIC, 10)
        self.depth_camera_info_pub = self.create_publisher(
            CameraInfo, config.OAKD_DEPTH_CAMERA_INFO_TOPIC, 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, config.OAKD_POINTCLOUD_TOPIC, 10)
        self.occupancy_grid_pub = self.create_publisher(
            Float64MultiArray, '/obstacle_avoidance/occupancy_grid', 10)
        self.front_camera_pub = self.create_publisher(
            CompressedImage, '/cameras/ground/image_raw/compressed', 10)

    def _on_drive(self, msg: Twist) -> None:
        half_width = config.TRACK_WIDTH_M / 2.0
        self.left_speed = msg.linear.x - msg.angular.z * half_width
        self.right_speed = msg.linear.x + msg.angular.z * half_width

    def _on_simulation_speed(self, msg: Float64) -> None:
        if self.simulation_speed.set_speed(msg.data):
            self.get_logger().info(f'Simulation speed target: {msg.data:g}x')

    def _on_state(self, msg: String) -> None:
        if msg.data == 'goal_reached':
            self.left_speed = 0.0
            self.right_speed = 0.0

    def _on_transformed_waypoints(self, msg: Float64MultiArray) -> None:
        latlons = _pairs(msg.data)
        if not latlons:
            return

        origin = self.origin_latlon or (config.LAT0, config.LON0)
        waypoints = []
        for lat, lon in latlons:
            x, y = transforms.latlon_to_xy(lat, lon, *origin)
            if config.TRANSFORMED_WAYPOINT_ROTATION_DEG != 0:
                x, y = transforms.rotate_xy(x, y, config.TRANSFORMED_WAYPOINT_ROTATION_DEG)
            if self.origin_world is not None:
                x += self.origin_world[0]
                y += self.origin_world[1]
            waypoints.append((x, y))

        self.get_logger().debug(f'transformed waypoints: {waypoints}')
        self.markers.draw_absolute_path(waypoints)

    def _on_obstacle_avoidance_path(self, msg: Float64MultiArray) -> None:
        waypoints = _pairs(msg.data)
        if not waypoints:
            return
        self.get_logger().debug(f'obstacle avoidance path: {waypoints}')
        self.markers.draw_relative_path(waypoints)

    def _on_competition_oa_path(self, msg: Path) -> None:
        """Draw the Theta* path emitted by the competition autonomy stack."""
        # Sim localization's odom axes are the inverse of Webots world X/Y because
        # the bridge converts world coordinates to the competition GPS convention.
        waypoints = [(-pose.pose.position.x, -pose.pose.position.y) for pose in msg.poses]
        if not waypoints:
            return
        self.get_logger().info(f'competition obstacle avoidance path: {len(waypoints)} points')
        self.markers.draw_oa_path(waypoints)

    def _on_goal(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 2:
            self.markers.draw_goal(msg.data[0], msg.data[1])

    def step(self) -> None:
        for wheel in self.left_wheels:
            wheel.setVelocity(-self.left_speed)
        for wheel in self.right_wheels:
            wheel.setVelocity(self.right_speed)

        nanoseconds = round(self.robot.getTime() * 1e9)
        stamp = Time(sec=nanoseconds // 1_000_000_000,
                     nanosec=nanoseconds % 1_000_000_000)
        self.clock_pub.publish(Clock(clock=stamp))
        self._publish_gps(stamp)
        self._publish_heading(stamp)
        self._publish_imu(stamp)
        # Keep depth at ~10 Hz of simulation time at either playback speed.
        if self.depth_enabled and self.depth_cam is not None and self.robot.getTime() - self.last_depth_time >= 0.096 - 1e-6:
            self._publish_depth(stamp)
        self._publish_front_camera(stamp)
        self._publish_contacts()

        # Export clean Webots 3D view directly via Supervisor
        record_dir = os.getenv('YONDER_RECORD_FRAMES_DIR')
        rec_interval = float(os.getenv('YONDER_RECORD_INTERVAL', '0.25'))
        if record_dir and (not hasattr(self, '_last_record_time') or (self.robot.getTime() - self._last_record_time >= rec_interval)):
            self._last_record_time = self.robot.getTime()
            if not hasattr(self, '_rec_frame_id'):
                self._rec_frame_id = 0
            self._rec_frame_id += 1
            try:
                os.makedirs(record_dir, exist_ok=True)
                self.robot.exportImage(f'{record_dir}/frame_{self._rec_frame_id:04d}.png', 95)
            except Exception:
                pass

        if not hasattr(self, '_last_export_img_time') or (self.robot.getTime() - self._last_export_img_time >= 0.15):
            self._last_export_img_time = self.robot.getTime()
            try:
                self.robot.exportImage('/tmp/webots_live_view.png', 95)
            except Exception:
                pass

    def _publish_contacts(self):
        if self.rocks is None:
            self.rocks = []
            try:
                root = self.robot.getRoot()
                if root:
                    children = root.getField('children')
                    if children:
                        self.rocks = [children.getMFNode(i) for i in range(children.getCount())
                                      if children.getMFNode(i) and children.getMFNode(i).getTypeName() == 'Rock']
                        for solid in [self.robot.getSelf(), *self.rocks]:
                            if solid:
                                solid.enableContactPointsTracking(self.timestep, True)
            except Exception:
                pass

        # Compare actual physics contact locations on both solids. This excludes
        # ordinary wheel/floor contacts and detects any rover/rock contact pair.
        rover_node = self.robot.getSelf()
        rover_points = [p.point for p in rover_node.getContactPoints(True)] if rover_node else []
        contacts = []
        for rock in self.rocks:
            for contact in rock.getContactPoints(True):
                point = contact.point
                if rover_points and any(
                        sum((a - b) ** 2 for a, b in zip(point, other)) < 0.005
                        for other in rover_points):
                    contacts.append(list(point))
        if contacts:
            self.rock_contact_steps += 1
        now = self.robot.getTime()
        if now - self.last_contact_report >= 0.2:
            self.last_contact_report = now
            self.contact_pub.publish(String(data=json.dumps({
                'simulation_time': now, 'rock_contact_steps': self.rock_contact_steps,
                'rock_count': len(self.rocks), 'contacts': contacts,
                'speed_target': self.simulation_speed.requested,
                'speed_actual': self.simulation_speed.actual})))

    def _publish_gps(self, stamp: Time) -> None:
        x, y, z = self.gps.getValues()
        lat, lon = transforms.xy_to_latlon(x, y)

        if self.origin_latlon is None:
            self.origin_latlon = (lat, lon)
            self.origin_world = (x, y)
            self.get_logger().info(
                f'origin set; GPS height {z:.3f} m, CAMERA_HEIGHT_M={config.CAMERA_HEIGHT_M}')

        msg = NavSatFix()
        msg.header = Header(stamp=stamp, frame_id='gps_link')
        msg.status.status = NavSatStatus.STATUS_FIX
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = z
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        self.gps_pub.publish(msg)

    def _publish_heading(self, stamp: Time) -> None:
        # ENU yaw: counter-clockwise from +X (east), per REP-105.
        north = self.compass.getValues()
        half_yaw = (math.pi - math.atan2(north[0], north[1])) / 2.0

        msg = Imu()
        msg.header = Header(stamp=stamp, frame_id=config.BASE_FRAME_ID)
        msg.orientation.z = math.sin(half_yaw)
        msg.orientation.w = math.cos(half_yaw)
        msg.angular_velocity_covariance[0] = -1.0
        msg.linear_acceleration_covariance[0] = -1.0
        self.heading_pub.publish(msg)

    def _publish_imu(self, stamp: Time) -> None:
        # Carries raw accelerometer/gyro readings; consumers expect this layout.
        ax, ay, az = self.accelerometer.getValues()
        gx, gy, gz = self.gyro.getValues()

        msg = TwistStamped()
        msg.header = Header(stamp=stamp, frame_id=config.BASE_FRAME_ID)
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = ax, ay, az - 9.81
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = gx, gy, gz
        self.velocity_pub.publish(msg)
        self.autonomy_velocity_pub.publish(msg)

    def _publish_depth(self, stamp: Time) -> None:
        if not self.depth_enabled or self.depth_cam is None:
            return
        # 'buffer' returns the raw pointer, avoiding a per-pixel Python list. Copied since Webots reuses it.
        buffer = self.depth_cam.getRangeImage(data_type='buffer')
        depth = np.ctypeslib.as_array(buffer, shape=(self.depth_height * self.depth_width,))
        depth = np.nan_to_num(
            depth.reshape((self.depth_height, self.depth_width)), nan=0.0, posinf=0.0, neginf=0.0)
        header = Header(stamp=stamp, frame_id=config.CAMERA_FRAME_ID)

        if config.ENABLE_DEPTH_IMAGE_PUBLISH:
            self._publish_depth_image(header, depth)
        if config.ENABLE_POINTCLOUD_PUBLISH:
            self._publish_pointcloud(header, depth)
        if config.ENABLE_OCCUPANCY_GRID_PUBLISH:
            self._publish_occupancy_grid(depth)

        # Export colorized depth visualization for mission HUD and inspection
        if not hasattr(self, '_last_depth_export_time') or (self.robot.getTime() - self._last_depth_export_time >= 0.5):
            self._last_depth_export_time = self.robot.getTime()
            try:
                d_vis = np.clip(depth, 0.1, 10.0)
                norm_d = ((10.0 - d_vis) / 9.9 * 255.0).astype(np.uint8)
                colored = cv2.applyColorMap(norm_d, cv2.COLORMAP_TURBO)
                cv2.putText(colored, "OAK-D Real-Time Depth Map [0.1m - 10.0m]", (14, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.imwrite('/tmp/latest_depth_view.png', colored)
            except Exception:
                pass

    def _publish_depth_image(self, header: Header, depth: np.ndarray) -> None:
        image = Image()
        image.header = header
        image.height = self.depth_height
        image.width = self.depth_width
        image.encoding = '32FC1'
        image.step = self.depth_width * 4
        image.data = array.array('B', depth.astype(np.float32).tobytes())
        self.depth_image_pub.publish(image)

        f, cx, cy = self.focal_px, self.cx, self.cy
        info = CameraInfo()
        info.header = header
        info.height = self.depth_height
        info.width = self.depth_width
        info.distortion_model = 'plumb_bob'
        info.d = [0.0] * 5
        info.k = [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [f, 0.0, cx, 0.0, 0.0, f, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.depth_camera_info_pub.publish(info)

    def _publish_pointcloud(self, header: Header, depth: np.ndarray) -> None:
        s = config.POINTCLOUD_STRIDE
        z = depth[::s, ::s]
        us, vs = np.meshgrid(np.arange(0, self.depth_width, s), np.arange(0, self.depth_height, s))
        x = (us - self.cx) * z / self.focal_px
        y = (vs - self.cy) * z / self.focal_px
        points = np.dstack((x, y, z)).reshape(-1, 3)
        valid = (z.ravel() > 0.01) & (z.ravel() < 100)
        # Reject the flat-world ground plane using the camera's measured full
        # pose. A fixed optical-Y cutoff admits distant floor points whenever
        # the suspension pitches during braking or turning.
        if self.depth_node is None:
            try:
                self.depth_node = self.robot.getFromDevice(self.depth_cam._tag)
            except Exception:
                self.depth_node = None
        if self.depth_node is not None:
            rotation = np.asarray(self.depth_node.getOrientation()).reshape(3, 3)
            camera_height = self.depth_node.getPosition()[2]
            # ROS optical (right, down, forward) -> Webots device
            # (forward, left, up) = (optical_z, -optical_x, -optical_y).
            world_height = camera_height + points @ np.array(
                [-rotation[2, 1], -rotation[2, 2], rotation[2, 0]])
            valid &= world_height > config.OCCUPANCY_FLOOR_THRESHOLD_M

        cloud = point_cloud2.create_cloud_xyz32(header, points[valid].astype(np.float32))
        cloud.is_dense = False
        self.pointcloud_pub.publish(cloud)

    def _publish_occupancy_grid(self, depth: np.ndarray) -> None:
        chunk = config.OCCUPANCY_CHUNK_SIZE_M
        map_depth = config.OCCUPANCY_MAP_DEPTH_M
        grid_x = int(config.OCCUPANCY_MAP_WIDTH_M / chunk)
        grid_z = int(map_depth / chunk)

        cols = np.arange(self.depth_width)
        rows = np.arange(self.depth_height).reshape(-1, 1)
        height_above_ground = config.CAMERA_HEIGHT_M - (rows - self.cy) * depth / self.focal_px
        valid = (
            (depth > 0.01) & (depth < map_depth)
            & (height_above_ground > config.OCCUPANCY_FLOOR_THRESHOLD_M)
            & (height_above_ground < config.OCCUPANCY_CEILING_THRESHOLD_M)
        )

        z = depth[valid]
        x = (np.broadcast_to(cols, depth.shape)[valid] - self.cx) / self.focal_px * z
        xi = (x / chunk).astype(np.int32) + grid_x // 2
        zi = (z / chunk).astype(np.int32)
        inside = (xi >= 0) & (xi < grid_x) & (zi >= 0) & (zi < grid_z)

        counts = np.zeros((grid_x, grid_z), dtype=np.int32)
        np.add.at(counts, (xi[inside], zi[inside]), 1)
        grid = (counts >= config.OCCUPANCY_MIN_POINTS_PER_CELL).astype(np.float64)

        msg = Float64MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label='x', size=grid_x, stride=grid_x * grid_z),
            MultiArrayDimension(label='z', size=grid_z, stride=grid_z),
        ]
        msg.data = array.array('d', grid.tobytes())
        self.occupancy_grid_pub.publish(msg)

    def _publish_front_camera(self, stamp: Time) -> None:
        if self.front_camera is None:
            return
        # JPEG because the frontend reads this through rosbridge.
        image = self.front_camera.getImage()
        if not image:
            return
        w, h = self.front_camera.getWidth(), self.front_camera.getHeight()
        bgr = np.frombuffer(image, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
        ok, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return

        msg = CompressedImage()
        msg.header = Header(stamp=stamp, frame_id='ground_camera')
        msg.format = 'jpeg'
        msg.data = array.array('B', jpeg.tobytes())
        self.front_camera_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    # Blocks until Webots accepts the extern controller connection.
    robot = Supervisor()
    node = WebotsBridge(robot)

    try:
        while rclpy.ok():
            step_res = robot.step(node.timestep)
            if step_res == -1:
                node.get_logger().warn("Webots robot.step() returned -1 (simulation stopped or reset)")
                break
            rclpy.spin_once(node, timeout_sec=0)
            node.step()
            node.simulation_speed.pace()
    except Exception as e:
        node.get_logger().error(f"Unhandled exception in WebotsBridge main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
