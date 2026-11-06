import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
import sys
from launch_ros.actions import Node


def generate_launch_description():
    ekf_path = os.path.join(
        os.path.dirname(__file__), "runtime", "rover_localization", "config", "ekf.yaml")

    return LaunchDescription([
        ExecuteProcess(cmd=[sys.executable, '-c',
            'from rover_localization.sim_localization import main; main()',
            '--ros-args', '-p', 'use_sim_time:=true'], output='screen'),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_static_map_odom",
            arguments=["--frame-id", "map", "--child-frame-id", "odom"],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_static_base_link_oakd_depth",
            arguments=[
                "--x", "0.3", "--y", "0.0", "--z", "0.48",
                "--roll", "-1.57079632679", "--pitch", "0.0",
                "--yaw", "-1.57079632679", "--frame-id", "base_link",
                "--child-frame-id", "oakd_depth_camera",
            ],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_static_base_link_gps_link",
            arguments=["--frame-id", "base_link", "--child-frame-id", "gps_link"],
        ),
        Node(
            package="robot_localization",
            executable="navsat_transform_node",
            name="navsat_transform",
            output="screen",
            parameters=[ekf_path, {"use_sim_time": True}],
            remappings=[
                ("gps/fix", "/autonomous/localization/gps/fix"),
                ("imu", "/autonomous/localization/gps/heading"),
                ("odometry/filtered", "/autonomous/localization/odometry/local"),
                ("odometry/gps", "/autonomous/localization/odometry/gps"),
            ],
        ),
    ])
