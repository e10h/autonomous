"""Launch the bundled simulation code against the existing ROS installation."""
import os
import sys
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    def python_node(module):
        return ExecuteProcess(cmd=[sys.executable, '-c',
            f'from {module} import main; main()',
            '--ros-args', '-p', 'use_sim_time:=true'], output='screen')
    return LaunchDescription([
        SetEnvironmentVariable('WEBOTS_CONTROLLER_URL', os.getenv(
            'WEBOTS_CONTROLLER_URL', 'tcp://host.docker.internal:1234/rover2')),
        SetEnvironmentVariable('WEBOTS_HOME', os.getenv('WEBOTS_HOME', '/opt/ros/jazzy')),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(os.path.dirname(__file__), 'localization.launch.py'))),
        python_node('autonomous_state_machine.autonomous_state_machine'),
        python_node('webots_bridge.bridge_node'),
    ])
