"""Launch the LAN-XI loadcell publisher in a vehicle namespace."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("vehicle_name", default_value="hippo02"),
            DeclareLaunchArgument(
                "calibration_file", default_value="loadcell_calibration.yaml"
            ),
            DeclareLaunchArgument("daq_ip", default_value="192.168.0.139"),
            DeclareLaunchArgument("input_filter", default_value="DC"),
            DeclareLaunchArgument("lowpass_hz", default_value="7.5"),
            DeclareLaunchArgument("tare_window_sec", default_value="0.5"),
            DeclareLaunchArgument("debug_period", default_value="2.0"),
            Node(
                package="loadcell_testing",
                executable="lanxi_loadcell_publisher",
                output="screen",
                emulate_tty=True,
                arguments=[
                    "--vehicle-name",
                    LaunchConfiguration("vehicle_name"),
                    "--calibration-file",
                    LaunchConfiguration("calibration_file"),
                    "--daq-ip",
                    LaunchConfiguration("daq_ip"),
                    "--input-filter",
                    LaunchConfiguration("input_filter"),
                    "--lowpass-hz",
                    LaunchConfiguration("lowpass_hz"),
                    "--tare-window-sec",
                    LaunchConfiguration("tare_window_sec"),
                    "--debug-period",
                    LaunchConfiguration("debug_period"),
                ],
            ),
        ]
    )
