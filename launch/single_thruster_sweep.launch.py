"""Launch a single-thruster sweep with an explicit arming confirmation."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_sweep(context):
    arm_motors = LaunchConfiguration("arm_motors").perform(context).lower()
    if arm_motors not in {"true", "false"}:
        raise ValueError("arm_motors must be 'true' or 'false'")

    arguments = [
        "--vehicle-name",
        LaunchConfiguration("vehicle_name").perform(context),
        "--wrench-topic",
        LaunchConfiguration("wrench_topic").perform(context),
        "--battery-topic",
        LaunchConfiguration("battery_topic").perform(context),
        "--esc-voltages-topic",
        LaunchConfiguration("esc_voltages_topic").perform(context),
        "--esc-rpms-topic",
        LaunchConfiguration("esc_rpms_topic").perform(context),
        "--arming-state-topic",
        LaunchConfiguration("arming_state_topic").perform(context),
        "--speeds",
        LaunchConfiguration("speeds").perform(context),
        "--repeat",
        LaunchConfiguration("repeat").perform(context),
        "--esc-index",
        LaunchConfiguration("esc_index").perform(context),
        "--output-dir",
        LaunchConfiguration("output_dir").perform(context),
        "--rate-hz",
        LaunchConfiguration("rate_hz").perform(context),
        "--pre-zero-sec",
        LaunchConfiguration("pre_zero_sec").perform(context),
        "--settle-sec",
        LaunchConfiguration("settle_sec").perform(context),
        "--sample-sec",
        LaunchConfiguration("sample_sec").perform(context),
        "--zero-sec",
        LaunchConfiguration("zero_sec").perform(context),
        "--post-zero-sec",
        LaunchConfiguration("post_zero_sec").perform(context),
    ]
    if arm_motors == "true":
        arguments.append("--yes")
    else:
        arguments.append("--dry-run")

    return [
        Node(
            package="loadcell_testing",
            executable="single_thruster_sweep",
            output="screen",
            emulate_tty=True,
            arguments=arguments,
        )
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("vehicle_name", default_value="hippo02"),
            DeclareLaunchArgument("wrench_topic", default_value="loadcell/wrench"),
            DeclareLaunchArgument("battery_topic", default_value="battery_state"),
            DeclareLaunchArgument(
                "esc_voltages_topic", default_value="esc_voltages"
            ),
            DeclareLaunchArgument("esc_rpms_topic", default_value="esc_rpms"),
            DeclareLaunchArgument(
                "arming_state_topic", default_value="arming_state"
            ),
            DeclareLaunchArgument(
                "speeds", default_value="0.1,0.2,0.3,0.4,0.5"
            ),
            DeclareLaunchArgument("repeat", default_value="1"),
            DeclareLaunchArgument("esc_index", default_value="0"),
            DeclareLaunchArgument(
                "output_dir", default_value="thruster_loadcell_runs"
            ),
            DeclareLaunchArgument("rate_hz", default_value="50.0"),
            DeclareLaunchArgument("pre_zero_sec", default_value="2.0"),
            DeclareLaunchArgument("settle_sec", default_value="1.0"),
            DeclareLaunchArgument("sample_sec", default_value="4.0"),
            DeclareLaunchArgument("zero_sec", default_value="1.0"),
            DeclareLaunchArgument("post_zero_sec", default_value="2.0"),
            DeclareLaunchArgument(
                "arm_motors",
                default_value="false",
                description=(
                    "Set true to permit ESC arming; false performs a dry run"
                ),
            ),
            OpaqueFunction(function=launch_sweep),
        ]
    )
