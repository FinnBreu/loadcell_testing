"""Launch a single-thruster sweep with an explicit arming confirmation."""

import json
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_sweep(context):
    arm_motors = LaunchConfiguration("arm_motors").perform(context).lower()
    if arm_motors not in {"true", "false"}:
        raise ValueError("arm_motors must be 'true' or 'false'")
    record = LaunchConfiguration("record").perform(context).lower()
    if record not in {"true", "false"}:
        raise ValueError("record must be 'true' or 'false'")

    arguments = [
        "--vehicle-name",
        LaunchConfiguration("vehicle_name").perform(context),
        "--arming-state-topic",
        LaunchConfiguration("arming_state_topic").perform(context),
        f"--speeds={LaunchConfiguration('speeds').perform(context)}",
        "--repeat",
        LaunchConfiguration("repeat").perform(context),
        "--esc-index",
        LaunchConfiguration("esc_index").perform(context),
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

    sweep = Node(
        package="loadcell_testing",
        executable="single_thruster_sweep",
        output="screen",
        emulate_tty=True,
        arguments=arguments,
    )
    if record == "false":
        return [sweep]

    metadata = {
        "arm_motors": arm_motors == "true",
        "esc_index": int(LaunchConfiguration("esc_index").perform(context)),
        "speeds": [
            float(value)
            for value in LaunchConfiguration("speeds").perform(context).split(",")
        ],
        "repeat": int(LaunchConfiguration("repeat").perform(context)),
        "rate_hz": float(LaunchConfiguration("rate_hz").perform(context)),
        "pre_zero_sec": float(
            LaunchConfiguration("pre_zero_sec").perform(context)
        ),
        "settle_sec": float(LaunchConfiguration("settle_sec").perform(context)),
        "sample_sec": float(LaunchConfiguration("sample_sec").perform(context)),
        "zero_sec": float(LaunchConfiguration("zero_sec").perform(context)),
        "post_zero_sec": float(
            LaunchConfiguration("post_zero_sec").perform(context)
        ),
    }
    tools_share = Path(get_package_share_directory("tools"))
    recorder = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(tools_share / "launch" / "record.launch.py")
        ),
        launch_arguments={
            "vehicle_name": LaunchConfiguration("vehicle_name").perform(context),
            "scenario": LaunchConfiguration("scenario").perform(context),
            "profile": "thruster_loadcell",
            "output_root": LaunchConfiguration("output_root").perform(context),
            "use_sim_time": "false",
            "metadata_json": json.dumps(metadata),
        }.items(),
    )
    stop_recording = RegisterEventHandler(
        OnProcessExit(
            target_action=sweep,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason="single-thruster sweep finished")
                )
            ],
        )
    )
    delayed_sweep = TimerAction(
        period=float(
            LaunchConfiguration("recorder_start_delay").perform(context)
        ),
        actions=[sweep],
    )
    return [stop_recording, recorder, delayed_sweep]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("vehicle_name", default_value="hippo02"),
            DeclareLaunchArgument(
                "arming_state_topic", default_value="arming_state"
            ),
            DeclareLaunchArgument(
                "speeds", default_value="0.1,0.2,0.3,0.4,0.5"
            ),
            DeclareLaunchArgument("repeat", default_value="1"),
            DeclareLaunchArgument("esc_index", default_value="0"),
            DeclareLaunchArgument(
                "output_root",
                default_value=str(
                    Path.home() / "ros2" / "data" / "thruster_loadcell_runs"
                ),
            ),
            DeclareLaunchArgument("scenario", default_value="thruster_sweep"),
            DeclareLaunchArgument("record", default_value="true"),
            DeclareLaunchArgument("recorder_start_delay", default_value="2.0"),
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
