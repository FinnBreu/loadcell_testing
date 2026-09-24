#!/usr/bin/env python3

"""Run one ESC channel through a speed sweep."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Iterable

import rclpy
from hippo_control_msgs.msg import ActuatorControls
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from std_msgs.msg import Bool
from std_srvs.srv import SetBool


def parse_float_list(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one speed is required")
    if any(speed < -1.0 or speed > 1.0 for speed in values):
        raise argparse.ArgumentTypeError("speeds must be in [-1.0, 1.0]")
    return values


def normalize_speed_argument(arguments: list[str]) -> list[str]:
    """Keep a negative speed list from being interpreted as an option."""
    normalized: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--speeds" and index + 1 < len(arguments):
            next_argument = arguments[index + 1]
            if next_argument.startswith("-"):
                normalized.append(f"--speeds={next_argument}")
                index += 2
                continue
        normalized.append(argument)
        index += 1
    return normalized


def resolve_ros_name(namespace: str, name: str) -> str:
    name = name.strip()
    namespace = namespace.strip("/")
    if name.startswith("/"):
        return name
    if not namespace:
        return f"/{name}"
    return f"/{namespace}/{name}"


class ThrusterBenchNode(Node):
    def __init__(
        self,
        namespace: str,
        command_topic: str,
        arm_service: str,
        arming_state_topic: str,
        esc_count: int,
    ) -> None:
        super().__init__("single_thruster_sweep")
        self.esc_count = esc_count
        self.armed_state: bool | None = None

        self.publisher = self.create_publisher(
            ActuatorControls,
            resolve_ros_name(namespace, command_topic),
            QoSProfile(depth=1),
        )
        self.create_subscription(
            Bool,
            resolve_ros_name(namespace, arming_state_topic),
            self._on_arming_state,
            QoSProfile(depth=20),
        )
        self.arm_client = self.create_client(
            SetBool, resolve_ros_name(namespace, arm_service)
        )

    def _on_arming_state(self, msg: Bool) -> None:
        self.armed_state = msg.data

    def publish_command(self, values: Iterable[float]) -> None:
        controls = list(values)
        if len(controls) != self.esc_count:
            raise ValueError(f"expected {self.esc_count} ESC commands")
        msg = ActuatorControls()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.control = controls
        self.publisher.publish(msg)

    def publish_single_esc(self, esc_index: int, speed: float) -> None:
        controls = [0.0] * self.esc_count
        controls[esc_index] = speed
        self.publish_command(controls)

    def publish_for(
        self,
        esc_index: int,
        speed: float,
        duration: float,
        rate_hz: float,
    ) -> None:
        period = 1.0 / rate_hz
        end = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < end:
            self.publish_single_esc(esc_index, speed)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

    def call_set_bool(
        self,
        client,
        value: bool,
        timeout_sec: float,
    ) -> None:
        service_name = client.srv_name
        deadline = time.monotonic() + timeout_sec
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(f"service {service_name} is not available")
        request = SetBool.Request()
        request.data = value
        future = client.call_async(request)
        remaining = max(deadline - time.monotonic(), 0.0)
        rclpy.spin_until_future_complete(self, future, timeout_sec=remaining)
        if not future.done():
            raise TimeoutError(f"service {service_name} timed out")
        if future.exception() is not None:
            raise RuntimeError(
                f"service {service_name} failed: {future.exception()}"
            )
        response = future.result()
        self.get_logger().info(f"{service_name}: {response.message}")
        while rclpy.ok() and self.armed_state is not value:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            rclpy.spin_once(self, timeout_sec=min(0.1, remaining))

        if self.armed_state is value:
            if not response.success:
                self.get_logger().warning(
                    f"{service_name} returned success=false but arming_state "
                    f"confirmed {value}; accepting the confirmed state"
                )
            return

        raise RuntimeError(
            f"{service_name} did not produce arming_state={value} within "
            f"{timeout_sec:.1f} s (response success={response.success}, "
            f"message={response.message!r}, observed={self.armed_state})"
        )

    def arm(self, timeout_sec: float) -> None:
        self.call_set_bool(self.arm_client, True, timeout_sec)

    def disarm(self, timeout_sec: float) -> None:
        self.call_set_bool(self.arm_client, False, timeout_sec)


def run_sweep(args: argparse.Namespace) -> int:
    if not args.yes and not args.dry_run:
        print(
            "Refusing to arm motors without --yes. Use --dry-run to test ROS "
            "commands without arming."
        )
        return 2

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = ThrusterBenchNode(
        namespace=args.vehicle_name,
        command_topic=args.command_topic,
        arm_service=args.arm_service,
        arming_state_topic=args.arming_state_topic,
        esc_count=args.esc_count,
    )
    armed_by_sweep = False
    exit_code = 0

    try:
        node.publish_for(args.esc_index, 0.0, args.pre_zero_sec, args.rate_hz)

        if not args.dry_run:
            print("Arming ESC commander")
            node.arm(args.service_timeout)
            armed_by_sweep = True
        else:
            print("Dry run: skipping arm/disarm calls")

        step_index = 0
        for repeat in range(args.repeat):
            for speed in args.speeds:
                commanded_speed = 0.0 if args.dry_run else speed
                print(
                    f"repeat {repeat + 1}/{args.repeat}, step {step_index}: "
                    f"ESC index {args.esc_index} -> {commanded_speed:+.3f}"
                )
                node.publish_for(
                    args.esc_index,
                    commanded_speed,
                    args.settle_sec,
                    args.rate_hz,
                )
                node.publish_for(
                    args.esc_index,
                    commanded_speed,
                    args.sample_sec,
                    args.rate_hz,
                )
                node.publish_for(args.esc_index, 0.0, args.zero_sec, args.rate_hz)
                step_index += 1

    except KeyboardInterrupt:
        print("Interrupted, commanding zero")
        exit_code = 130
    except (RuntimeError, TimeoutError) as exc:
        node.get_logger().error(f"Sweep stopped: {exc}")
        exit_code = 1
    finally:
        try:
            node.publish_for(args.esc_index, 0.0, args.post_zero_sec, args.rate_hz)
        finally:
            if armed_by_sweep:
                try:
                    print("Disarming ESC commander")
                    node.disarm(args.service_timeout)
                except Exception as exc:
                    node.get_logger().error(f"Disarm sequence reported an error: {exc}")
                    exit_code = 1

            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Command ESC 1 through the Pi-side ESC commander. Use the launch "
            "file to record the test to rosbag."
        )
    )
    default_speeds = parse_float_list("0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--yes", action="store_true", help="allow arming motors")
    parser.add_argument("--dry-run", action="store_true", help="skip arm/disarm calls")
    parser.add_argument(
        "--vehicle-name",
        "--esc-namespace",
        "--namespace",
        dest="vehicle_name",
        default="hippo02",
        help="vehicle namespace containing the ESC commander",
    )
    parser.add_argument("--command-topic", default="thruster_command")
    parser.add_argument("--arm-service", default="arm")
    parser.add_argument("--arming-state-topic", default="arming_state")
    parser.add_argument("--service-timeout", type=float, default=3.0)
    parser.add_argument(
        "--esc-index",
        type=int,
        default=0,
        help="zero-based ESC command index; 0 is ESC 1",
    )
    parser.add_argument("--esc-count", type=int, default=8)
    parser.add_argument("--speeds", type=parse_float_list, default=default_speeds)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--pre-zero-sec", type=float, default=2.0)
    parser.add_argument("--settle-sec", type=float, default=1.0)
    parser.add_argument("--sample-sec", type=float, default=4.0)
    parser.add_argument("--zero-sec", type=float, default=1.0)
    parser.add_argument("--post-zero-sec", type=float, default=2.0)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = normalize_speed_argument(remove_ros_args(args=sys.argv)[1:])
    args = parser.parse_args(arguments)
    if args.esc_index < 0 or args.esc_index >= args.esc_count:
        parser.error("--esc-index must be in [0, esc-count)")
    if args.esc_count != 8:
        parser.error("--esc-count must be 8 for ActuatorControls")
    if args.rate_hz <= 0.0:
        parser.error("--rate-hz must be positive")
    if args.service_timeout <= 0.0:
        parser.error("--service-timeout must be positive")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    for name in (
        "pre_zero_sec",
        "settle_sec",
        "sample_sec",
        "zero_sec",
        "post_zero_sec",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    raise SystemExit(run_sweep(args))


if __name__ == "__main__":
    main()
