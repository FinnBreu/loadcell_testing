#!/usr/bin/env python3

"""Run one ESC channel through a speed sweep and record an existing ROS wrench."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import rclpy
from geometry_msgs.msg import WrenchStamped
from hippo_control_msgs.msg import ActuatorControls
from hippo_msgs.msg import EscRpms, EscVoltages
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool
from std_srvs.srv import SetBool


@dataclass
class SweepStep:
    repeat: int
    index: int
    speed: float
    command_start_monotonic: float
    command_start_utc: str
    sample_start_monotonic: float
    sample_start_utc: str
    sample_end_monotonic: float
    sample_end_utc: str
    command_end_monotonic: float
    command_end_utc: str


@dataclass
class TelemetrySample:
    receipt_monotonic: float
    receipt_utc: str
    stamp_sec: int
    stamp_nanosec: int
    repeat: int
    step_index: int
    phase: str
    command_speed: float


@dataclass
class WrenchSample(TelemetrySample):
    force_x: float
    force_y: float
    force_z: float
    torque_x: float
    torque_y: float
    torque_z: float


@dataclass
class ThrusterCommandSample(TelemetrySample):
    control_0: float
    control_1: float
    control_2: float
    control_3: float
    control_4: float
    control_5: float
    control_6: float
    control_7: float


@dataclass
class BatterySample(TelemetrySample):
    voltage: float
    temperature: float
    current: float
    charge: float
    capacity: float
    design_capacity: float
    percentage: float
    power_supply_status: int
    power_supply_health: int
    power_supply_technology: int
    present: bool
    cell_voltage_json: str
    cell_temperature_json: str
    location: str
    serial_number: str


@dataclass
class EscVoltageSample(TelemetrySample):
    voltage_0: float
    voltage_1: float
    voltage_2: float
    voltage_3: float
    voltage_4: float
    voltage_5: float
    voltage_6: float
    voltage_7: float


@dataclass
class EscRpmSample(TelemetrySample):
    rpm_0: float
    rpm_1: float
    rpm_2: float
    rpm_3: float
    rpm_4: float
    rpm_5: float
    rpm_6: float
    rpm_7: float
    revolution_0: float
    revolution_1: float
    revolution_2: float
    revolution_3: float
    revolution_4: float
    revolution_5: float
    revolution_6: float
    revolution_7: float
    commutation_0: int
    commutation_1: int
    commutation_2: int
    commutation_3: int
    commutation_4: int
    commutation_5: int
    commutation_6: int
    commutation_7: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        wrench_topic: str,
        battery_topic: str,
        esc_voltages_topic: str,
        esc_rpms_topic: str,
        arm_service: str,
        arming_state_topic: str,
        esc_count: int,
    ) -> None:
        super().__init__("single_thruster_sweep")
        self.esc_count = esc_count
        self.current_speed = 0.0
        self.current_repeat = -1
        self.current_step = -1
        self.current_phase = "initializing"
        self.armed_state: bool | None = None
        self.wrench_samples: list[WrenchSample] = []
        self.command_samples: list[ThrusterCommandSample] = []
        self.battery_samples: list[BatterySample] = []
        self.esc_voltage_samples: list[EscVoltageSample] = []
        self.esc_rpm_samples: list[EscRpmSample] = []

        self.publisher = self.create_publisher(
            ActuatorControls,
            resolve_ros_name(namespace, command_topic),
            QoSProfile(depth=1),
        )
        self.create_subscription(
            WrenchStamped,
            resolve_ros_name(namespace, wrench_topic),
            self._on_wrench,
            QoSProfile(depth=20),
        )
        self.create_subscription(
            BatteryState,
            resolve_ros_name(namespace, battery_topic),
            self._on_battery,
            QoSProfile(depth=20),
        )
        self.create_subscription(
            EscVoltages,
            resolve_ros_name(namespace, esc_voltages_topic),
            self._on_esc_voltages,
            QoSProfile(depth=20),
        )
        self.create_subscription(
            EscRpms,
            resolve_ros_name(namespace, esc_rpms_topic),
            self._on_esc_rpms,
            QoSProfile(depth=20),
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

    def _sample_context(self, header) -> tuple:
        return (
            time.monotonic(),
            utc_now(),
            header.stamp.sec,
            header.stamp.nanosec,
            self.current_repeat,
            self.current_step,
            self.current_phase,
            self.current_speed,
        )

    def _on_wrench(self, msg: WrenchStamped) -> None:
        wrench = msg.wrench
        self.wrench_samples.append(
            WrenchSample(
                *self._sample_context(msg.header),
                force_x=wrench.force.x,
                force_y=wrench.force.y,
                force_z=wrench.force.z,
                torque_x=wrench.torque.x,
                torque_y=wrench.torque.y,
                torque_z=wrench.torque.z,
            )
        )

    def _on_battery(self, msg: BatteryState) -> None:
        self.battery_samples.append(
            BatterySample(
                *self._sample_context(msg.header),
                voltage=msg.voltage,
                temperature=msg.temperature,
                current=msg.current,
                charge=msg.charge,
                capacity=msg.capacity,
                design_capacity=msg.design_capacity,
                percentage=msg.percentage,
                power_supply_status=msg.power_supply_status,
                power_supply_health=msg.power_supply_health,
                power_supply_technology=msg.power_supply_technology,
                present=msg.present,
                cell_voltage_json=json.dumps(list(msg.cell_voltage)),
                cell_temperature_json=json.dumps(list(msg.cell_temperature)),
                location=msg.location,
                serial_number=msg.serial_number,
            )
        )

    def _on_esc_voltages(self, msg: EscVoltages) -> None:
        self.esc_voltage_samples.append(
            EscVoltageSample(*self._sample_context(msg.header), *msg.data)
        )

    def _on_esc_rpms(self, msg: EscRpms) -> None:
        self.esc_rpm_samples.append(
            EscRpmSample(
                *self._sample_context(msg.header),
                *msg.rpms,
                *msg.revolutions,
                *msg.commutations,
            )
        )

    def set_context(self, repeat: int, step: int, phase: str) -> None:
        self.current_repeat = repeat
        self.current_step = step
        self.current_phase = phase

    def publish_command(self, values: Iterable[float]) -> None:
        controls = list(values)
        if len(controls) != self.esc_count:
            raise ValueError(f"expected {self.esc_count} ESC commands")
        msg = ActuatorControls()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.control = controls
        self.publisher.publish(msg)
        self.command_samples.append(
            ThrusterCommandSample(*self._sample_context(msg.header), *controls)
        )

    def publish_single_esc(self, esc_index: int, speed: float) -> None:
        self.current_speed = speed
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


def write_dataclass_csv(path: Path, rows: list, row_type: type) -> None:
    fieldnames = list(row_type.__dataclass_fields__)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(vars(row))


def write_run_config(
    path: Path,
    args: argparse.Namespace,
    measurement_name: str,
    sample_counts: dict[str, int],
) -> None:
    data = vars(args).copy()
    data["measurement"] = measurement_name
    data["created_utc"] = utc_now()
    data["sample_counts"] = sample_counts
    data["wrench_sample_count"] = sample_counts["wrench"]
    data["esc_number_note"] = "ESC 1 is --esc-index 0 in the zero-based array."
    with path.open("w") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def run_sweep(args: argparse.Namespace) -> int:
    if not args.yes and not args.dry_run:
        print(
            "Refusing to arm motors without --yes. Use --dry-run to test ROS "
            "commands without arming."
        )
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    measurement_name = args.name or datetime.now().strftime("thruster_%Y%m%d_%H%M%S")

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = ThrusterBenchNode(
        namespace=args.vehicle_name,
        command_topic=args.command_topic,
        wrench_topic=args.wrench_topic,
        battery_topic=args.battery_topic,
        esc_voltages_topic=args.esc_voltages_topic,
        esc_rpms_topic=args.esc_rpms_topic,
        arm_service=args.arm_service,
        arming_state_topic=args.arming_state_topic,
        esc_count=args.esc_count,
    )
    steps: list[SweepStep] = []
    armed_by_sweep = False
    exit_code = 0

    try:
        node.set_context(-1, -1, "pre_zero")
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
                command_start_mono = time.monotonic()
                command_start_utc = utc_now()
                node.set_context(repeat, step_index, "settling")
                node.publish_for(
                    args.esc_index,
                    commanded_speed,
                    args.settle_sec,
                    args.rate_hz,
                )
                sample_start_mono = time.monotonic()
                sample_start_utc = utc_now()
                node.set_context(repeat, step_index, "sample")
                node.publish_for(
                    args.esc_index,
                    commanded_speed,
                    args.sample_sec,
                    args.rate_hz,
                )
                sample_end_mono = time.monotonic()
                sample_end_utc = utc_now()
                node.set_context(repeat, step_index, "zero")
                node.publish_for(args.esc_index, 0.0, args.zero_sec, args.rate_hz)
                command_end_mono = time.monotonic()
                command_end_utc = utc_now()
                steps.append(
                    SweepStep(
                        repeat=repeat,
                        index=step_index,
                        speed=commanded_speed,
                        command_start_monotonic=command_start_mono,
                        command_start_utc=command_start_utc,
                        sample_start_monotonic=sample_start_mono,
                        sample_start_utc=sample_start_utc,
                        sample_end_monotonic=sample_end_mono,
                        sample_end_utc=sample_end_utc,
                        command_end_monotonic=command_end_mono,
                        command_end_utc=command_end_utc,
                    )
                )
                step_index += 1

    except KeyboardInterrupt:
        print("Interrupted, commanding zero")
        exit_code = 130
    except (RuntimeError, TimeoutError) as exc:
        node.get_logger().error(f"Sweep stopped: {exc}")
        exit_code = 1
    finally:
        try:
            node.set_context(-1, -1, "post_zero")
            node.publish_for(args.esc_index, 0.0, args.post_zero_sec, args.rate_hz)
        finally:
            if armed_by_sweep:
                try:
                    print("Disarming ESC commander")
                    node.disarm(args.service_timeout)
                except Exception as exc:
                    node.get_logger().error(f"Disarm sequence reported an error: {exc}")
                    exit_code = 1

            steps_path = output_dir / f"{measurement_name}_steps.csv"
            commands_path = output_dir / f"{measurement_name}_commands.csv"
            wrench_path = output_dir / f"{measurement_name}_wrench.csv"
            battery_path = output_dir / f"{measurement_name}_battery_state.csv"
            esc_voltages_path = (
                output_dir / f"{measurement_name}_esc_voltages.csv"
            )
            esc_rpms_path = output_dir / f"{measurement_name}_esc_rpms.csv"
            config_path = output_dir / f"{measurement_name}_config.json"
            write_dataclass_csv(steps_path, steps, SweepStep)
            write_dataclass_csv(
                commands_path, node.command_samples, ThrusterCommandSample
            )
            write_dataclass_csv(wrench_path, node.wrench_samples, WrenchSample)
            write_dataclass_csv(
                battery_path, node.battery_samples, BatterySample
            )
            write_dataclass_csv(
                esc_voltages_path,
                node.esc_voltage_samples,
                EscVoltageSample,
            )
            write_dataclass_csv(
                esc_rpms_path, node.esc_rpm_samples, EscRpmSample
            )
            sample_counts = {
                "commands": len(node.command_samples),
                "wrench": len(node.wrench_samples),
                "battery_state": len(node.battery_samples),
                "esc_voltages": len(node.esc_voltage_samples),
                "esc_rpms": len(node.esc_rpm_samples),
            }
            write_run_config(
                config_path, args, measurement_name, sample_counts
            )
            telemetry_streams = (
                ("loadcell", args.wrench_topic, node.wrench_samples),
                ("battery", args.battery_topic, node.battery_samples),
                (
                    "ESC voltage",
                    args.esc_voltages_topic,
                    node.esc_voltage_samples,
                ),
                ("ESC RPM", args.esc_rpms_topic, node.esc_rpm_samples),
            )
            for label, topic, samples in telemetry_streams:
                if not samples:
                    node.get_logger().warning(
                        f"No {label} messages were received on "
                        f"{resolve_ros_name(args.vehicle_name, topic)}."
                    )
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            print(f"Wrote {steps_path}")
            print(f"Wrote {commands_path} ({len(node.command_samples)} samples)")
            print(f"Wrote {wrench_path} ({len(node.wrench_samples)} samples)")
            print(f"Wrote {battery_path} ({len(node.battery_samples)} samples)")
            print(
                f"Wrote {esc_voltages_path} "
                f"({len(node.esc_voltage_samples)} samples)"
            )
            print(f"Wrote {esc_rpms_path} ({len(node.esc_rpm_samples)} samples)")
            print(f"Wrote {config_path}")

    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Command ESC 1 through the Pi-side ESC commander and record an "
            "independently published ROS loadcell wrench."
        )
    )
    default_speeds = parse_float_list("0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--yes", action="store_true", help="allow arming motors")
    parser.add_argument("--dry-run", action="store_true", help="skip arm/disarm calls")
    parser.add_argument("--name", default="", help="measurement name prefix")
    parser.add_argument("--output-dir", default="thruster_loadcell_runs")
    parser.add_argument(
        "--vehicle-name",
        "--esc-namespace",
        "--namespace",
        dest="vehicle_name",
        default="hippo02",
        help="vehicle namespace containing the ESC commander",
    )
    parser.add_argument("--command-topic", default="thruster_command")
    parser.add_argument("--wrench-topic", default="loadcell/wrench")
    parser.add_argument("--battery-topic", default="battery_state")
    parser.add_argument("--esc-voltages-topic", default="esc_voltages")
    parser.add_argument("--esc-rpms-topic", default="esc_rpms")
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
    args = parser.parse_args(remove_ros_args(args=sys.argv)[1:])
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
