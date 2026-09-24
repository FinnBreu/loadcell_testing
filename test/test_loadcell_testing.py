import argparse
import csv
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from loadcell_testing.analysis_config import load_analysis_config
from loadcell_testing.analyze_thruster_run import (
    _default_analysis_dir,
    build_parser as build_analysis_parser,
)
from loadcell_testing.thruster_analysis import (
    AnalysisFits,
    CurveFit,
    RunSeries,
    actual_command_from_virtual,
    evaluate_symmetric_quadratic,
    fit_origin_linear,
    fit_origin_symmetric_quadratic,
    load_step_averages,
    virtual_command_from_actual,
)
from loadcell_testing.thruster_analysis_report import (
    _fit_dict,
    _write_plot_data,
    build_plot_data,
    fit_yaml_data,
)
from loadcell_testing.lanxi_loadcell_publisher import (
    HEADER_LENGTH,
    MESSAGE_SIGNAL_DATA,
    Calibration,
    OpenApiStreamParser,
    RuntimeWrenchTare,
    SignalMeta,
    configure_input_channels,
    build_parser as build_publisher_parser,
)
from loadcell_testing.single_thruster_sweep import (
    build_parser as build_sweep_parser,
    normalize_speed_argument,
    parse_float_list,
    resolve_ros_name,
)


def encode_i24le(value: int) -> bytes:
    encoded = value & 0xFFFFFF
    return bytes((encoded & 0xFF, (encoded >> 8) & 0xFF, encoded >> 16))


def signal_payload(channels: list[list[int]]) -> bytes:
    payload = bytearray(struct.pack("<HH", len(channels), 0))
    for signal_id, values in enumerate(channels, start=1):
        payload.extend(struct.pack("<hH", signal_id, len(values)))
        for value in values:
            payload.extend(encode_i24le(value))
    return bytes(payload)


def stream_message(payload: bytes) -> bytes:
    header = bytearray(HEADER_LENGTH)
    header[0:2] = b"BK"
    struct.pack_into("<H", header, 2, 20)
    struct.pack_into("<H", header, 4, MESSAGE_SIGNAL_DATA)
    struct.pack_into("<I", header, 24, len(payload))
    return bytes(header) + payload


class LoadcellTestingTest(unittest.TestCase):
    def test_analysis_accepts_multiple_runs(self) -> None:
        args = build_analysis_parser().parse_args(
            [
                "--run",
                "/tmp/run-a_ccw_fwd",
                "/tmp/run-b_cw_rev",
                "/tmp/run-c_ccw_rev",
                "--no-show",
            ]
        )

        self.assertEqual(
            args.run,
            ["/tmp/run-a_ccw_fwd", "/tmp/run-b_cw_rev", "/tmp/run-c_ccw_rev"],
        )
        self.assertTrue(args.no_show)

    def test_analysis_directory_is_independent_of_run_count(self) -> None:
        parent = Path("/tmp/thruster-runs")

        self.assertEqual(
            _default_analysis_dir([parent / "run-a"]),
            parent / "analysis",
        )
        self.assertEqual(
            _default_analysis_dir([parent / "run-a", parent / "run-b"]),
            parent / "analysis",
        )

    def test_fit_yaml_uses_simple_coefficient_name(self) -> None:
        fit = CurveFit(coefficients=(2.0, 3.0), r_squared=0.9)

        serialized = _fit_dict(fit)

        self.assertEqual(serialized["coefficients"], [2.0, 3.0])
        self.assertNotIn("coefficients_highest_power_first", serialized)

    def test_plot_data_uses_one_points_file(self) -> None:
        symmetric = CurveFit((2.0, 3.0), 1.0)
        proportional = CurveFit((1.0,), 1.0)
        command_fit = CurveFit((0.4, 0.6), 1.0)
        runs = [
            RunSeries(
                run_id=f"run-{index}",
                run_dir=Path("/tmp"),
                stand_direction=direction,
                steps=(),
                speeds=np.asarray([-0.5, 0.5]),
                thrust=np.asarray([-2.0, 2.0]),
                roll_torque=np.asarray([0.1, -0.1]),
                motor_torques=np.zeros((2, 3)),
                propeller_rotation="ccw",
                stand_sign=sign,
                propeller_sign=1,
            )
            for index, (direction, sign) in enumerate((("fwd", 1), ("rev", -1)))
        ]
        fits = AnalysisFits(
            thrust_proportional=proportional,
            thrust_quadratic=symmetric,
            roll_torque_proportional=proportional,
            roll_torque_quadratic=symmetric,
            roll_torque_per_thrust=proportional,
            virtual_command=command_fit,
            thrust_coefficient=5.0,
            torque_coefficient=1.0,
        )
        virtual_commands = tuple(
            virtual_command_from_actual(run.speeds, command_fit)
            for run in runs
        )
        plot_data = build_plot_data(fits, virtual_commands)
        models = fit_yaml_data(
            runs, fits, load_analysis_config()
        )["models"]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _write_plot_data(output, runs, plot_data)
            files = sorted(path.name for path in output.glob("*.csv"))
            with (output / "thruster_points.csv").open(
                newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(
            files,
            [
                "thruster_command_fits.csv",
                "thruster_points.csv",
                "thruster_virtual_command_fits.csv",
            ],
        )
        self.assertEqual(len(rows), 4)
        self.assertIn("thrust_quadratic", models)
        self.assertNotIn("thrust_quadratic_forward", models)
        self.assertEqual(
            models["thrust_proportional"]["coefficients"], [1.0]
        )
        self.assertEqual(
            models["thrust_quadratic"]["coefficients"], [2.0, 3.0]
        )
        self.assertEqual(
            models["virtual_command"]["coefficients"], [0.4, 0.6]
        )
        self.assertNotIn(
            "thruster_command_quadratic", models["virtual_command"]
        )
        self.assertEqual(models["virtual_command"]["thrust_coefficient"], 5.0)
        self.assertEqual(models["virtual_command"]["torque_coefficient"], 1.0)
        self.assertNotIn(
            "thrust_per_virtual_command", models["virtual_command"]
        )
        self.assertNotIn(
            "roll_torque_proportional", models["virtual_command"]
        )
        self.assertNotIn("actual_command_range", models["virtual_command"])
        self.assertNotIn("unit", models["roll_torque_per_thrust"])
        self.assertEqual({row["run_index"] for row in rows}, {"0", "1"})
        self.assertEqual(
            {row["loadcell_sign"] for row in rows}, {"-1", "1"}
        )

    def test_step_averages_respect_settling_and_sampling_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            csv_dir = run_dir / "csv"
            csv_dir.mkdir()
            manifest = {
                "vehicle_name": "uuv02",
                "metadata": {
                    "speeds": [0.2, -0.2],
                    "repeat": 1,
                    "rate_hz": 10.0,
                    "settle_sec": 1.0,
                    "sample_sec": 2.0,
                },
            }
            (run_dir / "manifest.yaml").write_text(
                yaml.safe_dump(manifest), encoding="utf-8"
            )
            base_ns = 1_000_000_000_000

            command_rows = []
            for tenth in range(100):
                time_sec = tenth / 10.0
                speed = 0.0
                if 2.0 <= time_sec < 5.0:
                    speed = 0.2
                elif 6.0 <= time_sec < 9.0:
                    speed = -0.2
                command_rows.append(
                    {
                        "t_ros_ns": base_ns + round(time_sec * 1e9),
                        "control.0": speed,
                    }
                )

            wrench_rows = []
            for half_second in range(20):
                time_sec = half_second / 2.0
                force_z = 0.0
                torque_z = 0.0
                if 3.0 <= time_sec < 5.0:
                    force_z = 2.0
                    torque_z = 0.2
                elif 7.0 <= time_sec < 9.0:
                    force_z = -3.0
                    torque_z = -0.3
                wrench_rows.append(
                    {
                        "t_ros_ns": base_ns + round(time_sec * 1e9),
                        "wrench.force.x": 1.0,
                        "wrench.force.y": 2.0,
                        "wrench.force.z": force_z,
                        "wrench.torque.x": 3.0,
                        "wrench.torque.y": 4.0,
                        "wrench.torque.z": torque_z,
                    }
                )

            battery_rows = [
                {"t_ros_ns": base_ns + 3_100_000_000, "voltage": 12.0},
                {"t_ros_ns": base_ns + 4_100_000_000, "voltage": 11.8},
                {"t_ros_ns": base_ns + 7_100_000_000, "voltage": 11.6},
                {"t_ros_ns": base_ns + 8_100_000_000, "voltage": 11.4},
            ]

            def write_rows(name: str, rows: list[dict]) -> None:
                with (csv_dir / name).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

            write_rows("uuv02_thruster_command.csv", command_rows)
            write_rows("uuv02_loadcell_wrench.csv", wrench_rows)
            write_rows("uuv02_battery_state.csv", battery_rows)

            _, steps = load_step_averages(run_dir)

        self.assertEqual(len(steps), 2)
        self.assertAlmostEqual(steps[0].force_z, 2.0)
        self.assertAlmostEqual(steps[0].torque_z, 0.2)
        self.assertAlmostEqual(steps[0].voltage, 11.9)
        self.assertEqual(steps[0].wrench_samples, 4)
        self.assertAlmostEqual(steps[1].force_z, -3.0)
        self.assertAlmostEqual(steps[1].torque_z, -0.3)
        self.assertAlmostEqual(steps[1].voltage, 11.5)

    def test_linear_fit_is_constrained_to_origin(self) -> None:
        speeds = np.asarray([-2.0, -1.0, 1.0, 2.0])
        values = 3.0 * speeds

        fit = fit_origin_linear(speeds, values)

        np.testing.assert_allclose(fit.coefficients, [3.0])
        self.assertAlmostEqual(fit.r_squared, 1.0)

    def test_symmetric_quadratic_fits_are_point_symmetric(self) -> None:
        speeds = np.asarray([-0.8, -0.4, 0.4, 0.8])
        values = 2.0 * speeds * np.abs(speeds) + 3.0 * speeds

        fit = fit_origin_symmetric_quadratic(speeds, values)

        np.testing.assert_allclose(fit.coefficients, [2.0, 3.0])
        predicted = evaluate_symmetric_quadratic(fit.coefficients, speeds)
        np.testing.assert_allclose(predicted, values)
        self.assertAlmostEqual(fit.r_squared, 1.0)

    def test_virtual_command_linearizes_thrust_fit(self) -> None:
        thrust_fit = CurveFit((2.0, 3.0), 1.0)
        actual = np.asarray([-0.8, -0.4, 0.0, 0.3, 0.7])

        scale = sum(thrust_fit.coefficients)
        virtual_fit = CurveFit(
            tuple(value / scale for value in thrust_fit.coefficients), 1.0
        )
        virtual = virtual_command_from_actual(actual, virtual_fit)
        restored = actual_command_from_virtual(virtual, virtual_fit)
        expected_thrust = evaluate_symmetric_quadratic(
            thrust_fit.coefficients, actual
        )
        np.testing.assert_allclose(scale * virtual, expected_thrust)
        np.testing.assert_allclose(restored, actual)

    def test_virtual_command_fit_is_normalized_forward_fit(self) -> None:
        thrust_fit = CurveFit((2.0, 3.0), 1.0)

        scale = sum(thrust_fit.coefficients)
        virtual_fit = CurveFit(
            tuple(value / scale for value in thrust_fit.coefficients), 1.0
        )

        self.assertEqual(scale, 5.0)
        np.testing.assert_allclose(virtual_fit.coefficients, [0.4, 0.6])
        endpoints = evaluate_symmetric_quadratic(
            virtual_fit.coefficients, np.asarray([-1.0, 1.0])
        )
        np.testing.assert_allclose(endpoints, [-1.0, 1.0])

    def test_calibration_applies_input_bias_then_wrench_tare(self) -> None:
        calibration = Calibration(
            matrix=[
                [1, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0],
                [0, 0, 3, 0, 0, 0],
                [0, 0, 0, 4, 0, 0],
                [0, 0, 0, 0, 5, 0],
                [0, 0, 0, 0, 0, 6],
            ],
            input_bias=[1, 1, 1, 1, 1, 1],
            wrench_tare=[0.5, 1, 1.5, 2, 2.5, 3],
        )

        self.assertEqual(
            calibration.apply([2, 3, 4, 5, 6, 7]),
            [0.5, 3, 7.5, 14, 22.5, 33],
        )

    def test_runtime_tare_averages_recent_samples_and_resets(self) -> None:
        tare = RuntimeWrenchTare(window_sec=0.5)
        tare.add(1.0, [1, 2, 3, 4, 5, 6])
        tare.add(1.4, [3, 4, 5, 6, 7, 8])

        offset, sample_count = tare.set_from_recent(1.5)

        self.assertEqual(sample_count, 2)
        self.assertEqual(offset, [2, 3, 4, 5, 6, 7])
        self.assertEqual(
            tare.apply([3, 5, 7, 9, 11, 13]),
            [1, 2, 3, 4, 5, 6],
        )
        tare.reset()
        self.assertEqual(
            tare.apply([3, 5, 7, 9, 11, 13]),
            [3, 5, 7, 9, 11, 13],
        )

    def test_runtime_tare_rejects_stale_samples(self) -> None:
        tare = RuntimeWrenchTare(window_sec=0.5)
        tare.add(1.0, [1, 2, 3, 4, 5, 6])

        self.assertEqual(tare.set_from_recent(2.0), (None, 0))

    def test_parser_decodes_split_signed_24_bit_message(self) -> None:
        parser = OpenApiStreamParser(channel_count=6)
        for signal_id in range(1, 7):
            parser.meta[signal_id] = SignalMeta(scale_factor=float(2**23))
        channels = [
            [-2, 4],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]
        message = stream_message(signal_payload(channels))

        self.assertEqual(parser.feed(message[:17]), [])
        self.assertEqual(len(parser.buffer), 17)
        self.assertEqual(
            parser.feed(message[17:]),
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
        )
        self.assertEqual(len(parser.buffer), 0)
        self.assertEqual(
            parser.signal_counts,
            {signal_id: 2 for signal_id in range(1, 7)},
        )
        self.assertEqual(parser.bad_magic_count, 0)

    def test_parse_float_list_validates_range(self) -> None:
        self.assertEqual(
            parse_float_list("-1, 0, 0.25, 1"),
            [-1.0, 0.0, 0.25, 1.0],
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_float_list("1.01")

    def test_channel_configuration_disables_high_pass_filter(self) -> None:
        setup = {
            "channels": [
                {"channel": 1, "filter": "7.0 Hz", "destinations": ["sd"]},
                {"channel": 2, "filter": "7.0 Hz", "destinations": ["sd"]},
            ]
        }

        configure_input_channels(setup, "DC")

        for channel in setup["channels"]:
            self.assertEqual(channel["filter"], "DC")
            self.assertEqual(channel["destinations"], ["socket"])

    def test_resolve_ros_name(self) -> None:
        self.assertEqual(
            resolve_ros_name("hippo02", "loadcell/wrench"),
            "/hippo02/loadcell/wrench",
        )
        self.assertEqual(
            resolve_ros_name("hippo02", "/shared/wrench"),
            "/shared/wrench",
        )

    def test_sweep_uses_vehicle_relative_command_topic(self) -> None:
        args = build_sweep_parser().parse_args(["--namespace", "uuv02"])

        self.assertEqual(args.vehicle_name, "uuv02")
        self.assertEqual(args.command_topic, "thruster_command")
        self.assertEqual(
            resolve_ros_name(args.vehicle_name, args.command_topic),
            "/uuv02/thruster_command",
        )

    def test_publisher_uses_vehicle_relative_names(self) -> None:
        args = build_publisher_parser().parse_args(["--vehicle-name", "uuv02"])

        self.assertEqual(args.vehicle_name, "uuv02")
        self.assertEqual(args.wrench_topic, "loadcell/wrench")
        self.assertEqual(args.tare_service, "loadcell/tare")

    def test_negative_speed_argument_is_unambiguous(self) -> None:
        arguments = normalize_speed_argument(["--speeds", "-0.1,-0.2"])
        args = build_sweep_parser().parse_args(arguments)

        self.assertEqual(args.speeds, [-0.1, -0.2])
