import argparse
import struct
import tempfile
import unittest
from pathlib import Path

from hippo_msgs.msg import EscRpms, EscVoltages
from sensor_msgs.msg import BatteryState

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
    BatterySample,
    ThrusterBenchNode,
    build_parser as build_sweep_parser,
    parse_float_list,
    resolve_ros_name,
    write_dataclass_csv,
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

    def test_sweep_separates_vehicle_name_from_wrench_topic(self) -> None:
        args = build_sweep_parser().parse_args(["--namespace", "uuv02"])

        self.assertEqual(args.vehicle_name, "uuv02")
        self.assertEqual(args.wrench_topic, "loadcell/wrench")
        self.assertEqual(
            resolve_ros_name(args.vehicle_name, args.wrench_topic),
            "/uuv02/loadcell/wrench",
        )

    def test_publisher_uses_vehicle_relative_names(self) -> None:
        args = build_publisher_parser().parse_args(["--vehicle-name", "uuv02"])

        self.assertEqual(args.vehicle_name, "uuv02")
        self.assertEqual(args.wrench_topic, "loadcell/wrench")
        self.assertEqual(args.tare_service, "loadcell/tare")

    def test_sweep_converts_battery_and_esc_telemetry(self) -> None:
        node = object.__new__(ThrusterBenchNode)
        node.current_repeat = 1
        node.current_step = 2
        node.current_phase = "sample"
        node.current_speed = 0.3
        node.battery_samples = []
        node.esc_voltage_samples = []
        node.esc_rpm_samples = []

        battery = BatteryState()
        battery.voltage = 14.8
        battery.current = -2.0
        battery.cell_voltage = [3.7] * 4
        ThrusterBenchNode._on_battery(node, battery)

        voltages = EscVoltages()
        voltages.data = [14.0 + index for index in range(8)]
        ThrusterBenchNode._on_esc_voltages(node, voltages)

        rpms = EscRpms()
        rpms.rpms = [100.0 + index for index in range(8)]
        rpms.revolutions = [200.0 + index for index in range(8)]
        rpms.commutations = [300 + index for index in range(8)]
        ThrusterBenchNode._on_esc_rpms(node, rpms)

        self.assertAlmostEqual(node.battery_samples[0].voltage, 14.8, places=5)
        self.assertIn("3.7", node.battery_samples[0].cell_voltage_json)
        self.assertEqual(node.esc_voltage_samples[0].voltage_7, 21.0)
        self.assertEqual(node.esc_rpm_samples[0].rpm_0, 100.0)
        self.assertEqual(node.esc_rpm_samples[0].commutation_7, 307)

    def test_empty_telemetry_csv_still_has_headers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "battery.csv"
            write_dataclass_csv(path, [], BatterySample)

            header = path.read_text().splitlines()[0]

        self.assertIn("receipt_monotonic", header)
        self.assertIn("voltage", header)
