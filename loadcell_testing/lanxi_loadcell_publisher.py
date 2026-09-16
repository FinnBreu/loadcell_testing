#!/usr/bin/env python3

"""Publish LAN-XI 6-DOF loadcell samples as ROS messages."""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_path
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from std_srvs.srv import Trigger

try:
    import requests
except ImportError as exc:  # pragma: no cover - exercised on target machine
    raise SystemExit(
        "Missing Python package 'requests'. Install python3-requests."
    ) from exc

try:
    import yaml
except ImportError:  # pragma: no cover - YAML calibration files become unavailable
    yaml = None


HEADER_LENGTH = 28
MESSAGE_SIGNAL_DATA = 1
MESSAGE_DATA_QUALITY = 2
MESSAGE_INTERPRETATION = 8
DESCRIPTOR_SCALE_FACTOR = 2
DESCRIPTOR_OFFSET = 3
DESCRIPTOR_PERIOD_TIME = 4
DESCRIPTOR_UNIT = 5
SIGNED_24BIT_FULL_SCALE = 2**23
INPUT_FILTERS = (
    "DC",
    "0.1 Hz 10%",
    "0.7 Hz",
    "1.0 Hz 10%",
    "7.0 Hz",
    "22.4 Hz",
    "Intensity",
)


@dataclass
class SignalMeta:
    scale_factor: float = 1.0
    offset: float = 0.0
    unit: str = ""
    period: float | None = None


@dataclass
class Calibration:
    matrix: list[list[float]]
    input_bias: list[float]
    wrench_tare: list[float]
    source_path: Path | None = None

    def apply(self, raw: list[float]) -> list[float]:
        corrected = [raw[i] - self.input_bias[i] for i in range(6)]
        calibrated = [
            sum(self.matrix[row][col] * corrected[col] for col in range(6))
            for row in range(6)
        ]
        return [calibrated[i] - self.wrench_tare[i] for i in range(6)]


class RuntimeWrenchTare:
    def __init__(self, window_sec: float, max_samples: int = 4096) -> None:
        self.window_sec = window_sec
        self.samples: deque[tuple[float, list[float]]] = deque(
            maxlen=max_samples
        )
        self.offset = [0.0] * 6
        self.lock = threading.Lock()

    def add(self, timestamp: float, wrench: list[float]) -> None:
        with self.lock:
            self.samples.append((timestamp, wrench.copy()))
            cutoff = timestamp - self.window_sec
            while self.samples and self.samples[0][0] < cutoff:
                self.samples.popleft()

    def set_from_recent(self, now: float) -> tuple[list[float] | None, int]:
        cutoff = now - self.window_sec
        with self.lock:
            recent = [
                wrench
                for timestamp, wrench in self.samples
                if timestamp >= cutoff
            ]
            if not recent:
                return None, 0
            self.offset = [
                sum(wrench[axis] for wrench in recent) / len(recent)
                for axis in range(6)
            ]
            return self.offset.copy(), len(recent)

    def reset(self) -> None:
        with self.lock:
            self.offset = [0.0] * 6

    def apply(self, wrench: list[float]) -> list[float]:
        with self.lock:
            return [wrench[i] - self.offset[i] for i in range(6)]


def calc_time_from_family(k: int, l: int, m: int, n: int, count: int) -> float:
    family = 2**k * 3**l * 5**m * 7**n
    return count / family


def load_calibration(path: str) -> Calibration:
    calibration_path = resolve_calibration_path(path)
    with calibration_path.open() as file:
        if calibration_path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError(
                    "YAML calibration files require python3-yaml. "
                    "Use JSON or install python3-yaml."
                )
            data = yaml.safe_load(file)
        else:
            data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("calibration file must contain a mapping")
    if "bias" in data:
        raise ValueError(
            "calibration key 'bias' is ambiguous; rename it to 'input_bias' "
            "for raw DAQ-channel offsets or 'wrench_tare' for force/torque tare"
        )

    matrix = data.get("matrix")
    input_bias = data.get("input_bias", [0.0] * 6)
    wrench_tare = data.get("wrench_tare", [0.0] * 6)
    if not is_matrix_6x6(matrix):
        raise ValueError("calibration 'matrix' must be a 6x6 array")
    if not isinstance(input_bias, list) or len(input_bias) != 6:
        raise ValueError("calibration 'input_bias' must be an array with 6 values")
    if not isinstance(wrench_tare, list) or len(wrench_tare) != 6:
        raise ValueError("calibration 'wrench_tare' must be an array with 6 values")
    return Calibration(
        matrix=[[float(value) for value in row] for row in matrix],
        input_bias=[float(value) for value in input_bias],
        wrench_tare=[float(value) for value in wrench_tare],
        source_path=calibration_path.resolve(),
    )


def resolve_calibration_path(path: str) -> Path:
    requested = Path(path).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append(Path.cwd() / requested)
        try:
            package_share = get_package_share_path("loadcell_testing")
        except PackageNotFoundError:
            package_share = None
        if package_share is not None:
            candidates.extend([
                package_share / requested,
                package_share / "config" / requested.name,
            ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    attempted = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find calibration file. Tried:\n"
        f"  {attempted}\n"
        "Tip: use an absolute path, a path relative to your shell's current "
        "directory, or an installed config filename such as "
        "'loadcell_calibration.yaml'."
    )


def is_matrix_6x6(value) -> bool:
    if not isinstance(value, list) or len(value) != 6:
        return False
    return all(isinstance(row, list) and len(row) == 6 for row in value)


def configure_input_channels(setup: dict, input_filter: str) -> dict:
    for channel in setup.get("channels", []):
        channel["destinations"] = ["socket"]
        channel["filter"] = input_filter
    return setup


class OpenApiStreamParser:
    """Incremental parser for the LAN-XI messages used by this publisher."""

    def __init__(self, channel_count: int) -> None:
        self.channel_count = channel_count
        self.buffer = bytearray()
        self.meta: dict[int, SignalMeta] = {}
        self.signal_order: list[int] = []
        self.latest_signal_values: dict[int, float] = {}
        self.updated_signals: set[int] = set()
        self.last_block_sizes: dict[int, int] = {}
        self.total_bytes = 0
        self.message_counts: dict[int, int] = {}
        self.signal_counts: dict[int, int] = {}
        self.vector_count = 0
        self.bad_magic_count = 0
        self.header_lengths: dict[int, int] = {}

    def feed(self, data: bytes) -> list[list[float]]:
        self.total_bytes += len(data)
        self.buffer.extend(data)
        samples: list[list[float]] = []
        while True:
            if len(self.buffer) < HEADER_LENGTH:
                break
            if self.buffer[0:2] != b"BK":
                self.bad_magic_count += 1
                del self.buffer[0]
                continue
            header_length = struct.unpack_from("<H", self.buffer, 2)[0]
            self.header_lengths[header_length] = (
                self.header_lengths.get(header_length, 0) + 1
            )
            payload_offset = max(header_length, HEADER_LENGTH)
            if len(self.buffer) < payload_offset:
                break
            message_type = struct.unpack_from("<H", self.buffer, 4)[0]
            message_length = struct.unpack_from("<I", self.buffer, 24)[0]
            total_length = payload_offset + message_length
            if len(self.buffer) < total_length:
                break
            payload = bytes(self.buffer[payload_offset:total_length])
            del self.buffer[:total_length]
            self.message_counts[message_type] = (
                self.message_counts.get(message_type, 0) + 1
            )

            if message_type == MESSAGE_INTERPRETATION:
                self._parse_interpretation(payload)
            elif message_type == MESSAGE_SIGNAL_DATA:
                samples.extend(self._parse_signal_data(payload))
            elif message_type == MESSAGE_DATA_QUALITY:
                pass
        return samples

    def _parse_interpretation(self, payload: bytes) -> None:
        pos = 0
        while pos + 8 <= len(payload):
            signal_id, descriptor, _, value_length = struct.unpack_from(
                "<HHHH", payload, pos
            )
            pos += 8
            end = pos + value_length
            if end > len(payload):
                return
            meta = self.meta.setdefault(signal_id, SignalMeta())
            value = payload[pos:end]
            if descriptor == DESCRIPTOR_SCALE_FACTOR and value_length >= 8:
                meta.scale_factor = struct.unpack_from("<d", value, 0)[0]
            elif descriptor == DESCRIPTOR_OFFSET and value_length >= 8:
                meta.offset = struct.unpack_from("<d", value, 0)[0]
            elif descriptor == DESCRIPTOR_PERIOD_TIME and value_length >= 12:
                k, l, m, n = struct.unpack_from("<BBBB", value, 0)
                count = struct.unpack_from("<Q", value, 4)[0]
                meta.period = calc_time_from_family(k, l, m, n, count)
            elif descriptor == DESCRIPTOR_UNIT and value_length >= 2:
                count = struct.unpack_from("<H", value, 0)[0]
                meta.unit = value[2:2 + count].decode("utf-8", errors="replace")
            pos = end
            pos += (4 - (pos % 4)) & 3

    def _parse_signal_data(self, payload: bytes) -> list[list[float]]:
        if len(payload) < 4:
            return []
        number_of_signals = struct.unpack_from("<H", payload, 0)[0]
        pos = 4
        vectors: list[list[float]] = []
        for _ in range(number_of_signals):
            if pos + 4 > len(payload):
                return []
            signal_id, number_of_values = struct.unpack_from("<hH", payload, pos)
            pos += 4
            value_bytes = number_of_values * 3
            if pos + value_bytes > len(payload):
                return []
            if number_of_values == 0:
                continue

            meta = self.meta.setdefault(signal_id, SignalMeta())
            encoded = np.frombuffer(
                payload, dtype=np.uint8, count=value_bytes, offset=pos
            ).reshape(number_of_values, 3)
            raw = (
                encoded[:, 0].astype(np.int32)
                | (encoded[:, 1].astype(np.int32) << 8)
                | (encoded[:, 2].astype(np.int32) << 16)
            )
            raw[raw >= 0x800000] -= 0x1000000
            mean_raw = float(np.mean(raw))
            self.latest_signal_values[signal_id] = (
                mean_raw * meta.scale_factor / SIGNED_24BIT_FULL_SCALE
                + meta.offset
            )
            self.updated_signals.add(signal_id)
            self.last_block_sizes[signal_id] = number_of_values
            self.signal_counts[signal_id] = (
                self.signal_counts.get(signal_id, 0) + number_of_values
            )
            pos += value_bytes

            if signal_id not in self.signal_order:
                self.signal_order.append(signal_id)
                self.signal_order.sort()

        ordered = self.signal_order[:self.channel_count]
        if len(ordered) < self.channel_count:
            return []
        if all(signal_id in self.updated_signals for signal_id in ordered):
            vectors.append([
                self.latest_signal_values[signal_id] for signal_id in ordered
            ])
            self.updated_signals.difference_update(ordered)
            self.vector_count += 1
        return vectors


class LanXiStreamClient:
    def __init__(
        self,
        host: str,
        input_filter: str = "DC",
        transducer_detection: bool = False,
        request_timeout: float = 5.0,
    ) -> None:
        self.host = host
        self.input_filter = input_filter
        self.transducer_detection = transducer_detection
        self.request_timeout = request_timeout
        self.session = requests.Session()
        self.base_url = self._base_url(host)
        self.sock: socket.socket | None = None
        self.configured_filters: list[str] = []
        self._owns_recorder = False

    @staticmethod
    def _base_url(host: str) -> str:
        ip_addr = host.split("%")[0]
        family = socket.getaddrinfo(ip_addr, port=0)[0][0]
        if family == socket.AF_INET6:
            return f"http://[{host}]/rest/rec"
        return f"http://{host}/rest/rec"

    def _request(self, method: str, path: str, **kwargs):
        kwargs.setdefault("timeout", self.request_timeout)
        response = self.session.request(method, f"{self.base_url}{path}", **kwargs)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            message = (
                f"LAN-XI {method} {path} failed with HTTP "
                f"{response.status_code} {response.reason}"
            )
            if detail:
                message += f": {detail}"
            raise RuntimeError(message) from exc
        return response

    def _status(self) -> dict:
        return self._request("GET", "/onchange").json()

    def _wait_for_state(self, expected: str, timeout: float = 10.0) -> dict:
        deadline = time.monotonic() + timeout
        last_state = "Unknown"
        while time.monotonic() < deadline:
            status = self._status()
            last_state = status.get("moduleState", "Unknown")
            if last_state == expected:
                return status
            time.sleep(0.1)
        raise RuntimeError(
            f"LAN-XI did not enter {expected} within {timeout:g} seconds "
            f"(current moduleState={last_state})"
        )

    def _request_transition(self, method: str, path: str, expected: str, **kwargs):
        try:
            response = self._request(method, path, **kwargs)
        except RuntimeError as exc:
            try:
                state = self._status().get("moduleState", "Unknown")
            except Exception:
                state = "Unavailable"
            raise RuntimeError(f"{exc}; moduleState={state}") from exc
        self._wait_for_state(expected)
        return response

    def open(self) -> None:
        initial_state = self._status().get("moduleState", "Unknown")
        if initial_state != "Idle":
            raise RuntimeError(
                "LAN-XI recorder is already in use "
                f"(moduleState={initial_state}). Stop the other recorder client "
                "or recording before starting this node."
            )
        self._request("PUT", "/module/time", data=str(int(time.time() * 1000)))
        try:
            self._request(
                "PUT",
                "/open",
                json={"performTransducerDetection": self.transducer_detection},
            )
        except RuntimeError as exc:
            try:
                state = self._status().get("moduleState", "Unknown")
            except Exception:
                state = "Unavailable"
            raise RuntimeError(f"{exc}; moduleState={state}") from exc
        self._owns_recorder = True
        self._wait_for_state("RecorderOpened")
        self._request_transition("PUT", "/create", "RecorderConfiguring")
        setup = self._request("GET", "/channels/input/default").json()
        configure_input_channels(setup, self.input_filter)
        self._request_transition(
            "PUT", "/channels/input", "RecorderStreaming", json=setup
        )
        applied_setup = self._request("GET", "/channels/input").json()
        self.configured_filters = [
            channel.get("filter", "")
            for channel in applied_setup.get("channels", [])
            if channel.get("enabled", False)
        ]
        if not self.configured_filters or any(
            value != self.input_filter for value in self.configured_filters
        ):
            raise RuntimeError(
                "LAN-XI did not apply the requested input filter "
                f"{self.input_filter!r}; reported filters="
                f"{self.configured_filters}"
            )
        port = self._request("GET", "/destination/socket").json()["tcpPort"]
        self.sock = socket.create_connection((self.host, port), timeout=10)
        self.sock.settimeout(0.5)
        self._request_transition(
            "POST", "/measurements", "RecorderRecording"
        )

    def read(self, size: int = 16384) -> bytes:
        if self.sock is None:
            raise RuntimeError("LAN-XI socket is not open")
        return self.sock.recv(size)

    def close(self) -> None:
        if self._owns_recorder:
            try:
                state = self._status().get("moduleState")
                if state == "RecorderRecording":
                    self._request_transition(
                        "PUT", "/measurements/stop", "RecorderStreaming"
                    )
            except Exception:
                pass
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        if self._owns_recorder:
            try:
                state = self._status().get("moduleState")
                if state == "RecorderConfiguring":
                    self._request_transition("PUT", "/cancel", "RecorderOpened")
                elif state == "RecorderStreaming":
                    self._request_transition("PUT", "/finish", "RecorderOpened")
                state = self._status().get("moduleState")
                if state == "RecorderOpened":
                    self._request_transition("PUT", "/close", "Idle")
            except Exception:
                pass
            finally:
                self._owns_recorder = False


class LoadcellPublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("lanxi_loadcell_publisher", namespace=args.vehicle_name)
        self.args = args
        self.frame_id = args.frame_id
        self.parser = OpenApiStreamParser(channel_count=6)
        self.client = LanXiStreamClient(
            args.daq_ip,
            input_filter=args.input_filter,
            transducer_detection=args.teds,
        )
        self.calibration = (
            load_calibration(args.calibration_file)
            if args.calibration_file
            else None
        )
        low_latency_qos = QoSProfile(depth=1)
        self.wrench_pub = self.create_publisher(
            WrenchStamped, args.wrench_topic, low_latency_qos
        )
        self.array_pub = self.create_publisher(
            Float64MultiArray, args.array_topic, low_latency_qos
        )
        self.runtime_tare = RuntimeWrenchTare(args.tare_window_sec)
        self.tare_service = self.create_service(
            Trigger, args.tare_service, self._tare_wrench
        )
        self.reset_tare_service = self.create_service(
            Trigger, args.reset_tare_service, self._reset_wrench_tare
        )
        self.parser_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.started_monotonic = time.monotonic()
        self.last_debug_monotonic = self.started_monotonic
        self.last_debug_parse_seconds = 0.0
        self.total_parse_seconds = 0.0
        self.max_parse_seconds = 0.0
        self.last_publish_age_ms = 0.0
        self.published_vectors = 0
        self.discarded_vectors = 0
        self.running = True
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.debug_timer = self.create_timer(args.debug_period, self._log_debug)

    def start(self) -> None:
        self.get_logger().info(f"Connecting to LAN-XI at {self.args.daq_ip}")
        self.client.open()
        self.reader_thread.start()
        self.get_logger().info(
            f"Publishing loadcell data on {self.wrench_pub.topic_name} and "
            f"{self.array_pub.topic_name}"
        )
        self.get_logger().info(
            f"LAN-XI input filters: {self.client.configured_filters}"
            + (" (high-pass disabled)" if self.args.input_filter == "DC" else "")
        )
        if self.calibration is None:
            self.get_logger().warning(
                "No calibration matrix supplied. Publishing raw strain channels "
                "only; WrenchStamped will not be published."
            )
        else:
            self.get_logger().info(
                f"Calibration file: {self.calibration.source_path}"
            )
            self.get_logger().info(
                "Calibration offsets: "
                f"input_bias={self.calibration.input_bias}, "
                f"wrench_tare={self.calibration.wrench_tare}"
            )
            self.get_logger().info(
                f"Runtime tare services: {self.tare_service.service_name} and "
                f"{self.reset_tare_service.service_name} "
                f"(window={self.args.tare_window_sec:.3f} s)"
            )

    def stop(self) -> None:
        self.running = False
        self.client.close()
        if self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2.0)

    def _read_loop(self) -> None:
        while self.running and rclpy.ok():
            try:
                data = self.client.read(262144)
            except socket.timeout:
                continue
            except OSError as exc:
                if self.running:
                    self.get_logger().error(f"LAN-XI socket error: {exc}")
                break
            if not data:
                if self.running:
                    self.get_logger().warning("LAN-XI stream socket closed")
                break
            received_monotonic = time.monotonic()
            try:
                with self.parser_lock:
                    vectors = self.parser.feed(data)
            except Exception as exc:
                self.get_logger().error(f"LAN-XI stream parse error: {exc}")
                break
            parse_seconds = time.monotonic() - received_monotonic
            with self.stats_lock:
                self.total_parse_seconds += parse_seconds
                self.max_parse_seconds = max(
                    self.max_parse_seconds, parse_seconds
                )
                if len(vectors) > 1:
                    self.discarded_vectors += len(vectors) - 1
            if vectors:
                self._publish_values(vectors[-1], received_monotonic)

    def _publish_values(
        self, values: list[float], received_monotonic: float
    ) -> None:
        stamp = self.get_clock().now().to_msg()

        if self.calibration is not None:
            calibrated_values = self.calibration.apply(values)
            self.runtime_tare.add(received_monotonic, calibrated_values)
            wrench_values = self.runtime_tare.apply(calibrated_values)
            wrench = WrenchStamped()
            wrench.header.stamp = stamp
            wrench.header.frame_id = self.frame_id
            wrench.wrench.force.x = wrench_values[0]
            wrench.wrench.force.y = wrench_values[1]
            wrench.wrench.force.z = wrench_values[2]
            wrench.wrench.torque.x = wrench_values[3]
            wrench.wrench.torque.y = wrench_values[4]
            wrench.wrench.torque.z = wrench_values[5]
            self.wrench_pub.publish(wrench)

        array = Float64MultiArray()
        array.layout.dim = [
            MultiArrayDimension(
                label="raw_strain_channels",
                size=len(values),
                stride=len(values),
            )
        ]
        array.data = values
        self.array_pub.publish(array)
        with self.stats_lock:
            self.published_vectors += 1
            self.last_publish_age_ms = (
                time.monotonic() - received_monotonic
            ) * 1000.0

    def _tare_wrench(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        if self.calibration is None:
            response.success = False
            response.message = "Cannot tare without a calibration matrix."
            return response

        offset, sample_count = self.runtime_tare.set_from_recent(time.monotonic())
        if offset is None:
            response.success = False
            response.message = (
                "No recent calibrated wrench samples are available; verify "
                "that the LAN-XI stream is running."
            )
            return response

        response.success = True
        response.message = (
            f"Runtime wrench tare set from {sample_count} samples: {offset}"
        )
        self.get_logger().info(response.message)
        return response

    def _reset_wrench_tare(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        self.runtime_tare.reset()
        response.success = True
        response.message = "Runtime wrench tare reset; using YAML tare only."
        self.get_logger().info(response.message)
        return response

    def _log_debug(self) -> None:
        with self.parser_lock:
            signal_ids = self.parser.signal_order.copy()
            units = {
                signal_id: self.parser.meta.get(signal_id, SignalMeta()).unit
                for signal_id in signal_ids
            }
            sample_rates = {
                signal_id: (
                    1.0 / self.parser.meta[signal_id].period
                    if self.parser.meta.get(signal_id, SignalMeta()).period
                    else None
                )
                for signal_id in signal_ids
            }
            total_bytes = self.parser.total_bytes
            header_lengths = self.parser.header_lengths.copy()
            message_counts = self.parser.message_counts.copy()
            signal_counts = self.parser.signal_counts.copy()
            block_sizes = self.parser.last_block_sizes.copy()
            bad_magic_count = self.parser.bad_magic_count
            parser_buffer_bytes = len(self.parser.buffer)
        now = time.monotonic()
        elapsed = max(now - self.started_monotonic, 1e-9)
        interval = max(now - self.last_debug_monotonic, 1e-9)
        with self.stats_lock:
            interval_parse_seconds = (
                self.total_parse_seconds - self.last_debug_parse_seconds
            )
            parse_load_interval = 100.0 * interval_parse_seconds / interval
            parse_load_total = 100.0 * self.total_parse_seconds / elapsed
            max_parse_ms = self.max_parse_seconds * 1000.0
            publish_age_ms = self.last_publish_age_ms
            published = self.published_vectors
            discarded = self.discarded_vectors
            self.last_debug_parse_seconds = self.total_parse_seconds
            self.last_debug_monotonic = now
        self.get_logger().info(
            "LAN-XI stream: "
            f"bytes={total_bytes}, "
            f"header_lengths={header_lengths}, "
            f"messages={message_counts}, "
            f"signals={signal_ids}, "
            f"samples={signal_counts}, "
            f"blocks={block_sizes}, "
            f"units={units}, "
            f"sample_rates={sample_rates}, "
            f"published={published}, "
            f"discarded={discarded}, "
            f"publish_age_ms={publish_age_ms:.2f}, "
            f"parse_load_interval={parse_load_interval:.1f}%, "
            f"parse_load_total={parse_load_total:.1f}%, "
            f"max_parse_ms={max_parse_ms:.2f}, "
            f"parser_buffer_bytes={parser_buffer_bytes}, "
            f"bad_magic={bad_magic_count}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish six LAN-XI input channels as loadcell ROS topics."
    )
    parser.add_argument(
        "--vehicle-name",
        "--namespace",
        dest="vehicle_name",
        default="hippo02",
        help="vehicle namespace for loadcell topics and services",
    )
    parser.add_argument("--daq-ip", default="192.168.0.139")
    parser.add_argument("--wrench-topic", default="loadcell/wrench")
    parser.add_argument("--array-topic", default="loadcell/channels")
    parser.add_argument("--tare-service", default="loadcell/tare")
    parser.add_argument("--reset-tare-service", default="loadcell/reset_tare")
    parser.add_argument(
        "--tare-window-sec",
        type=float,
        default=0.5,
        help="recent calibrated-wrench averaging window used by the tare service",
    )
    parser.add_argument("--frame-id", default="loadcell")
    parser.add_argument(
        "--input-filter",
        choices=INPUT_FILTERS,
        default="DC",
        help="LAN-XI channel high-pass filter; DC disables high-pass filtering",
    )
    parser.add_argument(
        "--calibration-file",
        default="",
        help=(
            "JSON/YAML file with a 6x6 'matrix' and optional six-value "
            "'input_bias' and 'wrench_tare' vectors. Wrench output is disabled "
            "when omitted."
        ),
    )
    parser.add_argument("--debug-period", type=float, default=2.0)
    parser.add_argument("--teds", action="store_true", help="run TEDS detection")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(remove_ros_args(args=sys.argv)[1:])
    if args.debug_period <= 0.0:
        parser.error("--debug-period must be positive")
    if args.tare_window_sec <= 0.0:
        parser.error("--tare-window-sec must be positive")
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = LoadcellPublisher(args)
    try:
        node.start()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
