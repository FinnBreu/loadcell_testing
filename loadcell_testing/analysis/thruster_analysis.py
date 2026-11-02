"""Load, average, and fit thruster sweep data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .analysis_config import AnalysisConfig


WRENCH_COLUMNS = {
    "force_x": "wrench.force.x",
    "force_y": "wrench.force.y",
    "force_z": "wrench.force.z",
    "torque_x": "wrench.torque.x",
    "torque_y": "wrench.torque.y",
    "torque_z": "wrench.torque.z",
}


@dataclass(frozen=True)
class CommandRun:
    speed: float
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class StepAverage:
    step_index: int
    repeat: int
    speed: float
    command_start_ns: int
    sample_start_ns: int
    sample_end_ns: int
    wrench_samples: int
    voltage_samples: int
    voltage: float
    force_x: float
    force_y: float
    force_z: float
    torque_x: float
    torque_y: float
    torque_z: float


@dataclass(frozen=True)
class CurveFit:
    coefficients: tuple[float, ...]
    r_squared: float


@dataclass(frozen=True)
class RunSeries:
    run_id: str
    run_dir: Path
    stand_direction: str
    steps: tuple[StepAverage, ...]
    speeds: np.ndarray
    thrust: np.ndarray
    roll_torque: np.ndarray
    motor_torques: np.ndarray
    propeller_rotation: str
    stand_sign: int
    propeller_thrust_sign: int = 1


@dataclass(frozen=True)
class AnalysisFits:
    thrust_proportional: CurveFit
    thrust_quadratic: CurveFit
    roll_torque_proportional: CurveFit
    roll_torque_quadratic: CurveFit
    roll_torque_per_thrust: CurveFit
    virtual_command: CurveFit
    thrust_coefficient: float
    torque_coefficient: float


def _load_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"run manifest not found: {path}")
    with path.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream) or {}
    if not isinstance(manifest.get("metadata"), dict):
        raise ValueError("manifest does not contain sweep metadata")
    return manifest


def _find_csv(run_dir: Path, vehicle_name: str, suffix: str) -> Path:
    csv_dir = run_dir / "csv"
    expected = csv_dir / f"{vehicle_name}_{suffix}.csv"
    if expected.is_file():
        return expected
    matches = sorted(csv_dir.glob(f"*_{suffix}.csv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"missing {suffix} CSV in {csv_dir}; run tools export_csv first"
        )
    raise ValueError(f"multiple {suffix} CSV files found in {csv_dir}")


def _read_columns(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    values = {column: [] for column in columns}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [column for column in columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        for row in reader:
            for column in columns:
                converter = int if column == "t_ros_ns" else float
                values[column].append(converter(row[column]))
    if not values[columns[0]]:
        raise ValueError(f"{path} contains no data rows")
    return {
        column: np.asarray(
            entries,
            dtype=np.int64 if column == "t_ros_ns" else np.float64,
        )
        for column, entries in values.items()
    }


def _command_runs(times_ns: np.ndarray, speeds: np.ndarray) -> list[CommandRun]:
    order = np.argsort(times_ns)
    times_ns = times_ns[order].astype(np.int64)
    speeds = speeds[order]
    runs: list[CommandRun] = []
    start = 0
    for index in range(1, len(speeds)):
        if not np.isclose(speeds[index], speeds[start], atol=1e-9, rtol=0.0):
            runs.append(
                CommandRun(
                    float(speeds[start]),
                    int(times_ns[start]),
                    int(times_ns[index]),
                )
            )
            start = index
    period_ns = int(np.median(np.diff(times_ns))) if len(times_ns) > 1 else 0
    runs.append(
        CommandRun(
            float(speeds[start]),
            int(times_ns[start]),
            int(times_ns[-1]) + max(period_ns, 1),
        )
    )
    return runs


def _match_steps(
    runs: list[CommandRun],
    expected_speeds: list[float],
    required_duration_sec: float,
    command_period_sec: float,
) -> list[CommandRun]:
    matched: list[CommandRun] = []
    cursor = 0
    tolerance = max(2.0 * command_period_sec, 0.05)
    for step_index, expected_speed in enumerate(expected_speeds):
        while cursor < len(runs):
            candidate = runs[cursor]
            cursor += 1
            duration = (candidate.end_ns - candidate.start_ns) * 1e-9
            if (
                np.isclose(candidate.speed, expected_speed, atol=1e-6, rtol=0.0)
                and duration + tolerance >= required_duration_sec
            ):
                matched.append(candidate)
                break
        else:
            raise ValueError(
                f"could not find command interval for step {step_index} "
                f"at speed {expected_speed:+.6g}"
            )
    return matched


def _window_mean(
    data: dict[str, np.ndarray],
    columns: list[str],
    start_ns: int,
    end_ns: int,
    label: str,
) -> tuple[dict[str, float], int]:
    selected = (data["t_ros_ns"] >= start_ns) & (data["t_ros_ns"] < end_ns)
    count = int(np.count_nonzero(selected))
    if count == 0:
        raise ValueError(f"no {label} samples in {start_ns} <= t < {end_ns}")
    return (
        {column: float(np.mean(data[column][selected])) for column in columns},
        count,
    )


def load_step_averages(run: str | Path) -> tuple[dict, list[StepAverage]]:
    run_dir = Path(run).expanduser().resolve()
    manifest = _load_manifest(run_dir)
    metadata = manifest["metadata"]
    vehicle_name = str(manifest.get("vehicle_name", "uuv02")).strip("/")
    speeds = [float(speed) for speed in metadata["speeds"]]
    expected_speeds = speeds * int(metadata.get("repeat", 1))
    settle_sec = float(metadata["settle_sec"])
    sample_sec = float(metadata["sample_sec"])
    rate_hz = float(metadata.get("rate_hz", 50.0))

    command = _read_columns(
        _find_csv(run_dir, vehicle_name, "thruster_command"),
        ["t_ros_ns", "control.0"],
    )
    wrench_fields = list(WRENCH_COLUMNS.values())
    wrench = _read_columns(
        _find_csv(run_dir, vehicle_name, "loadcell_wrench"),
        ["t_ros_ns", *wrench_fields],
    )
    battery = _read_columns(
        _find_csv(run_dir, vehicle_name, "battery_state"),
        ["t_ros_ns", "voltage"],
    )
    matched = _match_steps(
        _command_runs(command["t_ros_ns"], command["control.0"]),
        expected_speeds,
        settle_sec + sample_sec,
        1.0 / rate_hz,
    )

    settle_ns = round(settle_sec * 1e9)
    sample_ns = round(sample_sec * 1e9)
    steps: list[StepAverage] = []
    for step_index, (speed, command_run) in enumerate(
        zip(expected_speeds, matched, strict=True)
    ):
        sample_start_ns = command_run.start_ns + settle_ns
        sample_end_ns = sample_start_ns + sample_ns
        wrench_mean, wrench_count = _window_mean(
            wrench, wrench_fields, sample_start_ns, sample_end_ns, "wrench"
        )
        voltage_mean, voltage_count = _window_mean(
            battery, ["voltage"], sample_start_ns, sample_end_ns, "battery voltage"
        )
        steps.append(
            StepAverage(
                step_index=step_index,
                repeat=step_index // len(speeds),
                speed=speed,
                command_start_ns=command_run.start_ns,
                sample_start_ns=sample_start_ns,
                sample_end_ns=sample_end_ns,
                wrench_samples=wrench_count,
                voltage_samples=voltage_count,
                voltage=voltage_mean["voltage"],
                **{
                    name: wrench_mean[column]
                    for name, column in WRENCH_COLUMNS.items()
                },
            )
        )
    return manifest, steps


def _r_squared(y: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 if total == 0.0 else 1.0 - residual / total


def fit_origin_linear(x: np.ndarray, y: np.ndarray) -> CurveFit:
    if len(np.unique(x)) < 2 or not np.any(x != 0.0):
        raise ValueError("a linear fit requires at least two distinct values")
    slope = float(np.linalg.lstsq(x[:, np.newaxis], y, rcond=None)[0][0])
    return CurveFit((slope,), _r_squared(y, slope * x))


def evaluate_symmetric_quadratic(
    coefficients: tuple[float, ...], x: np.ndarray
) -> np.ndarray:
    quadratic, linear = coefficients
    return quadratic * x * np.abs(x) + linear * x


def fit_origin_symmetric_quadratic(x: np.ndarray, y: np.ndarray) -> CurveFit:
    positive = x > 0.0
    negative = x < 0.0
    if len(np.unique(x[positive])) < 2 or len(np.unique(x[negative])) < 2:
        raise ValueError(
            "a symmetric quadratic fit requires two nonzero values per direction"
        )
    coefficients = tuple(
        float(value)
        for value in np.linalg.lstsq(
            np.column_stack((x * np.abs(x), x)), y, rcond=None
        )[0]
    )
    return CurveFit(
        coefficients,
        _r_squared(y, evaluate_symmetric_quadratic(coefficients, x)),
    )


def virtual_command_from_actual(
    actual_command: np.ndarray, virtual_command_fit: CurveFit
) -> np.ndarray:
    return evaluate_symmetric_quadratic(
        virtual_command_fit.coefficients,
        np.asarray(actual_command, dtype=np.float64),
    )


def actual_command_from_virtual(
    virtual_command: np.ndarray, virtual_command_fit: CurveFit
) -> np.ndarray:
    quadratic, linear = virtual_command_fit.coefficients
    if linear <= 0.0 or linear + 2.0 * quadratic <= 0.0:
        raise ValueError("virtual-command model is not monotonic over [-1, 1]")
    virtual = np.asarray(virtual_command, dtype=np.float64)
    magnitude = np.abs(virtual)
    discriminant = linear**2 + 4.0 * quadratic * magnitude
    if np.any(discriminant < 0.0):
        raise ValueError("virtual-command model has no real inverse")
    return (
        np.sign(virtual)
        * 2.0
        * magnitude
        / (linear + np.sqrt(discriminant))
    )


def load_run_series(
    run: str | Path,
    config: AnalysisConfig,
) -> RunSeries:
    run_dir = Path(run).expanduser().resolve()
    manifest, steps = load_step_averages(run_dir)
    run_id = str(manifest.get("run_id", run_dir.name))
    name_tokens = set(run_id.casefold().replace("-", "_").split("_"))
    rotations = name_tokens & {"cw", "ccw"}
    directions = name_tokens & {"fwd", "rev"}
    if len(rotations) != 1 or len(directions) != 1:
        raise ValueError(
            f"run {run_id!r} must contain exactly one propeller marker "
            "(_cw or _ccw) and one stand-direction marker (_fwd or _rev)"
        )
    propeller_rotation = next(iter(rotations))
    stand_direction = next(iter(directions))
    propeller_thrust_sign = (
        config.propeller_signs[propeller_rotation]
        if config.propeller_signs is not None
        else 1
    )
    stand_sign = config.stand_signs[stand_direction]
    forces = np.asarray([[s.force_x, s.force_y, s.force_z] for s in steps])
    torques = np.asarray([[s.torque_x, s.torque_y, s.torque_z] for s in steps])
    # Fixed geometry makes translation commute with time averaging. Translate
    # in loadcell coordinates before applying the mounting sign.
    motor_torques = torques - np.cross(config.motor_position_m, forces)
    torque_axis = ("torque_x", "torque_y", "torque_z").index(config.roll_torque_component)
    return RunSeries(
        run_id=run_id,
        run_dir=run_dir,
        stand_direction=stand_direction,
        propeller_rotation=propeller_rotation,
        stand_sign=stand_sign,
        propeller_thrust_sign=propeller_thrust_sign,
        steps=tuple(steps),
        motor_torques=motor_torques,
        speeds=np.asarray([step.speed for step in steps]),
        thrust=config.thrust_sign
        * stand_sign
        * propeller_thrust_sign
        * np.asarray([getattr(step, config.thrust_component) for step in steps]),
        roll_torque=config.roll_torque_sign
        * stand_sign
        * motor_torques[:, torque_axis],
    )


def fit_run_series(
    runs: list[RunSeries],
) -> tuple[AnalysisFits, tuple[np.ndarray, ...]]:
    speeds = np.concatenate([run.speeds for run in runs])
    thrust = np.concatenate([run.thrust for run in runs])
    torque = np.concatenate([run.roll_torque for run in runs])
    if not np.any(speeds > 0.0) or not np.any(speeds < 0.0):
        raise ValueError("symmetric fits require positive and negative speeds")

    thrust_quadratic = fit_origin_symmetric_quadratic(speeds, thrust)
    thrust_coefficient = float(sum(thrust_quadratic.coefficients))
    if thrust_coefficient <= 0.0:
        raise ValueError("extrapolated thrust at command 1 must be positive")
    virtual_fit = CurveFit(
        tuple(value / thrust_coefficient for value in thrust_quadratic.coefficients),
        thrust_quadratic.r_squared,
    )
    virtual_commands = tuple(
        virtual_command_from_actual(run.speeds, virtual_fit) for run in runs
    )
    torque_fit = fit_origin_linear(np.concatenate(virtual_commands), torque)
    return (
        AnalysisFits(
            thrust_proportional=fit_origin_linear(speeds, thrust),
            thrust_quadratic=thrust_quadratic,
            roll_torque_proportional=fit_origin_linear(speeds, torque),
            roll_torque_quadratic=fit_origin_symmetric_quadratic(speeds, torque),
            roll_torque_per_thrust=fit_origin_linear(thrust, torque),
            virtual_command=virtual_fit,
            thrust_coefficient=thrust_coefficient,
            torque_coefficient=torque_fit.coefficients[0],
        ),
        virtual_commands,
    )
