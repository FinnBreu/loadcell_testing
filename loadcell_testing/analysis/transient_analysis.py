"""Extract normalized thruster step responses and identify dynamic models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, sosfilt, sosfreqz

from .analysis_config import AnalysisConfig
from .thruster_analysis import (
    CurveFit,
    RunSeries,
    _command_runs,
    _find_csv,
    _read_columns,
    virtual_command_from_actual,
)


@dataclass(frozen=True)
class TransientCurve:
    run_id: str
    step_index: int
    transition: str
    esc_command: float
    virtual_command: float
    steady_thrust_n: float
    baseline_thrust_n: float
    time_s: np.ndarray
    thrust_unit: np.ndarray


@dataclass(frozen=True)
class DynamicFit:
    model: str
    parameters: dict[str, float]
    loadcell_lowpass_hz: float | None
    predicted_start_peak: float
    start_peak_rmse: float
    rmse: float
    start_rmse: float
    stop_rmse: float
    r_squared: float


def first_order_step(time_s: np.ndarray, tau_s: float, delay_s: float) -> np.ndarray:
    elapsed = np.maximum(np.asarray(time_s, dtype=float) - delay_s, 0.0)
    return -np.expm1(-elapsed / tau_s)


def second_order_step(
    time_s: np.ndarray, natural_frequency_rad_s: float, damping_ratio: float,
    delay_s: float,
) -> np.ndarray:
    elapsed = np.maximum(np.asarray(time_s, dtype=float) - delay_s, 0.0)
    omega = natural_frequency_rad_s
    zeta = damping_ratio
    if zeta < 1.0 - 1e-5:
        root = np.sqrt(1.0 - zeta * zeta)
        phase = omega * root * elapsed
        response = 1.0 - np.exp(-zeta * omega * elapsed) * (
            np.cos(phase) + zeta / root * np.sin(phase)
        )
    elif zeta > 1.0 + 1e-5:
        root = np.sqrt(zeta * zeta - 1.0)
        slow = -omega * (zeta - root)
        fast = -omega * (zeta + root)
        response = 1.0 + (
            fast * np.exp(slow * elapsed) - slow * np.exp(fast * elapsed)
        ) / (slow - fast)
    else:
        response = 1.0 - np.exp(-omega * elapsed) * (1.0 + omega * elapsed)
    return response


FILTER_SIM_RATE_HZ = 1000


def inverse_loadcell_filter(
    time_s: np.ndarray, values: np.ndarray, cutoff_hz: float = 7.5,
    regularization: float = 0.02,
) -> np.ndarray:
    """Estimate the prefilter trace within the recorded 25 Hz bandwidth.

    Invert the publisher's causal low-pass response on a uniform 25 Hz grid.
    Regularization limits noise gain where the filter response is small.
    Constant end padding keeps the FFT's periodic boundary away from the step.
    """
    times = np.asarray(time_s, dtype=float)
    samples = np.asarray(values, dtype=float)
    if times.ndim != 1 or samples.shape != times.shape or len(times) < 3:
        raise ValueError("time and values must be matching one-dimensional traces")
    if not np.all(np.diff(times) > 0):
        raise ValueError("time must increase strictly")
    if not 0.0 < cutoff_hz < 12.5 or not 0.0 < regularization < 1.0:
        raise ValueError("invalid cutoff or inverse regularization")

    # ROS timestamps have arrival jitter. Evaluate the measured trace on the
    # publisher's nominal output grid, then return to the original timestamps.
    step_s = 1.0 / 25.0
    grid = np.arange(times[0], times[-1] + step_s / 2.0, step_s)
    uniform = np.interp(grid, times, samples)
    pad = 50  # 2 seconds on each side, well beyond the inverse impulse tail.
    padded = np.pad(uniform, pad, mode="edge")
    frequencies = np.fft.rfftfreq(len(padded), step_s)
    sos = butter(8, cutoff_hz, btype="lowpass",
                 fs=FILTER_SIM_RATE_HZ, output="sos")
    _, response = sosfreqz(sos, worN=frequencies,
                           fs=FILTER_SIM_RATE_HZ)
    inverse = np.conj(response) / (np.abs(response) ** 2 + regularization)
    inverse /= inverse[0].real  # Preserve constant baselines and step levels.
    estimate = np.fft.irfft(np.fft.rfft(padded) * inverse, len(padded))
    return np.interp(times, grid, estimate[pad:-pad])


def _ideal_step(model: str, time_s: np.ndarray,
                parameters: dict[str, float]) -> np.ndarray:
    if model == "first_order_no_delay":
        return first_order_step(time_s, parameters["tau_s"], 0.0)
    if model == "first_order":
        return first_order_step(time_s, parameters["tau_s"], parameters["delay_s"])
    if model == "second_order":
        return second_order_step(
            time_s, parameters["natural_frequency_rad_s"],
            parameters["damping_ratio"], parameters["delay_s"],
        )
    raise ValueError(f"unknown dynamic model: {model}")


def model_response(model: str, time_s: np.ndarray, transition: str,
                   parameters: dict[str, float],
                   loadcell_lowpass_hz: float | None = None) -> np.ndarray:
    times = np.asarray(time_s, dtype=float)
    if loadcell_lowpass_hz is None:
        step = _ideal_step(model, times, parameters)
    else:
        if not 0.0 < loadcell_lowpass_hz < 12.5:
            raise ValueError("loadcell low-pass cutoff must be between 0 and 12.5 Hz")
        # Match the publisher's causal eighth-order SOS filter at high sample rate.
        # Evaluate at the recorded 25 Hz timestamps, so the observed sampling
        # schedule is retained without inventing an acquisition phase.
        dt = 1.0 / FILTER_SIM_RATE_HZ
        start = min(-0.4, float(np.min(times)) - dt)
        stop = float(np.max(times)) + dt
        grid = np.arange(start, stop + dt, dt)
        sos = butter(8, loadcell_lowpass_hz, btype="lowpass",
                     fs=FILTER_SIM_RATE_HZ, output="sos")
        filtered = sosfilt(sos, _ideal_step(model, grid, parameters))
        step = np.interp(times, grid, filtered)
    return step if transition == "start" else 1.0 - step


def extract_transients(
    runs: list[RunSeries], config: AnalysisConfig, virtual_fit: CurveFit,
    start_horizon_s: float = 2.0, stop_horizon_s: float = 1.0,
) -> list[TransientCurve]:
    """Use original timestamps and normalize each step from its initial thrust."""
    curves: list[TransientCurve] = []
    before_ns = 350_000_000
    for run in runs:
        vehicle = "uuv02"  # _find_csv also accepts the single matching suffix.
        command = _read_columns(
            _find_csv(run.run_dir, vehicle, "thruster_command"),
            ["t_ros_ns", "control.0"],
        )
        force_column = "wrench." + config.thrust_component.replace("_", ".")
        wrench = _read_columns(
            _find_csv(run.run_dir, vehicle, "loadcell_wrench"),
            ["t_ros_ns", force_column],
        )
        intervals = _command_runs(command["t_ros_ns"], command["control.0"])
        interval_by_start = {item.start_ns: index for index, item in enumerate(intervals)}
        times_ns = wrench["t_ros_ns"]
        thrust = (
            config.thrust_sign * run.stand_sign * run.propeller_thrust_sign
            * wrench[force_column]
        )
        for step_index, step in enumerate(run.steps):
            index = interval_by_start.get(step.command_start_ns)
            if index is None or index + 1 >= len(intervals):
                raise ValueError(f"{run.run_id}: command interval missing for step {step_index}")
            interval = intervals[index]
            zero = intervals[index + 1]
            previous_zero = intervals[index - 1] if index > 0 else None
            if not np.isclose(zero.speed, 0.0, atol=1e-9):
                raise ValueError(f"{run.run_id}: step {step_index} has no stop interval")
            if previous_zero is not None and not np.isclose(previous_zero.speed, 0.0, atol=1e-9):
                raise ValueError(f"{run.run_id}: step {step_index} has no pre-step zero interval")
            start_sample = int(np.searchsorted(times_ns, interval.start_ns))
            if start_sample >= len(times_ns) or times_ns[start_sample] >= interval.end_ns:
                raise ValueError(f"{run.run_id}: no initial thrust sample for step {step_index}")
            baseline = float(thrust[start_sample])
            steady = float(run.thrust[step_index])
            amplitude = steady - baseline
            if abs(amplitude) < 0.05:
                raise ValueError(f"{run.run_id}: step {step_index} has near-zero thrust amplitude")
            virtual = float(virtual_command_from_actual(
                np.asarray([step.speed]), virtual_fit,
            )[0])
            for transition, event_ns, end_ns in (
                ("start", interval.start_ns, min(
                    interval.end_ns, interval.start_ns + round(start_horizon_s * 1e9),
                )),
                ("stop", zero.start_ns, min(
                    zero.end_ns, zero.start_ns + round(stop_horizon_s * 1e9),
                )),
            ):
                mask = (times_ns >= event_ns - before_ns) & (times_ns < end_ns)
                if np.count_nonzero(mask & (times_ns >= event_ns)) < 3:
                    raise ValueError(f"{run.run_id}: insufficient {transition} samples at step {step_index}")
                curves.append(TransientCurve(
                    run_id=run.run_id,
                    step_index=step_index,
                    transition=transition,
                    esc_command=float(step.speed),
                    virtual_command=virtual,
                    steady_thrust_n=steady,
                    baseline_thrust_n=baseline,
                    time_s=(times_ns[mask] - event_ns) * 1e-9,
                    thrust_unit=(thrust[mask] - baseline) / amplitude,
                ))
    return curves


def fit_dynamic_model(curves: list[TransientCurve], model: str,
                      loadcell_lowpass_hz: float | None = None) -> DynamicFit:
    if not curves:
        raise ValueError("at least one transient is required")
    if model == "first_order_no_delay":
        names = ("tau_s",)
        initial = (0.25,)
        lower = (0.01,)
        upper = (3.0,)
    elif model == "first_order":
        names = ("tau_s", "delay_s")
        initial = (0.25, 0.05)
        lower = (0.01, 0.0)
        upper = (3.0, 0.5)
    elif model == "second_order":
        names = ("natural_frequency_rad_s", "damping_ratio", "delay_s")
        initial = (8.0, 0.8, 0.05)
        lower = (0.5, 0.05, 0.0)
        upper = (100.0, 3.0, 0.5)
    else:
        raise ValueError(f"unknown dynamic model: {model}")

    fit_samples = [curve.time_s >= 0.0 for curve in curves]
    lengths = [int(np.count_nonzero(mask)) for mask in fit_samples]
    times = np.concatenate([curve.time_s[mask] for curve, mask in zip(curves, fit_samples)])
    actual = np.concatenate([curve.thrust_unit[mask] for curve, mask in zip(curves, fit_samples)])
    weights = np.concatenate([
        np.full(length, 1.0 / np.sqrt(length)) for length in lengths
    ])
    start_peaks = np.asarray([
        float(np.max(curve.thrust_unit[mask]))
        for curve, mask in zip(curves, fit_samples, strict=True)
        if curve.transition == "start"
    ])
    peak_times = np.linspace(0.0, 1.0, 1001)

    def predicted_start_peak(parameters: dict[str, float]) -> float:
        return float(np.max(model_response(
            model, peak_times, "start", parameters, loadcell_lowpass_hz,
        )))

    def prediction(parameters: dict[str, float]) -> np.ndarray:
        # The normalized start and stop share one step shape. Evaluate it once
        # per optimizer iteration rather than once for every measured curve.
        step = model_response(model, times, "start", parameters, loadcell_lowpass_hz)
        offset = 0
        for curve, length in zip(curves, lengths, strict=True):
            if curve.transition == "stop":
                step[offset:offset + length] = 1.0 - step[offset:offset + length]
            offset += length
        return step

    def residual(values: np.ndarray) -> np.ndarray:
        parameters = dict(zip(names, values, strict=True))
        return (prediction(parameters) - actual) * weights

    result = least_squares(
        residual, initial, bounds=(lower, upper), loss="soft_l1", f_scale=0.1,
    )
    if not result.success:
        raise RuntimeError(f"{model} optimization failed: {result.message}")
    parameters = dict(zip(names, (float(x) for x in result.x), strict=True))
    peak = predicted_start_peak(parameters)
    predicted = prediction(parameters)
    squared: dict[str, list[np.ndarray]] = {"start": [], "stop": []}
    offset = 0
    for curve, length in zip(curves, lengths, strict=True):
        squared[curve.transition].append(
            (predicted[offset:offset + length] - actual[offset:offset + length]) ** 2
        )
        offset += length
    all_squared = np.concatenate(squared["start"] + squared["stop"])
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    return DynamicFit(
        model=model,
        parameters=parameters,
        loadcell_lowpass_hz=loadcell_lowpass_hz,
        predicted_start_peak=peak,
        start_peak_rmse=(float(np.sqrt(np.mean((peak - start_peaks) ** 2)))
                         if start_peaks.size else float("nan")),
        rmse=float(np.sqrt(np.mean(all_squared))),
        start_rmse=float(np.sqrt(np.mean(np.concatenate(squared["start"])))),
        stop_rmse=float(np.sqrt(np.mean(np.concatenate(squared["stop"])))),
        r_squared=1.0 - float(np.sum(all_squared)) / total,
    )
