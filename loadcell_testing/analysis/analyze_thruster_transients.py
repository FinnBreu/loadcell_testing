#!/usr/bin/env python3
"""Fit pooled start and stop dynamics from exported thruster sweeps."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml

from .analysis_config import load_analysis_config
from .analyze_thruster_run import _default_analysis_root
from .thruster_analysis import fit_run_series, load_run_series
from .transient_analysis import (
    FILTER_SIM_RATE_HZ, extract_transients, first_order_step, fit_dynamic_model,
    inverse_loadcell_filter, model_response,
)


def analyze_transients(
    run_dirs: list[str | Path], config_file: str | Path | None = None,
    output_dir: str | Path | None = None, show: bool = True,
    loadcell_lowpass_hz: float = 7.5,
    inverse_regularization: float = 0.02,
) -> dict:
    if not run_dirs:
        raise ValueError("at least one run is required")
    paths = [Path(run).expanduser().resolve() for run in run_dirs]
    if len(paths) != len(set(paths)):
        raise ValueError("the same run was supplied more than once")
    config = load_analysis_config(config_file)
    runs = [load_run_series(path, config) for path in paths]
    steady_fits, _ = fit_run_series(runs)
    curves = extract_transients(runs, config, steady_fits.virtual_command)
    corrected = [inverse_loadcell_filter(
        curve.time_s, curve.thrust_unit, loadcell_lowpass_hz,
        inverse_regularization,
    ) for curve in curves]
    corrected_curves = [
        replace(curve, thrust_unit=estimated)
        for curve, estimated in zip(curves, corrected, strict=True)
    ]
    first_no_delay = fit_dynamic_model(corrected_curves, "first_order_no_delay")
    first = fit_dynamic_model(corrected_curves, "first_order")
    second = fit_dynamic_model(corrected_curves, "second_order")
    root = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None else _default_analysis_root(paths) / "transient"
    )
    csv_dir = root / "csv"
    plots_dir = root / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    response_csv = csv_dir / "thruster_transient_responses.csv"
    with response_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "run_id", "step_index", "transition", "esc_command",
            "virtual_command_amplitude", "steady_thrust_n", "baseline_thrust_n",
            "time_s", "virtual_command", "virtual_command_unit", "thrust_unit",
            "recorded_thrust_unit",
            "first_order_no_delay", "first_order", "second_order",
        ))
        for curve, corrected_curve in zip(curves, corrected_curves, strict=True):
            input_unit = (
                (curve.time_s >= 0).astype(float) if curve.transition == "start"
                else (curve.time_s < 0).astype(float)
            )
            predictions = (
                model_response("first_order_no_delay", curve.time_s, curve.transition,
                               first_no_delay.parameters),
                model_response("first_order", curve.time_s, curve.transition,
                               first.parameters),
                model_response("second_order", curve.time_s, curve.transition,
                               second.parameters),
            )
            for time, input_value, thrust, recorded, no_delay_value, first_value, second_value in zip(
                curve.time_s, input_unit, corrected_curve.thrust_unit,
                curve.thrust_unit, *predictions, strict=True,
            ):
                writer.writerow((
                    curve.run_id, curve.step_index, curve.transition, curve.esc_command,
                    curve.virtual_command, curve.steady_thrust_n,
                    curve.baseline_thrust_n, time,
                    input_value * curve.virtual_command, input_value, thrust,
                    recorded,
                    no_delay_value, first_value, second_value,
                ))

    overshoots = []

    inverse_csv = csv_dir / "thruster_transient_inverse.csv"
    with inverse_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("run_id", "step_index", "transition", "time_s",
                         "thrust_unit", "inverse_thrust_unit"))
        for curve, estimated in zip(curves, corrected, strict=True):
            for time, observed, inverse_value in zip(
                curve.time_s, curve.thrust_unit, estimated, strict=True,
            ):
                writer.writerow((curve.run_id, curve.step_index,
                                 curve.transition, time, observed, inverse_value))

    noise_ratios = []
    for curve, estimated in zip(curves, corrected, strict=True):
        before = curve.time_s < 0
        if np.count_nonzero(before) >= 3:
            original_noise = float(np.std(curve.thrust_unit[before]))
            if original_noise > 1e-8:
                noise_ratios.append(float(np.std(estimated[before]) / original_noise))
    for curve in corrected_curves:
        if curve.transition != "start":
            continue
        after_start = np.flatnonzero(curve.time_s >= 0.0)
        peak_index = int(after_start[np.argmax(curve.thrust_unit[after_start])])
        peak = float(curve.thrust_unit[peak_index])
        if peak > 1.5:
            amplitude = curve.steady_thrust_n - curve.baseline_thrust_n
            overshoots.append((
                curve.run_id, curve.step_index, curve.esc_command,
                curve.virtual_command, curve.steady_thrust_n,
                curve.baseline_thrust_n, float(curve.time_s[peak_index]),
                peak, float(curve.baseline_thrust_n + peak * amplitude),
                float((peak - 1.0) * abs(amplitude)),
            ))
    overshoots.sort(key=lambda row: row[7], reverse=True)
    overshoot_csv = csv_dir / "thruster_transient_overshoots.csv"
    with overshoot_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "run_id", "step_index", "esc_command", "virtual_command_amplitude",
            "steady_thrust_n", "baseline_thrust_n", "peak_time_s",
            "peak_thrust_unit", "peak_thrust_n", "excess_above_steady_n",
        ))
        writer.writerows(overshoots)

    # Separate tables keep each measured curve distinct in pgfplots. A NaN
    # separator prevents the final sample of one step joining the next step.
    tikz_curve_csvs = {}
    for transition in ("start", "stop"):
        path = csv_dir / f"thruster_transient_{transition}_curves.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("time_s", "thrust_unit"))
            for curve in corrected_curves:
                if curve.transition != transition:
                    continue
                for time, thrust in zip(curve.time_s, curve.thrust_unit, strict=True):
                    if -0.3 <= time <= 1.0:
                        writer.writerow((time, thrust))
                writer.writerow(("nan", "nan"))
        tikz_curve_csvs[transition] = path

    model_csv = csv_dir / "thruster_transient_models.csv"
    grid = np.linspace(-0.3, 2.0, 461)
    model_curves = {
        (transition, fit.model): model_response(
            fit.model, grid, transition, fit.parameters
        )
        for transition in ("start", "stop")
        for fit in (first_no_delay, first, second)
    }
    with model_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "time_s", "start_virtual_command_unit", "start_first_order_no_delay",
            "start_first_order",
            "start_second_order", "stop_virtual_command_unit",
            "stop_first_order_no_delay", "stop_first_order", "stop_second_order",
        ))
        for index, time in enumerate(grid):
            writer.writerow((
                time, float(time >= 0),
                model_curves["start", "first_order_no_delay"][index],
                model_curves["start", "first_order"][index],
                model_curves["start", "second_order"][index],
                float(time < 0),
                model_curves["stop", "first_order_no_delay"][index],
                model_curves["stop", "first_order"][index],
                model_curves["stop", "second_order"][index],
            ))

    fits_yaml = root / "thruster_transient_fits.yaml"
    measured_start_peaks = np.asarray([
        float(np.max(curve.thrust_unit[curve.time_s >= 0.0]))
        for curve in corrected_curves if curve.transition == "start"
    ])
    report = {
        "run_count": len(runs),
        "step_count": sum(len(run.steps) for run in runs),
        "start_count": sum(curve.transition == "start" for curve in curves),
        "stop_count": sum(curve.transition == "stop" for curve in curves),
        "large_start_overshoot_count": len(overshoots),
        "large_start_overshoot_threshold": "peak inverse-filtered normalized thrust > 1.5 after start",
        "normalization": "(thrust - first_start_sample) / (steady_step_mean - first_start_sample)",
        "input": "virtual_command / signed_step_virtual_command; start 0->1, stop 1->0",
        "fit_objective": "Equal event-weighted time-series residuals for all models",
        "inverse_filtered_start_peak_median": float(np.median(measured_start_peaks)),
        "delay_interpretation": (
            "Residual command-to-inverse-filtered-wrench offset; may include DAQ "
            "transport, command transport, ESC, and thruster response"
        ),
        "inverse_filter": {
            "method": "regularized frequency-domain inverse of publisher low-pass at 25 Hz",
            "publisher_filter": "causal 8th-order Butterworth low-pass",
            "cutoff_hz": loadcell_lowpass_hz,
            "frequency_response_simulation_rate_hz": FILTER_SIM_RATE_HZ,
            "regularization": inverse_regularization,
            "fit_target": "inverse-filtered normalized thrust",
            "median_precommand_noise_std_ratio": (
                float(np.median(noise_ratios)) if noise_ratios else None
            ),
            "limitation": "25 Hz sampling and suppressed high frequencies cannot be reversed",
        },
        "analysis_config": asdict(config),
        "steady_virtual_command_coefficients": list(steady_fits.virtual_command.coefficients),
        "first_order_no_delay": asdict(first_no_delay),
        "first_order": asdict(first),
        "second_order": asdict(second),
    }
    with fits_yaml.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)

    import matplotlib.pyplot as plt

    plot_xmin, plot_xmax = -0.3, 1.0
    visible = np.concatenate([
        curve.thrust_unit[(curve.time_s >= plot_xmin) & (curve.time_s <= plot_xmax)]
        for curve in corrected_curves
    ])
    plot_ymin = np.floor((min(0.0, float(np.min(visible))) - 0.08) * 10) / 10
    plot_ymax = np.ceil((max(1.0, float(np.max(visible))) + 0.10) * 10) / 10
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, sharey=True)
    for axis, transition, color in zip(axes, ("start", "stop"), ("#177E89", "#B65C38")):
        for curve in corrected_curves:
            if curve.transition == transition:
                axis.plot(curve.time_s, curve.thrust_unit, color=color, alpha=0.08, linewidth=0.8)
        axis.plot(grid, model_curves[transition, "first_order_no_delay"],
                  color="#4F71A3", linewidth=2.2, label="First order, no delay")
        axis.plot(grid, model_curves[transition, "first_order"],
                  color="#1B1B1B", linewidth=2.2, label="First order, residual delay")
        axis.plot(grid, model_curves[transition, "second_order"],
                  color="#C52F55", linewidth=2.2, label="Second order")
        axis.plot(grid, (grid >= 0).astype(float) if transition == "start" else (grid < 0).astype(float),
                  color="#666666", linestyle=":", label="Virtual command (unit)")
        axis.set(xlim=(plot_xmin, plot_xmax), ylim=(plot_ymin, plot_ymax),
                 ylabel=f"{transition.title()}: inverse-filtered thrust")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    axes[-1].set_xlabel("Time from command transition [s]")
    figure.tight_layout()
    plot_path = plots_dir / "thruster_transient_responses.png"
    figure.savefig(plot_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(figure)

    # Align every transition at t=0. Stop thrust is 1 -> 0 in the response
    # table, so 1 - thrust makes it a rising 0 -> 1 step like a start.
    combined_figure, combined_axis = plt.subplots(figsize=(11, 5.5))
    for transition, color in (
        ("start", "#177E89"),
        ("stop", "#B65C38"),
    ):
        selected = [curve for curve in corrected_curves
                    if curve.transition == transition]
        for curve in selected:
            mask = (curve.time_s >= plot_xmin) & (curve.time_s <= plot_xmax)
            values = (curve.thrust_unit if transition == "start"
                      else 1.0 - curve.thrust_unit)
            combined_axis.plot(curve.time_s[mask], values[mask], color=color,
                               alpha=0.07, linewidth=0.7)
        median_time = np.linspace(plot_xmin, plot_xmax, 301)
        median = np.median(np.stack([
            np.interp(median_time, curve.time_s,
                      curve.thrust_unit if transition == "start"
                      else 1.0 - curve.thrust_unit)
            for curve in selected
        ]), axis=0)
        combined_axis.plot(median_time, median, color=color,
                           linewidth=2.5, label=f"{transition.title()} median (n={len(selected)})")
    combined_time = np.linspace(plot_xmin, plot_xmax, 301)
    command_step = (combined_time >= 0).astype(float)
    simulated = first_order_step(combined_time, 0.4, 0.0)
    combined_axis.plot(combined_time, simulated, color="#53535A",
                       linewidth=2, linestyle="--", label="Simulation (τ = 0.4 s)")
    combined_axis.step(combined_time, command_step, where="post", color="#222222",
                       linewidth=1.5, linestyle=":", label="Normalized command")
    combined_axis.set(xlim=(plot_xmin, plot_xmax),
                      ylim=(plot_ymin, plot_ymax),
                      xlabel="Time from command transition [s]",
                      ylabel="Normalized step response",
                      title="Thruster start and transformed stop responses")
    combined_axis.grid(alpha=0.25)
    combined_axis.legend(loc="best")
    combined_figure.tight_layout()
    combined_plot_path = plots_dir / "thruster_transient_combined_step.png"
    combined_figure.savefig(combined_plot_path, dpi=180)
    plt.close(combined_figure)

    inverse_figure, inverse_axes = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, sharey=True,
    )
    corrected_visible = np.concatenate([
        estimated[(curve.time_s >= plot_xmin) & (curve.time_s <= plot_xmax)]
        for curve, estimated in zip(curves, corrected, strict=True)
    ])
    inverse_ymin = np.floor((min(plot_ymin, float(np.min(corrected_visible))) - 0.08) * 10) / 10
    inverse_ymax = np.ceil((max(plot_ymax, float(np.max(corrected_visible))) + 0.10) * 10) / 10
    for axis, transition in zip(inverse_axes, ("start", "stop"), strict=True):
        for curve, estimated in zip(curves, corrected, strict=True):
            if curve.transition == transition:
                axis.plot(curve.time_s, curve.thrust_unit, color="#177E89",
                          alpha=0.05, linewidth=0.8)
                axis.plot(curve.time_s, estimated, color="#C52F55",
                          alpha=0.09, linewidth=0.8)
        axis.plot([], [], color="#177E89", label="Recorded")
        axis.plot([], [], color="#C52F55", label="Regularized inverse")
        axis.set(xlim=(plot_xmin, plot_xmax), ylim=(inverse_ymin, inverse_ymax),
                 ylabel=f"{transition.title()}: normalized thrust")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    inverse_axes[-1].set_xlabel("Time from command transition [s]")
    inverse_figure.tight_layout()
    inverse_plot_path = plots_dir / "thruster_transient_inverse.png"
    inverse_figure.savefig(inverse_plot_path, dpi=180)
    plt.close(inverse_figure)
    report["files"] = {
        "responses_csv": str(response_csv),
        "inverse_csv": str(inverse_csv),
        "overshoots_csv": str(overshoot_csv),
        "start_curves_csv": str(tikz_curve_csvs["start"]),
        "stop_curves_csv": str(tikz_curve_csvs["stop"]),
        "models_csv": str(model_csv),
        "fits_yaml": str(fits_yaml),
        "plot": str(plot_path),
        "combined_step_plot": str(combined_plot_path),
        "inverse_plot": str(inverse_plot_path),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs="+", required=True, help="run directories")
    parser.add_argument("--config", help="steady-state analysis configuration")
    parser.add_argument("--output-dir", help="defaults to sibling analysis/transient")
    parser.add_argument("--loadcell-lowpass-hz", type=float, default=7.5,
                        help="cutoff used by loadcell publisher (default: 7.5 Hz)")
    parser.add_argument("--inverse-regularization", type=float, default=0.04,
                        help="regularization for inverse low-pass (default: 0.04)")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    try:
        result = analyze_transients(
            args.run, args.config, args.output_dir, not args.no_show,
            args.loadcell_lowpass_hz, args.inverse_regularization,
        )
    except Exception as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"Fitted {result['start_count']} starts and {result['stop_count']} stops")
    for key in ("first_order_no_delay", "first_order", "second_order"):
        fit = result[key]
        print(f"{key}: {fit['parameters']}, RMSE={fit['rmse']:.4f}, R²={fit['r_squared']:.4f}")
    for path in result["files"].values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
