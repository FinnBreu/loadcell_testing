"""Write tables and plots for thruster analysis results."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from .analysis_config import AnalysisConfig
from .thruster_analysis import (
    AnalysisFits,
    CurveFit,
    RunSeries,
    actual_command_from_virtual,
    evaluate_symmetric_quadratic,
)


@dataclass(frozen=True)
class PlotData:
    run_virtual_commands: tuple[np.ndarray, ...]
    command: np.ndarray
    thrust_proportional: np.ndarray
    thrust_quadratic: np.ndarray
    torque_proportional: np.ndarray
    torque_quadratic: np.ndarray
    actual_command: np.ndarray
    virtual_thrust: np.ndarray
    virtual_torque: np.ndarray


@dataclass(frozen=True)
class AnalysisPaths:
    averages_csv: Path
    plot_data_dir: Path
    fits_yaml: Path
    command_plot: Path
    virtual_plot: Path

    @classmethod
    def under(cls, analysis_dir: Path) -> "AnalysisPaths":
        csv_dir = analysis_dir / "csv"
        plots_dir = analysis_dir / "plots"
        return cls(
            averages_csv=csv_dir / "thruster_step_averages.csv",
            plot_data_dir=csv_dir / "plot_data",
            fits_yaml=plots_dir / "thruster_curve_fits.yaml",
            command_plot=plots_dir / "thruster_command_curves.png",
            virtual_plot=plots_dir / "thruster_virtual_command_curves.png",
        )


def build_plot_data(
    fits: AnalysisFits, run_virtual_commands: tuple[np.ndarray, ...]
) -> PlotData:
    command = np.linspace(-1.0, 1.0, 301)
    return PlotData(
        run_virtual_commands=run_virtual_commands,
        command=command,
        thrust_proportional=fits.thrust_proportional.coefficients[0] * command,
        thrust_quadratic=evaluate_symmetric_quadratic(
            fits.thrust_quadratic.coefficients, command
        ),
        torque_proportional=(
            fits.roll_torque_proportional.coefficients[0] * command
        ),
        torque_quadratic=evaluate_symmetric_quadratic(
            fits.roll_torque_quadratic.coefficients, command
        ),
        actual_command=actual_command_from_virtual(command, fits.virtual_command),
        virtual_thrust=fits.thrust_coefficient * command,
        virtual_torque=fits.torque_coefficient * command,
    )


def write_run_averages(run: RunSeries) -> None:
    _write_analysis_averages(
        run.run_dir / "csv" / "thruster_step_averages.csv", [run]
    )


def _write_analysis_averages(path: Path, runs: list[RunSeries]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    motor_columns = ["motor_torque_x", "motor_torque_y", "motor_torque_z"]
    fields = [
        "run_id",
        *asdict(runs[0].steps[0]),
        *motor_columns,
        "esc_command",
        "normalized_thruster_command",
        "vehicle_thrust",
        "vehicle_roll_torque",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for step, motor_torque, command, thrust, torque in zip(
                run.steps, run.motor_torques, run.speeds,
                run.thrust, run.roll_torque, strict=True
            ):
                writer.writerow(
                    {
                        "run_id": run.run_id,
                        **asdict(step),
                        **dict(zip(motor_columns, motor_torque, strict=True)),
                        "esc_command": command,
                        "normalized_thruster_command": command,
                        "vehicle_thrust": thrust,
                        "vehicle_roll_torque": torque,
                    }
                )


def _write_columns(path: Path, columns: dict[str, np.ndarray]) -> None:
    if len({len(values) for values in columns.values()}) != 1:
        raise ValueError(f"CSV columns have different lengths: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values(), strict=True))


def _write_plot_data(
    path: Path, runs: list[RunSeries], data: PlotData
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_index",
        "stand_direction",
        "loadcell_sign",
        "propeller_rotation",
        "propeller_thrust_sign",
        "esc_command",
        "normalized_thruster_command",
        "virtual_command",
        "vehicle_thrust",
        "vehicle_roll_torque",
    ]
    with (path / "thruster_points.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, run in enumerate(runs):
            writer.writerows(
                {
                    "run_index": index,
                    "stand_direction": run.stand_direction,
                    "loadcell_sign": run.stand_sign,
                    "propeller_rotation": run.propeller_rotation,
                    "propeller_thrust_sign": run.propeller_thrust_sign,
                    "esc_command": command,
                    "normalized_thruster_command": command,
                    "virtual_command": virtual,
                    "vehicle_thrust": thrust,
                    "vehicle_roll_torque": torque,
                }
                for command, virtual, thrust, torque in zip(
                    run.speeds,
                    data.run_virtual_commands[index],
                    run.thrust,
                    run.roll_torque,
                    strict=True,
                )
            )

    _write_columns(
        path / "thruster_command_fits.csv",
        {
            "esc_command": data.command,
            "normalized_thruster_command": data.command,
            "vehicle_thrust_proportional": data.thrust_proportional,
            "vehicle_thrust_quadratic": data.thrust_quadratic,
            "vehicle_roll_torque_proportional": data.torque_proportional,
            "vehicle_roll_torque_quadratic": data.torque_quadratic,
        },
    )
    _write_columns(
        path / "thruster_virtual_command_fits.csv",
        {
            "virtual_command": data.command,
            "esc_command": data.actual_command,
            "normalized_thruster_command": data.actual_command,
            "vehicle_thrust": data.virtual_thrust,
            "vehicle_roll_torque": data.virtual_torque,
        },
    )


def _fit_dict(fit: CurveFit) -> dict:
    return {
        "coefficients": list(fit.coefficients),
        "r_squared": fit.r_squared,
    }


def fit_yaml_data(
    runs: list[RunSeries],
    fits: AnalysisFits,
    config: AnalysisConfig,
) -> dict:
    return {
        "runs": [
            {
                "run_id": run.run_id,
                "path": str(run.run_dir),
                "stand_direction": run.stand_direction,
                "loadcell_sign": run.stand_sign,
                "propeller_rotation": run.propeller_rotation,
                "propeller_thrust_sign": run.propeller_thrust_sign,
            }
            for run in runs
        ],
        "analysis_config": asdict(config),
        "models": {
            "thrust_proportional": _fit_dict(fits.thrust_proportional),
            "roll_torque_proportional": _fit_dict(
                fits.roll_torque_proportional
            ),
            "roll_torque_per_thrust": _fit_dict(fits.roll_torque_per_thrust),
            "thrust_quadratic": _fit_dict(fits.thrust_quadratic),
            "roll_torque_quadratic": _fit_dict(fits.roll_torque_quadratic),
            "virtual_command": {
                **_fit_dict(fits.virtual_command),
                "thrust_coefficient": fits.thrust_coefficient,
                "torque_coefficient": fits.torque_coefficient,
            },
        },
    }


def write_analysis_data(
    paths: AnalysisPaths,
    runs: list[RunSeries],
    fits: AnalysisFits,
    data: PlotData,
    config: AnalysisConfig,
) -> None:
    _write_analysis_averages(paths.averages_csv, runs)
    _write_plot_data(paths.plot_data_dir, runs, data)
    paths.fits_yaml.parent.mkdir(parents=True, exist_ok=True)
    with paths.fits_yaml.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            fit_yaml_data(
                runs,
                fits,
                config,
            ),
            stream,
            sort_keys=False,
        )


def _legend(axis) -> None:
    axis.grid(True, alpha=0.25)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        markerscale=0.8,
    )


def plot_analysis(
    paths: AnalysisPaths,
    runs: list[RunSeries],
    fits: AnalysisFits,
    data: PlotData,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(2, 1, figsize=(12, 8))
    series = (
        (
            axes[0],
            "thrust",
            "Vehicle thrust [N]",
            data.thrust_proportional,
            data.thrust_quadratic,
            fits.thrust_proportional,
            fits.thrust_quadratic,
        ),
        (
            axes[1],
            "roll_torque",
            "Vehicle roll torque [N m]",
            data.torque_proportional,
            data.torque_quadratic,
            fits.roll_torque_proportional,
            fits.roll_torque_quadratic,
        ),
    )
    for axis, field, ylabel, linear_y, quadratic_y, linear, quadratic in series:
        for index, run in enumerate(runs):
            axis.scatter(
                run.speeds,
                getattr(run, field),
                color=colors(index % 10),
                label=run.run_id,
                alpha=0.85,
            )
        axis.plot(
            data.command,
            linear_y,
            "--",
            color="#202020",
            label=f"Proportional, R^2={linear.r_squared:.4f}",
        )
        axis.plot(
            data.command,
            quadratic_y,
            color="#C5522D",
            label=f"Symmetric quadratic, R^2={quadratic.r_squared:.4f}",
        )
        axis.set_ylabel(ylabel)
        _legend(axis)
    axes[0].set_xlabel("ESC command")
    axes[1].set_xlabel("ESC command")
    figure.tight_layout()
    figure.savefig(paths.command_plot, dpi=180)

    figure, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].plot(
        data.command,
        data.actual_command,
        color="#C5522D",
        label="Exact inverse",
    )
    for index, run in enumerate(runs):
        virtual = data.run_virtual_commands[index]
        color = colors(index % 10)
        axes[0].scatter(virtual, run.speeds, color=color, alpha=0.85)
        axes[1].scatter(
            virtual,
            run.thrust,
            color=color,
            label=run.run_id,
            alpha=0.85,
        )
        axes[2].scatter(
            virtual, run.roll_torque, color=color, alpha=0.85
        )
    axes[1].plot(
        data.command,
        data.virtual_thrust,
        "--",
        color="#202020",
        label=f"Linear target, {fits.thrust_coefficient:.4g} N / command",
    )
    axes[2].plot(
        data.command,
        data.virtual_torque,
        "--",
        color="#202020",
        label="Proportional",
    )
    axes[0].set_ylabel("ESC command")
    axes[1].set_ylabel("Vehicle thrust [N]")
    axes[2].set_ylabel("Vehicle roll torque [N m]")
    axes[0].set_xlabel("Thrust virtual command")
    axes[1].set_xlabel("Thrust virtual command")
    axes[2].set_xlabel("Virtual command")
    for axis in axes:
        _legend(axis)
    figure.tight_layout()
    figure.savefig(paths.virtual_plot, dpi=180)

    if show:
        plt.show()
    else:
        plt.close("all")
