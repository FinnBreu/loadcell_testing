#!/usr/bin/env python3

"""CLI for averaging and fitting recorded thruster sweeps."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from pathlib import Path

from .analysis_config import load_analysis_config
from .thruster_analysis import AnalysisFits, fit_run_series, load_run_series
from .thruster_analysis_report import (
    AnalysisPaths,
    build_plot_data,
    plot_analysis,
    write_analysis_data,
    write_run_averages,
)


@dataclass(frozen=True)
class AnalysisResult:
    fits: AnalysisFits
    paths: AnalysisPaths


def _default_analysis_root(run_dirs: list[Path]) -> Path:
    common_parent = Path(
        os.path.commonpath([str(run_dir.parent) for run_dir in run_dirs])
    )
    return common_parent / "analysis"


def _default_analysis_dir(run_dirs: list[Path]) -> Path:
    return _default_analysis_root(run_dirs) / "steady_state"


def analyze_runs(
    runs: list[str | Path],
    thrust_component: str | None = None,
    roll_torque_component: str | None = None,
    thrust_sign: int | None = None,
    roll_torque_sign: int | None = None,
    output_dir: str | Path | None = None,
    show: bool = True,
    config_file: str | Path | None = None,
) -> AnalysisResult:
    if not runs:
        raise ValueError("at least one run is required")
    overrides = {
        "thrust_component": thrust_component,
        "roll_torque_component": roll_torque_component,
        "thrust_sign": thrust_sign,
        "roll_torque_sign": roll_torque_sign,
    }
    config = replace(
        load_analysis_config(config_file),
        **{key: value for key, value in overrides.items() if value is not None},
    )

    run_dirs = [Path(run).expanduser().resolve() for run in runs]
    if len(set(run_dirs)) != len(run_dirs):
        raise ValueError("the same run was supplied more than once")

    series = [load_run_series(run_dir, config) for run_dir in run_dirs]
    for run in series:
        write_run_averages(run)

    fits, run_virtual_commands = fit_run_series(series)
    plot_data = build_plot_data(fits, run_virtual_commands)
    analysis_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else _default_analysis_dir(run_dirs)
    )
    paths = AnalysisPaths.under(analysis_dir)
    write_analysis_data(
        paths,
        series,
        fits,
        plot_data,
        config,
    )
    plot_analysis(paths, series, fits, plot_data, show)
    return AnalysisResult(fits=fits, paths=paths)


def _format_fit(fit) -> str:
    if len(fit.coefficients) == 1:
        return f"{fit.coefficients[0]:.8g} * speed"
    quadratic, linear = fit.coefficients
    sign = "+" if linear >= 0.0 else "-"
    return (
        f"{quadratic:.8g} * speed * abs(speed) "
        f"{sign} {abs(linear):.8g} * speed"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Average and fit one or more exported thruster sweeps."
    )
    parser.add_argument(
        "--config",
        help="analysis YAML; defaults to the package's config/thruster_analysis.yaml",
    )
    parser.add_argument(
        "--run",
        nargs="+",
        default=[],
        help="run directories containing _cw/_ccw and _fwd/_rev markers",
    )
    parser.add_argument(
        "--output-dir",
        help="steady-state output directory; defaults to sibling analysis/steady_state",
    )
    parser.add_argument(
        "--thrust-component",
        choices=("force_x", "force_y", "force_z"),
        help="override the YAML thrust component",
    )
    parser.add_argument(
        "--roll-torque-component",
        choices=("torque_x", "torque_y", "torque_z"),
        help="override the YAML roll torque component",
    )
    parser.add_argument(
        "--thrust-sign",
        type=int,
        choices=(-1, 1),
        help="override the YAML thrust sign",
    )
    parser.add_argument(
        "--roll-torque-sign",
        type=int,
        choices=(-1, 1),
        help="override the YAML roll torque sign",
    )
    parser.add_argument("--no-show", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = analyze_runs(
            args.run,
            config_file=args.config,
            thrust_component=args.thrust_component,
            roll_torque_component=args.roll_torque_component,
            thrust_sign=args.thrust_sign,
            roll_torque_sign=args.roll_torque_sign,
            output_dir=args.output_dir,
            show=not args.no_show,
        )
    except Exception as exc:
        parser.exit(1, f"error: {exc}\n")

    fits = result.fits
    for label, fit in (
        ("Thrust proportional", fits.thrust_proportional),
        ("Torque proportional", fits.roll_torque_proportional),
        ("Thrust symmetric quadratic", fits.thrust_quadratic),
        ("Torque symmetric quadratic", fits.roll_torque_quadratic),
    ):
        print(f"{label}: {_format_fit(fit)} (R^2={fit.r_squared:.6f})")
    print(
        "Roll torque / thrust: "
        f"{fits.roll_torque_per_thrust.coefficients[0]:.8g} "
        f"(R^2={fits.roll_torque_per_thrust.r_squared:.6f})"
    )
    print(f"Wrote {result.paths.averages_csv}")
    print(f"Wrote TikZ plot CSVs to {result.paths.plot_data_dir}")
    print(f"Wrote {result.paths.fits_yaml}")
    print(f"Wrote {result.paths.command_plot}")
    print(f"Wrote {result.paths.virtual_plot}")


if __name__ == "__main__":
    main()
