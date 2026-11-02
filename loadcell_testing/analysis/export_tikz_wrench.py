#!/usr/bin/env python3
"""Export a command step's motor-origin wrench for TikZ plotting."""

import argparse
import csv
from pathlib import Path

from .analysis_config import load_analysis_config


AXES = ("x", "y", "z")
FIELDS = [
    "t_rel_s",
    *(f"wrench.force.{axis}" for axis in AXES),
    *(f"wrench.torque.{axis}" for axis in AXES),
]


def export(source: Path, output: Path, config_path: Path, step_index: int = 0) -> int:
    with (source.parent / "thruster_step_averages.csv").open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        steps = list(csv.DictReader(stream))
    if not 0 <= step_index < len(steps):
        raise ValueError(f"step index {step_index} is outside 0..{len(steps) - 1}")
    # Keep a short baseline before the command and stop before the next step.
    start_ns = int(steps[step_index]["command_start_ns"]) - 830_000_000
    end_ns = (
        int(steps[step_index + 1]["command_start_ns"]) - 10_000_000
        if step_index + 1 < len(steps)
        else int(steps[step_index]["sample_end_ns"]) + 1_000_000_000
    )
    with source.open(newline="", encoding="utf-8-sig") as stream:
        raw = [
            row for row in csv.DictReader(stream)
            if start_ns <= int(row["t_ros_ns"]) < end_ns
        ]
    if not raw:
        raise ValueError(f"no wrench samples for step {step_index} in {source}")
    position = load_analysis_config(config_path).motor_position_m
    rows = []
    for sample in raw:
        force = [float(sample[f"wrench.force.{axis}"]) for axis in AXES]
        torque = [float(sample[f"wrench.torque.{axis}"]) for axis in AXES]
        rx, ry, rz = position
        fx, fy, fz = force
        correction = (ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx)
        rows.append({
            "t_rel_s": format(float(sample["t_rel_s"]), ".9f"),
            **{f"wrench.force.{axis}": format(value, ".9f")
               for axis, value in zip(AXES, force, strict=True)},
            **{f"wrench.torque.{axis}": format(value - offset, ".9f")
               for axis, value, offset in zip(AXES, torque, correction, strict=True)},
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="original run's loadcell wrench CSV")
    parser.add_argument("output", type=Path, help="TikZ wrench CSV to write")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--step-index", type=int, default=0)
    args = parser.parse_args()
    print(f"Wrote {export(args.source, args.output, args.config, args.step_index)} samples in {args.output}")


if __name__ == "__main__":
    main()
