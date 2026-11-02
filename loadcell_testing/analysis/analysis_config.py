"""Geometry and sign conventions for offline thruster analysis."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AnalysisConfig:
    motor_position_m: list[float]
    thrust_component: str
    roll_torque_component: str
    thrust_sign: int
    roll_torque_sign: int
    stand_signs: dict[str, int]
    propeller_signs: dict[str, int] | None = None

    def __post_init__(self) -> None:
        position = self.motor_position_m
        if (
            not isinstance(position, (list, tuple))
            or len(position) != 3
            or any(
                type(value) not in (int, float) or not isfinite(value)
                for value in position
            )
        ):
            raise ValueError("motor_position_m must contain three finite numbers")
        if self.thrust_component not in {"force_x", "force_y", "force_z"}:
            raise ValueError(f"invalid thrust component: {self.thrust_component}")
        if self.roll_torque_component not in {"torque_x", "torque_y", "torque_z"}:
            raise ValueError(f"invalid roll torque component: {self.roll_torque_component}")
        signs = [self.thrust_sign, self.roll_torque_sign]
        if not isinstance(self.stand_signs, dict) or set(self.stand_signs) != {"fwd", "rev"}:
            raise ValueError("stand_signs must define exactly ['fwd', 'rev']")
        signs.extend(self.stand_signs.values())
        if self.propeller_signs is not None:
            if (
                not isinstance(self.propeller_signs, dict)
                or set(self.propeller_signs) != {"cw", "ccw"}
            ):
                raise ValueError("propeller_signs must define exactly ['ccw', 'cw']")
            signs.extend(self.propeller_signs.values())
        if any(type(sign) is not int or sign not in (-1, 1) for sign in signs):
            raise ValueError("all analysis signs must be integers +1 or -1")


def load_analysis_config(path: str | Path | None = None) -> AnalysisConfig:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config/thruster_analysis.yaml"
        if not path.is_file():
            from ament_index_python.packages import get_package_share_path

            path = (
                get_package_share_path("loadcell_testing")
                / "config/thruster_analysis.yaml"
            )
    path = Path(path).expanduser()
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"analysis config must be a mapping: {path}")
    try:
        return AnalysisConfig(**data)
    except TypeError as exc:
        raise ValueError(f"invalid analysis config {path}: {exc}") from exc
