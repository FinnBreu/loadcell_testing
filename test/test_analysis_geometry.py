"""Regression checks for motor-origin moments and configurable signs."""

from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from loadcell_testing.analysis.analysis_config import load_analysis_config
from loadcell_testing.analysis.analyze_thruster_run import build_parser
from loadcell_testing.analysis.thruster_analysis import (
    RunSeries,
    StepAverage,
    fit_run_series,
    load_run_series,
)
from loadcell_testing.analysis.thruster_analysis_report import write_run_averages


class AnalysisGeometryTest(unittest.TestCase):
    def setUp(self):
        self.config = replace(load_analysis_config(), motor_position_m=[0.0, 0.22, 0.10])
        # F=(2,3,4) at r=(0,.22,.10) gives (.58,.20,-.44) Nm.
        # An additional .30 Nm motor couple leaves sensor Mz=-.14 Nm.
        self.step = StepAverage(
            step_index=0, repeat=0, speed=0.5,
            command_start_ns=0, sample_start_ns=0, sample_end_ns=1,
            wrench_samples=1, voltage_samples=1, voltage=12.0,
            force_x=2.0, force_y=3.0, force_z=4.0,
            torque_x=0.58, torque_y=0.20, torque_z=-0.14,
        )

    def series(self, prop="ccw", direction="fwd", config=None, step=None, run="/tmp"):
        manifest = {"run_id": f"example_{prop}_{direction}"}
        with patch(
            "loadcell_testing.analysis.thruster_analysis.load_step_averages",
            return_value=(manifest, [step or self.step]),
        ):
            return load_run_series(run, config or self.config)

    def test_translation_precedes_signs_and_keeps_fixed_position(self):
        for prop, p in (("ccw", 1), ("cw", -1)):
            for direction, d in (("fwd", 1), ("rev", -1)):
                with self.subTest(prop=prop, direction=direction):
                    run = self.series(prop, direction)
                    np.testing.assert_allclose(run.motor_torques, [[0, 0, .30]], atol=1e-15)
                    np.testing.assert_allclose(
                        run.roll_torque,
                        [self.config.roll_torque_sign * d * .30],
                    )
                    np.testing.assert_allclose(run.thrust, [p * d * 4.0])
                    np.testing.assert_allclose(run.speeds, [.5])
                    self.assertEqual(run.propeller_thrust_sign, p)
                    self.assertEqual(run.steps[0], self.step)

    def test_propeller_sign_corrects_thrust_without_changing_torque_pairing(self):
        runs = [
            RunSeries(
                run_id=f"example_{rotation}_fwd",
                run_dir=Path("/tmp"),
                stand_direction="fwd",
                steps=(),
                speeds=np.asarray([-1.0, -.5, .5, 1.0]),
                thrust=np.asarray([-10.0, -5.0, 5.0, 10.0]),
                roll_torque=np.asarray([-.1, -.05, .05, .1]),
                motor_torques=np.zeros((4, 3)),
                propeller_rotation=rotation,
                stand_sign=1,
                propeller_thrust_sign=sign,
            )
            for rotation, sign in (("ccw", 1), ("cw", -1))
        ]
        fits, virtual_commands = fit_run_series(runs)
        self.assertAlmostEqual(fits.thrust_proportional.coefficients[0], 10.0)
        self.assertAlmostEqual(fits.roll_torque_proportional.coefficients[0], .1)
        self.assertAlmostEqual(fits.roll_torque_per_thrust.coefficients[0], .01)
        np.testing.assert_allclose(virtual_commands[1], [-1.0, -.5, .5, 1.0])

    def test_general_offset_removes_pure_force_moment(self):
        config = replace(self.config, motor_position_m=[1.0, 2.0, 3.0])
        step = replace(
            self.step, force_x=4, force_y=5, force_z=6,
            torque_x=-3, torque_y=6, torque_z=-3,
        )
        for component in ("torque_x", "torque_y", "torque_z"):
            run = self.series(config=replace(config, roll_torque_component=component), step=step)
            np.testing.assert_allclose(run.motor_torques, [[0, 0, 0]])
            np.testing.assert_allclose(run.roll_torque, [0])

    def test_zero_offset_preserves_sensor_moments(self):
        run = self.series(config=replace(self.config, motor_position_m=[0, 0, 0]))
        np.testing.assert_allclose(run.motor_torques, [[.58, .20, -.14]])
        np.testing.assert_allclose(
            run.roll_torque, [-self.config.roll_torque_sign * .14]
        )

    def test_omitted_propeller_signs_leave_thrust_unchanged(self):
        config = replace(self.config, propeller_signs=None)
        run = self.series(prop="cw", config=config)
        self.assertEqual(run.propeller_thrust_sign, 1)
        np.testing.assert_allclose(run.speeds, [.5])
        np.testing.assert_allclose(run.thrust, [4.0])
        np.testing.assert_allclose(run.roll_torque, [config.roll_torque_sign * .30])

    def test_yaml_signs_are_used_and_cli_defaults_do_not_override(self):
        data = asdict(self.config)
        data.update(
            thrust_sign=-1, roll_torque_sign=1,
            stand_signs={"fwd": -1, "rev": 1},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.yaml"
            path.write_text(yaml.safe_dump(data))
            config = load_analysis_config(path)
        run = self.series(config=config)
        np.testing.assert_allclose(run.speeds, [.5])
        np.testing.assert_allclose(run.thrust, [4])
        np.testing.assert_allclose(run.roll_torque, [-.30])
        args = build_parser().parse_args(["--run", "/tmp/run_ccw_fwd"])
        self.assertIsNone(args.thrust_sign)
        self.assertIsNone(args.roll_torque_sign)
        self.assertIsNone(args.thrust_component)
        self.assertIsNone(args.roll_torque_component)

    def test_invalid_geometry_or_signs_fail_explicitly(self):
        for changes in (
            {"motor_position_m": [0, 1]},
            {"motor_position_m": [0, float("nan"), 1]},
            {"stand_signs": {"fwd": 1}},
            {"propeller_signs": {"cw": 0, "ccw": 1}},
            {"roll_torque_sign": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes)

    def test_averages_export_preserves_both_reference_points(self):
        import csv

        with tempfile.TemporaryDirectory() as directory:
            run = self.series(run=directory)
            write_run_averages(run)
            path = Path(directory) / "csv/thruster_step_averages.csv"
            with path.open() as stream:
                row = next(csv.DictReader(stream))
        self.assertAlmostEqual(float(row["torque_z"]), -.14)
        self.assertAlmostEqual(float(row["motor_torque_z"]), .30)
        self.assertAlmostEqual(
            float(row["vehicle_roll_torque"]),
            self.config.roll_torque_sign * .30,
        )
