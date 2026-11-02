"""Check transient model identification on known step responses."""

import unittest
from dataclasses import replace

import numpy as np

from loadcell_testing.analysis.transient_analysis import (
    TransientCurve,
    first_order_step,
    fit_dynamic_model,
    inverse_loadcell_filter,
    model_response,
    second_order_step,
)


def curves_from_step(step: np.ndarray, time: np.ndarray) -> list[TransientCurve]:
    return [
        TransientCurve(
            run_id="synthetic_cw_fwd", step_index=0, transition=transition,
            esc_command=0.5, virtual_command=0.35, steady_thrust_n=4.0,
            baseline_thrust_n=0.0, time_s=time,
            thrust_unit=step if transition == "start" else 1.0 - step,
        )
        for transition in ("start", "stop")
    ]


class TransientAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.time = np.linspace(-0.2, 1.0, 121)

    def test_first_order_recovers_time_constant_and_delay(self):
        step = first_order_step(self.time, tau_s=0.15, delay_s=0.08)
        fit = fit_dynamic_model(curves_from_step(step, self.time), "first_order")
        self.assertAlmostEqual(fit.parameters["tau_s"], 0.15, places=3)
        self.assertAlmostEqual(fit.parameters["delay_s"], 0.08, places=3)
        self.assertLess(fit.rmse, 1e-5)

    def test_first_order_without_delay_recovers_time_constant(self):
        step = first_order_step(self.time, tau_s=0.15, delay_s=0.0)
        fit = fit_dynamic_model(curves_from_step(step, self.time), "first_order_no_delay")
        self.assertAlmostEqual(fit.parameters["tau_s"], 0.15, places=3)
        self.assertNotIn("delay_s", fit.parameters)
        self.assertLess(fit.rmse, 1e-5)

    def test_second_order_recovers_frequency_damping_and_delay(self):
        step = second_order_step(
            self.time, natural_frequency_rad_s=12.0, damping_ratio=0.65,
            delay_s=0.04,
        )
        fit = fit_dynamic_model(curves_from_step(step, self.time), "second_order")
        self.assertAlmostEqual(fit.parameters["natural_frequency_rad_s"], 12.0, places=2)
        self.assertAlmostEqual(fit.parameters["damping_ratio"], 0.65, places=2)
        self.assertAlmostEqual(fit.parameters["delay_s"], 0.04, places=2)
        self.assertLess(fit.rmse, 1e-5)

    def test_filtered_fit_separates_filter_lag_from_thruster_delay(self):
        time = np.arange(-0.2, 1.0, 0.04)
        step = model_response(
            "first_order", time, "start", {"tau_s": 0.09, "delay_s": 0.03},
            loadcell_lowpass_hz=7.5,
        )
        fit = fit_dynamic_model(curves_from_step(step, time), "first_order", 7.5)
        self.assertAlmostEqual(fit.parameters["tau_s"], 0.09, places=3)
        self.assertAlmostEqual(fit.parameters["delay_s"], 0.03, places=3)
        self.assertLess(fit.rmse, 1e-5)

    def test_filter_has_step_lag_without_thruster_delay(self):
        time = np.arange(-0.2, 0.5, 0.001)
        filtered = model_response(
            "first_order_no_delay", time, "start", {"tau_s": 0.01},
            loadcell_lowpass_hz=7.5,
        )
        unfiltered = model_response(
            "first_order_no_delay", time, "start", {"tau_s": 0.01},
        )
        self.assertLess(filtered[np.argmin(abs(time - 0.05))], 0.05)
        self.assertGreater(unfiltered[np.argmin(abs(time - 0.05))], 0.99)
        self.assertGreater(filtered[np.argmin(abs(time - 0.2))], 0.95)

    def test_regularized_inverse_recovers_step_within_recorded_bandwidth(self):
        time = np.arange(-0.32, 1.8, 0.04)
        parameters = {"tau_s": 0.09, "delay_s": 0.04}
        ideal = model_response("first_order", time, "start", parameters)
        observed = model_response("first_order", time, "start", parameters, 7.5)
        recovered = inverse_loadcell_filter(time, observed, 7.5, 0.02)
        interior = (time >= -0.1) & (time <= 1.2)
        recorded_error = np.sqrt(np.mean((observed[interior] - ideal[interior]) ** 2))
        inverse_error = np.sqrt(np.mean((recovered[interior] - ideal[interior]) ** 2))
        self.assertLess(inverse_error, 0.02)
        self.assertLess(inverse_error, recorded_error / 5)

    def test_fit_to_inverse_filtered_curves_recovers_prefilter_dynamics(self):
        time = np.arange(-0.32, 1.8, 0.04)
        parameters = {"tau_s": 0.09, "delay_s": 0.04}
        corrected_curves = []
        templates = curves_from_step(np.zeros_like(time), time)
        for transition in ("start", "stop"):
            observed = model_response(
                "first_order", time, transition, parameters, 7.5,
            )
            index = 0 if transition == "start" else 1
            corrected_curves.append(replace(
                templates[index],
                thrust_unit=inverse_loadcell_filter(time, observed),
            ))
        fit = fit_dynamic_model(corrected_curves, "first_order")
        self.assertAlmostEqual(fit.parameters["tau_s"], 0.09, delta=0.01)
        self.assertAlmostEqual(fit.parameters["delay_s"], 0.04, delta=0.01)

if __name__ == "__main__":
    unittest.main()
