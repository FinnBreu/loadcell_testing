# loadcell_testing

ROS 2 tools for the Schunk/ATI six-axis loadcell, LAN-XI DAQ, and
single-thruster bench tests. The loadcell publisher and motor sweep run
independently.

The Python package separates the acquisition tools in `loadcell_testing/run/`
from fitting, reports, and the TikZ wrench export in
`loadcell_testing/analysis/`. The ROS 2 executable names stay the same.

## Loadcell publisher

```bash
ros2 launch loadcell_testing lanxi_loadcell_publisher.launch.py \
  vehicle_name:=uuv02 \
  calibration_file:=loadcell_calibration.yaml
```

The publisher owns the LAN-XI recorder session at `192.168.0.139`, configures
all six inputs. The publisher filters the full-rate stream before
reducing its rate. Filters and range are set in the `lanxi_loadcell_publisher.py` header.
Topics are:

- `/<vehicle_name>/loadcell/wrench`
- `/<vehicle_name>/loadcell/channels` (filtered strain channels)

Zero and reset the session-only runtime tare with:

```bash
ros2 service call /uuv02/loadcell/tare std_srvs/srv/Trigger "{}"
ros2 service call /uuv02/loadcell/reset_tare std_srvs/srv/Trigger "{}"
```

Keep the loadcell unloaded while taring. Calibration is applied as:

```text
wrench = matrix * (channels - input_bias) - wrench_tare - runtime_tare
```

`input_bias` is in raw DAQ units. `wrench_tare` is ordered
`[Fx, Fy, Fz, Tx, Ty, Tz]`.

## Thruster sweep

Start the loadcell publisher first, then run:

```bash
ros2 launch loadcell_testing single_thruster_sweep.launch.py \
  vehicle_name:=uuv02 \
  speeds:=0.10,0.20,0.30,0.40,0.50 \
  arm_motors:=true
```

The sweep commands ESC 1 and records thruster command, wrench, and battery
voltage to:

```text
/home/finnb/ros2/data/thruster_loadcell_runs/<run>/
```

Export the recorded MCAP data before analysis:

```bash
ros2 run tools export_csv \
  --run /home/finnb/ros2/data/thruster_loadcell_runs/<run>
```

## Thrust curve analysis

Use only runs whose names include exactly one propeller marker (`_cw` or
`_ccw`) and one stand marker (`_fwd` or `_rev`). Pass all runs together:

```bash
ros2 run loadcell_testing analyze_thruster_run \
  --config /home/finnb/ros2_underlay/src/loadcell_testing/config/thruster_analysis.yaml \
  --run /home/finnb/ros2/data/thruster_loadcell_runs/20260924_* \
  --no-show
```

`config/thruster_analysis.yaml` defines the motor position in loadcell axes, components, and signs. 

Step means use each manifest's settling and sampling periods. Before sign
normalization, all three torques are translated to the motor origin:

```text
M_motor = M_loadcell - cross(motor_position_m, F_loadcell)
```

The position stays fixed for FWD and REV. Translation preserves all measured points;
it changes the torque reference point. Forces are unchanged.

With `d = stand_signs[fwd/rev]` and `p = propeller_signs[cw/ccw]` (or `1`
when `propeller_signs` is omitted):

```text
thrust = thrust_sign * d * p * F_selected
roll_torque = roll_torque_sign * d * M_motor_selected
```

The September 2026 runs use `p = -1` for CW and `p = 1` for CCW so they can
be fitted together. Thrust and roll torque are fitted against the recorded ESC
command. REV mounting reverses the measured thrust and torque signs.

All runs are pooled for fitting. Results go to the sibling
`analysis/steady_state` directory (`--output-dir` overrides it). The fit YAML
includes the effective configuration.
Step CSVs retain the original loadcell wrench, add `motor_torque_x/y/z` in
loadcell axes, and store the ESC command, corrected vehicle thrust,
and mounting corrected roll torque used for fitting.

In `csv/plot_data/thruster_points.csv`, plot `vehicle_thrust` or
`vehicle_roll_torque` against `normalized_thruster_command` or
`virtual_command`. With the thrust sign correction applied to the measurement,
`normalized_thruster_command` has the same values as `esc_command`.
The separate `thruster_loadcell-wrench.csv` TikZ data keeps the measured forces
and time values; its torque columns refer to the motor origin using
`M_motor = M_loadcell - cross(motor_position_m, F_loadcell)`.
The current excerpt uses step 0 (command `+0.5`) of the
`20260924_162720_uuv02_real_shield_cw_fwd_12V0` run. To regenerate it after
changing `motor_position_m`, run from `~/ros2_underlay`:

```bash
PYTHONPATH=src/loadcell_testing python3 -m loadcell_testing.analysis.export_tikz_wrench \
  /home/finnb/ros2/data/thruster_loadcell_runs/20260924_162720_uuv02_real_shield_cw_fwd_12V0/csv/uuv02_loadcell_wrench.csv \
  /home/finnb/ros2/data/thruster_loadcell_runs/analysis/steady_state/csv/plot_data/thruster_loadcell-wrench.csv \
  --config src/loadcell_testing/config/thruster_analysis.yaml
```

The script selects the command step from that run's `thruster_step_averages.csv`
and includes 0.83 s of baseline before the step. Use `--step-index` to select
another step. Forces and torques are always read from the raw run, so rerunning
the command does not apply the correction twice. The separate step averages and
fit plots come from `analyze_thruster_run`.

All fits pass through the origin. Signed quadratics use:

```text
y = a * x * abs(x) + b * x
```

The fit YAML stores `[a, b]` under `coefficients`. The virtual command is based
on the thrust fit:

```text
n = esc_command
f(n) = q * n * abs(n) + l * n
C_f = f(1) = q + l
u = A * n * abs(n) + B * n
A = q / C_f
B = l / C_f
```

Under `models.virtual_command`, `coefficients` contains the forward mapping
`[A, B]`, `thrust_coefficient` contains `C_f`, and `torque_coefficient`
contains the proportional ESC virtual-command-to-roll-torque coefficient.

The exact inverse used for virtual-command plots and CSV data is:

```text
n = sign(u) * 2 * abs(u) / (B + sqrt(B^2 + 4 * A * abs(u)))
```

## Transient response analysis

The transient command extracts every start (`0` to a commanded speed) and stop
(commanded speed to `0`) from the raw wrench and command CSVs. It fits one set
of dynamics to all runs:

```bash
ros2 run loadcell_testing analyze_thruster_transients \
  --config /home/finnb/ros2_underlay/src/loadcell_testing/config/thruster_analysis.yaml \
  --run /home/finnb/ros2/data/thruster_loadcell_runs/20260924_* \
  --no-show
```

The analysis applies a regularized inverse of the publisher's causal
eighth-order Butterworth low-pass to each recorded 25 Hz wrench trace, then
fits unfiltered thruster models to those corrected traces. The inverse uses
the filter's frequency response within the recorded bandwidth. If the
publisher used a different cutoff, pass `--loadcell-lowpass-hz <Hz>`.
Use `--inverse-regularization <value>` (between 0 and 1) to adjust the
noise/sharpness tradeoff; the default is `0.02`. The original DAQ sample rate
and sampling phase were not recorded, so the inverse is approximate.

The input is the virtual command from the pooled steady-state thrust fit.
For each step, the virtual command is divided by its signed step magnitude.
Thrust is normalized as `(thrust - initial_thrust) /
(steady_thrust - initial_thrust)`, where `initial_thrust` is the first wrench
sample at or after the start command. The recorded start is zero at that
sample; the inverse-filtered estimate can differ slightly.
The normalized virtual input gives unit steps `0 → 1` for starts and `1 → 0`
for stops, including negative ESC commands. Start curves cover 2 s after the transition; stop
curves cover up to 1 s of the following zero interval. Each transient has
equal weight in the pooled fit.

The thruster transfer functions have unit DC gain. Plotted predictions are
unfiltered and fitted against the inverse-filtered thrust. The delay
parameters in the last two fits are residual command-to-corrected-wrench
delays:

```text
first order, no delay: 1 / (tau s + 1)
first order, residual delay: exp(-T_d s) / (tau s + 1)
second order, residual delay: exp(-T_d s) * omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)
```

All three models use the same time-series fitting objective, with equal
weight per transition. The second-order fit is a representative pooled model:
overshoot varies with command magnitude, and starts and stops are not exactly
symmetric. The fit report also lists the predicted start peak and its error.

Outputs are written under `analysis/transient/`: a plot overlaying all corrected
start and stop curves on shared time and thrust axes (`-0.3` to `1.0` s), a CSV
with corrected and recorded samples and the virtual input, a model-curve CSV for TikZ,
and a YAML file with parameters and fit errors. The fit still uses the full
2 s start window. `csv/thruster_transient_overshoots.csv` ranks start steps
whose corrected peak exceeds 1.5 times steady thrust and lists the run, step,
command, peak time, and excess thrust in newtons.
`plots/thruster_transient_combined_step.png` aligns every start and stop at
the same command transition. Since normalized stop thrust falls from 1 to 0,
the plot uses `1 - thrust_unit` to show each stop as a rising 0 to 1 step.
It overlays both medians and the current simulation's 0.4 s first-order response.
For pgfplots, `csv/thruster_transient_start_curves.csv` and
`csv/thruster_transient_stop_curves.csv` contain separate inverse-filtered
curves with NaN row separators. `docs/thruster_transient_plots.tex` uses those
two files and `csv/thruster_transient_models.csv` on one axis, transforming
stop thrust with `1 - thrust_unit`.

For a comparison with the recorded response,
`plots/thruster_transient_inverse.png` overlays the measured and inverse-filtered
start and stop curves. `csv/thruster_transient_inverse.csv` contains both values
at each recorded timestamp. The inverse cannot recover content discarded by
25 Hz sampling; the report records the median precommand noise increase.
