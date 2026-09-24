# loadcell_testing

ROS 2 tools for the Schunk/ATI six-axis loadcell, LAN-XI DAQ, and
single-thruster bench tests. The loadcell publisher and motor sweep run
independently.

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
  --config src/loadcell_testing/config/thruster_analysis.yaml \
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

With `d = stand_signs[fwd/rev]` and `p = propeller_signs[cw/ccw]`:

```text
n = p * esc_command
thrust = thrust_sign * d * F_selected
roll_torque = roll_torque_sign * d * p * M_motor_selected
```

All runs are pooled for fitting. Results go to the sibling `analysis` directory
(`--output-dir` overrides it). The fit YAML includes the effective configuration.
Step CSVs retain the original loadcell wrench, add `motor_torque_x/y/z` in
loadcell axes, and store the normalized thrust and roll torque used for fitting.

All fits pass through the origin. Signed quadratics use:

```text
y = a * x * abs(x) + b * x
```

The fit YAML stores `[a, b]` under `coefficients`. The virtual command is based
on the thrust fit:

```text
f(n) = q * n * abs(n) + l * n
C_f = f(1) = q + l
u = A * n * abs(n) + B * n
A = q / C_f
B = l / C_f
```

Under `models.virtual_command`, `coefficients` contains the forward mapping
`[A, B]`, `thrust_coefficient` contains `C_f`, and `torque_coefficient`
contains the proportional virtual-command-to-roll-torque coefficient.

The exact inverse used for virtual-command plots and CSV data is:

```text
n = sign(u) * 2 * abs(u) / (B + sqrt(B^2 + 4 * A * abs(u)))
```
