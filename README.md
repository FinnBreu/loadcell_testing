# loadcell_testing

ROS 2 tools for the Schunk/ATI six-axis loadcell on the LAN-XI DAQ and for
single-thruster bench sweeps. The loadcell publisher and motor sweep are
independent processes.

## Loadcell publisher

Start one publisher and leave it running for troubleshooting, plotting,
recording, and motor tests:

```bash
ros2 launch loadcell_testing lanxi_loadcell_publisher.launch.py \
  vehicle_name:=uuv02 \
  calibration_file:=loadcell_calibration.yaml
```

The publisher explicitly configures all six channels with the LAN-XI `DC`
filter mode, disabling the module's default 7 Hz high-pass filter. A different
supported mode can be selected with `--input-filter` when intentionally needed.

It owns the LAN-XI recorder session at `192.168.0.139` and publishes:

- `/<vehicle_name>/loadcell/wrench` (`geometry_msgs/msg/WrenchStamped`)
- `/<vehicle_name>/loadcell/channels` (`std_msgs/msg/Float64MultiArray`)

After the stream has started, zero the unloaded wrench with:

```bash
ros2 service call /uuv02/loadcell/tare std_srvs/srv/Trigger "{}"
```

The service averages the most recent 0.5 seconds of calibrated wrench data and
applies the result as a session-only runtime tare. Keep the loadcell unloaded
and still during this interval. Reset only the runtime tare, returning to the
YAML calibration and tare, with:

```bash
ros2 service call /uuv02/loadcell/reset_tare std_srvs/srv/Trigger "{}"
```

The averaging window and service names can be changed with
`--tare-window-sec`, `--tare-service`, and `--reset-tare-service`.

The channel topic contains the six conditioned bridge voltages. The wrench is
calculated as:

```text
wrench = matrix * (channels - input_bias) - wrench_tare - runtime_tare
```

`input_bias` contains offsets in raw DAQ channel units. `wrench_tare` is
ordered `[Fx, Fy, Fz, Tx, Ty, Tz]` in the calibrated force and torque units.
Both vectors are independent and default to zero.

ROS 2 executes the installed package, so a relative filename such as
`loadcell_calibration.yaml` resolves to the copy under
`install/loadcell_testing/share/loadcell_testing/config`. Rebuild after editing
the source configuration. During calibration development, an explicit source
path uses the edited file immediately:

```bash
ros2 run loadcell_testing lanxi_loadcell_publisher \
  --vehicle-name uuv02 \
  --calibration-file "$PWD/src/loadcell_testing/config/loadcell_calibration.yaml"
```

The publisher logs the fully resolved calibration path and both offset vectors
at startup.

The publisher emits the newest complete LAN-XI sample block immediately. Its
periodic diagnostic includes `publish_age_ms`, parser CPU load, block sizes,
source sample rates, and `parser_buffer_bytes`. Parse load is reported both for
the latest diagnostic interval and cumulatively. The parser retains only an
incomplete trailing LAN-XI message, not a queue of old measurement blocks.

## Thruster sweep

With the publisher already running, start the motor sweep separately:

```bash
ros2 launch loadcell_testing single_thruster_sweep.launch.py \
  vehicle_name:=uuv02 \
  speeds:=0.10,0.20,0.30,0.40,0.50 \
  arm_motors:=true
```

The sweep commands ESC 1 (zero-based `--esc-index 0`) through the Pi-side ESC
commander in the selected vehicle namespace. It arms and disarms only through
`/<vehicle_name>/arm` and verifies both transitions on
`/<vehicle_name>/arming_state`; it never calls the FCU services. The state
confirmation is authoritative because the ESC commander may return
`success=false` even after applying the requested state.

The wrench topic defaults to the relative `loadcell/wrench`, producing
`/<vehicle_name>/loadcell/wrench`. The launch file defaults `arm_motors` to
`false`, which performs a zero-command dry run. For direct invocation,
`--vehicle-name uuv02` is equivalent to the launch argument.

The sweep records the command and all relevant vehicle telemetry in separate,
timestamped CSV streams under `thruster_loadcell_runs`:

- `*_steps.csv`: command and sampling intervals
- `*_commands.csv`: all eight commanded thruster values
- `*_wrench.csv`: calibrated force and torque
- `*_battery_state.csv`: pack voltage, current, charge state, and cell data
- `*_esc_voltages.csv`: voltage reported by each ESC
- `*_esc_rpms.csv`: RPM, revolution, and commutation counters for each ESC
- `*_config.json`: run settings and per-stream sample counts

Every telemetry row includes receipt time, source timestamp, repeat, step,
phase, and the current selected-thruster command. The default subscriptions are
`/<vehicle_name>/loadcell/wrench`, `/<vehicle_name>/battery_state`,
`/<vehicle_name>/esc_voltages`, and `/<vehicle_name>/esc_rpms`.

Stopping or completing a sweep does not stop the loadcell publisher or touch
the LAN-XI recorder state. If no wrench publisher is available, the sweep logs
a warning and continues independently.
