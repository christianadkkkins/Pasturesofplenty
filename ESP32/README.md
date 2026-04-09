# ESP32 Cardiac V1

This folder holds the embedded-facing LTST cardiac scaffold. The public interface stays intentionally small: one frozen 12-byte beat packet on `UART1`, plus optional sideband debug and HMM state on `UART0`.

## What The ESP32 Does

In its current shape, the ESP32 is a **beat-to-primitive engine**:

1. ingest samples from one of three source modes
2. run causal beat detection
3. compute the three frozen LTST primitives
4. emit the unchanged V1 packet over `UART1`
5. optionally track a small local 3-state HMM on accepted beats and print that state on `UART0`

The main packet contract does **not** change when the live path is enabled.

## Public Packet Contract

The canonical packet is defined in [ltst_v1_feature_packet.h](C:/Users/Admin/Documents/Bearings/ESP32/ltst_v1_feature_packet.h):

- `beat_index`
- `poincare_b` as `UQ2.14`
- `drift_norm` as `UQ4.12`
- `energy_asym` as `SQ1.14`
- `quality_flags`

Packet size is fixed at `12` bytes.

The framed `UART1` transport is implemented in:

- [main/ltst_v1_transport.h](C:/Users/Admin/Documents/Bearings/ESP32/main/ltst_v1_transport.h)
- [main/ltst_v1_transport.c](C:/Users/Admin/Documents/Bearings/ESP32/main/ltst_v1_transport.c)

## Firmware Layout

Main ESP-IDF sources:

- [main/main.c](C:/Users/Admin/Documents/Bearings/ESP32/main/main.c)
- [main/ltst_v1_commands.h](C:/Users/Admin/Documents/Bearings/ESP32/main/ltst_v1_commands.h)
- [main/ltst_v1_commands.c](C:/Users/Admin/Documents/Bearings/ESP32/main/ltst_v1_commands.c)
- [main/ltst_v1_live_pipeline.h](C:/Users/Admin/Documents/Bearings/ESP32/main/ltst_v1_live_pipeline.h)
- [main/ltst_v1_live_pipeline.c](C:/Users\Admin/Documents/Bearings/ESP32/main/ltst_v1_live_pipeline.c)
- [main/ltst_v1_replay_samples.h](C:/Users\Admin/Documents/Bearings/ESP32/main/ltst_v1_replay_samples.h)
- [main/ltst_v1_replay_samples.c](C:/Users/Admin/Documents/Bearings/ESP32/main/ltst_v1_replay_samples.c)

Generated replay assets:

- [generated/ltst_v1_replay_samples_generated.h](C:/Users/Admin/Documents/Bearings/ESP32/generated/ltst_v1_replay_samples_generated.h)
- [generated/ltst_v1_replay_samples_manifest.json](C:/Users/Admin/Documents/Bearings/ESP32/generated/ltst_v1_replay_samples_manifest.json)

## Source Modes

The console command layer supports four named modes:

- `selftest`
  - emits the fixed packet self-test vectors
- `replay_packets`
  - the original float-to-packet bench path
- `replay_samples`
  - deterministic waveform replay from exported LTST traces
- `adc_live`
  - timer-driven single-lead ADC ingest at `250 Hz`

Mode selection is done on `UART0`:

```text
mode selftest
mode replay_packets
mode replay_samples
mode adc_live
```

Aliases still work:

- `mode replay` -> `replay_packets`
- `mode adc` -> `adc_live`

## Live Path

The live path in [main/ltst_v1_live_pipeline.c](C:/Users/Admin/Documents/Bearings/ESP32/main/ltst_v1_live_pipeline.c) does the following:

- sample cadence fixed to `250 Hz`
- circular sample buffer length `512`
- causal detector:
  - derivative
  - squaring
  - moving-window integrator
  - adaptive threshold
  - refractory and searchback
- beat primitives on accepted beats:
  - `poincare_b`
  - `drift_norm`
  - `energy_asym`

The formulas are aligned to the LTST host study in [ltst_full_86_study.py](C:/Users/Admin/Documents/Bearings/artifact/ltst_full_86_study.py).

## Small HMM

The firmware also carries a small advisory 3-state HMM:

- `baseline`
- `transition`
- `active`

Important boundary:

- HMM output is **sideband only**
- HMM state is **not** added to the V1 packet
- host tools that only consume `UART1` do not need to change

The HMM is enabled by default in `replay_samples` and `adc_live`. It is ignored in `replay_packets` and `selftest`.

Console commands:

```text
hmm on
hmm off
record s20021
status
```

`status` now reports:

- source mode
- ingest counters
- detector counters
- replay position
- HMM state
- transition score
- episode id and duration

## Host Tools

Packet / transport bench tools:

- [ltst_v1_uart_decoder.py](C:/Users/Admin/Documents/Bearings/ESP32/ltst_v1_uart_decoder.py)
- [ltst_v1_uart_replay.py](C:/Users/Admin/Documents/Bearings/ESP32/ltst_v1_uart_replay.py)
- [ltst_v1_uart_capture.py](C:/Users/Admin/Documents/Bearings/ESP32/ltst_v1_uart_capture.py)
- [ltst_v1_bench_session.py](C:/Users/Admin/Documents/Bearings/ESP32/ltst_v1_bench_session.py)
- [ltst_v1_bench_compare.py](C:/Users/Admin/Documents/Bearings/ESP32/ltst_v1_bench_compare.py)

Replay-sample and live-path tools:

- [ltst_v1_export_replay_samples.py](C:/Users/Admin/Documents/Bearings/ESP32/ltst_v1_export_replay_samples.py)
- [ltst_v1_live_compare.py](C:/Users/Admin/Documents/Bearings/ESP32/ltst_v1_live_compare.py)
- [ltst_v1_hmm_reference.py](C:/Users/Admin/Documents/Bearings/ESP32/ltst_v1_hmm_reference.py)
- [ltst_v1_hmm_log_parser.py](C:/Users/Admin/Documents/Bearings/ESP32/ltst_v1_hmm_log_parser.py)

## Recommended Bench Flow

### 1. Packet contract only

Use the original replay path when you only want to validate quantization, framing, and packet integrity:

1. send exemplar packets over `UART0` with [ltst_v1_uart_replay.py](C:/Users/Admin/Documents/Bearings/ESP32/ltst_v1_uart_replay.py)
2. capture `UART1` bytes with [ltst_v1_uart_capture.py](C:/Users/Admin/Documents/Bearings/ESP32/ltst_v1_uart_capture.py)
3. decode with [ltst_v1_uart_decoder.py](C:/Users/Admin/Documents/Bearings/ESP32/ltst_v1_uart_decoder.py)
4. score with [ltst_v1_bench_compare.py](C:/Users/Admin/Documents/Bearings/ESP32/ltst_v1_bench_compare.py)

### 2. Replay-sample live path

Use this when you want to exercise the actual detector + primitive engine:

1. generate replay traces with [ltst_v1_export_replay_samples.py](C:/Users/Admin/Documents/Bearings/ESP32/ltst_v1_export_replay_samples.py)
2. run a session in `replay_samples` mode with [ltst_v1_bench_session.py](C:/Users/Admin/Documents/Bearings/ESP32/ltst_v1_bench_session.py)
3. compare beat timing and primitives with [ltst_v1_live_compare.py](C:/Users/Admin/Documents/Bearings/ESP32/ltst_v1_live_compare.py)
4. build a host HMM reference with [ltst_v1_hmm_reference.py](C:/Users/Admin/Documents/Bearings/ESP32/ltst_v1_hmm_reference.py)
5. parse sideband HMM logs with [ltst_v1_hmm_log_parser.py](C:/Users/Admin/Documents/Bearings/ESP32/ltst_v1_hmm_log_parser.py)

## Example Commands

Generate replay assets:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe' 'C:\Users\Admin\Documents\Bearings\ESP32\ltst_v1_export_replay_samples.py' --run-dir 'C:\Users\Admin\Documents\Bearings\artifact\runs\ltst_full_86_20260405T035505Z'
```

Run a packet-only dry run:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe' 'C:\Users\Admin\Documents\Bearings\ESP32\ltst_v1_bench_session.py' --run-dir 'C:\Users\Admin\Documents\Bearings\artifact\runs\ltst_full_86_20260405T035505Z' --dry-run
```

Run a replay-sample bench session:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe' 'C:\Users\Admin\Documents\Bearings\ESP32\ltst_v1_bench_session.py' --run-dir 'C:\Users\Admin\Documents\Bearings\artifact\runs\ltst_full_86_20260405T035505Z' --source-mode replay_samples --record s20021 --console-serial COM5 --data-serial COM6
```

Compare replay-sample results:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe' 'C:\Users\Admin\Documents\Bearings\ESP32\ltst_v1_live_compare.py' --session-dir 'C:\Users\Admin\Documents\Bearings\artifact\runs\<session_dir>'
```

Parse HMM sideband logs:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe' 'C:\Users\Admin\Documents\Bearings\ESP32\ltst_v1_hmm_log_parser.py' --session-dir 'C:\Users\Admin\Documents\Bearings\artifact\runs\<session_dir>'
```

Build a host-side HMM reference for that same session:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe' 'C:\Users\Admin\Documents\Bearings\ESP32\ltst_v1_hmm_reference.py' --session-dir 'C:\Users\Admin\Documents\Bearings\artifact\runs\<session_dir>'
```

Then compare the device sideband against the host reference:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe' 'C:\Users\Admin\Documents\Bearings\ESP32\ltst_v1_hmm_log_parser.py' --session-dir 'C:\Users\Admin\Documents\Bearings\artifact\runs\<session_dir>' --reference-csv 'C:\Users\Admin\Documents\Bearings\artifact\runs\<session_dir>\hmm_reference.csv'
```

## Notes

- The ESP32 is still a **primitive engine**, not a clinical decision device.
- The main validation target remains packet and primitive fidelity.
- The HMM is intentionally advisory so the irreversible interface to future hardware stays simple.
- The corresponding host-side contract document is [ltst_v1_feature_contract.md](C:/Users/Admin/Documents/Bearings/artifact/ltst_v1_feature_contract.md).
