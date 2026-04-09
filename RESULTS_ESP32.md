# LTST ESP32 Contract Results

These experiments validate the frozen V1 LTST cardiac beat packet under ESP32-style fixed-point emulation and replay-sample export. They are derived from the same PhysioNet `ltstdb` cohort used by the host-side cardiac study, but the question here is different: does the embedded packet preserve primitive ordering and magnitude well enough to keep the host-side geometric interpretation intact?

Exact source-data and replay-asset notes are in [DATA_ACCESS.md](DATA_ACCESS.md#ltst-esp32-replay-and-contract-data).

## Frozen Artifact

- [`artifact/runs/ltst_full_86_20260405T035505Z/esp32_contract_sim/ltst_esp32_contract_report.md`](artifact/runs/ltst_full_86_20260405T035505Z/esp32_contract_sim/ltst_esp32_contract_report.md)
- [`artifact/runs/ltst_full_86_20260405T035505Z/esp32_contract_sim/ltst_esp32_contract_summary.csv`](artifact/runs/ltst_full_86_20260405T035505Z/esp32_contract_sim/ltst_esp32_contract_summary.csv)
- [`artifact/runs/ltst_full_86_20260405T035505Z/esp32_contract_sim/ltst_esp32_contract_ordering.csv`](artifact/runs/ltst_full_86_20260405T035505Z/esp32_contract_sim/ltst_esp32_contract_ordering.csv)
- [`artifact/runs/ltst_full_86_20260405T035505Z/esp32_contract_sim/ltst_esp32_contract_packets.csv`](artifact/runs/ltst_full_86_20260405T035505Z/esp32_contract_sim/ltst_esp32_contract_packets.csv)

## Main Finding

- On the four benchmark LTST exemplar records, the ESP32 fixed-point packet preserves the baseline ordering of `poincare_b`, `drift_norm`, and the derived stiffness proxy at the `500`-beat window exactly.

## Key Numbers

- Benchmark records: `4`
- Total beats evaluated: `388590`
- Mean median absolute error:
  - `poincare_b = 1.5317e-05`
  - `drift_norm = 6.0996e-05`
  - `energy_asym = 1.5275e-05`
- Minimum rank correlation across records:
  - `poincare_b_spearman = 0.999999`
  - `drift_norm_spearman = 0.999919`
- `energy_asym` sign agreement: `1.0` on all four records
- Maximum saturation fraction on any record: `0.0122`
- `div_guard_fraction = 0` on all four records
- `low_curr_energy_fraction = 0` on all four records
- `low_past_energy_fraction = 0` on all four records

## Interpretation

- This is a contract-fidelity result, not a detection result.
- The embedded packet appears numerically stable enough to preserve the host-side ordering story on the exemplar set.
- The main remaining step is hardware-session validation over UART and replay/live ingestion, not another host-side quantization pass.

## Entry Points

- Contract simulation: [`ESP32/ltst_esp32_contract_sim.py`](ESP32/ltst_esp32_contract_sim.py)
- Replay export: [`ESP32/ltst_v1_export_replay_samples.py`](ESP32/ltst_v1_export_replay_samples.py)
- Embedded workspace notes: [`ESP32/README.md`](ESP32/README.md)

Cross-platform repro from the frozen LTST cohort:

```bash
python results/reproduce.py esp32-contract --run-dir artifact/runs/ltst_full_86_20260405T035505Z
```
