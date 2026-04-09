# Pasturesofplenty Results

This repository currently has four main experiment tracks with frozen result artifacts:

- [RESULTS_LTST.md](RESULTS_LTST.md): cardiac experiments on the PhysioNet Long-Term ST Database (`ltstdb`)
- [RESULTS_ESP32.md](RESULTS_ESP32.md): embedded LTST packet-contract validation and replay assets for the ESP32 scaffold
- [RESULTS_SOLAR.md](RESULTS_SOLAR.md): solar-wind / geomagnetic storm experiments on OMNI HRO2 one-minute data
- [RESULTS_MICRO_PMU.md](RESULTS_MICRO_PMU.md): distribution-grid event experiments on the LBNL micro-PMU October 1 dataset

Exact data-access instructions, local file expectations, and official source links are in [DATA_ACCESS.md](DATA_ACCESS.md).

## Frozen Result Snapshot

| track | main result | frozen artifact |
| --- | --- | --- |
| LTST cardiac HMM | `7524 / 8000` ST episodes visible before onset, median lead `429` beats, lead-aware specificity `0.671` | `artifact/runs/ltst_full_86_20260405T035505Z/transition_hmm_onset_cooldown32` |
| LTST ESP32 contract | ordering preserved on all `4` benchmark exemplar records, with minimum rank correlation `0.999919` and max saturation `0.0122` | `artifact/runs/ltst_full_86_20260405T035505Z/esp32_contract_sim` |
| Solar layered HMM | `546 / 589` storms detected before onset, median lead `279.5` minutes, specificity `0.756` | `artifact/runs/solar_20260408T051653Z` |
| Micro-PMU jet | best jet AUROC `0.513`, lead-detected events `291 / 392`, but still weak separation overall | `artifact/runs/micro_pmu_jet_20260408T020759Z` |

## Reproduction Entry Points

- Cross-platform runner: [`results/reproduce.py`](results/reproduce.py)
- Repro package overview: [`results/README.md`](results/README.md)
- Frozen summary CSV: [`results/RESULTS_SUMMARY.csv`](results/RESULTS_SUMMARY.csv)
