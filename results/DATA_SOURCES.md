# Data Sources

## Raw Datasets

| key | formal name | domain | official source | official URL | local path used here | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `ltstdb` | Long-Term ST Database | cardiac ECG / ST analysis | PhysioNet | [https://www.physionet.org/physiobank/database/ltstdb/](https://www.physionet.org/physiobank/database/ltstdb/) | fetched through `wfdb` by database key `ltstdb`; derived local cohort at `artifact/runs/ltst_full_86_20260405T035505Z` | completed cohort uses 86 records with beat annotations `atr` and ST annotations `stc` |
| `omni_hro2_1min` | OMNI HRO2 1-minute solar wind / geomagnetic data | heliophysics / geomagnetic storm analysis | NASA SPDF OMNIWeb | [https://omniweb.gsfc.nasa.gov/html/omni_source.html](https://omniweb.gsfc.nasa.gov/html/omni_source.html) | `data/omni_cache` with files matching `omni_hro2_1min_*.cdf` | completed layered run used monthly CDFs spanning 2002-01 through 2009-12 |

## Derived Local Inputs Used By The Final HMM Runs

| study | derived input | local path | produced by |
| --- | --- | --- | --- |
| Cardiac LTST HMM | completed 86-record cohort with `beat_level/`, `regime.csv`, and summary tables | `artifact/runs/ltst_full_86_20260405T035505Z` | `artifact/ltst_full_86_study.py` |
| Solar layered HMM | local OMNI cache summary and monthly CDF collection | `data/omni_cache` | external OMNI HRO2 cache files consumed directly by `solar.py` |

## Final Result Artifacts

| study | final run directory | main report | main summary |
| --- | --- | --- | --- |
| Cardiac LTST HMM | `artifact/runs/ltst_full_86_20260405T035505Z/transition_hmm_onset_cooldown32` | `artifact/runs/ltst_full_86_20260405T035505Z/transition_hmm_onset_cooldown32/ltst_transition_hmm_report.md` | `artifact/runs/ltst_full_86_20260405T035505Z/transition_hmm_onset_cooldown32/ltst_transition_hmm_summary.csv` |
| Solar layered HMM | `artifact/runs/solar_20260408T051653Z` | `artifact/runs/solar_20260408T051653Z/solar_report.md` | `artifact/runs/solar_20260408T051653Z/solar_layered_summary.csv` |
