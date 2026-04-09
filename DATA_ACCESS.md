# Data Access

This page gives the exact dataset names, official source pages, and the local paths expected by the code in this repository.

## LTST Cardiac Data (`PhysioNet ltstdb`)

Official source:

- PhysioNet Long-Term ST Database: [https://physionet.org/content/ltstdb/](https://physionet.org/content/ltstdb/)

What the repo uses:

- Database key: `ltstdb`
- Beat annotations: `atr`
- ST annotations: `stc`
- Access mode used by the cardiac scripts: remote streaming through `wfdb` with `pn_dir="ltstdb"`

Exact script path:

- [`artifact/ltst_full_86_study.py`](artifact/ltst_full_86_study.py)

Two supported access modes:

1. Remote access, which is what the repo currently uses

```python
import wfdb
record = wfdb.rdrecord("s20011", pn_dir="ltstdb")
st_ann = wfdb.rdann("s20011", "stc", pn_dir="ltstdb")
beat_ann = wfdb.rdann("s20011", "atr", pn_dir="ltstdb")
```

2. Optional local mirror if you want the raw PhysioNet files on disk

```bash
python - <<'PY'
import wfdb
wfdb.dl_database("ltstdb", dl_dir="data/ltstdb")
PY
```

If you create a local mirror, the raw PhysioNet files will live under `data/ltstdb/`.

## Solar Data (OMNI HRO2 1-minute)

Official sources:

- NASA SPDF OMNIWeb landing page: [https://omniweb.gsfc.nasa.gov/](https://omniweb.gsfc.nasa.gov/)
- NASA open-data landing page for OMNI 1-minute data: [https://data.nasa.gov/dataset/omni-1-min-data-set](https://data.nasa.gov/dataset/omni-1-min-data-set)

What the repo uses:

- Monthly CDF files matching `omni_hro2_1min_*.cdf`
- Expected cache root: `data/omni_cache`
- Recursive discovery is enabled, so files can live either directly under `data/omni_cache/` or one level deeper as in `data/omni_cache/omni_cache/`

Exact script path:

- [`solar.py`](solar.py)

Exact local file pattern expected by the code:

- `data/omni_cache/omni_hro2_1min_20090801_v01.cdf`
- `data/omni_cache/omni_cache/omni_hro2_1min_20020101_v01.cdf`
- in general: `data/omni_cache/**/omni_hro2_1min_YYYYMM01_v01.cdf`

To use the frozen solar result path in this repo, place the monthly OMNI HRO2 CDF files under `data/omni_cache` and run:

```bash
python results/reproduce.py solar-hmm
```

## Micro-PMU October 1 Data

Official references:

- LBNL Power Data portal: [https://secpriv.lbl.gov/project/powerdata/](https://secpriv.lbl.gov/project/powerdata/)
- Open micro-PMU reference dataset page: [https://eta.lbl.gov/publications/open-pmu-real-world-reference](https://eta.lbl.gov/publications/open-pmu-real-world-reference)

What the repo uses:

- A single CSV file for the October 1, 2015 a6 bus1 stream
- Expected path:

`data/Micro PMU October 1 Dataset/_LBNL_a6_bus1_2015-10-01.csv`

Exact script paths:

- [`artifact/micro_pmu_oct1_experiment.py`](artifact/micro_pmu_oct1_experiment.py)
- [`artifact/micro_pmu_jet.py`](artifact/micro_pmu_jet.py)
- [`artifact/micro_pmu_sequence_episode_evaluator.py`](artifact/micro_pmu_sequence_episode_evaluator.py)

This repo does not automate the micro-PMU download. Acquire the October 1 a6 bus1 CSV from the LBNL Power Data / Open micro-PMU distribution path, then place it at the exact local path above before running the micro-PMU experiments.
