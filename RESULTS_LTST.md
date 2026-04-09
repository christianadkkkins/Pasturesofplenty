# LTST Cardiac Results

These experiments use the PhysioNet Long-Term ST Database (`ltstdb`) and focus on two questions:

1. Does baseline projective geometry reveal stable phenotype structure before ST events?
2. Can the same geometric signal be turned into an early warning detector with useful lead time?

Exact data-access instructions are in [DATA_ACCESS.md](DATA_ACCESS.md#ltst-cardiac-data-physionet-ltstdb).

## 1. Baseline Taxonomy

Frozen artifact:

- [`artifact/runs/ltst_full_86_20260405T035505Z/memory_taxonomy_v2/memory_taxonomy_report.md`](artifact/runs/ltst_full_86_20260405T035505Z/memory_taxonomy_v2/memory_taxonomy_report.md)

Main finding:

- The completed 86-record cohort supports a reproducible `loose_orbit` family, but does **not** support a single coherent `rigid_orbit` class.

Key numbers:

- Records analyzed: `86`
- Best loose-vs-rest AUROC:
  - `baseline_p95_temporal_volume_3 = 0.738`
  - `baseline_p95_temporal_area_loss = 0.715`
- Best rigid-vs-constrained AUROC:
  - `temporal_volume_3_st_abs_relative_change = 0.590`
- 3-way nearest-centroid performance:
  - macro-F1 `0.2815`
  - macro balanced accuracy `0.3393`
  - `rigid_orbit` recovered correctly: `0 / 26`

Interpretation:

- The geometry is informative, but the stable cardiac claim is currently `loose` vs non-`loose`, not a clean 3-class phenotype taxonomy.

## 2. Transition HMM

Frozen artifact:

- [`artifact/runs/ltst_full_86_20260405T035505Z/transition_hmm_onset_cooldown32/ltst_transition_hmm_report.md`](artifact/runs/ltst_full_86_20260405T035505Z/transition_hmm_onset_cooldown32/ltst_transition_hmm_report.md)
- [`artifact/runs/ltst_full_86_20260405T035505Z/transition_hmm_onset_cooldown32/ltst_transition_hmm_derived_metrics.csv`](artifact/runs/ltst_full_86_20260405T035505Z/transition_hmm_onset_cooldown32/ltst_transition_hmm_derived_metrics.csv)

Configuration:

- Onset-gated HMM alert extraction
- `16`-beat active tail
- `96`-beat alert cap
- `32`-beat cooldown
- Lead-aware positive window: `500` beats before each ST episode

Key numbers:

- ST episodes evaluated: `8000`
- HMM visible pre-ST: `7524 / 8000` = `0.9405`
- Median HMM lead: `429` beats
- Lead-aware specificity: `0.6710`
- Lead-aware sensitivity: `0.3428`
- Lead-aware precision: `0.4273`
- Alert occupancy fraction: `0.3347`
- False-positive episode fraction: `0.5741`

Reference comparisons from the report:

- Routed visible fraction: `0.116625`
- Hybrid visible fraction: `0.177`
- HMM improves over routed on `7264` events
- HMM improves over hybrid on `7190` events

Interpretation:

- The detector is early and broad, but still too active to be treated as a clean clinical alarm.
- The main translational bottleneck is false-alert burden, not event coverage.

## Entry Points

- Base cohort build: [`artifact/ltst_full_86_study.py`](artifact/ltst_full_86_study.py)
- HMM detector: [`artifact/ltst_transition_hmm.py`](artifact/ltst_transition_hmm.py)
- Cross-platform repro command:

```bash
python results/reproduce.py ltst-full
python results/reproduce.py ltst-hmm --run-dir artifact/runs/ltst_full_86_20260405T035505Z
```
