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

## 3. Three-Record Method Studies

These smaller LTST studies are not replacements for the 86-record cohort results above. They are method-facing probes used to test how newer geometry branches behave under tighter causal and reproducibility constraints.

### Graph Sidecar Pilot

Entry points:

- [`artifact/ltst_three_record_graph_sidecar_test.py`](artifact/ltst_three_record_graph_sidecar_test.py)
- [`tests/test_ltst_three_record_graph_sidecar_test.py`](tests/test_ltst_three_record_graph_sidecar_test.py)

Use:

- Compare the updated geometry observer against the older graph branch on three held-out LTST records.

Main finding:

- The updated geometry observer consistently beat the legacy graph branch in that pilot.
- Naive graph blending was often harmful.

Interpretation:

- This study was valuable as an architecture signal: it justified keeping the updated geometry observer and dropping the older graph score as a default partner.

### Audited Online DMD Rebuild

Entry points:

- [`artifact/ltst_three_record_online_dmd_clean_rebuild.py`](artifact/ltst_three_record_online_dmd_clean_rebuild.py)
- [`tests/test_ltst_three_record_online_dmd_clean_rebuild.py`](tests/test_ltst_three_record_online_dmd_clean_rebuild.py)

Use:

- Rebuild the remote three-record DMD replication under stricter conditions:
  - remove future-label leakage
  - enforce causal modal alignment
  - move feature selection inside the leave-one-record-out folds
  - filter unstable baseline features before scoring

Main finding:

- After audit, the earlier oversized observer win disappeared.
- The DMD substrate became the strongest default lane.
- The updated geometry sidecar still helped selectively, especially in a short-horizon structured pocket represented by `s30742`.

Interpretation:

- This is the more trustworthy result.
- The geometry observer is not a universal replacement for substrate; it behaves more like a pocket-sensitive short-horizon transition detector.

### Pocket-Gating Probe

Entry points:

- [`artifact/ltst_three_record_online_dmd_gate_probe.py`](artifact/ltst_three_record_online_dmd_gate_probe.py)
- [`tests/test_ltst_three_record_online_dmd_gate_probe.py`](tests/test_ltst_three_record_online_dmd_gate_probe.py)

Use:

- Probe whether short-horizon observer routing can improve on the audited DMD baseline without reintroducing leakage.

Main finding:

- A fast-plus-short multiscale route improved the audited pooled `5s` and `10s` horizons over substrate alone.
- Adding a long `100`-row gate overconstrained the route and did not improve the early-horizon result.

Interpretation:

- The useful observer effect in this setup appears to be short-memory rather than long-memory.
- Routing matters more than pooled observer complexity.

### S4 Effective Operator Probe

Entry points:

- [`artifact/ltst_s4_effective_operator_probe.py`](artifact/ltst_s4_effective_operator_probe.py)
- [`tests/test_ltst_s4_effective_operator_probe.py`](tests/test_ltst_s4_effective_operator_probe.py)

Use:

- Examine LTST dynamics through a structured state-space / effective-operator lens, including operator splits, projected diagnostics, and imminent-onset labeling.

Interpretation:

- This is an operator-facing companion probe rather than a replacement detector.
- Its main role in the public repo is to preserve the S4-style operator work in a tested, reproducible form next to the LTST geometry studies.

### Method Takeaway

Across these three-record studies, the main novel lesson is not simply that "the observer works" or that the original replication failed. The more useful conclusion is:

- broad Lie activation and structured directional geometry are different regimes
- the updated geometry observer is strongest in short-horizon structured pockets
- routing beats pooled complexity in this setting
- substrate should remain the default lane unless a structured short-memory geometry pocket is active

## Entry Points

- Base cohort build: [`artifact/ltst_full_86_study.py`](artifact/ltst_full_86_study.py)
- HMM detector: [`artifact/ltst_transition_hmm.py`](artifact/ltst_transition_hmm.py)
- Cross-platform repro command:

```bash
python results/reproduce.py ltst-full
python results/reproduce.py ltst-hmm --run-dir artifact/runs/ltst_full_86_20260405T035505Z
```
