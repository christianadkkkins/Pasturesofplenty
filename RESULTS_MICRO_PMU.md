# Micro-PMU Results

These experiments use the October 1, 2015 LBNL micro-PMU dataset and test three different views of event detection:

1. projective / memory features on the raw day file
2. ray-space / jet-space features
3. sequence-manifold episode decoding

Exact data-access instructions are in [DATA_ACCESS.md](DATA_ACCESS.md#micro-pmu-october-1-data).

## 1. Projective Baseline Experiment

Frozen artifact:

- [`artifact/runs/micro_pmu_oct1_20260407T010234Z/micro_pmu_report.md`](artifact/runs/micro_pmu_oct1_20260407T010234Z/micro_pmu_report.md)

Key numbers:

- Rows scanned: `10,368,000`
- Event onsets: `392`
- Events with a detected lead: `223 / 392`
- Best AUROC: `phase_current_spread = 0.5172`
- Best projective-family AUROC cluster: about `0.5163`

Interpretation:

- The projective state view is present, but only weakly separated from baseline on this file.

## 2. Jet Experiment

Frozen artifact:

- [`artifact/runs/micro_pmu_jet_20260408T020759Z/micro_pmu_jet_report.md`](artifact/runs/micro_pmu_jet_20260408T020759Z/micro_pmu_jet_report.md)

Key numbers:

- Event onsets: `392`
- Events with a detected lead: `291 / 392`
- Best raw AUROC: `voltage_angle_spread = 0.5272`
- Best jet AUROC: `jet_twist_delta = 0.5132`
- Lead-detected events vs prior PMU baseline: `291` vs `243`

Interpretation:

- Jet-space improves event lead coverage, but not overall separability.
- It looks more useful as a precursor-localization layer than as a decisive classifier.

## 3. Sequence Episode Evaluator

Frozen artifact:

- [`artifact/runs/micro_pmu_sequence_episode_20260407T023719Z/micro_pmu_sequence_episode_report.md`](artifact/runs/micro_pmu_sequence_episode_20260407T023719Z/micro_pmu_sequence_episode_report.md)

Key numbers:

- Episodes found: `1`
- Median episode duration: `86,400` seconds
- Event capture any-rate: `1.0`
- Event capture pre-rate: `1.0`
- Median earliest pre-event lead: `44,464` seconds

Interpretation:

- This decoder configuration collapsed into one day-long episode.
- It captures events only because it never really turns off, so it is not operationally usable in its current form.

## Entry Points

- Projective baseline: [`artifact/micro_pmu_oct1_experiment.py`](artifact/micro_pmu_oct1_experiment.py)
- Jet-space: [`artifact/micro_pmu_jet.py`](artifact/micro_pmu_jet.py)
- Sequence episode evaluator: [`artifact/micro_pmu_sequence_episode_evaluator.py`](artifact/micro_pmu_sequence_episode_evaluator.py)
