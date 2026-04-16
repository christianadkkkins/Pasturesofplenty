# Results Repro Pack

This repository contains a cross-domain geometric early-warning study in two settings: cardiac ST-event detection on PhysioNet `ltstdb`, and geomagnetic storm precursor detection on OMNI HRO2 one-minute solar wind data. The main publishable results live in [RESULTS_SUMMARY.csv](RESULTS_SUMMARY.csv): the finalized cardiac transition-HMM is visible before `7524/8000` ST episodes (`94.1%`) with median lead `429` beats and lead-aware specificity `0.671`, while the finalized layered solar HMM moves before `546/589` storms with median lead `279.5` minutes and minute-level specificity `0.756`.

If you only read one file first, read [RESULTS_SUMMARY.csv](RESULTS_SUMMARY.csv). If you only run one command first, run `python results/reproduce.py --help`.

## Contents

- `reproduce.py`: cross-platform Python runner for the cardiac cohort build, finalized cardiac HMM, finalized solar HMM, the exploratory transition-only solar dev pass, and Figure 2 solar plot reproduction.
- `reproduce_all.ps1`: runs the cardiac base cohort build, the cardiac transition-HMM, and the solar layered HMM in sequence.
- `reproduce_ltst_full.ps1`: rebuilds the 86-record LTST cohort from `ltstdb`.
- `reproduce_ltst_transition_hmm.ps1`: reruns the finalized onset-gated cardiac HMM and derives its beat-level specificity metrics.
- `reproduce_solar_hmm.ps1`: reruns the finalized layered solar HMM.
- `reproduce_solar_hmm_transition_only_dev.ps1`: runs the exploratory 2007-2008 solar transition-only dev pass with the selected beta pair and longer refractory.
- `reproduce_solar_alignment_figure.py`: regenerates the pooled solar alignment figure from a saved `solar_alignment.csv`.
- `derive_ltst_transition_hmm_metrics.py`: computes beat-level lead-aware and strict confusion-style metrics from a completed LTST HMM run.
- `requirements_minimal.txt`: pinned minimal Python dependencies for the reproduction entrypoints.
- `requirements_frozen.txt`: full `pip freeze` snapshot from the environment used to assemble this package.
- `DATA_SOURCES.md`: dataset names, official sources, local paths, and notes.
- `FIGURES.md`: figure-by-figure reproduction notes.
- `RESULTS_SUMMARY.csv`: frozen summary of the completed cardiac and solar HMM result sets.

## Minimal Toolchain

- Python 3.13 or another recent Python 3.x runtime
- Packages listed in `requirements_minimal.txt`
- Optional exact environment restore from `requirements_frozen.txt`
- Network access to PhysioNet if the LTST records are not already cached for `wfdb`
- Local OMNI HRO2 monthly CDF files under `data/omni_cache` for the solar run

The solar pipeline entrypoint, [solar.py](../solar.py), is included at repository root. The cardiac HMM entrypoint is [artifact/ltst_transition_hmm.py](../artifact/ltst_transition_hmm.py).

## Final Frozen Runs Summarized Here

- Cardiac HMM: `artifact/runs/ltst_full_86_20260405T035505Z/transition_hmm_onset_cooldown32`
- Solar layered HMM: `artifact/runs/solar_20260408T051653Z`

## Quick Start

```bash
python -m pip install -r results/requirements_minimal.txt
python results/reproduce.py --help
```

## Reproduce

1. Run `python results/reproduce.py ltst-full` to regenerate the base 86-record LTST cohort from `ltstdb`.
2. Run `python results/reproduce.py ltst-hmm --run-dir artifact/runs/<your_ltst_run>` to regenerate the finalized onset-gated cardiac HMM outputs.
3. That command also runs `python results/derive_ltst_transition_hmm_metrics.py --run-dir artifact/runs/<your_ltst_run>/transition_hmm_onset_cooldown32 --pre-event-beats 500` to write `ltst_transition_hmm_derived_metrics.csv`.
4. Run `python results/reproduce.py solar-hmm` to regenerate the finalized layered solar outputs.
5. Run `python results/reproduce.py solar-hmm-transition-dev` to test the anchored 2007-2008 transition-only solar variant with `beta_short=0.20`, `beta_long=0.003`, zero active tail, merge gap `5`, and cooldown `180`.
6. Or run `python results/reproduce.py all` to execute the full sequence.

The PowerShell scripts remain as Windows convenience wrappers, but `results/reproduce.py` is the platform-neutral entrypoint intended for GitHub users and CI.

## Tests

```bash
python -m unittest discover -s tests
```

The current test suite checks the LTST confusion-style metrics derivation on a fully synthetic mini-run.

## Figures

- Figure 2 solar alignment plot can be reproduced with `python results/reproduce.py solar-figure`.
- Figure-specific notes live in [FIGURES.md](FIGURES.md).

## Notes

- The cardiac HMM is a second-stage analysis. It consumes the derived LTST cohort produced by `artifact/ltst_full_86_study.py`, not raw PhysioNet records directly.
- The solar script reads OMNI HRO2 one-minute CDFs from `data/omni_cache` and writes a fresh `artifact/runs/solar_*` directory on each run.
- The cardiac HMM writes a very large `ltst_transition_hmm_windows.csv`. Keep that in mind before rerunning if disk space is tight.
