# Solar Results

These experiments test whether the same geometric precursor idea seen in cardiac data also appears in solar-wind dynamics before geomagnetic storms.

Exact data-access instructions are in [DATA_ACCESS.md](DATA_ACCESS.md#solar-data-omni-hro2-1-minute).

## 1. Baseline Geometric Routing

Frozen layered artifact:

- [`artifact/runs/solar_20260408T051653Z/solar_report.md`](artifact/runs/solar_20260408T051653Z/solar_report.md)
- [`artifact/runs/solar_20260408T051653Z/solar_layered_summary.csv`](artifact/runs/solar_20260408T051653Z/solar_layered_summary.csv)

Key numbers:

- Storm events evaluated: `589`
- `poincare_b` before storm: `555 / 589`
- `gram_spread_sq` before storm: `489 / 589`
- `poincare_b` before paired southward `Bz`: `424 / 589`
- `gram_spread_sq` before paired southward `Bz`: `385 / 589`
- Median `poincare_b` lead to storm: `693.0` minutes
- Median `gram_spread_sq` lead to storm: `701.0` minutes
- Median southward `Bz` lead to storm: `57.0` minutes

Interpretation:

- Geometric precursor motion is real in this domain, but it does not simply replace the canonical `Bz` cue.

## 2. Layered Solar HMM

Frozen artifact:

- [`artifact/runs/solar_20260408T051653Z/solar_report.md`](artifact/runs/solar_20260408T051653Z/solar_report.md)
- [`artifact/runs/solar_20260408T051653Z/solar_hmm_confusion_matrix.csv`](artifact/runs/solar_20260408T051653Z/solar_hmm_confusion_matrix.csv)

Key numbers:

- `solar_hmm` before storm: `546 / 589` = `0.9270`
- `solar_hmm` before paired southward `Bz`: `339 / 589`
- `solar_hmm` chosen as earliest combined lead: `125 / 589`
- Median HMM lead to storm: `279.5` minutes
- Minute-level sensitivity: `0.3629`
- Minute-level specificity: `0.7555`
- Minute-level precision: `0.2349`
- Alert occupancy fraction: `0.2648`

Interpretation:

- The layered HMM is a meaningful precursor detector, but still behaves more like a useful monitoring layer than a clean operational storm alarm.

## Figure

- Figure 2 alignment plot: [`artifact/figures/figure2_solar_alignment_plot.png`](artifact/figures/figure2_solar_alignment_plot.png)
- The minute-level `solar_alignment.csv` file is not tracked in git because it exceeds GitHub's file-size limit.
- To regenerate the figure from data, first run:

```bash
python results/reproduce.py solar-hmm
```

- Then regenerate the plot from the newly created alignment CSV:

```bash
python results/reproduce.py solar-figure \
  --alignment-csv artifact/runs/<your_solar_run>/solar_alignment.csv \
  --output artifact/figures/figure2_solar_alignment_plot.png
```

## Entry Point

- Main script: [`solar.py`](solar.py)
- Cross-platform repro command:

```bash
python results/reproduce.py solar-hmm
```
