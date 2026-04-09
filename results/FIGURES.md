# Figure Reproduction

## Figure 2

`Figure 2. Solar wind router alignment around geomagnetic storm onset.` can be regenerated directly from the saved alignment table or by rerunning the full solar layered pipeline.

### Regenerate from a local alignment CSV

```bash
python results/reproduce.py solar-figure \
  --alignment-csv artifact/runs/<your_solar_run>/solar_alignment.csv \
  --output artifact/figures/figure2_solar_alignment_plot.png
```

### Regenerate from raw OMNI HRO2 data

```bash
python results/reproduce.py solar-hmm
```

That command creates a fresh `artifact/runs/solar_*/solar_alignment.csv` and rewrites `artifact/runs/solar_*/solar_alignment_plot.png` as part of the full solar pipeline.

## Figure 1

`Figure 1. Regime dissociation in the 5-record LTST pilot.` is still backed by the pilot notebook at `artifact/minimal_5_record_dual_kernel_check_remote.ipynb`.

That figure has not yet been split into a standalone script in this repo. The notebook remains the authoritative source for the pilot visualization.
