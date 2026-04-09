# LTST Minimal Bench Prototype Definition

## Goal

Define the smallest cardiac bench reproduction that is still scientifically useful.

The bench prototype should reproduce the **V1 primitive feature stream**, not the full manuscript router. If we can match the Python reference on the primitive geometry, we preserve the validated cardiac lane and keep later routing choices open.

## Prototype Scope

Reproduce exactly these three beat-level outputs from the V1 contract:

1. `poincare_b`
2. `drift_norm`
3. `energy_asym`

Everything else stays host-side for this prototype:

- rolling medians
- stiffness ratios
- taxonomy thresholds
- routing decisions

## Reference Cases

Use these four LTST exemplars because they cover the phenotype ladder cleanly:

| Record | Regime | `poincare_b_p50` | `median_drift_norm` |
|:--|:--|--:|--:|
| `s20021` | `angle_only` | `0.8599` | `0.1229` |
| `s20041` | `long_only` | `0.9742` | `0.0175` |
| `s20151` | `neither` | `0.9955` | `0.0110` |
| `s30742` | `neither` | `0.9984` | `0.0090` |

These values come from [invariant_summary.csv](C:/Users/Admin/Documents/Bearings/artifact/runs/ltst_full_86_20260405T035505Z/invariant_summary.csv) and [regime.csv](C:/Users/Admin/Documents/Bearings/artifact/runs/ltst_full_86_20260405T035505Z/regime.csv).

## First Bench Build

### Input Path

- replay the same accepted-beat sequence into:
  - the Python reference kernel
  - the hardware/emulator implementation
- align outputs by `beat_index`
- ignore beats where either side has `FEATURE_VALID = 0`

### Comparison Windows

- smoke window: first `64` valid beats after warmup
- intermediate window: first `250` valid non-ST beats
- phenotype window: first `500` valid non-ST beats

## Good Enough Criteria

The bench prototype is good enough if all of the following hold on the four exemplar records.

### Beat-Level Agreement

- valid-beat overlap after warmup: `>= 98%`
- `poincare_b`
  - median absolute error `<= 0.03`
  - Spearman correlation `>= 0.95`
- `drift_norm`
  - median absolute error `<= 0.015`
  - Spearman correlation `>= 0.93`
- `energy_asym`
  - sign agreement `>= 90%` on beats where `|reference_energy_asym| >= 0.05`

### Summary-Level Agreement

Using host-side summaries computed from the streamed beat packets:

- `500`-beat median `poincare_b` absolute error `<= 0.03`
- `500`-beat median `drift_norm` absolute error `<= 0.015`
- exemplar ordering must be preserved:
  - `poincare_b`: `s20021 < s20041 < s20151 < s30742`
  - `drift_norm`: `s20021 > s20041 > s20151 > s30742`

### Phenotype-Preserving Agreement

The hardware beat stream must preserve the coarse phenotype separation that matters for the paper:

- the loose exemplar `s20021` remains visibly lower in `poincare_b` and higher in `drift_norm` than the constrained/rigid exemplars
- the rigid exemplars remain in the low-drift, high-`poincare_b` corner
- host-derived stiffness proxy from the streamed hardware features does not invert the loose vs high-stiffness ordering

## What Counts As Failure

The bench prototype is **not** good enough if any of these happen:

- fixed-point clipping regularly sets `SATURATED`
- `poincare_b` agreement is only good after heavy smoothing
- `drift_norm` collapses toward zero on all exemplars
- the host-derived summary ordering flips the loose/constrained/rigid ladder
- agreement depends on on-device quantile estimation or a blended score that is not in the V1 contract

## Explicit Out Of Scope

For this first bench prototype, do **not** require:

- on-device quantiles or medians
- on-device stiffness ratio
- on-device taxonomy thresholds
- on-device phenotype routing
- PMU-inspired sequence models or any cross-domain transfer work

## Exit Criterion For The August Milestone

We can call the cardiac hardware lane unblocked for the August milestone once:

1. the V1 beat packet is frozen,
2. the four exemplar replay passes the good-enough checks above,
3. the host can reconstruct the same loose vs high-stiffness separation from the hardware packet stream.

That is enough to justify moving from digital replay to the first analog bench comparison without pretending we already have a full embedded router.
