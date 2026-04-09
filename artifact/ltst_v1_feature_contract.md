# LTST V1 Cardiac Feature Contract

## Purpose

Freeze the first hardware-facing cardiac feature interface for the LTST study so the bench prototype reproduces the validated kernel primitives, not a prematurely compressed decision score.

This contract is grounded in the completed LTST triage run at [ltst_feature_triage.csv](C:/Users/Admin/Documents/Bearings/artifact/runs/ltst_full_86_20260405T035505Z/feature_triage/ltst_feature_triage.csv) and [ltst_feature_task_metrics.csv](C:/Users/Admin/Documents/Bearings/artifact/runs/ltst_full_86_20260405T035505Z/feature_triage/ltst_feature_task_metrics.csv).

## Decision Summary

- `median_drift_norm` is the most balanced hardware candidate in the cohort triage:
  - best task: `loose_orbit`
  - CV AUROC: `0.700`
  - analog score: `3.0`
- `poincare_b_p50` is still one of the strongest single features even though the strict rubric marks it as `drop` because fold-to-fold variability is high:
  - best task: `loose_orbit`
  - CV AUROC: `0.712`
  - analog score: `3.0`
- `median_energy_asym` is weaker, but it is still useful as a signed diagnostic and sanity-check feature.

Because the validated cardiac story depends on routing over primitive geometry rather than an early blended scalar, V1 freezes the **primitive per-beat outputs** and leaves summary quantiles, stiffness ratios, and routing decisions offboard.

## V1 Streamed Features

### Required Beat Packet

Emit one packet per **accepted beat** after delay-state warmup.

| Field | Type | Q format | Cadence | Meaning |
|:--|:--|:--|:--|:--|
| `beat_index` | `uint32` | integer | every accepted beat | Monotonic beat counter after boot/reset |
| `poincare_b` | `uint16` | `UQ2.14` | every accepted beat | Return coefficient primitive behind `poincare_b_p50` |
| `drift_norm` | `uint16` | `UQ4.12` | every accepted beat | Positive drift norm primitive behind `median_drift_norm` |
| `energy_asym` | `int16` | `SQ1.14` | every accepted beat | Signed energy asymmetry diagnostic |
| `quality_flags` | `uint16` | bitfield | every accepted beat | Validity and saturation state |

### Fixed Ranges

- `poincare_b`: `UQ2.14`
  - range `[0, 4)`
  - reason: bounded but occasionally above `1.0`, so V1 keeps headroom instead of squeezing to `UQ1.15`
- `drift_norm`: `UQ4.12`
  - range `[0, 16)`
  - reason: positive norm with enough room for transient excursions and safe fixed-point division later
- `energy_asym`: `SQ1.14`
  - range `[-2, 2)`
  - reason: signed normalized asymmetry with more than enough headroom for V1 replay and bench comparison

## Frozen Quality Flags

`quality_flags` is a 16-bit word with these frozen low bits:

| Bit | Name | Meaning |
|:--|:--|:--|
| `0` | `FEATURE_VALID` | This beat has sufficient history and passed kernel validity checks |
| `1` | `WARMUP` | Delay-state or EMA history not yet initialized |
| `2` | `LOW_CURR_ENERGY` | Current-state energy below floor |
| `3` | `LOW_PAST_ENERGY` | Past-state energy below floor |
| `4` | `SATURATED` | Any fixed-point intermediate or output clipped |
| `5` | `DIV_GUARD` | Epsilon/denominator guard engaged |
| `6` | `REJECTED_BEAT` | Upstream beat rejected or unusable |
| `7-15` | `RESERVED` | Reserved for later protocol growth |

## Offboard-Only in V1

These stay off the hardware stream in V1:

- rolling medians and quantiles such as `poincare_b_p50`, `p90_drift_norm`, `poincare_b_p05`
- long-window fractions such as `frac_negative_phase`
- high-dynamic-range quadratic summaries such as `gram_spread_sq`
- derived stiffness ratios such as `poincare_b / (drift_norm + eps)`
- phenotype/router outputs
- any blended score meant to stand in for routing

The host/reference side computes:

- rolling `median(poincare_b)`
- rolling `median(drift_norm)`
- stiffness proxy `median(poincare_b / (drift_norm + eps))`
- any downstream phenotype or routing logic

## Why `poincare_b` Stays In Scope

The strict triage rubric downgraded `poincare_b_p50` because its CV spread was wider than the hard threshold, not because the feature lacked value. It remained:

- the highest CV AUROC feature for `loose_orbit`
- one of the strongest features for `detected_any`
- a core primitive in the already-validated cardiac taxonomy work

So V1 keeps `poincare_b` as a streamed primitive and postpones the unstable part, which is the **on-device summary decision**, not the raw feature itself.

## Contracted Derived Comparisons

When comparing hardware output to the Python reference, the frozen host-side summary windows are:

- `64` accepted beats for first-pass smoke plots
- `250` accepted beats for intermediate summary checks
- `500` accepted beats for baseline phenotype comparisons

## Explicit Non-Goals

V1 is **not** trying to:

- implement the full router on-device
- emit final regime labels from hardware
- reproduce cohort quantiles on-device
- rescue the PMU sequence work inside the cardiac milestone

The point of V1 is simpler: stream the few primitives that preserve the cardiac result and keep the irreversible compression steps offboard until the bench agreement is solid.
