# LTST O(d) Gram-Memory Taxonomy v2

Full-cohort baseline-first Gram-memory analysis using two EMA memory horizons and quadratic-form summaries only.

## Setup
- Run directory: `C:\Users\Admin\Documents\Bearings\artifact\runs\ltst_full_86_20260405T035505Z`
- Output directory: `C:\Users\Admin\Documents\Bearings\artifact\runs\ltst_full_86_20260405T035505Z\memory_taxonomy_v2`
- Short beta: `0.1`
- Long beta: `0.01`
- Baseline beats per record: `500`
- Records analyzed: `86`

## Cohort Thresholds
- `temporal_area_loss_low` = `0.000174164`
- `present_transverse_freedom_low` = `0.0028293`
- `log_temporal_volume_3_low` = `-14.9161`
- `memory_lock_barrier_high` = `8.65551`
- `present_lock_barrier_high` = `5.86772`
- `abs_short_long_explanation_imbalance_high` = `0.0754005`

## Phenotype Summary
| phenotype_target   |   n_records |   median_proj_line_lock_sl |   median_area_sl |   median_proj_line_lock_xl |   median_area_xl |   median_barrier_sl |   median_barrier_xl |   median_log_volume |   median_abs_delta_explain |   median_memory_align |   median_linger |   median_lock_barrier_st_change |   median_present_lock_barrier_st_change |
|:-------------------|------------:|---------------------------:|-----------------:|---------------------------:|-----------------:|--------------------:|--------------------:|--------------------:|---------------------------:|----------------------:|----------------:|--------------------------------:|----------------------------------------:|
| loose_orbit        |          19 |                   0.994119 |       0.00588121 |                   0.949824 |        0.050176  |             5.13601 |             2.99222 |            -10.117  |                 0.0176032  |              0.997055 |        0.974589 |                       0.118611  |                               0.0647249 |
| constrained_orbit  |          41 |                   0.998319 |       0.00168132 |                   0.982317 |        0.0176831 |             6.38818 |             4.03515 |            -11.9192 |                 0.0062266  |              0.999159 |        0.991119 |                       0.0858204 |                               0.0574464 |
| rigid_orbit        |          26 |                   0.998441 |       0.00155878 |                   0.979756 |        0.0202441 |             6.47175 |             3.902   |            -11.4076 |                 0.00488053 |              0.99922  |        0.989826 |                       0.0595865 |                               0.0922435 |

## Top Loose-vs-Rest Metrics
| task          | metric                                   | positive_class   | direction   |    auroc |
|:--------------|:-----------------------------------------|:-----------------|:------------|---------:|
| loose_vs_rest | baseline_p95_temporal_volume_3           | loose_orbit      | higher      | 0.738413 |
| loose_vs_rest | baseline_p95_temporal_area_loss          | loose_orbit      | higher      | 0.714847 |
| loose_vs_rest | proj_line_lock_sl_st_abs_relative_change | loose_orbit      | higher      | 0.71249  |
| loose_vs_rest | baseline_median_log_temporal_volume_3    | loose_orbit      | higher      | 0.703849 |
| loose_vs_rest | baseline_median_temporal_volume_3        | loose_orbit      | higher      | 0.703849 |

## High-Stiffness Split (Rigid vs Constrained)
| task                 | metric                                     | positive_class   | direction   |    auroc |
|:---------------------|:-------------------------------------------|:-----------------|:------------|---------:|
| rigid_vs_constrained | temporal_volume_3_st_abs_relative_change   | rigid_orbit      | lower       | 0.590056 |
| rigid_vs_constrained | present_lock_barrier_st_over_baseline      | rigid_orbit      | higher      | 0.58818  |
| rigid_vs_constrained | temporal_area_loss_st_abs_relative_change  | rigid_orbit      | lower       | 0.586304 |
| rigid_vs_constrained | proj_line_lock_sl_st_abs_relative_change   | rigid_orbit      | lower       | 0.584428 |
| rigid_vs_constrained | memory_lock_barrier_st_abs_relative_change | rigid_orbit      | lower       | 0.574109 |

## 3-Way Nearest-Centroid
- Macro-F1: `0.2815`
- Macro balanced accuracy: `0.3393`

| actual            |   loose_orbit |   constrained_orbit |   rigid_orbit |
|:------------------|--------------:|--------------------:|--------------:|
| loose_orbit       |            11 |                   6 |             2 |
| constrained_orbit |            15 |                  18 |             8 |
| rigid_orbit       |            10 |                  16 |             0 |

## Regime Summary
| regime            |   n_records |   median_proj_line_lock_xl |   median_area_xl |   median_barrier_xl |   median_log_volume |   median_memory_align |   median_linger |
|:------------------|------------:|---------------------------:|-----------------:|--------------------:|--------------------:|----------------------:|----------------:|
| long_first        |          14 |                   0.983827 |        0.016173  |             4.12497 |           -12.7228  |              0.999447 |        0.991881 |
| long_only         |          16 |                   0.983739 |        0.0162612 |             4.1218  |           -11.4584  |              0.998904 |        0.991836 |
| neither           |          26 |                   0.979756 |        0.0202441 |             3.902   |           -11.4076  |              0.99922  |        0.989826 |
| both_same_horizon |          11 |                   0.953668 |        0.046332  |             3.07193 |            -9.75286 |              0.998253 |        0.976559 |
| angle_only        |          13 |                   0.949824 |        0.050176  |             2.99222 |           -10.117   |              0.997055 |        0.974589 |
| angle_first       |           6 |                   0.944989 |        0.0550114 |             2.91829 |            -9.60616 |              0.99724  |        0.972091 |

## Per-Record Predictions
| record   | phenotype_target   | predicted_phenotype   | regime            |
|:---------|:-------------------|:----------------------|:------------------|
| s20011   | constrained_orbit  | rigid_orbit           | long_first        |
| s20031   | constrained_orbit  | constrained_orbit     | both_same_horizon |
| s20041   | constrained_orbit  | constrained_orbit     | long_only         |
| s20051   | constrained_orbit  | loose_orbit           | long_only         |
| s20071   | constrained_orbit  | constrained_orbit     | long_only         |
| s20091   | constrained_orbit  | rigid_orbit           | both_same_horizon |
| s20101   | constrained_orbit  | constrained_orbit     | long_only         |
| s20111   | constrained_orbit  | rigid_orbit           | long_only         |
| s20121   | constrained_orbit  | constrained_orbit     | long_only         |
| s20161   | constrained_orbit  | constrained_orbit     | long_only         |
| s20171   | constrained_orbit  | rigid_orbit           | long_first        |
| s20201   | constrained_orbit  | rigid_orbit           | long_only         |
| s20231   | constrained_orbit  | loose_orbit           | long_first        |
| s20251   | constrained_orbit  | constrained_orbit     | both_same_horizon |
| s20261   | constrained_orbit  | rigid_orbit           | long_only         |
| s20271   | constrained_orbit  | rigid_orbit           | long_first        |
| s20272   | constrained_orbit  | loose_orbit           | long_first        |
| s20274   | constrained_orbit  | constrained_orbit     | long_only         |
| s20281   | constrained_orbit  | constrained_orbit     | long_first        |
| s20291   | constrained_orbit  | loose_orbit           | both_same_horizon |
| s20341   | constrained_orbit  | loose_orbit           | both_same_horizon |
| s20351   | constrained_orbit  | constrained_orbit     | both_same_horizon |
| s20361   | constrained_orbit  | constrained_orbit     | long_first        |
| s20371   | constrained_orbit  | constrained_orbit     | both_same_horizon |
| s20401   | constrained_orbit  | constrained_orbit     | long_first        |
| s20411   | constrained_orbit  | constrained_orbit     | long_first        |
| s20431   | constrained_orbit  | loose_orbit           | long_only         |
| s20441   | constrained_orbit  | loose_orbit           | long_only         |
| s20451   | constrained_orbit  | constrained_orbit     | both_same_horizon |
| s20461   | constrained_orbit  | loose_orbit           | long_first        |
| s20501   | constrained_orbit  | rigid_orbit           | long_only         |
| s20511   | constrained_orbit  | loose_orbit           | long_only         |
| s20551   | constrained_orbit  | loose_orbit           | long_only         |
| s20581   | constrained_orbit  | loose_orbit           | long_first        |
| s20591   | constrained_orbit  | constrained_orbit     | long_first        |
| s20651   | constrained_orbit  | loose_orbit           | both_same_horizon |
| s30671   | constrained_orbit  | constrained_orbit     | long_first        |
| s30711   | constrained_orbit  | loose_orbit           | long_only         |
| s30732   | constrained_orbit  | loose_orbit           | both_same_horizon |
| s30741   | constrained_orbit  | constrained_orbit     | long_first        |
| s30781   | constrained_orbit  | loose_orbit           | both_same_horizon |
| s20021   | loose_orbit        | loose_orbit           | angle_only        |
| s20081   | loose_orbit        | constrained_orbit     | angle_first       |
| s20273   | loose_orbit        | constrained_orbit     | angle_only        |
| s20301   | loose_orbit        | constrained_orbit     | angle_only        |
| s20311   | loose_orbit        | loose_orbit           | angle_only        |
| s20381   | loose_orbit        | constrained_orbit     | angle_only        |
| s20391   | loose_orbit        | loose_orbit           | angle_only        |
| s20471   | loose_orbit        | constrained_orbit     | angle_only        |
| s20541   | loose_orbit        | loose_orbit           | angle_only        |
| s20561   | loose_orbit        | loose_orbit           | angle_first       |
| s20611   | loose_orbit        | loose_orbit           | angle_only        |
| s20621   | loose_orbit        | loose_orbit           | angle_first       |
| s20641   | loose_orbit        | rigid_orbit           | angle_first       |
| s30681   | loose_orbit        | rigid_orbit           | angle_only        |
| s30691   | loose_orbit        | loose_orbit           | angle_only        |
| s30721   | loose_orbit        | loose_orbit           | angle_first       |
| s30731   | loose_orbit        | loose_orbit           | angle_first       |
| s30751   | loose_orbit        | constrained_orbit     | angle_only        |
| s30801   | loose_orbit        | loose_orbit           | angle_only        |
| s20061   | rigid_orbit        | constrained_orbit     | neither           |
| s20131   | rigid_orbit        | constrained_orbit     | neither           |
| s20141   | rigid_orbit        | constrained_orbit     | neither           |
| s20151   | rigid_orbit        | constrained_orbit     | neither           |
| s20181   | rigid_orbit        | loose_orbit           | neither           |
| s20191   | rigid_orbit        | loose_orbit           | neither           |
| s20211   | rigid_orbit        | constrained_orbit     | neither           |
| s20221   | rigid_orbit        | constrained_orbit     | neither           |
| s20241   | rigid_orbit        | constrained_orbit     | neither           |
| s20321   | rigid_orbit        | constrained_orbit     | neither           |
| s20331   | rigid_orbit        | loose_orbit           | neither           |
| s20421   | rigid_orbit        | constrained_orbit     | neither           |
| s20481   | rigid_orbit        | loose_orbit           | neither           |
| s20491   | rigid_orbit        | constrained_orbit     | neither           |
| s20521   | rigid_orbit        | constrained_orbit     | neither           |
| s20531   | rigid_orbit        | loose_orbit           | neither           |
| s20571   | rigid_orbit        | constrained_orbit     | neither           |
| s20601   | rigid_orbit        | constrained_orbit     | neither           |
| s20631   | rigid_orbit        | constrained_orbit     | neither           |
| s30661   | rigid_orbit        | loose_orbit           | neither           |
| s30701   | rigid_orbit        | loose_orbit           | neither           |
| s30742   | rigid_orbit        | constrained_orbit     | neither           |
| s30752   | rigid_orbit        | loose_orbit           | neither           |
| s30761   | rigid_orbit        | loose_orbit           | neither           |
| s30771   | rigid_orbit        | loose_orbit           | neither           |
| s30791   | rigid_orbit        | constrained_orbit     | neither           |
