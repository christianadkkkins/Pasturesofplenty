# Micro PMU October 1 Real-Data Experiment

Chunked two-pass projective analysis of the full October 1 micro-PMU day file.

## Setup
- Data path: `C:\Users\Admin\Documents\Bearings\data\Micro PMU October 1 Dataset\_LBNL_a6_bus1_2015-10-01.csv`
- Rows scanned: `10368000`
- Event rows (`Events != 0`): `47400`
- Event onsets: `392`
- Short beta: `0.1`
- Long beta: `0.01`
- Pre-event window: `600` samples
- Post-event window: `240` samples
- Baseline window: `240` samples
- Lead rule: `z > 1.5` for `6` samples after rearming below `0.5` for `12` samples

## Top Metrics
| metric                           | direction   |    auroc | family     |
|:---------------------------------|:------------|---------:|:-----------|
| phase_current_spread             | higher      | 0.517249 | raw        |
| memory_align                     | higher      | 0.516315 | projective |
| proj_area_sl                     | lower       | 0.516315 | projective |
| proj_line_lock_sl                | higher      | 0.516315 | projective |
| proj_lock_barrier_sl             | higher      | 0.516315 | projective |
| linger                           | higher      | 0.516307 | projective |
| novelty                          | lower       | 0.516307 | projective |
| proj_line_lock_xl                | higher      | 0.516307 | projective |
| proj_lock_barrier_xl             | higher      | 0.516307 | projective |
| proj_transverse_xl               | lower       | 0.516307 | projective |
| short_long_explanation_imbalance | lower       | 0.516164 | projective |
| log_proj_volume_xsl              | lower       | 0.515077 | raw        |

## Event Lead Summary
- Events with a detected lead: `223` / `392`
- `phase_current_spread`: `162` events
- `proj_lock_barrier_xl`: `39` events
- `proj_area_sl`: `22` events

|   event_index |   event_onset_step |   event_onset_time_ns |   proj_lock_barrier_xl_lead_steps |   proj_transverse_xl_lead_steps |   proj_volume_xsl_lead_steps |   proj_area_sl_lead_steps |   phase_current_spread_lead_steps | lead_feature         |   lead_steps | event_onset_time_utc      |
|--------------:|-------------------:|----------------------:|----------------------------------:|--------------------------------:|-----------------------------:|--------------------------:|----------------------------------:|:---------------------|-------------:|:--------------------------|
|             0 |              38039 |           1.44366e+18 |                               nan |                             nan |                          nan |                       nan |                               nan | nan                  |          nan | 2015-10-01 00:05:17+00:00 |
|             1 |              40919 |           1.44366e+18 |                               nan |                             nan |                          nan |                       nan |                               205 | phase_current_spread |          205 | 2015-10-01 00:05:41+00:00 |
|             2 |              60959 |           1.44366e+18 |                               nan |                             nan |                          nan |                       nan |                               157 | phase_current_spread |          157 | 2015-10-01 00:08:28+00:00 |
|             3 |             123119 |           1.44366e+18 |                               nan |                             nan |                          nan |                       nan |                                95 | phase_current_spread |           95 | 2015-10-01 00:17:06+00:00 |
|             4 |             137399 |           1.44366e+18 |                               nan |                             nan |                          nan |                       nan |                               nan | nan                  |          nan | 2015-10-01 00:19:05+00:00 |
|             5 |             142799 |           1.44366e+18 |                               nan |                             nan |                          nan |                       nan |                               171 | phase_current_spread |          171 | 2015-10-01 00:19:50+00:00 |
|             6 |             154079 |           1.44366e+18 |                               nan |                             nan |                          nan |                       nan |                               304 | phase_current_spread |          304 | 2015-10-01 00:21:24+00:00 |
|             7 |             168359 |           1.44366e+18 |                               nan |                             nan |                          nan |                       nan |                               183 | phase_current_spread |          183 | 2015-10-01 00:23:23+00:00 |
|             8 |             178679 |           1.44366e+18 |                               nan |                             nan |                          nan |                       nan |                               nan | nan                  |          nan | 2015-10-01 00:24:49+00:00 |
|             9 |             194879 |           1.44366e+18 |                               nan |                             nan |                          nan |                       nan |                               nan | nan                  |          nan | 2015-10-01 00:27:04+00:00 |
|            10 |             219119 |           1.44366e+18 |                               nan |                             nan |                          nan |                       nan |                                78 | phase_current_spread |           78 | 2015-10-01 00:30:26+00:00 |
|            11 |             234839 |           1.44366e+18 |                               nan |                             nan |                          nan |                       nan |                               nan | nan                  |          nan | 2015-10-01 00:32:37+00:00 |
|            12 |             293879 |           1.44366e+18 |                               nan |                             nan |                          nan |                       nan |                               nan | nan                  |          nan | 2015-10-01 00:40:49+00:00 |
|            13 |             336719 |           1.44366e+18 |                               nan |                             nan |                          nan |                       nan |                               nan | nan                  |          nan | 2015-10-01 00:46:46+00:00 |
|            14 |             409079 |           1.44366e+18 |                               209 |                             209 |                          nan |                       202 |                               nan | proj_lock_barrier_xl |          209 | 2015-10-01 00:56:49+00:00 |
|            15 |             428999 |           1.44366e+18 |                               nan |                             nan |                          nan |                       nan |                               216 | phase_current_spread |          216 | 2015-10-01 00:59:35+00:00 |
|            16 |             505559 |           1.44366e+18 |                                17 |                              17 |                          nan |                        29 |                               nan | proj_area_sl         |           29 | 2015-10-01 01:10:13+00:00 |
|            17 |             592439 |           1.44366e+18 |                               nan |                             nan |                          nan |                       nan |                               nan | nan                  |          nan | 2015-10-01 01:22:17+00:00 |
|            18 |             628679 |           1.44366e+18 |                               nan |                             nan |                          nan |                       nan |                               nan | nan                  |          nan | 2015-10-01 01:27:19+00:00 |
|            19 |             651839 |           1.44366e+18 |                               nan |                             nan |                          nan |                       nan |                               nan | nan                  |          nan | 2015-10-01 01:30:32+00:00 |
