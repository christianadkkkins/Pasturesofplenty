# Micro PMU Sequence Episode Evaluator

Episode-level evaluation over the sequence-manifold score using EMA smoothing, hysteretic detection, and post-hoc episode merging.

## Setup
- Source sequence run: `C:\Users\Admin\Documents\Bearings\artifact\runs\micro_pmu_sequence_20260407T014634Z`
- Data path: `C:\Users\Admin\Documents\Bearings\data\Micro PMU October 1 Dataset\_LBNL_a6_bus1_2015-10-01.csv`
- Chosen orientation: `ABC`
- Score metrics: `seq_pos_frac, seq_neg_frac, seq_zero_frac, seq_unbalance_barrier`
- Decoder mode: `twist_shmm`
- EMA alpha: `0.05`
- Episode on threshold: `2.0` for `12` samples
- Episode off threshold: `0.75` for `24` samples
- Merge gap: `120` samples
- Twist enter/exit thresholds: `0.6` / `0.35`
- Twist checkpoint: `C:\Users\Admin\Documents\Bearings\artifact\runs\micro_pmu_twist_train_20260407T023134Z\micro_pmu_twist_transition_head.pt`

## Baseline Stats
| metric                | direction   |       center |         mad |       scale |   non_event_count |
|:----------------------|:------------|-------------:|------------:|------------:|------------------:|
| seq_pos_frac          | lower       |  0.999994    | 7.36968e-07 | 1.09263e-06 |             51625 |
| seq_neg_frac          | higher      |  5.27215e-06 | 7.26093e-07 | 1.07651e-06 |             51625 |
| seq_zero_frac         | higher      |  2.17156e-07 | 2.42161e-08 | 3.59027e-08 |             51625 |
| seq_unbalance_barrier | higher      | 12.1086      | 0.135881    | 0.201457    |             51625 |

## Summary
| metric                                 |           value |
|:---------------------------------------|----------------:|
| baseline_center_seq_pos_frac           |     0.999994    |
| baseline_scale_seq_pos_frac            |     1.09263e-06 |
| baseline_center_seq_neg_frac           |     5.27215e-06 |
| baseline_scale_seq_neg_frac            |     1.07651e-06 |
| baseline_center_seq_zero_frac          |     2.17156e-07 |
| baseline_scale_seq_zero_frac           |     3.59027e-08 |
| baseline_center_seq_unbalance_barrier  |    12.1086      |
| baseline_scale_seq_unbalance_barrier   |     0.201457    |
| n_episodes                             |     1           |
| median_episode_duration_steps          |     1.0368e+07  |
| median_episode_duration_seconds        | 86400           |
| mean_episode_peak_score                |     0.836186    |
| event_capture_any_rate                 |     1           |
| event_capture_pre_rate                 |     1           |
| median_earliest_pre_event_lead_steps   |     5.33568e+06 |
| median_earliest_pre_event_lead_seconds | 44464           |
| mean_overlap_fraction_of_event         |     1           |

## Episode Dominant Metrics
- `negative_sequence`: `1` episodes

## Event Capture
|   event_index |   event_start_step |   event_end_step | event_start_time_utc      | event_end_time_utc                  | captured_by_any_episode   | captured_by_pre_event_episode   |   earliest_pre_event_lead_steps |   earliest_pre_event_lead_seconds |   matching_episode_count |   overlap_steps |   overlap_fraction_of_event |   earliest_episode_id |   earliest_episode_duration_steps | earliest_episode_dominant_metric   |
|--------------:|-------------------:|-----------------:|:--------------------------|:------------------------------------|:--------------------------|:--------------------------------|--------------------------------:|----------------------------------:|-------------------------:|----------------:|----------------------------:|----------------------:|----------------------------------:|:-----------------------------------|
|             0 |              38039 |            38158 | 2015-10-01 00:05:17+00:00 | 2015-10-01 00:05:17.991666688+00:00 | True                      | True                            |                           38039 |                           316.992 |                        1 |             120 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |
|             1 |              40919 |            41038 | 2015-10-01 00:05:41+00:00 | 2015-10-01 00:05:41.991666688+00:00 | True                      | True                            |                           40919 |                           340.992 |                        1 |             120 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |
|             2 |              60959 |            61078 | 2015-10-01 00:08:28+00:00 | 2015-10-01 00:08:28.991666688+00:00 | True                      | True                            |                           60959 |                           507.992 |                        1 |             120 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |
|             3 |             123119 |           123238 | 2015-10-01 00:17:06+00:00 | 2015-10-01 00:17:06.991666688+00:00 | True                      | True                            |                          123119 |                          1025.99  |                        1 |             120 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |
|             4 |             137399 |           137518 | 2015-10-01 00:19:05+00:00 | 2015-10-01 00:19:05.991666688+00:00 | True                      | True                            |                          137399 |                          1144.99  |                        1 |             120 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |
|             5 |             142799 |           142918 | 2015-10-01 00:19:50+00:00 | 2015-10-01 00:19:50.991666688+00:00 | True                      | True                            |                          142799 |                          1189.99  |                        1 |             120 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |
|             6 |             154079 |           154198 | 2015-10-01 00:21:24+00:00 | 2015-10-01 00:21:24.991666688+00:00 | True                      | True                            |                          154079 |                          1283.99  |                        1 |             120 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |
|             7 |             168359 |           168478 | 2015-10-01 00:23:23+00:00 | 2015-10-01 00:23:23.991666688+00:00 | True                      | True                            |                          168359 |                          1402.99  |                        1 |             120 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |
|             8 |             178679 |           178798 | 2015-10-01 00:24:49+00:00 | 2015-10-01 00:24:49.991666688+00:00 | True                      | True                            |                          178679 |                          1488.99  |                        1 |             120 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |
|             9 |             194879 |           194998 | 2015-10-01 00:27:04+00:00 | 2015-10-01 00:27:04.991666688+00:00 | True                      | True                            |                          194879 |                          1623.99  |                        1 |             120 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |
|            10 |             219119 |           219238 | 2015-10-01 00:30:26+00:00 | 2015-10-01 00:30:26.991666688+00:00 | True                      | True                            |                          219119 |                          1825.99  |                        1 |             120 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |
|            11 |             234839 |           234958 | 2015-10-01 00:32:37+00:00 | 2015-10-01 00:32:37.991666688+00:00 | True                      | True                            |                          234839 |                          1956.99  |                        1 |             120 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |
|            12 |             293879 |           294358 | 2015-10-01 00:40:49+00:00 | 2015-10-01 00:40:52.991666688+00:00 | True                      | True                            |                          293879 |                          2448.99  |                        1 |             480 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |
|            13 |             336719 |           336838 | 2015-10-01 00:46:46+00:00 | 2015-10-01 00:46:46.991666688+00:00 | True                      | True                            |                          336719 |                          2805.99  |                        1 |             120 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |
|            14 |             409079 |           409198 | 2015-10-01 00:56:49+00:00 | 2015-10-01 00:56:49.991666688+00:00 | True                      | True                            |                          409079 |                          3408.99  |                        1 |             120 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |
|            15 |             428999 |           429118 | 2015-10-01 00:59:35+00:00 | 2015-10-01 00:59:35.991666688+00:00 | True                      | True                            |                          428999 |                          3574.99  |                        1 |             120 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |
|            16 |             505559 |           505678 | 2015-10-01 01:10:13+00:00 | 2015-10-01 01:10:13.991666688+00:00 | True                      | True                            |                          505559 |                          4212.99  |                        1 |             120 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |
|            17 |             592439 |           592558 | 2015-10-01 01:22:17+00:00 | 2015-10-01 01:22:17.991666688+00:00 | True                      | True                            |                          592439 |                          4936.99  |                        1 |             120 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |
|            18 |             628679 |           628798 | 2015-10-01 01:27:19+00:00 | 2015-10-01 01:27:19.991666688+00:00 | True                      | True                            |                          628679 |                          5238.99  |                        1 |             120 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |
|            19 |             651839 |           651958 | 2015-10-01 01:30:32+00:00 | 2015-10-01 01:30:32.991666688+00:00 | True                      | True                            |                          651839 |                          5431.99  |                        1 |             120 |                           1 |                     0 |                        1.0368e+07 | negative_sequence                  |

## Episodes
|   episode_id |   start_step |   start_time_ns |   end_step |   end_time_ns |   peak_score |   peak_step |   peak_time_ns | feature_counts                           |   n_points |   duration_steps |   duration_seconds | dominant_metric   | start_time_utc                      | end_time_utc              |
|-------------:|-------------:|----------------:|-----------:|--------------:|-------------:|------------:|---------------:|:-----------------------------------------|-----------:|-----------------:|-------------------:|:------------------|:------------------------------------|:--------------------------|
|            0 |            0 |     1.44366e+18 |   10367999 |   1.44374e+18 |     0.836186 |    10105413 |    1.44374e+18 | Counter({'negative_sequence': 10368000}) |   10368000 |         10368000 |              86400 | negative_sequence | 2015-10-01 00:00:00.008333056+00:00 | 2015-10-02 00:00:00+00:00 |

## Decoder Note
- `baseline` mode is the original smoothed hysteretic baseline.
- `twist_shmm` mode uses the fixed-symmetry Twist front end plus a sticky semi-HMM-style state decoder over `stable / negative_sequence / zero_sequence / mixed`.
