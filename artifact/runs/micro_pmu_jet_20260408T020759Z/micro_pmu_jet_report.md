# Micro PMU October 1 Phasor Jet Experiment

Ray-space / jet-space analysis on the full October 1 micro-PMU day file using the positive-sequence phasor and its first two local differences.

## Setup
- Data path: `C:\Users\Admin\Documents\Bearings\data\Micro PMU October 1 Dataset\_LBNL_a6_bus1_2015-10-01.csv`
- Rows scanned: `10368000`
- Event rows (`Events != 0`): `47400`
- Event onsets: `392`
- Chosen healthy orientation: `ABC`
- Orientation sample count: `51615`
- Local jet state: `z = V+`, `dz = z[t]-z[t-1]`, `ddz = dz[t]-dz[t-1]`
- Raw jet channels: `g11 g22 g33 g12 g13 g23 q12 q13 q23`
- Pre-event window: `600` samples
- Post-event window: `240` samples
- Baseline window: `240` samples
- Lead rule: `z > 1.5` for `6` samples after rearming below `0.5` for `12` samples

## Orientation Inference
| orientation   |   sample_median_seq_pos_frac |   sample_mean_seq_pos_frac |   sample_count | chosen   |
|:--------------|-----------------------------:|---------------------------:|---------------:|:---------|
| ABC           |                  0.999994    |                 0.999994   |          51615 | True     |
| ACB           |                  5.26766e-06 |                 5.3259e-06 |          51615 | False    |

## Synthetic Sanity Checks
| scenario                   |   jet_g11 |   jet_area_zd |   jet_twist_norm |
|:---------------------------|----------:|--------------:|-----------------:|
| balanced_positive_sequence |       nan |           nan |              nan |
| reversed_order_under_ABC   |       nan |           nan |              nan |
| common_mode                |       nan |           nan |              nan |

## Top Metrics
| metric               | direction   |    auroc | family   |
|:---------------------|:------------|---------:|:---------|
| voltage_angle_spread | higher      | 0.527191 | raw      |
| phase_current_spread | higher      | 0.517249 | raw      |
| jet_twist_delta      | absdev      | 0.513183 | jet      |
| jet_area_zd          | absdev      | 0.504686 | jet      |
| current_angle_spread | higher      | 0.502546 | raw      |
| jet_area_za          | absdev      | 0.499327 | jet      |
| jet_q13              | absdev      | 0.49764  | jet      |
| jet_g33              | higher      | 0.497573 | jet      |
| jet_volume           | absdev      | 0.496739 | jet      |
| jet_area_da          | absdev      | 0.496423 | jet      |
| jet_rocof_proxy      | absdev      | 0.489844 | jet      |
| jet_delta            | higher      | 0.486411 | jet      |
| jet_q12              | absdev      | 0.48552  | jet      |
| jet_freq_proxy       | absdev      | 0.48552  | jet      |
| jet_q23              | absdev      | 0.485103 | jet      |

## Comparison To Prior PMU Baselines
- Best jet AUROC vs cross-phase baseline: `0.513183` vs `0.527191`
- Lead-detected events vs cross-phase baseline: `291` vs `243`

## Event Lead Summary
- Events with a detected lead: `291` / `392`
- `jet_twist_norm`: `134` events
- `phase_current_spread`: `120` events
- `jet_freq_proxy`: `27` events
- `jet_area_zd`: `6` events
- `jet_rocof_proxy`: `1` events
- `jet_area_za`: `1` events
- `jet_area_da`: `1` events
- `jet_twist_delta`: `1` events

## Alignment Windows
| window    |   jet_twist_norm_median_z |   jet_area_zd_median_z |   jet_volume_median_z |   phase_current_spread_median_z |
|:----------|--------------------------:|-----------------------:|----------------------:|--------------------------------:|
| [-120,-1] |                -0.0903614 |               0.875642 |           3.82175e-22 |                      0.0014319  |
| [-60,-1]  |                -0.0602366 |               0.907602 |           3.85171e-22 |                     -0.00176827 |
| [-30,-1]  |                -0.0487479 |               0.92503  |           3.63984e-22 |                      0.0518884  |
| [0,60]    |                -0.0317886 |               0.995622 |           3.82176e-22 |                      0.0436659  |

- Interpretation: Jet-space alignment still looks plateau-like; largest median-window shift is only `jet_area_zd` = `0.049` from `[-120,-1]` to `[-30,-1]`.

|   event_index |   event_onset_step |   event_onset_time_ns |   jet_area_zd_lead_steps |   jet_area_za_lead_steps |   jet_area_da_lead_steps |   jet_delta_lead_steps |   jet_volume_lead_steps |   jet_twist_norm_lead_steps |   jet_twist_delta_lead_steps |   jet_freq_proxy_lead_steps |   jet_rocof_proxy_lead_steps |   phase_current_spread_lead_steps | lead_feature         |   lead_steps | event_onset_time_utc      |
|--------------:|-------------------:|----------------------:|-------------------------:|-------------------------:|-------------------------:|-----------------------:|------------------------:|----------------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------------------------:|:---------------------|-------------:|:--------------------------|
|             0 |              38039 |           1.44366e+18 |                      nan |                      nan |                      nan |                    nan |                     nan |                         nan |                          nan |                         nan |                          nan |                               nan | nan                  |          nan | 2015-10-01 00:05:17+00:00 |
|             1 |              40919 |           1.44366e+18 |                      nan |                      nan |                      nan |                    nan |                     nan |                         nan |                          nan |                         nan |                          nan |                               205 | phase_current_spread |          205 | 2015-10-01 00:05:41+00:00 |
|             2 |              60959 |           1.44366e+18 |                      nan |                      nan |                      nan |                    nan |                     nan |                         nan |                          nan |                         nan |                          nan |                               157 | phase_current_spread |          157 | 2015-10-01 00:08:28+00:00 |
|             3 |             123119 |           1.44366e+18 |                      nan |                      nan |                      nan |                    nan |                     nan |                         nan |                          nan |                         nan |                          nan |                               nan | nan                  |          nan | 2015-10-01 00:17:06+00:00 |
|             4 |             137399 |           1.44366e+18 |                      nan |                      nan |                      nan |                    nan |                     nan |                         200 |                          124 |                         125 |                          nan |                               nan | jet_twist_norm       |          200 | 2015-10-01 00:19:05+00:00 |
|             5 |             142799 |           1.44366e+18 |                      nan |                      nan |                      nan |                    nan |                     nan |                         nan |                          nan |                         nan |                          nan |                               171 | phase_current_spread |          171 | 2015-10-01 00:19:50+00:00 |
|             6 |             154079 |           1.44366e+18 |                      nan |                      nan |                      nan |                    nan |                     nan |                         nan |                          nan |                         nan |                          nan |                               304 | phase_current_spread |          304 | 2015-10-01 00:21:24+00:00 |
|             7 |             168359 |           1.44366e+18 |                      nan |                      nan |                      nan |                    nan |                     nan |                         nan |                          nan |                         nan |                          nan |                               183 | phase_current_spread |          183 | 2015-10-01 00:23:23+00:00 |
|             8 |             178679 |           1.44366e+18 |                      nan |                      nan |                      nan |                    nan |                     nan |                         217 |                          nan |                         nan |                          nan |                               nan | jet_twist_norm       |          217 | 2015-10-01 00:24:49+00:00 |
|             9 |             194879 |           1.44366e+18 |                      nan |                      nan |                      nan |                    nan |                     nan |                         nan |                          nan |                         nan |                          nan |                               nan | nan                  |          nan | 2015-10-01 00:27:04+00:00 |
|            10 |             219119 |           1.44366e+18 |                      nan |                      nan |                      nan |                    nan |                     nan |                         274 |                          nan |                         nan |                          nan |                                78 | jet_twist_norm       |          274 | 2015-10-01 00:30:26+00:00 |
|            11 |             234839 |           1.44366e+18 |                      nan |                      nan |                      nan |                    nan |                     nan |                         nan |                          267 |                         268 |                          nan |                                60 | jet_freq_proxy       |          268 | 2015-10-01 00:32:37+00:00 |
|            12 |             293879 |           1.44366e+18 |                      nan |                      nan |                      nan |                    nan |                     nan |                         nan |                          nan |                         nan |                          nan |                               nan | nan                  |          nan | 2015-10-01 00:40:49+00:00 |
|            13 |             336719 |           1.44366e+18 |                      nan |                      nan |                      nan |                    nan |                     nan |                         nan |                          nan |                         nan |                          nan |                               nan | nan                  |          nan | 2015-10-01 00:46:46+00:00 |
|            14 |             409079 |           1.44366e+18 |                      nan |                      nan |                      nan |                    nan |                     nan |                         nan |                          nan |                         nan |                          nan |                               nan | nan                  |          nan | 2015-10-01 00:56:49+00:00 |
|            15 |             428999 |           1.44366e+18 |                      nan |                      nan |                      nan |                    nan |                     nan |                         nan |                          nan |                         nan |                          nan |                               231 | phase_current_spread |          231 | 2015-10-01 00:59:35+00:00 |
|            16 |             505559 |           1.44366e+18 |                      nan |                      nan |                      nan |                    nan |                     nan |                         230 |                          nan |                         nan |                          nan |                               nan | jet_twist_norm       |          230 | 2015-10-01 01:10:13+00:00 |
|            17 |             592439 |           1.44366e+18 |                      nan |                      nan |                      nan |                    nan |                     nan |                         343 |                          nan |                         nan |                          nan |                               nan | jet_twist_norm       |          343 | 2015-10-01 01:22:17+00:00 |
|            18 |             628679 |           1.44366e+18 |                      nan |                      nan |                        7 |                    nan |                     nan |                         343 |                          nan |                         nan |                          nan |                               nan | jet_twist_norm       |          343 | 2015-10-01 01:27:19+00:00 |
|            19 |             651839 |           1.44366e+18 |                      nan |                      nan |                      nan |                    nan |                     nan |                         217 |                          216 |                         217 |                          nan |                               nan | jet_twist_norm       |          217 | 2015-10-01 01:30:32+00:00 |
