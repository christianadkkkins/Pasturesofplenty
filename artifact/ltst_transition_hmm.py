from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_lie import (
    CANONICAL_LIE_FEATURE_COLUMNS,
    CANONICAL_PROJECTIVE_REFERENCE_COLUMNS,
    canonical_lie_transition_score,
    canonical_projective_transition_score,
)


EPS = 1e-9
STATE_ORDER = ["baseline", "transition", "active"]
STATE_TO_INDEX = {name: idx for idx, name in enumerate(STATE_ORDER)}
REGIME_TO_PHENOTYPE = {
    "angle_only": "loose_orbit",
    "angle_first": "loose_orbit",
    "long_only": "constrained_orbit",
    "long_first": "constrained_orbit",
    "both_same_horizon": "constrained_orbit",
    "neither": "rigid_orbit",
}
EXEMPLAR_RECORDS = ("s20021", "s20041", "s20151", "s30742")
BEAT_COLUMNS = [
    "record",
    "beat_sample",
    "energy_asym",
    "drift_norm",
    "poincare_b",
    "kernel_score_angle",
    "kernel_score_long",
    "kernel_score_hybrid",
    "memory_align",
    "novelty",
    "proj_lock_barrier_sl",
    "proj_lock_barrier_xl",
    "proj_volume_xsl",
    "gram_trace",
    "gram_logdet",
    "lie_orbit_norm",
    "lie_strain_norm",
    "lie_commutator_norm",
    "lie_metric_drift",
    "st_event",
    "st_episode_id",
]


@dataclass(frozen=True)
class TransitionHMMConfig:
    run_dir: Path
    out_dir_name: str = "transition_hmm"
    baseline_non_st_beats: int = 500
    min_baseline_beats: int = 128
    rolling_window_beats: int = 8
    pre_event_beats: int = 500
    post_event_beats: int = 250
    merge_gap_beats: int = 8
    detector_on_threshold: float = 1.5
    detector_off_threshold: float = 0.5
    detector_on_sustain: int = 4
    detector_off_sustain: int = 8
    level_weight: float = 0.5
    velocity_weight: float = 1.0
    curvature_weight: float = 1.5
    stiffness_weight: float = 0.75
    alert_active_tail_beats: int = 16
    alert_max_beats: int = 96
    alert_cooldown_beats: int = 32
    exemplar_records: tuple[str, ...] = EXEMPLAR_RECORDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relative-geometry LTST transition model with windowed HMM smoothing.")
    parser.add_argument("--run-dir", default=str(Path("artifact") / "runs" / "ltst_full_86_20260405T035505Z"))
    parser.add_argument("--out-dir-name", default="transition_hmm")
    parser.add_argument("--baseline-non-st-beats", type=int, default=500)
    parser.add_argument("--rolling-window-beats", type=int, default=8)
    parser.add_argument("--pre-event-beats", type=int, default=500)
    parser.add_argument("--post-event-beats", type=int, default=250)
    parser.add_argument("--alert-active-tail-beats", type=int, default=16)
    parser.add_argument("--alert-max-beats", type=int, default=96)
    parser.add_argument("--alert-cooldown-beats", type=int, default=32)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def ensure_output_dir(run_dir: Path, out_dir_name: str) -> Path:
    out_dir = run_dir / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def phenotype_from_regime(regime: str) -> str:
    return REGIME_TO_PHENOTYPE.get(str(regime), "rigid_orbit")


def read_beat_frame(beat_level_dir: Path, record: str) -> pd.DataFrame:
    parquet_path = beat_level_dir / f"{record}.parquet"
    csv_path = beat_level_dir / f"{record}.csv"
    if parquet_path.exists():
        try:
            frame = pd.read_parquet(parquet_path)
        except Exception:
            frame = None
        else:
            for column in BEAT_COLUMNS:
                if column not in frame.columns:
                    frame[column] = np.nan
            return frame[BEAT_COLUMNS]
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
        for column in BEAT_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        return frame[BEAT_COLUMNS]
    raise FileNotFoundError(f"Missing beat-level file for {record}.")


def robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return 0.0, 1.0
    center = float(np.median(clean))
    mad = float(np.median(np.abs(clean - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(np.std(clean, ddof=0))
    if not np.isfinite(scale) or scale <= EPS:
        scale = 1.0
    return center, scale


def trailing_roll(values: np.ndarray, window: int, reducer: str = "mean") -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    if window <= 1:
        return values.astype(float)
    for idx in range(len(values)):
        lo = max(0, idx - window + 1)
        seg = values[lo : idx + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) == 0:
            continue
        if reducer == "mean":
            out[idx] = float(np.mean(seg))
        elif reducer == "max":
            out[idx] = float(np.max(seg))
        else:
            raise ValueError(f"Unknown reducer: {reducer}")
    return out


def max_true_run(mask: np.ndarray) -> int:
    arr = np.asarray(mask, dtype=bool)
    if len(arr) == 0 or not arr.any():
        return 0
    padded = np.concatenate(([False], arr, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return int(np.max(ends - starts)) if len(starts) else 0


def select_reference_baseline(df: pd.DataFrame, cfg: TransitionHMMConfig) -> pd.DataFrame:
    df = df.copy()
    st_mask = df["st_event"].fillna(False).astype(bool)
    non_st = df.loc[~st_mask].copy()

    if st_mask.any():
        first_st_sample = int(df.loc[st_mask, "beat_sample"].min())
        pre_first_st = non_st.loc[non_st["beat_sample"] < first_st_sample].copy()
        if len(pre_first_st) >= cfg.min_baseline_beats:
            return pre_first_st.head(cfg.baseline_non_st_beats).copy()

    if len(non_st) >= cfg.min_baseline_beats:
        return non_st.head(cfg.baseline_non_st_beats).copy()

    valid = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["poincare_b", "drift_norm", "energy_asym"])
    return valid.head(max(cfg.min_baseline_beats, min(len(valid), cfg.baseline_non_st_beats))).copy()


def routed_reference_score(df: pd.DataFrame, phenotype_target: str) -> np.ndarray:
    if phenotype_target == "loose_orbit":
        return df["kernel_score_angle"].to_numpy(dtype=float)
    if phenotype_target == "constrained_orbit":
        return df["kernel_score_long"].to_numpy(dtype=float)
    return df["kernel_score_hybrid"].to_numpy(dtype=float)


def build_relative_feature_frame(df: pd.DataFrame, cfg: TransitionHMMConfig) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["st_event"] = out["st_event"].fillna(False).astype(bool)
    out["stiffness_proxy"] = out["poincare_b"] / (out["drift_norm"] + EPS)

    baseline = select_reference_baseline(out, cfg)
    reference_cols = ["poincare_b", "drift_norm", "energy_asym", "stiffness_proxy"]
    baseline_idx = baseline.index.to_numpy(dtype=int)

    for column in reference_cols:
        center, scale = robust_center_scale(baseline[column].to_numpy(dtype=float))
        out[f"baseline_center_{column}"] = center
        out[f"baseline_scale_{column}"] = scale
        out[f"rel_{column}"] = (out[column].to_numpy(dtype=float) - center) / (scale + EPS)
        out[f"vel_{column}"] = np.r_[0.0, np.diff(out[f"rel_{column}"].to_numpy(dtype=float))]
        out[f"acc_{column}"] = np.r_[0.0, np.diff(out[f"vel_{column}"].to_numpy(dtype=float))]

    rel_cols = [f"rel_{column}" for column in reference_cols]
    vel_cols = [f"vel_{column}" for column in reference_cols]
    acc_cols = [f"acc_{column}" for column in reference_cols]

    out["level_norm_raw"] = np.sqrt(np.sum(np.square(out[rel_cols].to_numpy(dtype=float)), axis=1))
    out["velocity_norm_raw"] = np.sqrt(np.sum(np.square(out[vel_cols].to_numpy(dtype=float)), axis=1))
    out["curvature_norm_raw"] = np.sqrt(np.sum(np.square(out[acc_cols].to_numpy(dtype=float)), axis=1))

    window = max(1, int(cfg.rolling_window_beats))
    for column in ["level_norm_raw", "velocity_norm_raw", "curvature_norm_raw"]:
        out[column.replace("_raw", "")] = trailing_roll(out[column].to_numpy(dtype=float), window=window, reducer="mean")

    out["hmm_transition_score_raw"] = np.sqrt(
        cfg.level_weight * np.square(out["level_norm"].to_numpy(dtype=float))
        + cfg.velocity_weight * np.square(out["velocity_norm"].to_numpy(dtype=float))
        + cfg.curvature_weight * np.square(out["curvature_norm"].to_numpy(dtype=float))
        + cfg.stiffness_weight * np.square(out["acc_stiffness_proxy"].to_numpy(dtype=float))
    )
    out["hmm_transition_score"] = trailing_roll(
        out["hmm_transition_score_raw"].to_numpy(dtype=float),
        window=window,
        reducer="mean",
    )

    for metric in ["level_norm", "velocity_norm", "curvature_norm", "hmm_transition_score"]:
        center, scale = robust_center_scale(out.loc[baseline_idx, metric].to_numpy(dtype=float))
        out[f"{metric}_z"] = (out[metric].to_numpy(dtype=float) - center) / (scale + EPS)
        out[f"{metric}_z"] = np.where(np.isfinite(out[f"{metric}_z"].to_numpy(dtype=float)), out[f"{metric}_z"], 0.0)

    projective_reference_cols = list(CANONICAL_PROJECTIVE_REFERENCE_COLUMNS)
    for column in projective_reference_cols:
        center, scale = robust_center_scale(out.loc[baseline_idx, column].to_numpy(dtype=float))
        out[f"baseline_center_{column}"] = center
        out[f"baseline_scale_{column}"] = scale
        out[f"projective_rel_{column}"] = (out[column].to_numpy(dtype=float) - center) / (scale + EPS)
        out[f"projective_vel_{column}"] = np.r_[0.0, np.diff(out[f"projective_rel_{column}"].to_numpy(dtype=float))]
        out[f"projective_acc_{column}"] = np.r_[0.0, np.diff(out[f"projective_vel_{column}"].to_numpy(dtype=float))]

    projective_rel_cols = [f"projective_rel_{column}" for column in projective_reference_cols]
    projective_vel_cols = [f"projective_vel_{column}" for column in projective_reference_cols]
    projective_acc_cols = [f"projective_acc_{column}" for column in projective_reference_cols]
    out["projective_level_norm_raw"] = np.sqrt(np.sum(np.square(out[projective_rel_cols].to_numpy(dtype=float)), axis=1))
    out["projective_velocity_norm_raw"] = np.sqrt(np.sum(np.square(out[projective_vel_cols].to_numpy(dtype=float)), axis=1))
    out["projective_curvature_norm_raw"] = np.sqrt(np.sum(np.square(out[projective_acc_cols].to_numpy(dtype=float)), axis=1))

    for column in ["projective_level_norm_raw", "projective_velocity_norm_raw", "projective_curvature_norm_raw"]:
        out[column.replace("_raw", "")] = trailing_roll(out[column].to_numpy(dtype=float), window=window, reducer="mean")

    barrier_acc = np.nan_to_num(out["projective_acc_proj_lock_barrier_sl"].to_numpy(dtype=float), nan=0.0)
    novelty_vel = np.nan_to_num(out["projective_vel_novelty"].to_numpy(dtype=float), nan=0.0)
    out["transition_score_raw"] = canonical_projective_transition_score(
        level=out["projective_level_norm"].to_numpy(dtype=float),
        velocity=out["projective_velocity_norm"].to_numpy(dtype=float),
        curvature=out["projective_curvature_norm"].to_numpy(dtype=float),
        barrier_acc=barrier_acc,
        novelty_vel=novelty_vel,
    )
    out["transition_score"] = trailing_roll(out["transition_score_raw"].to_numpy(dtype=float), window=window, reducer="mean")

    for metric in ["projective_level_norm", "projective_velocity_norm", "projective_curvature_norm", "transition_score"]:
        center, scale = robust_center_scale(out.loc[baseline_idx, metric].to_numpy(dtype=float))
        out[f"{metric}_z"] = (out[metric].to_numpy(dtype=float) - center) / (scale + EPS)
        out[f"{metric}_z"] = np.where(np.isfinite(out[f"{metric}_z"].to_numpy(dtype=float)), out[f"{metric}_z"], 0.0)

    for metric in CANONICAL_LIE_FEATURE_COLUMNS:
        center, scale = robust_center_scale(out.loc[baseline_idx, metric].to_numpy(dtype=float))
        out[f"{metric}_z"] = (out[metric].to_numpy(dtype=float) - center) / (scale + EPS)
        out[f"{metric}_z"] = np.where(np.isfinite(out[f"{metric}_z"].to_numpy(dtype=float)), out[f"{metric}_z"], 0.0)

    out["lie_transition_score"] = canonical_lie_transition_score(
        orbit_z=out["lie_orbit_norm_z"].to_numpy(dtype=float),
        strain_z=out["lie_strain_norm_z"].to_numpy(dtype=float),
        commutator_z=out["lie_commutator_norm_z"].to_numpy(dtype=float),
        metric_drift_z=out["lie_metric_drift_z"].to_numpy(dtype=float),
        gram_logdet_z=out["gram_logdet_z"].to_numpy(dtype=float),
    )

    phenotype = str(out["phenotype_target"].iloc[0])
    out["routed_reference_score"] = routed_reference_score(out, phenotype)
    out["hybrid_reference_score"] = out["kernel_score_hybrid"].to_numpy(dtype=float)

    for metric in ["routed_reference_score", "hybrid_reference_score"]:
        center, scale = robust_center_scale(out.loc[baseline_idx, metric].to_numpy(dtype=float))
        out[f"{metric}_z"] = (out[metric].to_numpy(dtype=float) - center) / (scale + EPS)
        out[f"{metric}_z"] = np.where(np.isfinite(out[f"{metric}_z"].to_numpy(dtype=float)), out[f"{metric}_z"], 0.0)

    return out


def emission_logits(frame: pd.DataFrame) -> np.ndarray:
    score_z = frame["transition_score_z"].to_numpy(dtype=float)
    level_z = frame["level_norm_z"].to_numpy(dtype=float)
    velocity_z = frame["velocity_norm_z"].to_numpy(dtype=float)
    curvature_z = frame["curvature_norm_z"].to_numpy(dtype=float)

    baseline_logit = -1.15 * score_z - 0.60 * level_z - 0.35 * curvature_z
    transition_logit = 0.90 * score_z + 0.95 * curvature_z + 0.45 * velocity_z - 0.25 * level_z
    active_logit = 0.80 * score_z + 0.90 * level_z + 0.20 * velocity_z - 0.20 * curvature_z
    return np.vstack([baseline_logit, transition_logit, active_logit]).T


def viterbi_decode(logits: np.ndarray) -> np.ndarray:
    trans = np.array(
        [
            [0.94, 0.06, 1e-6],
            [0.08, 0.78, 0.14],
            [1e-6, 0.12, 0.88],
        ],
        dtype=float,
    )
    trans = np.log(trans + EPS)
    start = np.log(np.array([0.995, 0.005, 1e-6], dtype=float) + EPS)

    n_steps, n_states = logits.shape
    dp = np.full((n_steps, n_states), -np.inf, dtype=float)
    back = np.zeros((n_steps, n_states), dtype=int)
    dp[0] = start + logits[0]

    for idx in range(1, n_steps):
        for state in range(n_states):
            prev = dp[idx - 1] + trans[:, state]
            back[idx, state] = int(np.argmax(prev))
            dp[idx, state] = float(np.max(prev) + logits[idx, state])

    states = np.zeros(n_steps, dtype=int)
    states[-1] = int(np.argmax(dp[-1]))
    for idx in range(n_steps - 2, -1, -1):
        states[idx] = int(back[idx + 1, states[idx + 1]])
    return states


def build_episode_table(active_mask: np.ndarray, step_values: np.ndarray, sample_values: np.ndarray, merge_gap: int) -> pd.DataFrame:
    arr = np.asarray(active_mask, dtype=bool)
    rows: list[dict[str, Any]] = []
    if len(arr) == 0:
        return pd.DataFrame(columns=["episode_id", "start_idx", "end_idx", "start_step", "end_step", "start_sample", "end_sample", "duration_beats"])

    start_idx: int | None = None
    for idx, flag in enumerate(arr):
        if flag and start_idx is None:
            start_idx = idx
        elif not flag and start_idx is not None:
            rows.append({"start_idx": int(start_idx), "end_idx": int(idx - 1)})
            start_idx = None
    if start_idx is not None:
        rows.append({"start_idx": int(start_idx), "end_idx": int(len(arr) - 1)})

    if not rows:
        return pd.DataFrame(columns=["episode_id", "start_idx", "end_idx", "start_step", "end_step", "start_sample", "end_sample", "duration_beats"])

    merged: list[dict[str, Any]] = []
    current = rows[0]
    for row in rows[1:]:
        gap = row["start_idx"] - current["end_idx"] - 1
        if gap <= int(merge_gap):
            current["end_idx"] = row["end_idx"]
        else:
            merged.append(current)
            current = row
    merged.append(current)

    final_rows: list[dict[str, Any]] = []
    for episode_id, row in enumerate(merged):
        start_idx = int(row["start_idx"])
        end_idx = int(row["end_idx"])
        final_rows.append(
            {
                "episode_id": int(episode_id),
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_step": int(step_values[start_idx]),
                "end_step": int(step_values[end_idx]),
                "start_sample": int(sample_values[start_idx]),
                "end_sample": int(sample_values[end_idx]),
                "duration_beats": int(end_idx - start_idx + 1),
            }
        )
    return pd.DataFrame(final_rows)


def build_hmm_alert_mask(states: np.ndarray, cfg: TransitionHMMConfig) -> np.ndarray:
    transition_idx = STATE_TO_INDEX["transition"]
    active_idx = STATE_TO_INDEX["active"]
    active_tail = max(0, int(cfg.alert_active_tail_beats))
    max_alert_beats = max(1, int(cfg.alert_max_beats))

    arr = np.asarray(states, dtype=int)
    mask = np.zeros(len(arr), dtype=bool)
    alert_start: int | None = None
    last_transition_idx: int | None = None

    for idx, state in enumerate(arr):
        if state == transition_idx:
            if alert_start is None:
                alert_start = idx
            last_transition_idx = idx
        elif (
            alert_start is not None
            and state == active_idx
            and last_transition_idx is not None
            and (idx - last_transition_idx) <= active_tail
        ):
            pass
        else:
            alert_start = None
            last_transition_idx = None
            continue

        if alert_start is not None and (idx - alert_start + 1) <= max_alert_beats:
            mask[idx] = True
        else:
            alert_start = None
            last_transition_idx = None
    return mask


def apply_episode_cooldown(episodes_df: pd.DataFrame, cooldown_beats: int) -> pd.DataFrame:
    if episodes_df.empty or cooldown_beats <= 0:
        return episodes_df.copy()

    kept_rows: list[dict[str, Any]] = []
    last_end_idx: int | None = None
    for row in episodes_df.sort_values("start_idx").to_dict("records"):
        start_idx = int(row["start_idx"])
        if last_end_idx is not None and start_idx <= (last_end_idx + int(cooldown_beats)):
            continue
        kept_rows.append(row)
        last_end_idx = int(row["end_idx"])

    if not kept_rows:
        return episodes_df.iloc[0:0].copy()

    out = pd.DataFrame(kept_rows).reset_index(drop=True)
    out["episode_id"] = np.arange(len(out), dtype=int)
    return out[episodes_df.columns]


def episode_table_to_mask(length: int, episodes_df: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(int(length), dtype=bool)
    if episodes_df.empty:
        return mask
    for row in episodes_df.itertuples(index=False):
        start_idx = max(0, int(row.start_idx))
        end_idx = min(int(length) - 1, int(row.end_idx))
        if end_idx >= start_idx:
            mask[start_idx : end_idx + 1] = True
    return mask


def detect_score_episodes(
    scores: np.ndarray,
    step_values: np.ndarray,
    sample_values: np.ndarray,
    on_threshold: float,
    off_threshold: float,
    on_sustain: int,
    off_sustain: int,
    merge_gap: int,
) -> pd.DataFrame:
    active = np.zeros(len(scores), dtype=bool)
    in_episode = False
    on_run = 0
    off_run = 0
    for idx, score in enumerate(np.nan_to_num(scores.astype(float), nan=0.0)):
        if not in_episode:
            if score >= on_threshold:
                on_run += 1
            else:
                on_run = 0
            if on_run >= on_sustain:
                start = idx - on_sustain + 1
                active[start : idx + 1] = True
                in_episode = True
                off_run = 0
        else:
            active[idx] = True
            if score <= off_threshold:
                off_run += 1
            else:
                off_run = 0
            if off_run >= off_sustain:
                stop = max(0, idx - off_sustain + 1)
                active[stop : idx + 1] = False
                in_episode = False
                on_run = 0
                off_run = 0
    return build_episode_table(active, step_values=step_values, sample_values=sample_values, merge_gap=merge_gap)


def first_pre_event_lead(event_start_idx: int, episodes_df: pd.DataFrame, pre_event_beats: int) -> tuple[float, int | None]:
    if episodes_df.empty:
        return np.nan, None
    start_min = int(event_start_idx - pre_event_beats)
    eligible = episodes_df.loc[(episodes_df["start_idx"] >= start_min) & (episodes_df["start_idx"] < event_start_idx)].sort_values("start_idx")
    if eligible.empty:
        return np.nan, None
    best = eligible.iloc[0]
    return float(event_start_idx - int(best["start_idx"])), int(best["episode_id"])


def overlap_fraction(event_start_idx: int, event_end_idx: int, episodes_df: pd.DataFrame) -> float:
    if episodes_df.empty:
        return 0.0
    total = max(1, event_end_idx - event_start_idx + 1)
    covered = 0
    for _, row in episodes_df.iterrows():
        lo = max(int(row["start_idx"]), int(event_start_idx))
        hi = min(int(row["end_idx"]), int(event_end_idx))
        if hi >= lo:
            covered += hi - lo + 1
    return float(min(covered / total, 1.0))


def pre_event_continuity(event_start_idx: int, episodes_df: pd.DataFrame, pre_event_beats: int) -> tuple[float, int]:
    if episodes_df.empty:
        return 0.0, 0
    pre_lo = max(0, int(event_start_idx - pre_event_beats))
    pre_hi = int(event_start_idx - 1)
    if pre_hi < pre_lo:
        return 0.0, 0
    mask = np.zeros(pre_hi - pre_lo + 1, dtype=bool)
    for _, row in episodes_df.iterrows():
        lo = max(int(row["start_idx"]), pre_lo)
        hi = min(int(row["end_idx"]), pre_hi)
        if hi >= lo:
            mask[(lo - pre_lo) : (hi - pre_lo + 1)] = True
    max_run = max_true_run(mask)
    return float(mask.mean()) if len(mask) else 0.0, int(max_run)


def state_runs(states: np.ndarray) -> dict[str, int]:
    out = {}
    for idx, name in enumerate(STATE_ORDER):
        out[f"max_run_{name}"] = max_true_run(states == idx)
    return out


def evaluate_record(record_df: pd.DataFrame, cfg: TransitionHMMConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = build_relative_feature_frame(record_df, cfg)
    hmm_frame = frame.copy()
    hmm_frame["transition_score_z"] = hmm_frame["hmm_transition_score_z"]
    logits = emission_logits(hmm_frame)
    state_idx = viterbi_decode(logits)
    frame["state"] = [STATE_ORDER[int(idx)] for idx in state_idx]
    frame["state_code"] = state_idx

    hmm_alert_mask = build_hmm_alert_mask(state_idx, cfg)
    frame["hmm_alert"] = hmm_alert_mask
    hmm_episodes = build_episode_table(
        active_mask=hmm_alert_mask,
        step_values=np.arange(len(frame), dtype=int),
        sample_values=frame["beat_sample"].to_numpy(dtype=int),
        merge_gap=cfg.merge_gap_beats,
    )
    hmm_episodes = apply_episode_cooldown(hmm_episodes, cfg.alert_cooldown_beats)
    frame["hmm_alert"] = episode_table_to_mask(len(frame), hmm_episodes)
    routed_episodes = detect_score_episodes(
        frame["routed_reference_score_z"].to_numpy(dtype=float),
        step_values=np.arange(len(frame), dtype=int),
        sample_values=frame["beat_sample"].to_numpy(dtype=int),
        on_threshold=cfg.detector_on_threshold,
        off_threshold=cfg.detector_off_threshold,
        on_sustain=cfg.detector_on_sustain,
        off_sustain=cfg.detector_off_sustain,
        merge_gap=cfg.merge_gap_beats,
    )
    hybrid_episodes = detect_score_episodes(
        frame["hybrid_reference_score_z"].to_numpy(dtype=float),
        step_values=np.arange(len(frame), dtype=int),
        sample_values=frame["beat_sample"].to_numpy(dtype=int),
        on_threshold=cfg.detector_on_threshold,
        off_threshold=cfg.detector_off_threshold,
        on_sustain=cfg.detector_on_sustain,
        off_sustain=cfg.detector_off_sustain,
        merge_gap=cfg.merge_gap_beats,
    )

    for name, table in [("hmm", hmm_episodes), ("routed", routed_episodes), ("hybrid", hybrid_episodes)]:
        if not table.empty:
            table["record"] = str(frame["record"].iloc[0])
            table["regime"] = str(frame["regime"].iloc[0])
            table["phenotype_target"] = str(frame["phenotype_target"].iloc[0])
            table["method"] = name

    st_ids = sorted(v for v in frame["st_episode_id"].dropna().astype(int).unique().tolist() if v >= 0)
    event_rows: list[dict[str, Any]] = []
    for st_id in st_ids:
        event_frame = frame.loc[frame["st_episode_id"].astype(int) == int(st_id)].copy()
        if event_frame.empty:
            continue
        start_idx = int(event_frame.index.min())
        end_idx = int(event_frame.index.max())
        hmm_lead, hmm_episode_id = first_pre_event_lead(start_idx, hmm_episodes, cfg.pre_event_beats)
        routed_lead, routed_episode_id = first_pre_event_lead(start_idx, routed_episodes, cfg.pre_event_beats)
        hybrid_lead, hybrid_episode_id = first_pre_event_lead(start_idx, hybrid_episodes, cfg.pre_event_beats)
        hmm_pre_fraction, hmm_pre_max_run = pre_event_continuity(start_idx, hmm_episodes, cfg.pre_event_beats)
        routed_pre_fraction, routed_pre_max_run = pre_event_continuity(start_idx, routed_episodes, cfg.pre_event_beats)
        hybrid_pre_fraction, hybrid_pre_max_run = pre_event_continuity(start_idx, hybrid_episodes, cfg.pre_event_beats)
        event_rows.append(
            {
                "record": str(frame["record"].iloc[0]),
                "regime": str(frame["regime"].iloc[0]),
                "phenotype_target": str(frame["phenotype_target"].iloc[0]),
                "st_episode_id": int(st_id),
                "event_start_idx": start_idx,
                "event_end_idx": end_idx,
                "event_start_sample": int(frame.at[start_idx, "beat_sample"]),
                "event_end_sample": int(frame.at[end_idx, "beat_sample"]),
                "event_duration_beats": int(end_idx - start_idx + 1),
                "hmm_lead_beats": hmm_lead,
                "routed_lead_beats": routed_lead,
                "hybrid_lead_beats": hybrid_lead,
                "hmm_visible": int(np.isfinite(hmm_lead)),
                "routed_visible": int(np.isfinite(routed_lead)),
                "hybrid_visible": int(np.isfinite(hybrid_lead)),
                "hmm_improves_over_routed": int((np.isfinite(hmm_lead) and not np.isfinite(routed_lead)) or (np.isfinite(hmm_lead) and np.isfinite(routed_lead) and float(hmm_lead) > float(routed_lead))),
                "hmm_improves_over_hybrid": int((np.isfinite(hmm_lead) and not np.isfinite(hybrid_lead)) or (np.isfinite(hmm_lead) and np.isfinite(hybrid_lead) and float(hmm_lead) > float(hybrid_lead))),
                "hmm_overlap_fraction": overlap_fraction(start_idx, end_idx, hmm_episodes),
                "routed_overlap_fraction": overlap_fraction(start_idx, end_idx, routed_episodes),
                "hybrid_overlap_fraction": overlap_fraction(start_idx, end_idx, hybrid_episodes),
                "hmm_pre_event_fraction": hmm_pre_fraction,
                "routed_pre_event_fraction": routed_pre_fraction,
                "hybrid_pre_event_fraction": hybrid_pre_fraction,
                "hmm_pre_event_max_run": int(hmm_pre_max_run),
                "routed_pre_event_max_run": int(routed_pre_max_run),
                "hybrid_pre_event_max_run": int(hybrid_pre_max_run),
                "hmm_episode_id": hmm_episode_id,
                "routed_episode_id": routed_episode_id,
                "hybrid_episode_id": hybrid_episode_id,
            }
        )

    record_summary = {
        "record": str(frame["record"].iloc[0]),
        "regime": str(frame["regime"].iloc[0]),
        "phenotype_target": str(frame["phenotype_target"].iloc[0]),
        "n_beats": int(len(frame)),
        "n_st_episodes": int(len(st_ids)),
        "baseline_center_poincare_b": float(frame["baseline_center_poincare_b"].iloc[0]),
        "baseline_center_drift_norm": float(frame["baseline_center_drift_norm"].iloc[0]),
        "baseline_center_energy_asym": float(frame["baseline_center_energy_asym"].iloc[0]),
        "baseline_center_stiffness_proxy": float(frame["baseline_center_stiffness_proxy"].iloc[0]),
        "median_hmm_transition_score_z": float(np.nanmedian(frame["hmm_transition_score_z"].to_numpy(dtype=float))),
        "p95_hmm_transition_score_z": float(np.nanquantile(frame["hmm_transition_score_z"].to_numpy(dtype=float), 0.95)),
        "median_transition_score_z": float(np.nanmedian(frame["transition_score_z"].to_numpy(dtype=float))),
        "p95_transition_score_z": float(np.nanquantile(frame["transition_score_z"].to_numpy(dtype=float), 0.95)),
        "median_lie_transition_score": float(np.nanmedian(frame["lie_transition_score"].to_numpy(dtype=float))),
        "p95_lie_transition_score": float(np.nanquantile(frame["lie_transition_score"].to_numpy(dtype=float), 0.95)),
        **state_runs(state_idx),
    }
    if event_rows:
        event_df = pd.DataFrame(event_rows)
        record_summary["median_hmm_lead_beats"] = float(np.nanmedian(event_df["hmm_lead_beats"].to_numpy(dtype=float)))
        record_summary["median_routed_lead_beats"] = float(np.nanmedian(event_df["routed_lead_beats"].to_numpy(dtype=float)))
        record_summary["median_hybrid_lead_beats"] = float(np.nanmedian(event_df["hybrid_lead_beats"].to_numpy(dtype=float)))
        record_summary["fraction_events_hmm_visible"] = float(event_df["hmm_visible"].mean())
        record_summary["fraction_events_routed_visible"] = float(event_df["routed_visible"].mean())
        record_summary["fraction_events_hybrid_visible"] = float(event_df["hybrid_visible"].mean())
    else:
        event_df = pd.DataFrame()

    episode_tables = pd.concat([table for table in [hmm_episodes, routed_episodes, hybrid_episodes] if not table.empty], ignore_index=True) if any(not t.empty for t in [hmm_episodes, routed_episodes, hybrid_episodes]) else pd.DataFrame()
    return frame, event_df, pd.DataFrame([record_summary]), episode_tables


def plot_exemplar(record: str, frame: pd.DataFrame, out_dir: Path, cfg: TransitionHMMConfig) -> None:
    st_idx = frame.index[frame["st_event"]].to_numpy(dtype=int)
    if len(st_idx) == 0:
        return
    onset_idx = int(st_idx[0])
    lo = max(0, onset_idx - cfg.pre_event_beats)
    hi = min(len(frame), onset_idx + cfg.post_event_beats)
    window = frame.iloc[lo:hi].copy()
    x = np.arange(lo - onset_idx, hi - onset_idx, dtype=int)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(x, window["hmm_transition_score_z"], label="hmm_transition_score_z", color="#1b9e77")
    axes[0].plot(x, window["transition_score_z"], label="projective_transition_score_z", color="#66a61e", alpha=0.75)
    axes[0].plot(x, window["lie_transition_score"], label="lie_transition_score", color="#7570b3", alpha=0.75)
    axes[0].plot(x, window["routed_reference_score_z"], label="routed_score_z", color="#d95f02", alpha=0.8)
    axes[0].plot(x, window["hybrid_reference_score_z"], label="hybrid_score_z", color="#e7298a", alpha=0.8)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Robust z")
    axes[0].legend(loc="upper left")

    axes[1].plot(x, window["rel_poincare_b"], label="rel_poincare_b", color="#1f78b4")
    axes[1].plot(x, window["rel_drift_norm"], label="rel_drift_norm", color="#e31a1c")
    axes[1].plot(x, window["rel_stiffness_proxy"], label="rel_stiffness_proxy", color="#33a02c")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Relative level")
    axes[1].legend(loc="upper left")

    state_colors = {"baseline": "#d9d9d9", "transition": "#fdae61", "active": "#2b83ba"}
    axes[2].plot(x, window["velocity_norm_z"], label="velocity_norm_z", color="#7570b3")
    axes[2].plot(x, window["curvature_norm_z"], label="curvature_norm_z", color="#66a61e")
    axes[2].axvline(0, color="black", linestyle="--", linewidth=1.0)
    for state_name, color in state_colors.items():
        mask = window["state"] == state_name
        if mask.any():
            axes[2].fill_between(x, 0, 1, where=mask.to_numpy(dtype=bool), color=color, alpha=0.15, transform=axes[2].get_xaxis_transform())
    axes[2].set_ylabel("Velocity/Curvature z")
    axes[2].set_xlabel("Beats from first ST beat")
    axes[2].legend(loc="upper left")

    fig.suptitle(f"{record} relative-geometry HMM")
    fig.tight_layout()
    fig.savefig(out_dir / f"{record}_transition_hmm.png", dpi=160)
    plt.close(fig)


def write_report(cfg: TransitionHMMConfig, out_dir: Path, event_df: pd.DataFrame, record_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    lines = [
        "# LTST Relative-Geometry Transition HMM",
        "",
        "First cardiac port of the PMU-style architecture: legacy relative primitive geometry for the frozen HMM, plus canonical projective and derivative Gram/Lie sidecar features over beat windows.",
        "",
        "## Inputs",
        "- `rel/vel/acc` of `poincare_b`, `drift_norm`, `energy_asym`",
        "- host-only `stiffness_proxy = poincare_b / (drift_norm + eps)`",
        "- canonical projective sidecar: `memory_align`, `novelty`, `proj_lock_barrier_sl`, `proj_lock_barrier_xl`, `proj_volume_xsl`",
        "- derivative Gram/Lie sidecar: `gram_trace`, `gram_logdet`, `lie_orbit_norm`, `lie_strain_norm`, `lie_commutator_norm`, `lie_metric_drift`",
        "- HMM states: `baseline`, `transition`, `active`",
        f"- HMM alert extraction: `transition` onset plus `{int(cfg.alert_active_tail_beats)}`-beat `active` tail, capped at `{int(cfg.alert_max_beats)}` beats",
        f"- HMM alert cooldown: suppress re-arming for `{int(cfg.alert_cooldown_beats)}` beats after each alert",
        "",
    ]
    if not event_df.empty:
        event_count = int(len(event_df))
        lines += [
            "## Event Summary",
            f"- ST episodes evaluated: `{event_count}`",
            f"- HMM visible pre-ST: `{int(event_df['hmm_visible'].sum())}` / `{event_count}`",
            f"- Routed reference visible pre-ST: `{int(event_df['routed_visible'].sum())}` / `{event_count}`",
            f"- Hybrid reference visible pre-ST: `{int(event_df['hybrid_visible'].sum())}` / `{event_count}`",
            f"- HMM improves over routed reference on lead: `{int(event_df['hmm_improves_over_routed'].sum())}` events",
            f"- HMM improves over hybrid reference on lead: `{int(event_df['hmm_improves_over_hybrid'].sum())}` events",
            f"- Median HMM lead (visible events): `{float(np.nanmedian(event_df['hmm_lead_beats'])):.1f}` beats",
            f"- Median routed lead (visible events): `{float(np.nanmedian(event_df['routed_lead_beats'])):.1f}` beats",
            f"- Median hybrid lead (visible events): `{float(np.nanmedian(event_df['hybrid_lead_beats'])):.1f}` beats",
            "",
            "## Top Record Summaries",
            record_df.sort_values(["fraction_events_hmm_visible", "median_hmm_lead_beats"], ascending=[False, False]).head(12).to_markdown(index=False),
            "",
            "## Event Table Head",
            event_df.sort_values(["hmm_visible", "hmm_lead_beats"], ascending=[False, False]).head(20).to_markdown(index=False),
            "",
        ]
    if not summary_df.empty:
        lines += ["## State Summary", summary_df.to_markdown(index=False), ""]
    (out_dir / "ltst_transition_hmm_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = TransitionHMMConfig(
        run_dir=Path(args.run_dir).resolve(),
        out_dir_name=str(args.out_dir_name),
        baseline_non_st_beats=int(args.baseline_non_st_beats),
        rolling_window_beats=int(args.rolling_window_beats),
        pre_event_beats=int(args.pre_event_beats),
        post_event_beats=int(args.post_event_beats),
        alert_active_tail_beats=int(args.alert_active_tail_beats),
        alert_max_beats=int(args.alert_max_beats),
        alert_cooldown_beats=int(args.alert_cooldown_beats),
    )
    out_dir = ensure_output_dir(cfg.run_dir, cfg.out_dir_name)

    regime_df = pd.read_csv(cfg.run_dir / "regime.csv")[["record", "regime"]].copy()
    regime_df["phenotype_target"] = regime_df["regime"].map(phenotype_from_regime)
    beat_level_dir = cfg.run_dir / "beat_level"

    frame_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []
    record_parts: list[pd.DataFrame] = []
    episode_parts: list[pd.DataFrame] = []

    for idx, row in regime_df.sort_values("record").iterrows():
        record = str(row["record"])
        regime = str(row["regime"])
        print(f"[LTST-HMM] {idx + 1}/{len(regime_df)} {record} ({regime})", flush=True)
        beat_df = read_beat_frame(beat_level_dir, record)
        beat_df = beat_df.replace([np.inf, -np.inf], np.nan)
        beat_df["record"] = record
        beat_df["regime"] = regime
        beat_df["phenotype_target"] = phenotype_from_regime(regime)
        frame, event_df, record_df, episode_df = evaluate_record(beat_df, cfg)
        frame_parts.append(frame)
        record_parts.append(record_df)
        if not event_df.empty:
            event_parts.append(event_df)
        if not episode_df.empty:
            episode_parts.append(episode_df)
        if record in cfg.exemplar_records:
            plot_exemplar(record, frame, out_dir, cfg)

    frame_df = pd.concat(frame_parts, ignore_index=True)
    event_df = pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame()
    record_df = pd.concat(record_parts, ignore_index=True) if record_parts else pd.DataFrame()
    episode_df = pd.concat(episode_parts, ignore_index=True) if episode_parts else pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    if not event_df.empty:
        summary_rows.extend(
            [
                {"metric": "events_visible_fraction", "hmm": float(event_df["hmm_visible"].mean()), "routed": float(event_df["routed_visible"].mean()), "hybrid": float(event_df["hybrid_visible"].mean())},
                {"metric": "median_lead_beats", "hmm": float(np.nanmedian(event_df["hmm_lead_beats"].to_numpy(dtype=float))), "routed": float(np.nanmedian(event_df["routed_lead_beats"].to_numpy(dtype=float))), "hybrid": float(np.nanmedian(event_df["hybrid_lead_beats"].to_numpy(dtype=float)))},
                {"metric": "median_overlap_fraction", "hmm": float(np.nanmedian(event_df["hmm_overlap_fraction"].to_numpy(dtype=float))), "routed": float(np.nanmedian(event_df["routed_overlap_fraction"].to_numpy(dtype=float))), "hybrid": float(np.nanmedian(event_df["hybrid_overlap_fraction"].to_numpy(dtype=float)))},
                {"metric": "median_pre_event_fraction", "hmm": float(np.nanmedian(event_df["hmm_pre_event_fraction"].to_numpy(dtype=float))), "routed": float(np.nanmedian(event_df["routed_pre_event_fraction"].to_numpy(dtype=float))), "hybrid": float(np.nanmedian(event_df["hybrid_pre_event_fraction"].to_numpy(dtype=float)))},
                {"metric": "events_hmm_improves_over_routed", "hmm": int(event_df["hmm_improves_over_routed"].sum()), "routed": np.nan, "hybrid": np.nan},
                {"metric": "events_hmm_improves_over_hybrid", "hmm": int(event_df["hmm_improves_over_hybrid"].sum()), "routed": np.nan, "hybrid": np.nan},
            ]
        )
    summary_df = pd.DataFrame(summary_rows)

    frame_df.to_csv(out_dir / "ltst_transition_hmm_windows.csv", index=False)
    event_df.to_csv(out_dir / "ltst_transition_hmm_event_table.csv", index=False)
    record_df.to_csv(out_dir / "ltst_transition_hmm_record_summary.csv", index=False)
    episode_df.to_csv(out_dir / "ltst_transition_hmm_episodes.csv", index=False)
    summary_df.to_csv(out_dir / "ltst_transition_hmm_summary.csv", index=False)
    write_report(cfg, out_dir, event_df=event_df, record_df=record_df, summary_df=summary_df)

    metadata = {
        "config": json_safe(asdict(cfg)),
        "n_records": int(len(record_df)),
        "n_events": int(len(event_df)),
        "n_episodes": int(len(episode_df)),
        "states": STATE_ORDER,
        "inputs": [
            "rel/vel/acc poincare_b",
            "rel/vel/acc drift_norm",
            "rel/vel/acc energy_asym",
            "rel/vel/acc stiffness_proxy",
            *CANONICAL_PROJECTIVE_REFERENCE_COLUMNS,
            *CANONICAL_LIE_FEATURE_COLUMNS,
        ],
    }
    with (out_dir / "ltst_transition_hmm_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"[LTST-HMM] Output directory: {out_dir}", flush=True)
    if not summary_df.empty:
        print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
