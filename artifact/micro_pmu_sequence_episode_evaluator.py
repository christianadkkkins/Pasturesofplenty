from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.signal import lfilter
except Exception:
    lfilter = None

from micro_pmu_sequence_experiment import (
    EPS,
    MicroPMUSequenceConfig,
    compute_sequence_chunk_features,
    json_safe,
    sequence_basis,
    voltage_phasors_from_chunk,
)
from micro_pmu_twist_net import (
    RelativeGeometryConfig,
    RelativeGeometryState,
    TwistFeatureConfig,
    TwistFeatureState,
    compute_heuristic_twist_outputs,
    compute_relative_geometry_frame,
    infer_checkpoint_twist_outputs,
    compute_twist_feature_frame,
    decode_twist_sequence,
)


SCORE_METRICS = [
    "seq_pos_frac",
    "seq_neg_frac",
    "seq_zero_frac",
    "seq_unbalance_barrier",
]
SCORE_DIRECTIONS = {
    "seq_pos_frac": "lower",
    "seq_neg_frac": "higher",
    "seq_zero_frac": "higher",
    "seq_unbalance_barrier": "higher",
}


@dataclass(frozen=True)
class EpisodeEvalConfig:
    source_run_dir: Path
    run_root: Path
    data_path: Path
    chosen_orientation: str
    chunk_size: int
    pre_event_steps: int
    post_event_steps: int
    baseline_steps: int
    non_event_stride: int
    ema_alpha: float = 0.05
    episode_on_threshold: float = 2.0
    episode_off_threshold: float = 0.75
    on_sustain_steps: int = 12
    off_sustain_steps: int = 24
    merge_gap_steps: int = 120
    decoder_mode: str = "twist_shmm"
    twist_enter_threshold: float = 0.60
    twist_exit_threshold: float = 0.35
    twist_fault_min_dwell: int = 12
    twist_stable_min_dwell: int = 24
    twist_checkpoint_path: Path | None = None
    twist_baseline_alpha: float = 0.01


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_run_dir(run_root: Path) -> Path:
    run_dir = run_root / f"micro_pmu_sequence_episode_{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_source_run_context(source_run_dir: Path, run_root: Path) -> tuple[EpisodeEvalConfig, pd.DataFrame]:
    metadata = json.loads((source_run_dir / "micro_pmu_sequence_metadata.json").read_text(encoding="utf-8"))
    source_cfg = MicroPMUSequenceConfig(
        data_path=Path(metadata["config"]["data_path"]),
        run_root=Path(metadata["config"]["run_root"]),
        chunk_size=int(metadata["config"]["chunk_size"]),
        min_voltage_norm=float(metadata["config"]["min_voltage_norm"]),
        pre_event_steps=int(metadata["config"]["pre_event_steps"]),
        post_event_steps=int(metadata["config"]["post_event_steps"]),
        baseline_steps=int(metadata["config"]["baseline_steps"]),
        threshold_sigma=float(metadata["config"]["threshold_sigma"]),
        sustain_steps=int(metadata["config"]["sustain_steps"]),
        rearm_sigma=float(metadata["config"]["rearm_sigma"]),
        rearm_steps=int(metadata["config"]["rearm_steps"]),
        non_event_stride=int(metadata["config"]["non_event_stride"]),
    )
    sampled_df = pd.read_csv(source_run_dir / "micro_pmu_sequence_sampled_features.csv")
    cfg = EpisodeEvalConfig(
        source_run_dir=source_run_dir,
        run_root=run_root,
        data_path=source_cfg.data_path,
        chosen_orientation=str(metadata["chosen_orientation"]),
        chunk_size=source_cfg.chunk_size,
        pre_event_steps=source_cfg.pre_event_steps,
        post_event_steps=source_cfg.post_event_steps,
        baseline_steps=source_cfg.baseline_steps,
        non_event_stride=source_cfg.non_event_stride,
    )
    return cfg, sampled_df


def compute_baseline_stats(sampled_df: pd.DataFrame) -> pd.DataFrame:
    non_event_df = sampled_df.loc[~sampled_df["event_flag"]].copy()
    rows = []
    for metric in SCORE_METRICS:
        values = non_event_df[metric].to_numpy(dtype=float)
        center = float(np.nanmedian(values))
        mad = float(np.nanmedian(np.abs(values - center)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= EPS:
            scale = float(np.nanstd(values, ddof=0))
        if not np.isfinite(scale) or scale <= EPS:
            scale = 1.0
        rows.append(
            {
                "metric": metric,
                "direction": SCORE_DIRECTIONS[metric],
                "center": center,
                "mad": mad,
                "scale": scale,
                "non_event_count": int(np.isfinite(values).sum()),
            }
        )
    return pd.DataFrame(rows)


def ema_chunk(values: np.ndarray, alpha: float, prev_state: float) -> tuple[np.ndarray, float]:
    if len(values) == 0:
        return np.array([], dtype=float), prev_state
    x = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if lfilter is not None:
        filtered, zf = lfilter([alpha], [1.0, -(1.0 - alpha)], x, zi=[(1.0 - alpha) * prev_state])
        return filtered.astype(np.float64), float(zf[0])

    out = np.zeros_like(x, dtype=np.float64)
    state = float(prev_state)
    for idx, value in enumerate(x):
        state = alpha * value + (1.0 - alpha) * state
        out[idx] = state
    return out, state


def compose_scores(feature_df: pd.DataFrame, baseline_stats_df: pd.DataFrame) -> tuple[np.ndarray, list[str | None], dict[str, np.ndarray]]:
    component_scores: dict[str, np.ndarray] = {}
    for _, row in baseline_stats_df.iterrows():
        metric = str(row["metric"])
        direction = str(row["direction"])
        center = float(row["center"])
        scale = float(row["scale"])
        values = feature_df[metric].to_numpy(dtype=float)
        if direction == "higher":
            z = (values - center) / scale
        else:
            z = (center - values) / scale
        component_scores[metric] = np.where(np.isfinite(z), np.maximum(z, 0.0), np.nan)

    stack = np.vstack([component_scores[m] for m in SCORE_METRICS]).T
    raw_score = np.nanmax(stack, axis=1)
    valid = np.isfinite(stack).any(axis=1)
    raw_score[~valid] = np.nan

    dominant_features: list[str | None] = []
    for row in stack:
        if not np.isfinite(row).any():
            dominant_features.append(None)
        else:
            idx = int(np.nanargmax(row))
            dominant_features.append(SCORE_METRICS[idx])
    return raw_score, dominant_features, component_scores


def update_event_windows(
    chunk_df: pd.DataFrame,
    cfg: EpisodeEvalConfig,
    recent_tail: pd.DataFrame,
    active_collectors: list[dict[str, Any]],
    event_windows: list[pd.DataFrame],
    prev_event: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[pd.DataFrame], bool]:
    def finalize(col: dict[str, Any]) -> None:
        event_df = pd.concat(col["parts"], ignore_index=True)
        needed = cfg.pre_event_steps + 1 + cfg.post_event_steps
        if len(event_df) < needed:
            return
        event_df = event_df.iloc[:needed].copy().reset_index(drop=True)
        event_df["event_index"] = int(col["event_index"])
        event_df["event_onset_step"] = int(col["onset_step"])
        event_df["event_onset_time_ns"] = float(col["onset_time_ns"])
        event_df["steps_from_onset"] = np.arange(-cfg.pre_event_steps, cfg.post_event_steps + 1, dtype=int)
        event_windows.append(event_df)

    if active_collectors:
        carry_active: list[dict[str, Any]] = []
        for col in active_collectors:
            take = min(col["remaining"], len(chunk_df))
            if take > 0:
                col["parts"].append(chunk_df.iloc[:take].copy())
                col["remaining"] -= take
            if col["remaining"] <= 0:
                finalize(col)
            else:
                carry_active.append(col)
        active_collectors = carry_active

    event_flag = chunk_df["event_flag"].to_numpy(dtype=bool)
    onset_mask = event_flag & ~np.concatenate(([prev_event], event_flag[:-1]))
    onset_positions = np.flatnonzero(onset_mask)
    tail_plus_chunk = pd.concat([recent_tail, chunk_df], ignore_index=True)
    tail_len = len(recent_tail)

    for pos in onset_positions:
        onset_in_combined = tail_len + int(pos)
        pre_df = tail_plus_chunk.iloc[max(0, onset_in_combined - cfg.pre_event_steps):onset_in_combined].copy()
        if len(pre_df) < cfg.pre_event_steps:
            continue
        current_df = chunk_df.iloc[pos:pos + 1].copy()
        collector = {
            "event_index": int(chunk_df.iloc[pos]["event_index_hint"]),
            "onset_step": int(chunk_df.iloc[pos]["row_index"]),
            "onset_time_ns": float(chunk_df.iloc[pos]["time_ns"]),
            "parts": [pre_df, current_df],
            "remaining": cfg.post_event_steps,
        }
        available = min(collector["remaining"], len(chunk_df) - pos - 1)
        if available > 0:
            collector["parts"].append(chunk_df.iloc[pos + 1:pos + 1 + available].copy())
            collector["remaining"] -= available
        if collector["remaining"] <= 0:
            finalize(collector)
        else:
            active_collectors.append(collector)

    recent_tail = tail_plus_chunk.tail(cfg.pre_event_steps).copy().reset_index(drop=True)
    prev_event = bool(event_flag[-1]) if len(event_flag) else prev_event
    return recent_tail, active_collectors, event_windows, prev_event


def detect_episodes(
    cfg: EpisodeEvalConfig,
    baseline_stats_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    basis = sequence_basis(cfg.chosen_orientation)
    alpha = cfg.ema_alpha
    prev_ema = 0.0
    twist_state = TwistFeatureState()
    relative_state = RelativeGeometryState()
    relative_cfg = RelativeGeometryConfig(baseline_alpha=cfg.twist_baseline_alpha)

    history = deque(maxlen=cfg.on_sustain_steps)
    in_episode = False
    on_run = 0
    off_run = 0
    current_episode: dict[str, Any] | None = None
    episodes: list[dict[str, Any]] = []

    event_intervals: list[dict[str, Any]] = []
    event_active = False
    event_start_step = -1
    event_start_time = np.nan
    prev_step = -1
    prev_time = np.nan
    event_index_counter = 0
    event_onset_counter = 0

    recent_tail = pd.DataFrame()
    active_collectors: list[dict[str, Any]] = []
    event_windows: list[pd.DataFrame] = []
    prev_event_flag_for_windows = False

    score_sample_parts: list[pd.DataFrame] = []
    global_row = 0

    for chunk_id, chunk in enumerate(pd.read_csv(cfg.data_path, usecols=["Time", "VL1", "VL2", "VL3", "AL1", "AL2", "AL3", "IC1", "IC2", "IC3", "AC1", "AC2", "AC3", "Events"], chunksize=cfg.chunk_size)):
        v, aux = voltage_phasors_from_chunk(chunk)
        if cfg.decoder_mode == "twist_shmm":
            feature_df, twist_state = compute_twist_feature_frame(
                chunk,
                cfg=TwistFeatureConfig(orientation=cfg.chosen_orientation, ema_alpha=cfg.ema_alpha),
                state=twist_state,
            )
            feature_df, relative_state = compute_relative_geometry_frame(
                feature_df,
                cfg=relative_cfg,
                state=relative_state,
            )
            if cfg.twist_checkpoint_path is not None:
                twist_scored = infer_checkpoint_twist_outputs(feature_df, checkpoint_path=cfg.twist_checkpoint_path)
            else:
                twist_scored = compute_heuristic_twist_outputs(feature_df)
            twist_df = decode_twist_sequence(
                twist_scored,
                enter_threshold=cfg.twist_enter_threshold,
                exit_threshold=cfg.twist_exit_threshold,
                min_fault_dwell=cfg.twist_fault_min_dwell,
                min_stable_dwell=cfg.twist_stable_min_dwell,
            )
            raw_score = twist_df["twist_wake_prob"].to_numpy(dtype=float)
            dominant_features = twist_df["decoded_state"].astype(object).tolist()
            component_scores: dict[str, np.ndarray] = {
                "seq_pos_frac": twist_df["twist_state_prob_stable"].to_numpy(dtype=float),
                "seq_neg_frac": twist_df["twist_state_prob_negative_sequence"].to_numpy(dtype=float),
                "seq_zero_frac": twist_df["twist_state_prob_zero_sequence"].to_numpy(dtype=float),
                "seq_unbalance_barrier": twist_df["twist_state_prob_mixed"].to_numpy(dtype=float),
            }
        else:
            feat = compute_sequence_chunk_features(v, basis, MicroPMUSequenceConfig(data_path=cfg.data_path))
            feature_df = pd.DataFrame(feat)
            raw_score, dominant_features, component_scores = compose_scores(feature_df, baseline_stats_df)
            twist_df = None
        ema_score, prev_ema = ema_chunk(raw_score, alpha, prev_ema)

        times = chunk["Time"].to_numpy(dtype=np.float64)
        event_codes = chunk["Events"].to_numpy(dtype=np.float64)
        event_flag = event_codes != 0
        row_index = np.arange(global_row, global_row + len(chunk), dtype=int)
        global_row += len(chunk)

        event_index_hint = np.full(len(chunk), -1, dtype=int)
        onset_mask_hint = event_flag & ~np.concatenate(([prev_event_flag_for_windows], event_flag[:-1]))
        for pos in np.flatnonzero(onset_mask_hint):
            event_index_hint[pos] = event_onset_counter
            event_onset_counter += 1

        score_df = pd.DataFrame(
            {
                "row_index": row_index,
                "time_ns": times,
                "event_code": event_codes,
                "event_flag": event_flag,
                "event_index_hint": event_index_hint,
                "phase_current_spread": aux["phase_current_spread"],
                "score_raw": raw_score,
                "score_ema": ema_score,
                "dominant_metric": dominant_features,
                "seq_pos_frac": feature_df["seq_pos_frac"].to_numpy(dtype=float),
                "seq_neg_frac": feature_df["seq_neg_frac"].to_numpy(dtype=float),
                "seq_zero_frac": feature_df["seq_zero_frac"].to_numpy(dtype=float),
                "seq_unbalance_barrier": feature_df["seq_unbalance_barrier"].to_numpy(dtype=float),
            }
        )
        if twist_df is not None:
            score_df["twist_wake_prob"] = twist_df["twist_wake_prob"].to_numpy(dtype=float)
            score_df["decoded_state"] = twist_df["decoded_state"].astype(object).to_numpy()
            score_df["episode_active"] = twist_df["episode_active"].to_numpy(dtype=bool)
            score_df["twist_state_prob_stable"] = twist_df["twist_state_prob_stable"].to_numpy(dtype=float)
            score_df["twist_state_prob_negative_sequence"] = twist_df["twist_state_prob_negative_sequence"].to_numpy(dtype=float)
            score_df["twist_state_prob_zero_sequence"] = twist_df["twist_state_prob_zero_sequence"].to_numpy(dtype=float)
            score_df["twist_state_prob_mixed"] = twist_df["twist_state_prob_mixed"].to_numpy(dtype=float)
        for metric in SCORE_METRICS:
            score_df[f"{metric}_score"] = component_scores[metric]

        score_sample_parts.append(pd.concat([score_df.loc[event_flag].copy(), score_df.loc[~event_flag].iloc[::cfg.non_event_stride].copy()], ignore_index=True))

        recent_tail, active_collectors, event_windows, prev_event_flag_for_windows = update_event_windows(
            chunk_df=score_df[[col for col in ["row_index", "time_ns", "event_flag", "phase_current_spread", "score_raw", "score_ema", "event_index_hint", "decoded_state", "twist_wake_prob"] if col in score_df.columns]].copy(),
            cfg=cfg,
            recent_tail=recent_tail,
            active_collectors=active_collectors,
            event_windows=event_windows,
            prev_event=prev_event_flag_for_windows,
        )

        for idx in range(len(score_df)):
            step = int(score_df.at[idx, "row_index"])
            time_ns = float(score_df.at[idx, "time_ns"])
            is_event = bool(score_df.at[idx, "event_flag"])
            score = float(score_df.at[idx, "score_ema"]) if np.isfinite(score_df.at[idx, "score_ema"]) else 0.0
            dominant_metric = score_df.at[idx, "dominant_metric"]

            if is_event and not event_active:
                event_active = True
                event_start_step = step
                event_start_time = time_ns
            elif (not is_event) and event_active:
                event_intervals.append(
                    {
                        "event_index": event_index_counter,
                        "event_start_step": event_start_step,
                        "event_end_step": prev_step,
                        "event_start_time_ns": event_start_time,
                        "event_end_time_ns": prev_time,
                        "duration_steps": prev_step - event_start_step + 1,
                    }
                )
                event_index_counter += 1
                event_active = False

            history.append((step, time_ns, score, dominant_metric))

            active_flag = bool(score_df.at[idx, "episode_active"]) if cfg.decoder_mode == "twist_shmm" and "episode_active" in score_df.columns else None

            if cfg.decoder_mode == "twist_shmm":
                if not in_episode:
                    if active_flag:
                        seeded = list(history)[-cfg.on_sustain_steps:] if len(history) >= cfg.on_sustain_steps else list(history)
                        counts = Counter()
                        for _, _, _, feat_name in seeded:
                            if feat_name:
                                counts[str(feat_name)] += 1
                        current_episode = {
                            "episode_id": len(episodes),
                            "start_step": seeded[0][0],
                            "start_time_ns": seeded[0][1],
                            "end_step": seeded[-1][0],
                            "end_time_ns": seeded[-1][1],
                            "peak_score": max(s[2] for s in seeded),
                            "peak_step": max(seeded, key=lambda t: t[2])[0],
                            "peak_time_ns": max(seeded, key=lambda t: t[2])[1],
                            "feature_counts": counts,
                            "n_points": len(seeded),
                        }
                        in_episode = True
                        off_run = 0
                        on_run = 0
                        prev_step = step
                        prev_time = time_ns
                        continue
                else:
                    assert current_episode is not None
                    current_episode["end_step"] = step
                    current_episode["end_time_ns"] = time_ns
                    current_episode["n_points"] += 1
                    if dominant_metric:
                        current_episode["feature_counts"][str(dominant_metric)] += 1
                    if score > float(current_episode["peak_score"]):
                        current_episode["peak_score"] = score
                        current_episode["peak_step"] = step
                        current_episode["peak_time_ns"] = time_ns
                    if not active_flag:
                        episodes.append(current_episode)
                        current_episode = None
                        in_episode = False
                        off_run = 0
                        on_run = 0
            elif not in_episode:
                if score >= cfg.episode_on_threshold:
                    on_run += 1
                else:
                    on_run = 0
                if on_run >= cfg.on_sustain_steps:
                    seeded = list(history)[-cfg.on_sustain_steps:]
                    counts = Counter()
                    for _, _, _, feat_name in seeded:
                        if feat_name:
                            counts[str(feat_name)] += 1
                    current_episode = {
                        "episode_id": len(episodes),
                        "start_step": seeded[0][0],
                        "start_time_ns": seeded[0][1],
                        "end_step": seeded[-1][0],
                        "end_time_ns": seeded[-1][1],
                        "peak_score": max(s[2] for s in seeded),
                        "peak_step": max(seeded, key=lambda t: t[2])[0],
                        "peak_time_ns": max(seeded, key=lambda t: t[2])[1],
                        "feature_counts": counts,
                        "n_points": len(seeded),
                    }
                    in_episode = True
                    off_run = 0
                    on_run = 0
                    prev_step = step
                    prev_time = time_ns
                    continue
            else:
                assert current_episode is not None
                current_episode["end_step"] = step
                current_episode["end_time_ns"] = time_ns
                current_episode["n_points"] += 1
                if dominant_metric:
                    current_episode["feature_counts"][str(dominant_metric)] += 1
                if score > float(current_episode["peak_score"]):
                    current_episode["peak_score"] = score
                    current_episode["peak_step"] = step
                    current_episode["peak_time_ns"] = time_ns

                if score <= cfg.episode_off_threshold:
                    off_run += 1
                else:
                    off_run = 0

                if off_run >= cfg.off_sustain_steps:
                    current_episode["end_step"] = step - cfg.off_sustain_steps
                    current_episode["end_time_ns"] = prev_time
                    episodes.append(current_episode)
                    current_episode = None
                    in_episode = False
                    off_run = 0
                    on_run = 0

            prev_step = step
            prev_time = time_ns

        if (chunk_id + 1) % 10 == 0:
            print(
                f"[PMU-EPISODE] chunk {chunk_id + 1} | rows {global_row} | episodes {len(episodes)} | events {len(event_intervals)}",
                flush=True,
            )

    if event_active:
        event_intervals.append(
            {
                "event_index": event_index_counter,
                "event_start_step": event_start_step,
                "event_end_step": prev_step,
                "event_start_time_ns": event_start_time,
                "event_end_time_ns": prev_time,
                "duration_steps": prev_step - event_start_step + 1,
            }
        )

    if in_episode and current_episode is not None:
        episodes.append(current_episode)

    episode_df = pd.DataFrame(episodes)
    if not episode_df.empty:
        episode_df["duration_steps"] = episode_df["end_step"] - episode_df["start_step"] + 1
        episode_df["duration_seconds"] = episode_df["duration_steps"] / 120.0
        episode_df["dominant_metric"] = episode_df["feature_counts"].apply(lambda c: max(c.items(), key=lambda kv: kv[1])[0] if c else None)
        episode_df["start_time_utc"] = pd.to_datetime(episode_df["start_time_ns"], unit="ns", utc=True)
        episode_df["end_time_utc"] = pd.to_datetime(episode_df["end_time_ns"], unit="ns", utc=True)

    labeled_event_df = pd.DataFrame(event_intervals)
    if not labeled_event_df.empty:
        labeled_event_df["duration_seconds"] = labeled_event_df["duration_steps"] / 120.0
        labeled_event_df["event_start_time_utc"] = pd.to_datetime(labeled_event_df["event_start_time_ns"], unit="ns", utc=True)
        labeled_event_df["event_end_time_utc"] = pd.to_datetime(labeled_event_df["event_end_time_ns"], unit="ns", utc=True)

    score_sample_df = pd.concat(score_sample_parts, ignore_index=True) if score_sample_parts else pd.DataFrame()
    event_window_df = pd.concat(event_windows, ignore_index=True) if event_windows else pd.DataFrame()
    return episode_df, labeled_event_df, score_sample_df, event_window_df


def merge_nearby_episodes(episode_df: pd.DataFrame, merge_gap_steps: int) -> pd.DataFrame:
    if episode_df.empty:
        return episode_df.copy()

    rows = episode_df.sort_values("start_step").to_dict(orient="records")
    merged: list[dict[str, Any]] = []
    current = rows[0].copy()
    for nxt in rows[1:]:
        gap = int(nxt["start_step"]) - int(current["end_step"]) - 1
        if gap <= merge_gap_steps:
            current["end_step"] = max(int(current["end_step"]), int(nxt["end_step"]))
            if int(nxt["end_step"]) >= int(current["end_step"]):
                current["end_time_ns"] = nxt["end_time_ns"]
            if float(nxt["peak_score"]) > float(current["peak_score"]):
                current["peak_score"] = nxt["peak_score"]
                current["peak_step"] = nxt["peak_step"]
                current["peak_time_ns"] = nxt["peak_time_ns"]
            counts = Counter(current["feature_counts"])
            counts.update(nxt["feature_counts"])
            current["feature_counts"] = counts
            current["n_points"] = int(current["n_points"]) + int(nxt["n_points"])
        else:
            merged.append(current)
            current = nxt.copy()
    merged.append(current)

    merged_df = pd.DataFrame(merged)
    merged_df["episode_id"] = np.arange(len(merged_df), dtype=int)
    merged_df["duration_steps"] = merged_df["end_step"] - merged_df["start_step"] + 1
    merged_df["duration_seconds"] = merged_df["duration_steps"] / 120.0
    merged_df["dominant_metric"] = merged_df["feature_counts"].apply(lambda c: max(c.items(), key=lambda kv: kv[1])[0] if c else None)
    merged_df["start_time_utc"] = pd.to_datetime(merged_df["start_time_ns"], unit="ns", utc=True)
    merged_df["end_time_utc"] = pd.to_datetime(merged_df["end_time_ns"], unit="ns", utc=True)
    return merged_df


def build_event_episode_metrics(
    labeled_event_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    pre_event_steps: int,
) -> pd.DataFrame:
    if labeled_event_df.empty:
        return pd.DataFrame()

    rows = []
    for _, event in labeled_event_df.iterrows():
        if episode_df.empty:
            matches = pd.DataFrame()
        else:
            matches = episode_df[
                (episode_df["start_step"] <= event["event_end_step"])
                & (episode_df["end_step"] >= event["event_start_step"] - pre_event_steps)
            ]
        pre_matches = matches[matches["start_step"] < event["event_start_step"]] if not matches.empty else matches
        any_capture = not matches.empty
        pre_capture = not pre_matches.empty
        overlap_steps = 0
        if any_capture:
            for _, ep in matches.iterrows():
                overlap_steps += max(0, min(int(ep["end_step"]), int(event["event_end_step"])) - max(int(ep["start_step"]), int(event["event_start_step"])) + 1)

        earliest_lead = float(event["event_start_step"] - pre_matches["start_step"].min()) if pre_capture else np.nan
        earliest_episode_id = int(pre_matches.sort_values("start_step").iloc[0]["episode_id"]) if pre_capture else np.nan
        earliest_duration = float(pre_matches.sort_values("start_step").iloc[0]["duration_steps"]) if pre_capture else np.nan
        earliest_feature = pre_matches.sort_values("start_step").iloc[0]["dominant_metric"] if pre_capture else np.nan

        rows.append(
            {
                "event_index": int(event["event_index"]),
                "event_start_step": int(event["event_start_step"]),
                "event_end_step": int(event["event_end_step"]),
                "event_start_time_utc": event["event_start_time_utc"],
                "event_end_time_utc": event["event_end_time_utc"],
                "captured_by_any_episode": any_capture,
                "captured_by_pre_event_episode": pre_capture,
                "earliest_pre_event_lead_steps": earliest_lead,
                "earliest_pre_event_lead_seconds": earliest_lead / 120.0 if np.isfinite(earliest_lead) else np.nan,
                "matching_episode_count": int(len(matches)),
                "overlap_steps": int(overlap_steps),
                "overlap_fraction_of_event": float(overlap_steps / max(int(event["duration_steps"]), 1)),
                "earliest_episode_id": earliest_episode_id,
                "earliest_episode_duration_steps": earliest_duration,
                "earliest_episode_dominant_metric": earliest_feature,
            }
        )
    return pd.DataFrame(rows)


def build_episode_summary(
    baseline_stats_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    event_episode_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in baseline_stats_df.iterrows():
        rows.append({"metric": f"baseline_center_{row['metric']}", "value": float(row["center"])})
        rows.append({"metric": f"baseline_scale_{row['metric']}", "value": float(row["scale"])})

    rows.append({"metric": "n_episodes", "value": int(len(episode_df))})
    rows.append({"metric": "median_episode_duration_steps", "value": float(episode_df["duration_steps"].median()) if not episode_df.empty else np.nan})
    rows.append({"metric": "median_episode_duration_seconds", "value": float(episode_df["duration_seconds"].median()) if not episode_df.empty else np.nan})
    rows.append({"metric": "mean_episode_peak_score", "value": float(episode_df["peak_score"].mean()) if not episode_df.empty else np.nan})
    if not event_episode_df.empty:
        rows.append({"metric": "event_capture_any_rate", "value": float(event_episode_df["captured_by_any_episode"].mean())})
        rows.append({"metric": "event_capture_pre_rate", "value": float(event_episode_df["captured_by_pre_event_episode"].mean())})
        rows.append({"metric": "median_earliest_pre_event_lead_steps", "value": float(event_episode_df["earliest_pre_event_lead_steps"].median())})
        rows.append({"metric": "median_earliest_pre_event_lead_seconds", "value": float(event_episode_df["earliest_pre_event_lead_seconds"].median())})
        rows.append({"metric": "mean_overlap_fraction_of_event", "value": float(event_episode_df["overlap_fraction_of_event"].mean())})
    return pd.DataFrame(rows)


def build_episode_alignment(event_window_df: pd.DataFrame, cfg: EpisodeEvalConfig) -> pd.DataFrame:
    if event_window_df.empty:
        return pd.DataFrame()
    parts = []
    for _, g in event_window_df.groupby("event_index", sort=True):
        g = g.sort_values("steps_from_onset").copy()
        baseline = g[
            (g["steps_from_onset"] >= -cfg.pre_event_steps)
            & (g["steps_from_onset"] < -cfg.pre_event_steps + cfg.baseline_steps)
        ]
        if len(baseline) < 30:
            continue
        mu = float(np.nanmedian(baseline["score_ema"]))
        sigma = float(np.nanstd(baseline["score_ema"], ddof=0))
        sigma = sigma if sigma > EPS else 1.0
        g["score_ema_z"] = (g["score_ema"] - mu) / sigma
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def plot_episode_alignment(aligned_df: pd.DataFrame, output_path: Path) -> None:
    if aligned_df.empty:
        return
    grouped = aligned_df.groupby("steps_from_onset")["score_ema_z"].median()
    time_s = grouped.index.to_numpy(dtype=float) / 120.0
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_s, grouped.to_numpy(dtype=float), color="#1b9e77", label="episode score EMA (median z)")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Seconds from labeled event onset")
    ax.set_ylabel("Median baseline-normalized score")
    ax.set_title("Sequence episode score around labeled event onset")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(
    run_dir: Path,
    cfg: EpisodeEvalConfig,
    baseline_stats_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    labeled_event_df: pd.DataFrame,
    event_episode_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    dominant_counts = episode_df["dominant_metric"].value_counts().to_dict() if not episode_df.empty else {}
    lines = [
        "# Micro PMU Sequence Episode Evaluator",
        "",
        "Episode-level evaluation over the sequence-manifold score using EMA smoothing, hysteretic detection, and post-hoc episode merging.",
        "",
        "## Setup",
        f"- Source sequence run: `{cfg.source_run_dir}`",
        f"- Data path: `{cfg.data_path}`",
        f"- Chosen orientation: `{cfg.chosen_orientation}`",
        f"- Score metrics: `{', '.join(SCORE_METRICS)}`",
        f"- Decoder mode: `{cfg.decoder_mode}`",
        f"- EMA alpha: `{cfg.ema_alpha}`",
        f"- Episode on threshold: `{cfg.episode_on_threshold}` for `{cfg.on_sustain_steps}` samples",
        f"- Episode off threshold: `{cfg.episode_off_threshold}` for `{cfg.off_sustain_steps}` samples",
        f"- Merge gap: `{cfg.merge_gap_steps}` samples",
        f"- Twist enter/exit thresholds: `{cfg.twist_enter_threshold}` / `{cfg.twist_exit_threshold}`",
        f"- Twist local reference alpha: `{cfg.twist_baseline_alpha}`",
        f"- Twist checkpoint: `{cfg.twist_checkpoint_path}`",
        "",
        "## Baseline Stats",
        baseline_stats_df.to_markdown(index=False),
        "",
        "## Summary",
        summary_df.to_markdown(index=False),
        "",
        "## Episode Dominant Metrics",
        "\n".join(f"- `{k}`: `{v}` episodes" for k, v in dominant_counts.items()) if dominant_counts else "- No episodes detected.",
        "",
        "## Event Capture",
        event_episode_df.head(20).to_markdown(index=False) if not event_episode_df.empty else "No event metrics available.",
        "",
        "## Episodes",
        episode_df.head(20).to_markdown(index=False) if not episode_df.empty else "No episodes detected.",
        "",
        "## Decoder Note",
        "- `baseline` mode is the original smoothed hysteretic baseline.",
        "- `twist_shmm` mode uses the fixed-symmetry Twist front end, a causal local-reference geometry chart, and a sticky semi-HMM-style state decoder over `stable / negative_sequence / zero_sequence / mixed`.",
        "",
    ]
    (run_dir / "micro_pmu_sequence_episode_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate micro-PMU sequence features at the episode level.")
    parser.add_argument("--sequence-run-dir", default=None)
    parser.add_argument("--run-root", default=str(Path("artifact") / "runs"))
    parser.add_argument("--ema-alpha", type=float, default=0.05)
    parser.add_argument("--episode-on-threshold", type=float, default=2.0)
    parser.add_argument("--episode-off-threshold", type=float, default=0.75)
    parser.add_argument("--on-sustain-steps", type=int, default=12)
    parser.add_argument("--off-sustain-steps", type=int, default=24)
    parser.add_argument("--merge-gap-steps", type=int, default=120)
    parser.add_argument("--decoder-mode", default="twist_shmm", choices=["baseline", "twist_shmm"])
    parser.add_argument("--twist-enter-threshold", type=float, default=0.60)
    parser.add_argument("--twist-exit-threshold", type=float, default=0.35)
    parser.add_argument("--twist-fault-min-dwell", type=int, default=12)
    parser.add_argument("--twist-stable-min-dwell", type=int, default=24)
    parser.add_argument("--twist-checkpoint", default=None)
    parser.add_argument("--twist-baseline-alpha", type=float, default=0.01)
    return parser.parse_args()


def latest_sequence_run(run_root: Path) -> Path:
    dirs = sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("micro_pmu_sequence_")], key=lambda p: p.stat().st_mtime)
    if not dirs:
        raise FileNotFoundError("No micro_pmu_sequence_* run found.")
    return dirs[-1]


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    source_run_dir = Path(args.sequence_run_dir).resolve() if args.sequence_run_dir else latest_sequence_run(run_root)
    cfg, sampled_df = load_source_run_context(source_run_dir, run_root)
    cfg = EpisodeEvalConfig(
        source_run_dir=cfg.source_run_dir,
        run_root=cfg.run_root,
        data_path=cfg.data_path,
        chosen_orientation=cfg.chosen_orientation,
        chunk_size=cfg.chunk_size,
        pre_event_steps=cfg.pre_event_steps,
        post_event_steps=cfg.post_event_steps,
        baseline_steps=cfg.baseline_steps,
        non_event_stride=cfg.non_event_stride,
        ema_alpha=float(args.ema_alpha),
        episode_on_threshold=float(args.episode_on_threshold),
        episode_off_threshold=float(args.episode_off_threshold),
        on_sustain_steps=int(args.on_sustain_steps),
        off_sustain_steps=int(args.off_sustain_steps),
        merge_gap_steps=int(args.merge_gap_steps),
        decoder_mode=str(args.decoder_mode),
        twist_enter_threshold=float(args.twist_enter_threshold),
        twist_exit_threshold=float(args.twist_exit_threshold),
        twist_fault_min_dwell=int(args.twist_fault_min_dwell),
        twist_stable_min_dwell=int(args.twist_stable_min_dwell),
        twist_checkpoint_path=Path(args.twist_checkpoint).resolve() if args.twist_checkpoint else None,
        twist_baseline_alpha=float(args.twist_baseline_alpha),
    )
    run_dir = ensure_run_dir(cfg.run_root)

    baseline_stats_df = compute_baseline_stats(sampled_df)
    raw_episode_df, labeled_event_df, score_sample_df, event_window_df = detect_episodes(cfg, baseline_stats_df)
    episode_df = merge_nearby_episodes(raw_episode_df, cfg.merge_gap_steps)
    event_episode_df = build_event_episode_metrics(labeled_event_df, episode_df, cfg.pre_event_steps)
    summary_df = build_episode_summary(baseline_stats_df, episode_df, event_episode_df)
    aligned_df = build_episode_alignment(event_window_df, cfg)

    baseline_stats_df.to_csv(run_dir / "micro_pmu_sequence_baseline_stats.csv", index=False)
    score_sample_df.to_csv(run_dir / "micro_pmu_sequence_episode_score_samples.csv", index=False)
    raw_episode_df.to_csv(run_dir / "micro_pmu_sequence_raw_episode_table.csv", index=False)
    episode_df.to_csv(run_dir / "micro_pmu_sequence_episode_table.csv", index=False)
    labeled_event_df.to_csv(run_dir / "micro_pmu_sequence_labeled_event_intervals.csv", index=False)
    event_episode_df.to_csv(run_dir / "micro_pmu_sequence_event_episode_metrics.csv", index=False)
    summary_df.to_csv(run_dir / "micro_pmu_sequence_episode_summary.csv", index=False)
    aligned_df.to_csv(run_dir / "micro_pmu_sequence_episode_alignment.csv", index=False)
    plot_episode_alignment(aligned_df, run_dir / "micro_pmu_sequence_episode_alignment_plot.png")

    metadata = {
        "config": json_safe(cfg.__dict__),
        "output_dir": str(run_dir),
        "n_raw_episodes": int(len(raw_episode_df)),
        "n_merged_episodes": int(len(episode_df)),
        "n_labeled_events": int(len(labeled_event_df)),
    }
    with (run_dir / "micro_pmu_sequence_episode_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(json_safe(metadata), fp, indent=2)

    write_report(run_dir, cfg, baseline_stats_df, episode_df, labeled_event_df, event_episode_df, summary_df)

    print(f"[PMU-EPISODE] Output directory: {run_dir}", flush=True)
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
