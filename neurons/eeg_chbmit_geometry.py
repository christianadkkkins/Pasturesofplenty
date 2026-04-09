from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_lie import (
    CANONICAL_PROJECTIVE_REFERENCE_COLUMNS,
    EPS,
    canonical_projective_transition_score,
    compute_projective_state_features,
    ema_prior_states,
)
from eeg_sleep_edf_geometry import (
    find_project_root,
    json_safe,
    robust_center_scale,
    trailing_roll,
)


DEFAULT_CHBMIT_CHANNEL_CANDIDATES = (
    "FZ-CZ",
    "CZ-PZ",
    "FP1-F7",
    "FP2-F8",
    "F3-C3",
    "F4-C4",
    "C3-P3",
    "C4-P4",
)


@dataclass(frozen=True)
class CHBMITConfig:
    run_root: Path
    edf: Path | None = None
    subject_dir: Path | None = None
    summary_txt: Path | None = None
    channel: str | None = None
    channel_mode: str = "sweep"
    target_sfreq: float = 256.0
    epoch_seconds: float = 1.0
    embedding_dim: int = 16
    lag_samples: int = 4
    beta_short: float = 0.10
    beta_long: float = 0.01
    rolling_seconds: float = 5.0
    pre_ictal_seconds: float = 300.0
    post_ictal_seconds: float = 30.0
    min_state_energy: float = 1e-8
    alert_threshold: float = 1.5
    velocity_threshold: float = 0.25
    active_score_threshold: float = 0.5
    active_level_threshold: float = 0.25
    alert_open_sustain_epochs: int = 2
    alert_min_duration_epochs: int = 12
    alert_max_duration_epochs: int = 90
    alert_cooldown_epochs: int = 45
    alert_merge_gap_epochs: int = 2
    write_epoch_csv: bool = False
    verbose: bool = True


def log(cfg: CHBMITConfig, message: str) -> None:
    if cfg.verbose:
        print(f"[CHBMIT] {message}", flush=True)


def prepare_chbmit_run_directory(base_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("neurons_chbmit_%Y%m%dT%H%M%SZ")
    run_dir = base_root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def infer_summary_path_from_edf(edf_path: Path) -> Path:
    match = re.match(r"^(chb\d+)_\d+\.edf$", edf_path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not infer CHB-MIT summary file from EDF name: {edf_path.name}")
    return edf_path.parent / f"{match.group(1).lower()}-summary.txt"


def parse_chbmit_summary(summary_path: Path) -> dict[str, dict[str, Any]]:
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    rows: dict[str, dict[str, Any]] = {}
    current_file: str | None = None
    pending_start: float | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"File Name:\s*(\S+)", line, flags=re.IGNORECASE)
        if match:
            current_file = match.group(1)
            rows.setdefault(current_file, {"n_seizures": 0, "seizures": []})
            pending_start = None
            continue

        if current_file is None:
            continue

        match = re.match(r"Number of Seizures in File:\s*(\d+)", line, flags=re.IGNORECASE)
        if match:
            rows[current_file]["n_seizures"] = int(match.group(1))
            continue

        match = re.match(r"Seizure Start Time:\s*([0-9.]+)\s*seconds", line, flags=re.IGNORECASE)
        if match:
            pending_start = float(match.group(1))
            continue

        match = re.match(r"Seizure End Time:\s*([0-9.]+)\s*seconds", line, flags=re.IGNORECASE)
        if match and pending_start is not None:
            rows[current_file]["seizures"].append((pending_start, float(match.group(1))))
            pending_start = None

    return rows


def select_chbmit_channel(ch_names: list[str], requested: str | None) -> str:
    cleaned = [name for name in ch_names if str(name).strip() not in {"", "-"}]
    if requested:
        if requested not in ch_names:
            raise ValueError(f"Requested channel '{requested}' not found. Available channels: {ch_names}")
        return requested
    for candidate in DEFAULT_CHBMIT_CHANNEL_CANDIDATES:
        if candidate in cleaned:
            return candidate
    for name in cleaned:
        upper = str(name).upper()
        if "ECG" not in upper and "VNS" not in upper:
            return name
    if not cleaned:
        raise ValueError("No usable channels found in EDF.")
    return cleaned[0]


def list_available_chbmit_channels(edf_path: Path) -> list[str]:
    try:
        import mne
    except Exception as exc:
        raise RuntimeError("mne is required for CHB-MIT loading. Install neurons/requirements_minimal.txt first.") from exc

    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    ch_names = [str(name) for name in raw.ch_names]
    cleaned = []
    for name in ch_names:
        upper = name.upper()
        if name.strip() in {"", "-"}:
            continue
        if "ECG" in upper or "VNS" in upper:
            continue
        cleaned.append(name)
    return cleaned


def candidate_chbmit_channels(edf_path: Path, cfg: CHBMITConfig) -> list[str]:
    available = list_available_chbmit_channels(edf_path)
    if cfg.channel is not None:
        if cfg.channel not in available:
            raise ValueError(f"Requested channel '{cfg.channel}' not found in {edf_path.name}. Available channels: {available}")
        return [cfg.channel]

    preferred = [name for name in DEFAULT_CHBMIT_CHANNEL_CANDIDATES if name in available]
    if cfg.channel_mode == "auto":
        return [preferred[0] if preferred else select_chbmit_channel(available, None)]
    if cfg.channel_mode == "all":
        return available
    if preferred:
        return preferred
    return available[: min(8, len(available))]


def choose_best_channel(channel_summary_df: pd.DataFrame) -> str:
    ranked = channel_summary_df.sort_values(
        [
            "visible_fraction",
            "precision",
            "specificity",
            "median_lead_seconds",
            "alert_occupancy_fraction",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    return str(ranked.iloc[0]["channel"])


def load_edf_signal(edf_path: Path, channel: str | None, target_sfreq: float) -> tuple[np.ndarray, float, str]:
    try:
        import mne
    except Exception as exc:
        raise RuntimeError("mne is required for CHB-MIT loading. Install neurons/requirements_minimal.txt first.") from exc

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    selected_channel = select_chbmit_channel(list(raw.ch_names), channel)
    raw.pick([selected_channel])
    if target_sfreq > 0 and abs(float(raw.info["sfreq"]) - float(target_sfreq)) > 1e-6:
        raw.resample(target_sfreq)

    signal = raw.get_data(picks=[0])[0].astype(np.float64)
    sfreq = float(raw.info["sfreq"])
    return signal, sfreq, selected_channel


def compute_sample_features(signal: np.ndarray, sfreq: float, cfg: CHBMITConfig) -> pd.DataFrame:
    arr = np.asarray(signal, dtype=float)
    center, scale = robust_center_scale(arr)
    arr = (arr - center) / (scale + EPS)

    lag_offsets = cfg.lag_samples * np.arange(cfg.embedding_dim, dtype=int)
    max_offset = int(lag_offsets.max()) if len(lag_offsets) else 0
    if len(arr) <= max_offset + 2:
        raise RuntimeError("Signal is too short for the requested embedding_dim and lag_samples.")

    state_end_indices = np.arange(max_offset, len(arr), dtype=int)
    state_indices = state_end_indices[:, None] - lag_offsets[None, :]
    x_raw = arr[state_indices]
    x_center = x_raw - np.mean(x_raw, axis=1, keepdims=True)
    n_x = np.einsum("td,td->t", x_center, x_center)

    ms_prior = ema_prior_states(x_center, beta=cfg.beta_short)
    ml_prior = ema_prior_states(x_center, beta=cfg.beta_long)
    features = compute_projective_state_features(
        x_center,
        ms_prior=ms_prior,
        ml_prior=ml_prior,
        min_state_energy=cfg.min_state_energy,
    )

    frame = pd.DataFrame(
        {
            "sample_index": state_end_indices,
            "time_seconds": state_end_indices.astype(float) / float(sfreq),
            "epoch_index": np.floor((state_end_indices.astype(float) / float(sfreq)) / float(cfg.epoch_seconds)).astype(int),
            "state_energy": n_x,
        }
    )
    for name, values in features.items():
        frame[name] = values

    reference_cols = list(CANONICAL_PROJECTIVE_REFERENCE_COLUMNS)
    for column in reference_cols:
        values = frame[column].to_numpy(dtype=float)
        c0, s0 = robust_center_scale(values)
        rel = (values - c0) / (s0 + EPS)
        frame[f"rel_{column}"] = np.where(np.isfinite(rel), rel, 0.0)
        frame[f"vel_{column}"] = np.r_[0.0, np.diff(frame[f"rel_{column}"].to_numpy(dtype=float))]
        frame[f"acc_{column}"] = np.r_[0.0, np.diff(frame[f"vel_{column}"].to_numpy(dtype=float))]

    rel_cols = [f"rel_{column}" for column in reference_cols]
    vel_cols = [f"vel_{column}" for column in reference_cols]
    acc_cols = [f"acc_{column}" for column in reference_cols]

    frame["projective_level_norm_raw"] = np.sqrt(np.sum(np.square(frame[rel_cols].to_numpy(dtype=float)), axis=1))
    frame["projective_velocity_norm_raw"] = np.sqrt(np.sum(np.square(frame[vel_cols].to_numpy(dtype=float)), axis=1))
    frame["projective_curvature_norm_raw"] = np.sqrt(np.sum(np.square(frame[acc_cols].to_numpy(dtype=float)), axis=1))

    smooth_window = max(1, int(round(float(cfg.rolling_seconds) * float(sfreq))))
    for column in ["projective_level_norm_raw", "projective_velocity_norm_raw", "projective_curvature_norm_raw"]:
        frame[column.replace("_raw", "")] = trailing_roll(frame[column].to_numpy(dtype=float), window=smooth_window, reducer="mean")

    barrier_acc = np.nan_to_num(frame["acc_proj_lock_barrier_sl"].to_numpy(dtype=float), nan=0.0)
    novelty_vel = np.nan_to_num(frame["vel_novelty"].to_numpy(dtype=float), nan=0.0)
    frame["transition_score_raw"] = canonical_projective_transition_score(
        level=frame["projective_level_norm"].to_numpy(dtype=float),
        velocity=frame["projective_velocity_norm"].to_numpy(dtype=float),
        curvature=frame["projective_curvature_norm"].to_numpy(dtype=float),
        barrier_acc=barrier_acc,
        novelty_vel=novelty_vel,
    )
    frame["transition_score"] = trailing_roll(
        frame["transition_score_raw"].to_numpy(dtype=float),
        window=smooth_window,
        reducer="mean",
    )
    return frame


def aggregate_epoch_features(sample_df: pd.DataFrame, cfg: CHBMITConfig) -> pd.DataFrame:
    metric_cols = list(CANONICAL_PROJECTIVE_REFERENCE_COLUMNS) + [
        "projective_level_norm",
        "projective_velocity_norm",
        "projective_curvature_norm",
        "transition_score",
    ]
    agg_map: dict[str, str] = {column: "median" for column in metric_cols}
    agg_map["transition_score_peak"] = "max"

    work = sample_df.copy()
    work["transition_score_peak"] = work["transition_score"].to_numpy(dtype=float)
    out = work.groupby("epoch_index", as_index=False).agg(agg_map)
    out["start_second"] = out["epoch_index"].astype(float) * float(cfg.epoch_seconds)
    out["end_second"] = (out["epoch_index"].astype(float) + 1.0) * float(cfg.epoch_seconds)

    for metric in metric_cols:
        center, scale = robust_center_scale(out[metric].to_numpy(dtype=float))
        z = (out[metric].to_numpy(dtype=float) - center) / (scale + EPS)
        out[f"{metric}_z"] = np.where(np.isfinite(z), z, 0.0)

    center, scale = robust_center_scale(out["transition_score_peak"].to_numpy(dtype=float))
    z = (out["transition_score_peak"].to_numpy(dtype=float) - center) / (scale + EPS)
    out["transition_score_peak_z"] = np.where(np.isfinite(z), z, 0.0)
    return out


def build_episode_table(mask: np.ndarray, merge_gap: int) -> pd.DataFrame:
    arr = np.asarray(mask, dtype=bool)
    rows: list[dict[str, int]] = []
    start: int | None = None
    gap = 0
    episode_id = 0

    for idx, active in enumerate(arr):
        if active:
            if start is None:
                start = idx
            gap = 0
            continue
        if start is None:
            continue
        gap += 1
        if gap > merge_gap:
            end = idx - gap
            rows.append({"episode_id": episode_id, "start_epoch": int(start), "end_epoch": int(end), "n_epochs": int(end - start + 1)})
            episode_id += 1
            start = None
            gap = 0

    if start is not None:
        end = len(arr) - 1
        rows.append({"episode_id": episode_id, "start_epoch": int(start), "end_epoch": int(end), "n_epochs": int(end - start + 1)})

    return pd.DataFrame(rows, columns=["episode_id", "start_epoch", "end_epoch", "n_epochs"])


def apply_alert_logic(epoch_df: pd.DataFrame, cfg: CHBMITConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = epoch_df.copy()
    score_z = out["transition_score_z"].to_numpy(dtype=float)
    velocity_z = out["projective_velocity_norm_z"].to_numpy(dtype=float)
    level_z = out["projective_level_norm_z"].to_numpy(dtype=float)

    open_seed = (
        (score_z >= float(cfg.alert_threshold))
        & (velocity_z >= float(cfg.velocity_threshold))
    )
    sustain_support = (
        (score_z >= float(cfg.active_score_threshold))
        | (level_z >= float(cfg.active_level_threshold))
    )

    min_duration = max(1, int(cfg.alert_min_duration_epochs))
    max_duration = max(min_duration, int(cfg.alert_max_duration_epochs))
    open_sustain = max(1, int(cfg.alert_open_sustain_epochs))
    cooldown = max(0, int(cfg.alert_cooldown_epochs))

    final_alert = np.zeros(len(out), dtype=bool)
    active = False
    start_idx = -1
    cooldown_until = -1
    open_streak = 0

    for idx in range(len(out)):
        if active:
            duration = idx - start_idx
            if duration < min_duration:
                final_alert[idx] = True
                continue
            if duration >= max_duration:
                active = False
                cooldown_until = idx + cooldown
                open_streak = 0
                continue
            if sustain_support[idx]:
                final_alert[idx] = True
                continue
            active = False
            cooldown_until = idx + cooldown
            open_streak = 0

        if idx < cooldown_until:
            continue

        if open_seed[idx]:
            open_streak += 1
        else:
            open_streak = 0

        if open_streak >= open_sustain:
            start_idx = idx - open_streak + 1
            final_alert[start_idx : idx + 1] = True
            active = True

    raw_alert = open_seed
    episodes_df = build_episode_table(final_alert, merge_gap=max(0, int(cfg.alert_merge_gap_epochs)))
    out["alert"] = final_alert
    out["alert_raw"] = raw_alert
    return out, episodes_df


def seizures_to_epoch_rows(
    seizures: list[tuple[float, float]],
    cfg: CHBMITConfig,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for idx, (start_sec, end_sec) in enumerate(seizures):
        onset_epoch = int(np.floor(float(start_sec) / float(cfg.epoch_seconds)))
        end_epoch = int(np.floor(float(end_sec) / float(cfg.epoch_seconds)))
        pre_start_epoch = max(0, onset_epoch - int(np.floor(float(cfg.pre_ictal_seconds) / float(cfg.epoch_seconds))))
        post_end_epoch = end_epoch + int(np.ceil(float(cfg.post_ictal_seconds) / float(cfg.epoch_seconds)))
        rows.append(
            {
                "seizure_id": idx,
                "onset_second": float(start_sec),
                "end_second": float(end_sec),
                "onset_epoch": onset_epoch,
                "end_epoch": end_epoch,
                "pre_start_epoch": pre_start_epoch,
                "post_end_epoch": post_end_epoch,
            }
        )
    return rows


def build_event_table(
    edf_name: str,
    channel: str,
    epoch_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    seizures: list[tuple[float, float]],
    cfg: CHBMITConfig,
) -> pd.DataFrame:
    seizure_rows = seizures_to_epoch_rows(seizures, cfg)
    rows: list[dict[str, Any]] = []
    for seizure in seizure_rows:
        onset_epoch = int(seizure["onset_epoch"])
        candidate = episodes_df.loc[
            (episodes_df["start_epoch"] >= int(seizure["pre_start_epoch"]))
            & (episodes_df["start_epoch"] <= onset_epoch)
        ].copy()
        if candidate.empty:
            visible = 0
            lead_epochs = np.nan
            alert_episode_id = np.nan
            alert_start_epoch = np.nan
            alert_end_epoch = np.nan
        else:
            selected = candidate.sort_values("start_epoch").iloc[-1]
            visible = 1
            lead_epochs = float(onset_epoch - int(selected["start_epoch"]))
            alert_episode_id = int(selected["episode_id"])
            alert_start_epoch = int(selected["start_epoch"])
            alert_end_epoch = int(selected["end_epoch"])

        pre_window = epoch_df.loc[
            (epoch_df["epoch_index"] >= int(seizure["pre_start_epoch"]))
            & (epoch_df["epoch_index"] < onset_epoch)
        ].copy()

        rows.append(
            {
                "edf_file": edf_name,
                "channel": channel,
                "seizure_id": int(seizure["seizure_id"]),
                "onset_second": float(seizure["onset_second"]),
                "end_second": float(seizure["end_second"]),
                "onset_epoch": onset_epoch,
                "end_epoch": int(seizure["end_epoch"]),
                "visible": visible,
                "lead_epochs": lead_epochs,
                "lead_seconds": lead_epochs * float(cfg.epoch_seconds) if np.isfinite(lead_epochs) else np.nan,
                "alert_episode_id": alert_episode_id,
                "alert_start_epoch": alert_start_epoch,
                "alert_end_epoch": alert_end_epoch,
                "pre_ictal_transition_score_peak_z": float(pre_window["transition_score_peak_z"].max()) if not pre_window.empty else np.nan,
                "pre_ictal_transition_score_median": float(pre_window["transition_score"].median()) if not pre_window.empty else np.nan,
                "pre_ictal_memory_align_median": float(pre_window["memory_align"].median()) if not pre_window.empty else np.nan,
                "pre_ictal_novelty_median": float(pre_window["novelty"].median()) if not pre_window.empty else np.nan,
            }
        )

    columns = [
        "edf_file",
        "channel",
        "seizure_id",
        "onset_second",
        "end_second",
        "onset_epoch",
        "end_epoch",
        "visible",
        "lead_epochs",
        "lead_seconds",
        "alert_episode_id",
        "alert_start_epoch",
        "alert_end_epoch",
        "pre_ictal_transition_score_peak_z",
        "pre_ictal_transition_score_median",
        "pre_ictal_memory_align_median",
        "pre_ictal_novelty_median",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_file_confusion_counts(
    epoch_df: pd.DataFrame,
    seizures: list[tuple[float, float]],
    cfg: CHBMITConfig,
) -> dict[str, int | float]:
    positive = np.zeros(len(epoch_df), dtype=bool)
    for seizure in seizures_to_epoch_rows(seizures, cfg):
        lo = max(0, int(seizure["pre_start_epoch"]))
        hi = min(len(epoch_df) - 1, int(seizure["post_end_epoch"]))
        positive[lo : hi + 1] = True

    predicted = epoch_df["alert"].fillna(False).astype(bool).to_numpy()
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & (~positive)))
    tn = int(np.sum((~predicted) & (~positive)))
    fn = int(np.sum((~predicted) & positive))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "alert_occupancy_fraction": float(np.mean(predicted)) if len(predicted) else np.nan,
        "positive_epoch_fraction": float(np.mean(positive)) if len(positive) else np.nan,
    }


def combine_confusion_counts(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tp = int(rows["tp"].sum()) if not rows.empty else 0
    fp = int(rows["fp"].sum()) if not rows.empty else 0
    tn = int(rows["tn"].sum()) if not rows.empty else 0
    fn = int(rows["fn"].sum()) if not rows.empty else 0

    matrix_df = pd.DataFrame(
        [
            {"actual": "negative", "predicted_negative": tn, "predicted_positive": fp},
            {"actual": "positive", "predicted_negative": fn, "predicted_positive": tp},
        ]
    )
    metrics_df = pd.DataFrame(
        [
            {
                "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
                "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                "precision": float(tp / (tp + fp)) if (tp + fp) else np.nan,
                "alert_occupancy_fraction": float((tp + fp) / (tp + fp + tn + fn)) if (tp + fp + tn + fn) else np.nan,
                "positive_epoch_fraction": float((tp + fn) / (tp + fp + tn + fn)) if (tp + fp + tn + fn) else np.nan,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        ]
    )
    return matrix_df, metrics_df


def list_edf_files(cfg: CHBMITConfig) -> list[Path]:
    if cfg.edf is not None:
        return [cfg.edf.resolve()]
    if cfg.subject_dir is None:
        raise ValueError("Either edf or subject_dir must be provided.")
    return sorted(path for path in cfg.subject_dir.resolve().glob("*.edf") if path.is_file())


def infer_summary_path(cfg: CHBMITConfig) -> Path:
    if cfg.summary_txt is not None:
        return cfg.summary_txt.resolve()
    if cfg.edf is not None:
        return infer_summary_path_from_edf(cfg.edf.resolve())
    if cfg.subject_dir is not None:
        subject_name = cfg.subject_dir.resolve().name
        return cfg.subject_dir.resolve() / f"{subject_name.lower()}-summary.txt"
    raise ValueError("Could not infer summary path.")


def build_file_summary_row(
    edf_name: str,
    channel: str,
    sfreq: float,
    epoch_df: pd.DataFrame,
    seizures: list[tuple[float, float]],
    events_df: pd.DataFrame,
    confusion_counts: dict[str, int | float],
) -> dict[str, Any]:
    n_seizures = len(seizures)
    return {
        "edf_file": edf_name,
        "channel": channel,
        "sfreq_hz": sfreq,
        "n_epochs": int(len(epoch_df)),
        "n_seizures": n_seizures,
        "n_visible_pre_ictal": int(events_df["visible"].sum()) if not events_df.empty else 0,
        "visible_fraction": float(events_df["visible"].mean()) if not events_df.empty else np.nan,
        "median_lead_epochs": float(events_df["lead_epochs"].median()) if not events_df.empty else np.nan,
        "median_lead_seconds": float(events_df["lead_seconds"].median()) if not events_df.empty else np.nan,
        "specificity": float(confusion_counts["tn"] / (confusion_counts["tn"] + confusion_counts["fp"])) if (int(confusion_counts["tn"]) + int(confusion_counts["fp"])) else np.nan,
        "sensitivity": float(confusion_counts["tp"] / (confusion_counts["tp"] + confusion_counts["fn"])) if (int(confusion_counts["tp"]) + int(confusion_counts["fn"])) else np.nan,
        "precision": float(confusion_counts["tp"] / (confusion_counts["tp"] + confusion_counts["fp"])) if (int(confusion_counts["tp"]) + int(confusion_counts["fp"])) else np.nan,
        "alert_occupancy_fraction": float(confusion_counts["alert_occupancy_fraction"]),
        "positive_epoch_fraction": float(confusion_counts["positive_epoch_fraction"]),
        "tp": int(confusion_counts["tp"]),
        "fp": int(confusion_counts["fp"]),
        "tn": int(confusion_counts["tn"]),
        "fn": int(confusion_counts["fn"]),
    }


def evaluate_file_channel(
    edf_path: Path,
    channel: str,
    seizures: list[tuple[float, float]],
    cfg: CHBMITConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal, sfreq, resolved_channel = load_edf_signal(edf_path, channel=channel, target_sfreq=cfg.target_sfreq)
    sample_df = compute_sample_features(signal, sfreq=sfreq, cfg=cfg)
    epoch_df = aggregate_epoch_features(sample_df, cfg=cfg)
    epoch_df, episodes_df = apply_alert_logic(epoch_df, cfg)
    events_df = build_event_table(edf_path.name, resolved_channel, epoch_df, episodes_df, seizures, cfg)
    confusion_counts = build_file_confusion_counts(epoch_df, seizures, cfg)
    summary_row = build_file_summary_row(edf_path.name, resolved_channel, sfreq, epoch_df, seizures, events_df, confusion_counts)
    return summary_row, events_df, epoch_df, episodes_df


def build_channel_summary(file_channel_df: pd.DataFrame) -> pd.DataFrame:
    if file_channel_df.empty:
        return pd.DataFrame(
            columns=[
                "channel",
                "n_files",
                "n_seizures",
                "n_visible_pre_ictal",
                "visible_fraction",
                "median_lead_seconds",
                "specificity",
                "sensitivity",
                "precision",
                "alert_occupancy_fraction",
                "positive_epoch_fraction",
                "tp",
                "fp",
                "tn",
                "fn",
            ]
        )

    rows: list[dict[str, Any]] = []
    for channel, group in file_channel_df.groupby("channel"):
        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        tn = int(group["tn"].sum())
        fn = int(group["fn"].sum())
        n_seizures = int(group["n_seizures"].sum())
        n_visible = int(group["n_visible_pre_ictal"].sum())
        total_epochs = tp + fp + tn + fn
        rows.append(
            {
                "channel": str(channel),
                "n_files": int(group["edf_file"].nunique()),
                "n_seizures": n_seizures,
                "n_visible_pre_ictal": n_visible,
                "visible_fraction": float(n_visible / n_seizures) if n_seizures else np.nan,
                "median_lead_seconds": float(group["median_lead_seconds"].median()) if "median_lead_seconds" in group else np.nan,
                "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
                "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                "precision": float(tp / (tp + fp)) if (tp + fp) else np.nan,
                "alert_occupancy_fraction": float((tp + fp) / total_epochs) if total_epochs else np.nan,
                "positive_epoch_fraction": float((tp + fn) / total_epochs) if total_epochs else np.nan,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    return pd.DataFrame(rows).sort_values(["visible_fraction", "precision", "specificity", "median_lead_seconds"], ascending=[False, False, False, False]).reset_index(drop=True)


def write_report(
    run_dir: Path,
    cfg: CHBMITConfig,
    summary_path: Path,
    channel_summary_df: pd.DataFrame,
    selected_channel: str,
    file_summary_df: pd.DataFrame,
    overall_metrics_df: pd.DataFrame,
    event_df: pd.DataFrame,
) -> None:
    lines = [
        "# CHB-MIT Geometry Starter",
        "",
        "Takens-style delay-embedding geometry on CHB-MIT scalp EEG, evaluated against seizure onsets from the PhysioNet summary annotations.",
        "",
        "## Configuration",
        f"- Summary file: `{summary_path}`",
        f"- Channel request: `{cfg.channel if cfg.channel is not None else cfg.channel_mode}`",
        f"- Selected channel: `{selected_channel}`",
        f"- Epoch length: `{cfg.epoch_seconds:g}` seconds",
        f"- Embedding: `dim={cfg.embedding_dim}`, `lag_samples={cfg.lag_samples}`",
        f"- Memory betas: short=`{cfg.beta_short:g}`, long=`{cfg.beta_long:g}`",
        f"- Pre-ictal window: `{cfg.pre_ictal_seconds:g}` seconds",
        f"- Post-ictal window: `{cfg.post_ictal_seconds:g}` seconds",
        f"- Excitable alert open: `transition_score_z >= {cfg.alert_threshold:g}` and `projective_velocity_norm_z >= {cfg.velocity_threshold:g}` for `{cfg.alert_open_sustain_epochs}` epochs",
        f"- Excitable alert sustain: `transition_score_z >= {cfg.active_score_threshold:g}` or `projective_level_norm_z >= {cfg.active_level_threshold:g}`",
        f"- Alert duration/cooldown: min `{cfg.alert_min_duration_epochs}` epochs, max `{cfg.alert_max_duration_epochs}` epochs, cooldown `{cfg.alert_cooldown_epochs}` epochs",
        "",
        "## Overall Metrics",
        overall_metrics_df.to_markdown(index=False),
        "",
    ]

    if not channel_summary_df.empty:
        lines.extend(
            [
                "## Channel Sweep",
                channel_summary_df.to_markdown(index=False),
                "",
            ]
        )

    if not file_summary_df.empty:
        lines.extend(
            [
                "## File Summary",
                file_summary_df.to_markdown(index=False),
                "",
            ]
        )

    if not event_df.empty:
        cols = [
            "edf_file",
            "seizure_id",
            "onset_second",
            "end_second",
            "visible",
            "lead_seconds",
            "pre_ictal_transition_score_peak_z",
        ]
        lines.extend(
            [
                "## Seizure Event Table",
                event_df[cols].head(60).to_markdown(index=False),
                "",
            ]
        )

    (run_dir / "chbmit_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CHB-MIT seizure-onset starter experiment for the geometric EEG pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--edf", default=None, help="Single CHB-MIT EDF file.")
    group.add_argument("--subject-dir", default=None, help="Subject directory containing CHB-MIT EDF files.")
    parser.add_argument("--summary-txt", default=None, help="Optional explicit path to the subject summary text.")
    parser.add_argument("--channel", default=None)
    parser.add_argument("--channel-mode", choices=["auto", "sweep", "all"], default="sweep")
    parser.add_argument("--target-sfreq", type=float, default=256.0)
    parser.add_argument("--epoch-seconds", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--lag-samples", type=int, default=4)
    parser.add_argument("--beta-short", type=float, default=0.10)
    parser.add_argument("--beta-long", type=float, default=0.01)
    parser.add_argument("--rolling-seconds", type=float, default=5.0)
    parser.add_argument("--pre-ictal-seconds", type=float, default=300.0)
    parser.add_argument("--post-ictal-seconds", type=float, default=30.0)
    parser.add_argument("--min-state-energy", type=float, default=1e-8)
    parser.add_argument("--alert-threshold", type=float, default=1.5)
    parser.add_argument("--velocity-threshold", type=float, default=0.25)
    parser.add_argument("--active-score-threshold", type=float, default=0.5)
    parser.add_argument("--active-level-threshold", type=float, default=0.25)
    parser.add_argument("--alert-open-sustain-epochs", type=int, default=2)
    parser.add_argument("--alert-min-duration-epochs", type=int, default=12)
    parser.add_argument("--alert-max-duration-epochs", type=int, default=90)
    parser.add_argument("--alert-cooldown-epochs", type=int, default=45)
    parser.add_argument("--alert-merge-gap-epochs", type=int, default=2)
    parser.add_argument("--write-epoch-csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root(Path(__file__).resolve().parent)
    run_dir = prepare_chbmit_run_directory(project_root / "artifact" / "runs")
    cfg = CHBMITConfig(
        run_root=project_root / "artifact" / "runs",
        edf=Path(args.edf).resolve() if args.edf else None,
        subject_dir=Path(args.subject_dir).resolve() if args.subject_dir else None,
        summary_txt=Path(args.summary_txt).resolve() if args.summary_txt else None,
        channel=args.channel,
        channel_mode=args.channel_mode,
        target_sfreq=args.target_sfreq,
        epoch_seconds=args.epoch_seconds,
        embedding_dim=args.embedding_dim,
        lag_samples=args.lag_samples,
        beta_short=args.beta_short,
        beta_long=args.beta_long,
        rolling_seconds=args.rolling_seconds,
        pre_ictal_seconds=args.pre_ictal_seconds,
        post_ictal_seconds=args.post_ictal_seconds,
        min_state_energy=args.min_state_energy,
        alert_threshold=args.alert_threshold,
        velocity_threshold=args.velocity_threshold,
        active_score_threshold=args.active_score_threshold,
        active_level_threshold=args.active_level_threshold,
        alert_open_sustain_epochs=args.alert_open_sustain_epochs,
        alert_min_duration_epochs=args.alert_min_duration_epochs,
        alert_max_duration_epochs=args.alert_max_duration_epochs,
        alert_cooldown_epochs=args.alert_cooldown_epochs,
        alert_merge_gap_epochs=args.alert_merge_gap_epochs,
        write_epoch_csv=bool(args.write_epoch_csv),
    )

    summary_path = infer_summary_path(cfg)
    seizure_map = parse_chbmit_summary(summary_path)
    edf_files = list_edf_files(cfg)
    log(cfg, f"Loaded summary from {summary_path.name}; processing {len(edf_files)} EDF files")

    epoch_dir = run_dir / "epoch_features"
    episode_dir = run_dir / "alert_episodes"
    if cfg.write_epoch_csv:
        epoch_dir.mkdir(parents=True, exist_ok=True)
        episode_dir.mkdir(parents=True, exist_ok=True)

    candidate_map = {edf_path.name: candidate_chbmit_channels(edf_path, cfg) for edf_path in edf_files}
    all_channels = sorted({channel for channels in candidate_map.values() for channel in channels})
    log(cfg, f"Evaluating channels: {', '.join(all_channels)}")

    file_channel_rows: list[dict[str, Any]] = []
    event_frames: list[pd.DataFrame] = []
    epoch_cache: dict[tuple[str, str], pd.DataFrame] = {}
    episode_cache: dict[tuple[str, str], pd.DataFrame] = {}

    for edf_path in edf_files:
        seizures = list(seizure_map.get(edf_path.name, {}).get("seizures", []))
        for channel in candidate_map[edf_path.name]:
            log(cfg, f"Processing {edf_path.name} [{channel}]")
            summary_row, events_df, epoch_df, episodes_df = evaluate_file_channel(edf_path, channel, seizures, cfg)
            file_channel_rows.append(summary_row)
            event_frames.append(events_df)
            epoch_cache[(edf_path.name, str(summary_row["channel"]))] = epoch_df
            episode_cache[(edf_path.name, str(summary_row["channel"]))] = episodes_df

    file_channel_df = pd.DataFrame(file_channel_rows)
    channel_summary_df = build_channel_summary(file_channel_df)
    selected_channel = choose_best_channel(channel_summary_df) if not channel_summary_df.empty else (cfg.channel or "")
    file_summary_df = file_channel_df.loc[file_channel_df["channel"] == selected_channel].copy().reset_index(drop=True)
    event_df_all = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    event_df = event_df_all.loc[event_df_all["channel"] == selected_channel].copy().reset_index(drop=True) if not event_df_all.empty else pd.DataFrame()
    matrix_df, metrics_df = combine_confusion_counts(file_summary_df)

    if cfg.write_epoch_csv:
        for _, row in file_summary_df.iterrows():
            key = (str(row["edf_file"]), str(row["channel"]))
            safe_channel = re.sub(r"[^A-Za-z0-9]+", "_", str(row["channel"])).strip("_")
            epoch_cache[key].to_csv(epoch_dir / f"{Path(str(row['edf_file'])).stem}_{safe_channel}_epoch_features.csv", index=False)
            episode_cache[key].to_csv(episode_dir / f"{Path(str(row['edf_file'])).stem}_{safe_channel}_alert_episodes.csv", index=False)

    channel_summary_df.to_csv(run_dir / "chbmit_channel_summary.csv", index=False)
    file_summary_df.to_csv(run_dir / "chbmit_file_summary.csv", index=False)
    event_df.to_csv(run_dir / "chbmit_event_table.csv", index=False)
    event_df_all.to_csv(run_dir / "chbmit_channel_event_table.csv", index=False)
    matrix_df.to_csv(run_dir / "chbmit_confusion_matrix.csv", index=False)
    metrics_df.to_csv(run_dir / "chbmit_confusion_metrics.csv", index=False)

    write_report(
        run_dir=run_dir,
        cfg=cfg,
        summary_path=summary_path,
        channel_summary_df=channel_summary_df,
        selected_channel=selected_channel,
        file_summary_df=file_summary_df,
        overall_metrics_df=metrics_df,
        event_df=event_df,
    )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": json_safe(asdict(cfg)),
        "summary_path": str(summary_path),
        "n_edf_files": int(len(edf_files)),
        "selected_channel": selected_channel,
        "n_events": int(len(event_df)),
    }
    (run_dir / "chbmit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Run directory: {run_dir}")
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
