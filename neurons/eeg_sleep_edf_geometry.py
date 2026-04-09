from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_lie import (
    CANONICAL_LIE_FEATURE_COLUMNS,
    CANONICAL_PROJECTIVE_REFERENCE_COLUMNS,
    EPS,
    canonical_lie_transition_score,
    canonical_projective_transition_score,
    compute_lie_state_features,
    compute_projective_state_features,
    ema_prior_states,
)

VALID_STAGE_LABELS = {"W", "N1", "N2", "N3", "REM"}
DEFAULT_CHANNEL_CANDIDATES = (
    "EEG Fpz-Cz",
    "EEG Pz-Oz",
    "EEG Cz-Oz",
    "EEG C4-A1",
    "EEG C3-A2",
)
SLEEP_STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Sleep stage ?": "UNKNOWN",
    "Movement time": "MOVEMENT",
}


@dataclass(frozen=True)
class SleepEDFConfig:
    psg_edf: Path
    hypnogram_edf: Path
    run_root: Path
    channel: str | None = None
    target_sfreq: float = 100.0
    epoch_seconds: float = 30.0
    embedding_dim: int = 16
    lag_samples: int = 4
    beta_short: float = 0.10
    beta_long: float = 0.01
    rolling_seconds: float = 30.0
    pre_transition_epochs: int = 20
    post_transition_epochs: int = 1
    baseline_epochs: int = 20
    min_stage_run_epochs: int = 2
    min_state_energy: float = 1e-8
    alert_threshold: float = 1.5
    velocity_threshold: float = 0.25
    lie_alert_threshold: float = 1.0
    lie_orbit_gate: float = 0.0
    alert_merge_gap_epochs: int = 1
    sample_context_epochs: int = 2
    write_sample_features: bool = False
    write_full_sample_features: bool = False
    verbose: bool = True


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


def parse_float_grid(spec: str | None) -> list[float]:
    if spec is None:
        return []
    values: list[float] = []
    for token in str(spec).split(","):
        part = token.strip()
        if part:
            values.append(float(part))
    return values


def find_project_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "artifact").exists() and (candidate / "data").exists():
            return candidate
    return here


def prepare_run_directory(base_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("neurons_sleep_edf_%Y%m%dT%H%M%SZ")
    run_dir = base_root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def log(cfg: SleepEDFConfig, message: str) -> None:
    if cfg.verbose:
        print(f"[NEURONS] {message}", flush=True)


def resolve_existing_input_path(path: Path, label: str, project_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    data_root = project_root / "data"
    if data_root.exists():
        matches = [p for p in data_root.rglob(candidate.name) if p.is_file()]
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            options = "\n".join(f"- {match}" for match in matches[:10])
            raise FileNotFoundError(
                f"{label} was not found at '{candidate}', but multiple files named '{candidate.name}' "
                f"were found under '{data_root}'. Please pass the exact path.\n{options}"
            )

    expected_dir = data_root / "sleep_edf"
    raise FileNotFoundError(
        f"{label} does not exist: '{candidate}'. No matching file named '{candidate.name}' was found under "
        f"'{data_root}'. Download Sleep-EDF Expanded and place the files under '{expected_dir}', for example:\n"
        f"- {expected_dir / 'sc4002e0' / 'sc4002e0-PSG.edf'}\n"
        f"- {expected_dir / 'sc4002e0' / 'sc4002e0-Hypnogram.edf'}\n"
        f"See neurons/DATA_ACCESS.md for the expected layout."
    )


def robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
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
    arr = np.asarray(values, dtype=float)
    if window <= 1:
        return arr.copy()
    out = np.full(len(arr), np.nan, dtype=float)
    if reducer == "mean":
        finite = np.isfinite(arr)
        clean = np.where(finite, arr, 0.0)
        counts = np.cumsum(finite.astype(np.int64), dtype=np.int64)
        totals = np.cumsum(clean, dtype=float)
        if window < len(arr):
            counts[window:] = counts[window:] - counts[:-window]
            totals[window:] = totals[window:] - totals[:-window]
        valid = counts > 0
        out[valid] = totals[valid] / counts[valid]
        return out
    for idx in range(len(arr)):
        lo = max(0, idx - window + 1)
        seg = arr[lo : idx + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) == 0:
            continue
        if reducer == "mean":
            out[idx] = float(np.mean(seg))
        elif reducer == "max":
            out[idx] = float(np.max(seg))
        elif reducer == "median":
            out[idx] = float(np.median(seg))
        else:
            raise ValueError(f"Unknown reducer: {reducer}")
    return out


def select_channel(ch_names: list[str], requested: str | None) -> str:
    if requested:
        if requested not in ch_names:
            raise ValueError(f"Requested channel '{requested}' not found. Available channels: {ch_names}")
        return requested
    for candidate in DEFAULT_CHANNEL_CANDIDATES:
        if candidate in ch_names:
            return candidate
    if not ch_names:
        raise ValueError("No channels available in EDF.")
    return ch_names[0]


def load_sleep_edf_pair(cfg: SleepEDFConfig) -> tuple[np.ndarray, float, str, pd.DataFrame]:
    try:
        import mne
    except Exception as exc:
        raise RuntimeError("mne is required for Sleep-EDF loading. Install neurons/requirements_minimal.txt first.") from exc

    project_root = find_project_root(Path(__file__).resolve().parent)
    psg_path = resolve_existing_input_path(cfg.psg_edf, "PSG EDF", project_root)
    hypnogram_path = resolve_existing_input_path(cfg.hypnogram_edf, "Hypnogram EDF", project_root)

    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose="ERROR")
    annotations = mne.read_annotations(str(hypnogram_path))
    raw.set_annotations(annotations)

    channel = select_channel(list(raw.ch_names), cfg.channel)
    raw.pick([channel])
    if cfg.target_sfreq > 0 and abs(float(raw.info["sfreq"]) - float(cfg.target_sfreq)) > 1e-6:
        raw.resample(cfg.target_sfreq)

    signal = raw.get_data(picks=[0])[0].astype(np.float64)
    sfreq = float(raw.info["sfreq"])
    epochs_df = build_epoch_table_from_annotations(
        annotations=raw.annotations,
        n_samples=len(signal),
        sfreq=sfreq,
        epoch_seconds=cfg.epoch_seconds,
    )
    return signal, sfreq, channel, epochs_df


def build_epoch_table_from_annotations(
    annotations: Any,
    n_samples: int,
    sfreq: float,
    epoch_seconds: float,
) -> pd.DataFrame:
    epoch_samples = int(round(float(epoch_seconds) * float(sfreq)))
    if epoch_samples <= 0:
        raise ValueError("epoch_seconds must produce a positive number of samples")
    n_epochs = int(n_samples // epoch_samples)
    rows = [
        {
            "epoch_index": idx,
            "start_second": float(idx * epoch_seconds),
            "end_second": float((idx + 1) * epoch_seconds),
            "stage_label": "UNKNOWN",
            "annotation_desc": "UNKNOWN",
        }
        for idx in range(n_epochs)
    ]

    for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
        label = SLEEP_STAGE_MAP.get(str(desc), "UNKNOWN")
        start_epoch = max(0, int(np.floor(float(onset) / float(epoch_seconds))))
        stop_time = float(onset) + max(float(duration), float(epoch_seconds))
        stop_epoch = min(n_epochs, int(np.ceil(stop_time / float(epoch_seconds))))
        for epoch_idx in range(start_epoch, stop_epoch):
            rows[epoch_idx]["stage_label"] = label
            rows[epoch_idx]["annotation_desc"] = str(desc)

    df = pd.DataFrame(rows)
    df["stage_valid"] = df["stage_label"].isin(VALID_STAGE_LABELS)
    return df


def compute_sample_features(signal: np.ndarray, sfreq: float, cfg: SleepEDFConfig) -> pd.DataFrame:
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
    lie_features = compute_lie_state_features(
        x_center,
        ms_prior=ms_prior,
        ml_prior=ml_prior,
        sfreq=sfreq,
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
    for name, values in lie_features.items():
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


def aggregate_epoch_features(sample_df: pd.DataFrame, epochs_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = list(CANONICAL_PROJECTIVE_REFERENCE_COLUMNS) + [
        "projective_level_norm",
        "projective_velocity_norm",
        "projective_curvature_norm",
        "transition_score",
    ] + list(CANONICAL_LIE_FEATURE_COLUMNS)
    agg_map: dict[str, str] = {column: "median" for column in metric_cols}
    agg_map["transition_score_peak"] = "max"
    agg_map["lie_orbit_norm_peak"] = "max"
    agg_map["lie_strain_norm_peak"] = "max"

    work = sample_df.copy()
    work["transition_score_peak"] = work["transition_score"].to_numpy(dtype=float)
    work["lie_orbit_norm_peak"] = work["lie_orbit_norm"].to_numpy(dtype=float)
    work["lie_strain_norm_peak"] = work["lie_strain_norm"].to_numpy(dtype=float)
    epoch_features = work.groupby("epoch_index", as_index=False).agg(agg_map)
    out = epochs_df.merge(epoch_features, on="epoch_index", how="left")

    for metric in metric_cols:
        center, scale = robust_center_scale(out[metric].to_numpy(dtype=float))
        z = (out[metric].to_numpy(dtype=float) - center) / (scale + EPS)
        out[f"{metric}_z"] = np.where(np.isfinite(z), z, 0.0)

    peak_cols = ["transition_score_peak", "lie_orbit_norm_peak", "lie_strain_norm_peak"]
    for metric in peak_cols:
        center, scale = robust_center_scale(out[metric].to_numpy(dtype=float))
        z = (out[metric].to_numpy(dtype=float) - center) / (scale + EPS)
        out[f"{metric}_z"] = np.where(np.isfinite(z), z, 0.0)

    out["lie_transition_score"] = canonical_lie_transition_score(
        orbit_z=out["lie_orbit_norm_z"].to_numpy(dtype=float),
        strain_z=out["lie_strain_norm_z"].to_numpy(dtype=float),
        commutator_z=out["lie_commutator_norm_z"].to_numpy(dtype=float),
        metric_drift_z=out["lie_metric_drift_z"].to_numpy(dtype=float),
        gram_logdet_z=out["gram_logdet_z"].to_numpy(dtype=float),
    )
    return out


def build_stage_runs(epoch_df: pd.DataFrame) -> pd.DataFrame:
    labels = epoch_df["stage_label"].astype(str).tolist()
    if not labels:
        return pd.DataFrame(columns=["run_id", "stage_label", "start_epoch", "end_epoch", "n_epochs", "stage_valid"])

    rows: list[dict[str, Any]] = []
    run_start = 0
    run_id = 0
    for idx in range(1, len(labels) + 1):
        if idx == len(labels) or labels[idx] != labels[run_start]:
            stage = labels[run_start]
            rows.append(
                {
                    "run_id": run_id,
                    "stage_label": stage,
                    "start_epoch": run_start,
                    "end_epoch": idx - 1,
                    "n_epochs": idx - run_start,
                    "stage_valid": stage in VALID_STAGE_LABELS,
                }
            )
            run_start = idx
            run_id += 1
    return pd.DataFrame(rows)


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
    return pd.DataFrame(rows)


def finalize_alerts(
    epoch_df: pd.DataFrame,
    raw_alert: np.ndarray,
    alert_col: str,
    alert_raw_col: str,
    merge_gap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = epoch_df.copy()
    episodes_df = build_episode_table(raw_alert, merge_gap=max(0, int(merge_gap)))
    final_alert = np.zeros(len(out), dtype=bool)
    for row in episodes_df.itertuples(index=False):
        final_alert[int(row.start_epoch) : int(row.end_epoch) + 1] = True
    out[alert_col] = final_alert
    out[alert_raw_col] = raw_alert
    return out, episodes_df


def apply_alert_logic(epoch_df: pd.DataFrame, cfg: SleepEDFConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_alert = (
        (epoch_df["transition_score_z"].to_numpy(dtype=float) >= float(cfg.alert_threshold))
        & (epoch_df["projective_velocity_norm_z"].to_numpy(dtype=float) >= float(cfg.velocity_threshold))
        & epoch_df["stage_valid"].to_numpy(dtype=bool)
    )
    return finalize_alerts(epoch_df, raw_alert, alert_col="alert", alert_raw_col="alert_raw", merge_gap=cfg.alert_merge_gap_epochs)


def apply_lie_alert_logic(epoch_df: pd.DataFrame, cfg: SleepEDFConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_alert = (
        (epoch_df["lie_transition_score"].to_numpy(dtype=float) >= float(cfg.lie_alert_threshold))
        & (epoch_df["lie_orbit_norm_z"].to_numpy(dtype=float) >= float(cfg.lie_orbit_gate))
        & epoch_df["stage_valid"].to_numpy(dtype=bool)
    )
    return finalize_alerts(
        epoch_df,
        raw_alert,
        alert_col="lie_alert",
        alert_raw_col="lie_alert_raw",
        merge_gap=cfg.alert_merge_gap_epochs,
    )


def evaluate_alert_setting(
    epoch_features_df: pd.DataFrame,
    cfg: SleepEDFConfig,
    channel: str,
    sfreq: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    epoch_alert_df, episodes_df = apply_alert_logic(epoch_features_df, cfg)
    transitions_df = build_transition_table(
        epoch_alert_df,
        episodes_df,
        cfg,
        score_col="transition_score",
        score_peak_col="transition_score_peak_z",
        align_metric_col="memory_align",
        secondary_metric_col="novelty",
    )
    pair_summary_df = build_pair_summary(transitions_df)
    confusion_df, metrics_df = build_confusion_summary(epoch_alert_df, transitions_df, cfg, alert_col="alert")
    summary_df = build_summary_row(transitions_df, metrics_df, channel=channel, sfreq=sfreq, epoch_df=epoch_alert_df)
    return epoch_alert_df, episodes_df, transitions_df, pair_summary_df, confusion_df, summary_df


def evaluate_lie_alert_setting(
    epoch_features_df: pd.DataFrame,
    cfg: SleepEDFConfig,
    channel: str,
    sfreq: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    epoch_alert_df, episodes_df = apply_lie_alert_logic(epoch_features_df, cfg)
    transitions_df = build_transition_table(
        epoch_alert_df,
        episodes_df,
        cfg,
        score_col="lie_transition_score",
        score_peak_col="lie_orbit_norm_peak_z",
        align_metric_col="lie_orbit_norm",
        secondary_metric_col="lie_strain_norm",
    )
    pair_summary_df = build_pair_summary(transitions_df)
    confusion_df, metrics_df = build_confusion_summary(epoch_alert_df, transitions_df, cfg, alert_col="lie_alert")
    summary_df = build_summary_row(transitions_df, metrics_df, channel=channel, sfreq=sfreq, epoch_df=epoch_alert_df)
    return epoch_alert_df, episodes_df, transitions_df, pair_summary_df, confusion_df, summary_df


def build_transition_table(
    epoch_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    cfg: SleepEDFConfig,
    score_col: str,
    score_peak_col: str,
    align_metric_col: str,
    secondary_metric_col: str,
) -> pd.DataFrame:
    runs_df = build_stage_runs(epoch_df)
    rows: list[dict[str, Any]] = []
    for left, right in zip(runs_df.iloc[:-1].itertuples(index=False), runs_df.iloc[1:].itertuples(index=False)):
        if not bool(left.stage_valid) or not bool(right.stage_valid):
            continue
        if int(left.n_epochs) < int(cfg.min_stage_run_epochs) or int(right.n_epochs) < int(cfg.min_stage_run_epochs):
            continue
        if str(left.stage_label) == str(right.stage_label):
            continue

        event_epoch = int(right.start_epoch)
        pre_start = max(0, event_epoch - int(cfg.pre_transition_epochs))
        candidate = episodes_df.loc[
            (episodes_df["start_epoch"] >= pre_start)
            & (episodes_df["start_epoch"] <= event_epoch)
        ].copy()
        if candidate.empty:
            visible = 0
            lead_epochs = np.nan
            alert_start_epoch = np.nan
            alert_end_epoch = np.nan
            alert_episode_id = np.nan
        else:
            selected = candidate.sort_values("start_epoch").iloc[-1]
            visible = 1
            lead_epochs = float(event_epoch - int(selected["start_epoch"]))
            alert_start_epoch = int(selected["start_epoch"])
            alert_end_epoch = int(selected["end_epoch"])
            alert_episode_id = int(selected["episode_id"])

        pre_window = epoch_df.loc[(epoch_df["epoch_index"] >= pre_start) & (epoch_df["epoch_index"] < event_epoch)].copy()
        baseline_lo = max(0, int(left.end_epoch) - int(cfg.baseline_epochs) + 1)
        baseline_hi = int(left.end_epoch)
        baseline = epoch_df.loc[(epoch_df["epoch_index"] >= baseline_lo) & (epoch_df["epoch_index"] <= baseline_hi)].copy()

        rows.append(
            {
                "transition_id": len(rows),
                "from_stage": str(left.stage_label),
                "to_stage": str(right.stage_label),
                "event_epoch": event_epoch,
                "visible": visible,
                "lead_epochs": lead_epochs,
                "lead_minutes": lead_epochs * float(cfg.epoch_seconds) / 60.0 if np.isfinite(lead_epochs) else np.nan,
                "alert_episode_id": alert_episode_id,
                "alert_start_epoch": alert_start_epoch,
                "alert_end_epoch": alert_end_epoch,
                f"pre_transition_{score_col}_median": float(pre_window[score_col].median()) if not pre_window.empty else np.nan,
                f"pre_transition_{score_col}_peak": float(pre_window[score_col].max()) if not pre_window.empty else np.nan,
                f"pre_transition_{score_peak_col}": float(pre_window[score_peak_col].max()) if not pre_window.empty else np.nan,
                f"pre_transition_{align_metric_col}_median": float(pre_window[align_metric_col].median()) if not pre_window.empty else np.nan,
                f"pre_transition_{secondary_metric_col}_median": float(pre_window[secondary_metric_col].median()) if not pre_window.empty else np.nan,
                f"baseline_{score_col}_median": float(baseline[score_col].median()) if not baseline.empty else np.nan,
                f"baseline_{align_metric_col}_median": float(baseline[align_metric_col].median()) if not baseline.empty else np.nan,
                f"baseline_{secondary_metric_col}_median": float(baseline[secondary_metric_col].median()) if not baseline.empty else np.nan,
            }
        )
    columns = [
        "transition_id",
        "from_stage",
        "to_stage",
        "event_epoch",
        "visible",
        "lead_epochs",
        "lead_minutes",
        "alert_episode_id",
        "alert_start_epoch",
        "alert_end_epoch",
        f"pre_transition_{score_col}_median",
        f"pre_transition_{score_col}_peak",
        f"pre_transition_{score_peak_col}",
        f"pre_transition_{align_metric_col}_median",
        f"pre_transition_{secondary_metric_col}_median",
        f"baseline_{score_col}_median",
        f"baseline_{align_metric_col}_median",
        f"baseline_{secondary_metric_col}_median",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_confusion_summary(
    epoch_df: pd.DataFrame,
    transitions_df: pd.DataFrame,
    cfg: SleepEDFConfig,
    alert_col: str = "alert",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive = np.zeros(len(epoch_df), dtype=bool)
    if "event_epoch" in transitions_df.columns:
        event_epochs = transitions_df["event_epoch"].dropna().astype(int).tolist()
    else:
        event_epochs = []
    for event_epoch in event_epochs:
        lo = max(0, event_epoch - int(cfg.pre_transition_epochs))
        hi = min(len(epoch_df) - 1, event_epoch + int(cfg.post_transition_epochs))
        positive[lo : hi + 1] = True

    predicted = epoch_df[alert_col].fillna(False).astype(bool).to_numpy()
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & (~positive)))
    tn = int(np.sum((~predicted) & (~positive)))
    fn = int(np.sum((~predicted) & positive))

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
                "alert_occupancy_fraction": float(np.mean(predicted)) if len(predicted) else np.nan,
                "positive_epoch_fraction": float(np.mean(positive)) if len(positive) else np.nan,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        ]
    )
    return matrix_df, metrics_df


def build_pair_summary(transitions_df: pd.DataFrame) -> pd.DataFrame:
    if transitions_df.empty:
        return pd.DataFrame(columns=["from_stage", "to_stage", "n_transitions", "visible_fraction", "median_lead_epochs", "median_lead_minutes"])
    rows = (
        transitions_df.groupby(["from_stage", "to_stage"])
        .agg(
            n_transitions=("transition_id", "count"),
            visible_fraction=("visible", "mean"),
            median_lead_epochs=("lead_epochs", "median"),
            median_lead_minutes=("lead_minutes", "median"),
        )
        .reset_index()
        .sort_values(["n_transitions", "from_stage", "to_stage"], ascending=[False, True, True])
    )
    return rows


def build_detection_sample_export(
    sample_df: pd.DataFrame,
    epoch_df: pd.DataFrame,
    baseline_episodes_df: pd.DataFrame,
    baseline_transitions_df: pd.DataFrame,
    lie_episodes_df: pd.DataFrame,
    lie_transitions_df: pd.DataFrame,
    cfg: SleepEDFConfig,
) -> pd.DataFrame:
    if sample_df.empty:
        return sample_df.copy()

    if epoch_df.empty:
        max_epoch = int(sample_df["epoch_index"].max())
    else:
        max_epoch = int(max(sample_df["epoch_index"].max(), epoch_df["epoch_index"].max()))

    selected_epochs = np.zeros(max_epoch + 1, dtype=bool)
    baseline_near_alert_epochs = np.zeros(max_epoch + 1, dtype=bool)
    baseline_near_transition_epochs = np.zeros(max_epoch + 1, dtype=bool)
    lie_near_alert_epochs = np.zeros(max_epoch + 1, dtype=bool)
    lie_near_transition_epochs = np.zeros(max_epoch + 1, dtype=bool)
    baseline_alert_tags = [""] * (max_epoch + 1)
    baseline_transition_tags = [""] * (max_epoch + 1)
    lie_alert_tags = [""] * (max_epoch + 1)
    lie_transition_tags = [""] * (max_epoch + 1)
    context = max(0, int(cfg.sample_context_epochs))

    def append_tag(tag_list: list[str], idx: int, value: str) -> None:
        if not tag_list[idx]:
            tag_list[idx] = value
        elif value not in tag_list[idx].split(","):
            tag_list[idx] = f"{tag_list[idx]},{value}"

    def mark_episode_context(
        episodes_df: pd.DataFrame,
        near_alert_epochs: np.ndarray,
        alert_tags: list[str],
    ) -> None:
        for row in episodes_df.itertuples(index=False):
            lo = max(0, int(row.start_epoch) - context)
            hi = min(max_epoch, int(row.end_epoch) + context)
            selected_epochs[lo : hi + 1] = True
            near_alert_epochs[lo : hi + 1] = True
            for epoch_idx in range(lo, hi + 1):
                append_tag(alert_tags, epoch_idx, str(int(row.episode_id)))

    def mark_transition_context(
        transitions_df: pd.DataFrame,
        near_transition_epochs: np.ndarray,
        transition_tags: list[str],
    ) -> None:
        for row in transitions_df.itertuples(index=False):
            event_epoch = int(row.event_epoch)
            lo = max(0, event_epoch - context)
            hi = min(max_epoch, event_epoch + max(context, int(cfg.post_transition_epochs)))
            selected_epochs[lo : hi + 1] = True
            near_transition_epochs[lo : hi + 1] = True
            transition_label = f"{int(row.transition_id)}:{row.from_stage}->{row.to_stage}"
            for epoch_idx in range(lo, hi + 1):
                append_tag(transition_tags, epoch_idx, transition_label)

    mark_episode_context(baseline_episodes_df, baseline_near_alert_epochs, baseline_alert_tags)
    mark_transition_context(baseline_transitions_df, baseline_near_transition_epochs, baseline_transition_tags)
    mark_episode_context(lie_episodes_df, lie_near_alert_epochs, lie_alert_tags)
    mark_transition_context(lie_transitions_df, lie_near_transition_epochs, lie_transition_tags)

    sample_epoch_idx = sample_df["epoch_index"].to_numpy(dtype=int)
    export_df = sample_df.loc[selected_epochs[sample_epoch_idx]].copy()
    if export_df.empty:
        return export_df

    export_epoch_idx = export_df["epoch_index"].to_numpy(dtype=int)
    export_df["near_alert_context"] = baseline_near_alert_epochs[export_epoch_idx]
    export_df["near_transition_context"] = baseline_near_transition_epochs[export_epoch_idx]
    export_df["alert_episode_ids"] = [baseline_alert_tags[idx] for idx in export_epoch_idx]
    export_df["transition_ids"] = [baseline_transition_tags[idx] for idx in export_epoch_idx]
    export_df["baseline_near_alert_context"] = baseline_near_alert_epochs[export_epoch_idx]
    export_df["baseline_near_transition_context"] = baseline_near_transition_epochs[export_epoch_idx]
    export_df["baseline_alert_episode_ids"] = [baseline_alert_tags[idx] for idx in export_epoch_idx]
    export_df["baseline_transition_ids"] = [baseline_transition_tags[idx] for idx in export_epoch_idx]
    export_df["lie_near_alert_context"] = lie_near_alert_epochs[export_epoch_idx]
    export_df["lie_near_transition_context"] = lie_near_transition_epochs[export_epoch_idx]
    export_df["lie_alert_episode_ids"] = [lie_alert_tags[idx] for idx in export_epoch_idx]
    export_df["lie_transition_ids"] = [lie_transition_tags[idx] for idx in export_epoch_idx]

    epoch_annotations = epoch_df.loc[
        :,
        [
            "epoch_index",
            "stage_label",
            "alert",
            "lie_alert",
            "transition_score",
            "transition_score_z",
            "lie_transition_score",
            "lie_orbit_norm",
            "lie_strain_norm",
        ],
    ].copy()
    return export_df.merge(epoch_annotations, on="epoch_index", how="left")


def run_alert_sweep(
    epoch_features_df: pd.DataFrame,
    cfg: SleepEDFConfig,
    channel: str,
    sfreq: float,
    alert_thresholds: list[float],
    velocity_thresholds: list[float],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    clean_epoch_df = epoch_features_df.drop(columns=["alert", "alert_raw"], errors="ignore")
    for alert_threshold in alert_thresholds:
        for velocity_threshold in velocity_thresholds:
            sweep_cfg = replace(cfg, alert_threshold=float(alert_threshold), velocity_threshold=float(velocity_threshold))
            epoch_alert_df, episodes_df, transitions_df, _, confusion_df, summary_df = evaluate_alert_setting(
                clean_epoch_df,
                cfg=sweep_cfg,
                channel=channel,
                sfreq=sfreq,
            )
            sweep_row = summary_df.copy()
            sweep_row["alert_threshold"] = float(alert_threshold)
            sweep_row["velocity_threshold"] = float(velocity_threshold)
            sweep_row["n_alert_episodes"] = int(len(episodes_df))
            sweep_row["n_alert_epochs"] = int(epoch_alert_df["alert"].fillna(False).astype(bool).sum())
            if not confusion_df.empty:
                sweep_row["tn"] = int(confusion_df.loc[confusion_df["actual"] == "negative", "predicted_negative"].iloc[0])
                sweep_row["fp"] = int(confusion_df.loc[confusion_df["actual"] == "negative", "predicted_positive"].iloc[0])
                sweep_row["fn"] = int(confusion_df.loc[confusion_df["actual"] == "positive", "predicted_negative"].iloc[0])
                sweep_row["tp"] = int(confusion_df.loc[confusion_df["actual"] == "positive", "predicted_positive"].iloc[0])
            rows.append(sweep_row)
    if not rows:
        return pd.DataFrame()
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(
            [
                "visible_fraction",
                "sensitivity",
                "precision",
                "specificity",
                "alert_threshold",
                "velocity_threshold",
            ],
            ascending=[False, False, False, False, True, True],
        )
        .reset_index(drop=True)
    )


def build_summary_row(
    transitions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    channel: str,
    sfreq: float,
    epoch_df: pd.DataFrame,
) -> pd.DataFrame:
    if transitions_df.empty:
        return pd.DataFrame(
            [
                {
                    "channel": channel,
                    "sfreq_hz": sfreq,
                    "n_epochs": int(len(epoch_df)),
                    "n_transitions": 0,
                    "n_visible_pre_transition": 0,
                    "visible_fraction": np.nan,
                    "median_lead_epochs": np.nan,
                    "median_lead_minutes": np.nan,
                    "specificity": float(metrics_df["specificity"].iloc[0]) if not metrics_df.empty else np.nan,
                    "sensitivity": float(metrics_df["sensitivity"].iloc[0]) if not metrics_df.empty else np.nan,
                    "precision": float(metrics_df["precision"].iloc[0]) if not metrics_df.empty else np.nan,
                    "alert_occupancy_fraction": float(metrics_df["alert_occupancy_fraction"].iloc[0]) if not metrics_df.empty else np.nan,
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "channel": channel,
                "sfreq_hz": sfreq,
                "n_epochs": int(len(epoch_df)),
                "n_transitions": int(len(transitions_df)),
                "n_visible_pre_transition": int(transitions_df["visible"].sum()),
                "visible_fraction": float(transitions_df["visible"].mean()),
                "median_lead_epochs": float(transitions_df["lead_epochs"].median()),
                "median_lead_minutes": float(transitions_df["lead_minutes"].median()),
                "specificity": float(metrics_df["specificity"].iloc[0]) if not metrics_df.empty else np.nan,
                "sensitivity": float(metrics_df["sensitivity"].iloc[0]) if not metrics_df.empty else np.nan,
                "precision": float(metrics_df["precision"].iloc[0]) if not metrics_df.empty else np.nan,
                "alert_occupancy_fraction": float(metrics_df["alert_occupancy_fraction"].iloc[0]) if not metrics_df.empty else np.nan,
            }
        ]
    )


def build_detector_comparison_row(
    detector_name: str,
    summary_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    score_threshold: float,
    gate_threshold: float,
    score_column: str,
    gate_column: str,
) -> pd.DataFrame:
    row = summary_df.copy()
    row["detector"] = detector_name
    row["score_threshold"] = float(score_threshold)
    row["gate_threshold"] = float(gate_threshold)
    row["score_column"] = score_column
    row["gate_column"] = gate_column
    row["n_alert_episodes"] = int(len(episodes_df))
    ordered_cols = [
        "detector",
        "score_column",
        "gate_column",
        "score_threshold",
        "gate_threshold",
        "channel",
        "sfreq_hz",
        "n_epochs",
        "n_transitions",
        "n_visible_pre_transition",
        "visible_fraction",
        "median_lead_epochs",
        "median_lead_minutes",
        "specificity",
        "sensitivity",
        "precision",
        "alert_occupancy_fraction",
        "n_alert_episodes",
    ]
    return row.loc[:, ordered_cols]


def write_report(
    run_dir: Path,
    cfg: SleepEDFConfig,
    channel: str,
    sfreq: float,
    summary_df: pd.DataFrame,
    pair_summary_df: pd.DataFrame,
    transitions_df: pd.DataFrame,
) -> None:
    lines = [
        "# Sleep-EDF Geometry Starter",
        "",
        "Takens-style delay-embedding geometry on a single Sleep-EDF EEG channel, evaluated against labeled sleep-stage transitions.",
        "",
        "## Configuration",
        f"- PSG EDF: `{cfg.psg_edf}`",
        f"- Hypnogram EDF: `{cfg.hypnogram_edf}`",
        f"- Channel: `{channel}`",
        f"- Sample rate after resampling: `{sfreq:g}` Hz",
        f"- Epoch length: `{cfg.epoch_seconds:g}` seconds",
        f"- Embedding: `dim={cfg.embedding_dim}`, `lag_samples={cfg.lag_samples}`",
        f"- Memory betas: short=`{cfg.beta_short:g}`, long=`{cfg.beta_long:g}`",
        f"- Pre-transition window: `{cfg.pre_transition_epochs}` epochs",
        f"- Alert rule: `transition_score_z >= {cfg.alert_threshold:g}` and `projective_velocity_norm_z >= {cfg.velocity_threshold:g}`",
        "",
        "## Summary",
        summary_df.to_markdown(index=False),
        "",
    ]

    if not pair_summary_df.empty:
        lines.extend(
            [
                "## Transition Pairs",
                pair_summary_df.to_markdown(index=False),
                "",
            ]
        )

    if not transitions_df.empty:
        cols = [
            "from_stage",
            "to_stage",
            "event_epoch",
            "visible",
            "lead_epochs",
            "lead_minutes",
            "pre_transition_transition_score_peak_z",
        ]
        lines.extend(
            [
                "## Event Table",
                transitions_df[cols].head(40).to_markdown(index=False),
                "",
            ]
        )

    (run_dir / "sleep_edf_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Sleep-EDF starter experiment for the geometric EEG pipeline.")
    parser.add_argument("--psg-edf", required=True)
    parser.add_argument("--hypnogram-edf", required=True)
    parser.add_argument("--channel", default=None)
    parser.add_argument("--target-sfreq", type=float, default=100.0)
    parser.add_argument("--epoch-seconds", type=float, default=30.0)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--lag-samples", type=int, default=4)
    parser.add_argument("--beta-short", type=float, default=0.10)
    parser.add_argument("--beta-long", type=float, default=0.01)
    parser.add_argument("--rolling-seconds", type=float, default=30.0)
    parser.add_argument("--pre-transition-epochs", type=int, default=20)
    parser.add_argument("--post-transition-epochs", type=int, default=1)
    parser.add_argument("--baseline-epochs", type=int, default=20)
    parser.add_argument("--min-stage-run-epochs", type=int, default=2)
    parser.add_argument("--min-state-energy", type=float, default=1e-8)
    parser.add_argument("--alert-threshold", type=float, default=1.5)
    parser.add_argument("--velocity-threshold", type=float, default=0.25)
    parser.add_argument("--alert-merge-gap-epochs", type=int, default=1)
    parser.add_argument(
        "--sample-context-epochs",
        type=int,
        default=2,
        help="Epoch padding to include before and after detected alert or transition windows in sample-level exports.",
    )
    parser.add_argument(
        "--write-sample-features",
        action="store_true",
        help="Write only the sample-level rows near alert episodes and transition events.",
    )
    parser.add_argument(
        "--write-full-sample-features",
        action="store_true",
        help="Write the full sample-level feature table to CSV. This is much larger and slower.",
    )
    parser.add_argument(
        "--alert-threshold-grid",
        default=None,
        help="Comma-separated alert-threshold values to sweep in one run, for example '0.5,1.0,1.5,2.0'.",
    )
    parser.add_argument(
        "--velocity-threshold-grid",
        default=None,
        help="Comma-separated velocity-threshold values to sweep in one run, for example '0.0,0.25,0.5,0.75'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root(Path(__file__).resolve().parent)
    run_dir = prepare_run_directory(project_root / "artifact" / "runs")
    cfg = SleepEDFConfig(
        psg_edf=Path(args.psg_edf).resolve(),
        hypnogram_edf=Path(args.hypnogram_edf).resolve(),
        run_root=project_root / "artifact" / "runs",
        channel=args.channel,
        target_sfreq=args.target_sfreq,
        epoch_seconds=args.epoch_seconds,
        embedding_dim=args.embedding_dim,
        lag_samples=args.lag_samples,
        beta_short=args.beta_short,
        beta_long=args.beta_long,
        rolling_seconds=args.rolling_seconds,
        pre_transition_epochs=args.pre_transition_epochs,
        post_transition_epochs=args.post_transition_epochs,
        baseline_epochs=args.baseline_epochs,
        min_stage_run_epochs=args.min_stage_run_epochs,
        min_state_energy=args.min_state_energy,
        alert_threshold=args.alert_threshold,
        velocity_threshold=args.velocity_threshold,
        alert_merge_gap_epochs=args.alert_merge_gap_epochs,
        sample_context_epochs=args.sample_context_epochs,
        write_sample_features=bool(args.write_sample_features),
        write_full_sample_features=bool(args.write_full_sample_features),
    )

    log(cfg, f"Loading PSG {cfg.psg_edf.name} and hypnogram {cfg.hypnogram_edf.name}")
    signal, sfreq, channel, epochs_df = load_sleep_edf_pair(cfg)
    log(cfg, f"Loaded {len(signal)} samples at {sfreq:g} Hz on channel {channel}")

    sample_df = compute_sample_features(signal, sfreq=sfreq, cfg=cfg)
    log(cfg, f"Built {len(sample_df)} delay-state rows")

    epoch_features_df = aggregate_epoch_features(sample_df, epochs_df)

    baseline_epoch_df, episodes_df, transitions_df, pair_summary_df, confusion_df, summary_df = evaluate_alert_setting(
        epoch_features_df,
        cfg=cfg,
        channel=channel,
        sfreq=sfreq,
    )
    log(cfg, f"Built {len(episodes_df)} alert episodes")
    _, metrics_df = build_confusion_summary(baseline_epoch_df, transitions_df, cfg, alert_col="alert")

    lie_epoch_df, lie_episodes_df, lie_transitions_df, _, lie_confusion_df, lie_summary_df = evaluate_lie_alert_setting(
        epoch_features_df,
        cfg=cfg,
        channel=channel,
        sfreq=sfreq,
    )
    log(cfg, f"Built {len(lie_episodes_df)} Lie alert episodes")
    _, lie_metrics_df = build_confusion_summary(lie_epoch_df, lie_transitions_df, cfg, alert_col="lie_alert")

    combined_epoch_df = epoch_features_df.copy()
    combined_epoch_df["alert"] = baseline_epoch_df["alert"].to_numpy(dtype=bool)
    combined_epoch_df["alert_raw"] = baseline_epoch_df["alert_raw"].to_numpy(dtype=bool)
    combined_epoch_df["lie_alert"] = lie_epoch_df["lie_alert"].to_numpy(dtype=bool)
    combined_epoch_df["lie_alert_raw"] = lie_epoch_df["lie_alert_raw"].to_numpy(dtype=bool)

    if cfg.write_sample_features:
        detection_sample_df = build_detection_sample_export(
            sample_df,
            combined_epoch_df,
            episodes_df,
            transitions_df,
            lie_episodes_df,
            lie_transitions_df,
            cfg,
        )
        detection_sample_df.to_csv(run_dir / "sleep_edf_detection_sample_features.csv", index=False)
        log(cfg, f"Wrote {len(detection_sample_df)} detection-local sample rows")
    else:
        log(cfg, "Skipped detection-local sample export (pass --write-sample-features to enable)")

    if cfg.write_full_sample_features:
        sample_df.to_csv(run_dir / "sleep_edf_sample_features.csv", index=False)
        log(cfg, "Wrote full sample-level CSV export")
    del sample_df

    combined_epoch_df.to_csv(run_dir / "sleep_edf_epoch_features.csv", index=False)
    episodes_df.to_csv(run_dir / "sleep_edf_alert_episodes.csv", index=False)
    transitions_df.to_csv(run_dir / "sleep_edf_transition_table.csv", index=False)
    pair_summary_df.to_csv(run_dir / "sleep_edf_transition_pair_summary.csv", index=False)
    confusion_df.to_csv(run_dir / "sleep_edf_confusion_matrix.csv", index=False)
    metrics_df.to_csv(run_dir / "sleep_edf_confusion_metrics.csv", index=False)
    summary_df.to_csv(run_dir / "sleep_edf_summary.csv", index=False)
    lie_episodes_df.to_csv(run_dir / "sleep_edf_lie_alert_episodes.csv", index=False)
    lie_transitions_df.to_csv(run_dir / "sleep_edf_lie_transition_table.csv", index=False)
    lie_confusion_df.to_csv(run_dir / "sleep_edf_lie_confusion_matrix.csv", index=False)
    lie_metrics_df.to_csv(run_dir / "sleep_edf_lie_confusion_metrics.csv", index=False)
    lie_summary_df.to_csv(run_dir / "sleep_edf_lie_summary.csv", index=False)

    detector_comparison_df = pd.concat(
        [
            build_detector_comparison_row(
                detector_name="baseline",
                summary_df=summary_df,
                episodes_df=episodes_df,
                score_threshold=cfg.alert_threshold,
                gate_threshold=cfg.velocity_threshold,
                score_column="transition_score_z",
                gate_column="projective_velocity_norm_z",
            ),
            build_detector_comparison_row(
                detector_name="lie",
                summary_df=lie_summary_df,
                episodes_df=lie_episodes_df,
                score_threshold=cfg.lie_alert_threshold,
                gate_threshold=cfg.lie_orbit_gate,
                score_column="lie_transition_score",
                gate_column="lie_orbit_norm_z",
            ),
        ],
        ignore_index=True,
    )
    detector_comparison_df.to_csv(run_dir / "sleep_edf_detector_comparison.csv", index=False)

    alert_threshold_grid = parse_float_grid(args.alert_threshold_grid)
    velocity_threshold_grid = parse_float_grid(args.velocity_threshold_grid)
    if alert_threshold_grid or velocity_threshold_grid:
        if not alert_threshold_grid:
            alert_threshold_grid = [float(cfg.alert_threshold)]
        if not velocity_threshold_grid:
            velocity_threshold_grid = [float(cfg.velocity_threshold)]
        sweep_df = run_alert_sweep(
            epoch_features_df=combined_epoch_df,
            cfg=cfg,
            channel=channel,
            sfreq=sfreq,
            alert_thresholds=alert_threshold_grid,
            velocity_thresholds=velocity_threshold_grid,
        )
        sweep_df.to_csv(run_dir / "sleep_edf_threshold_sweep.csv", index=False)
        log(cfg, f"Wrote threshold sweep with {len(sweep_df)} settings")
        if not sweep_df.empty:
            print("\nTop sweep settings:")
            print(
                sweep_df[
                    [
                        "alert_threshold",
                        "velocity_threshold",
                        "visible_fraction",
                        "median_lead_minutes",
                        "specificity",
                        "sensitivity",
                        "precision",
                        "alert_occupancy_fraction",
                        "n_alert_episodes",
                    ]
                ]
                .head(10)
                .to_string(index=False)
            )

    write_report(
        run_dir=run_dir,
        cfg=cfg,
        channel=channel,
        sfreq=sfreq,
        summary_df=summary_df,
        pair_summary_df=pair_summary_df,
        transitions_df=transitions_df,
    )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": json_safe(asdict(cfg)),
        "channel": channel,
        "sfreq_hz": sfreq,
        "n_signal_samples": int(len(signal)),
        "n_epochs": int(len(combined_epoch_df)),
        "n_transitions": int(len(transitions_df)),
    }
    (run_dir / "sleep_edf_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Run directory: {run_dir}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
