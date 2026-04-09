from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cdflib import CDF, cdfepoch


EPS = 1e-9
SOLAR_HMM_STATE_ORDER = ["baseline", "transition", "active"]
SOLAR_HMM_STATE_TO_INDEX = {name: idx for idx, name in enumerate(SOLAR_HMM_STATE_ORDER)}


OMNI_COLUMNS = {
    "bz_gsm": "BZ_GSM",
    "flow_speed": "flow_speed",
    "proton_density": "proton_density",
    "sym_h": "SYM_H",
}

OMNI_BOUNDS = {
    "bz_gsm": {"fill": 9999.99, "min": -99.9, "max": 99.9},
    "flow_speed": {"fill": 99999.9, "min": 0.0, "max": 2000.0},
    "proton_density": {"fill": 999.99, "min": -100.0, "max": 100.0},
    "sym_h": {"fill": 99999.0, "min": -1000.0, "max": 1000.0},
}


@dataclass(frozen=True)
class SolarConfig:
    cache_dir: Path
    run_root: Path
    file_glob: str = "omni_hro2_1min_*.cdf"
    recursive: bool = True
    start_time: str | None = None
    end_time: str | None = None
    verbose: bool = True
    write_feature_csv: bool = False
    write_alignment_csv: bool = False
    lag_minutes: int = 60
    interpolation_limit_minutes: int = 30
    storm_threshold: float = -40.0
    storm_sustain_minutes: int = 30
    storm_min_gap_minutes: int = 720
    bz_south_threshold: float = -5.0
    bz_sustain_minutes: int = 15
    bz_lookback_minutes: int = 1440
    baseline_minutes: int = 720
    pre_event_minutes: int = 1440
    post_event_minutes: int = 360
    early_move_sigma: float = 1.5
    early_move_sustain_minutes: int = 3
    beta_short: float = 0.10
    beta_long: float = 0.01
    projective_min_state_energy: float = 1e-8
    hmm_rolling_window_minutes: int = 15
    hmm_merge_gap_minutes: int = 15
    hmm_alert_active_tail_minutes: int = 30
    hmm_alert_max_minutes: int = 180
    hmm_alert_cooldown_minutes: int = 60


def find_project_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "data").exists() and (candidate / "artifact").exists():
            return candidate
    raise FileNotFoundError("Could not locate the Bearings project root.")


def default_config() -> SolarConfig:
    project_root = find_project_root()
    return SolarConfig(
        cache_dir=project_root / "data" / "omni_cache",
        run_root=project_root / "artifact" / "runs",
    )


def prepare_run_directory(base_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("solar_%Y%m%dT%H%M%SZ")
    run_dir = base_root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_optional_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    return pd.Timestamp(value)


def log(cfg: SolarConfig, message: str) -> None:
    if cfg.verbose:
        print(f"[SOLAR] {message}", flush=True)


def discover_omni_files(cache_dir: Path, file_glob: str, recursive: bool) -> list[Path]:
    iterator = cache_dir.rglob(file_glob) if recursive else cache_dir.glob(file_glob)
    return sorted(path for path in iterator if path.is_file())


def extract_month_start_from_filename(path: Path) -> pd.Timestamp | None:
    name = path.name
    marker = "_1min_"
    idx = name.find(marker)
    if idx < 0:
        return None
    token = name[idx + len(marker) : idx + len(marker) + 8]
    if len(token) != 8 or not token.isdigit():
        return None
    try:
        return pd.Timestamp(f"{token[:4]}-{token[4:6]}-{token[6:8]}")
    except Exception:
        return None


def file_overlaps_time_range(path: Path, start_ts: pd.Timestamp | None, end_ts: pd.Timestamp | None) -> bool:
    month_start = extract_month_start_from_filename(path)
    if month_start is None:
        return True
    month_end = month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    if start_ts is not None and month_end < start_ts:
        return False
    if end_ts is not None and month_start > end_ts:
        return False
    return True


def load_omni_cache(cfg: SolarConfig) -> pd.DataFrame:
    start_ts = parse_optional_timestamp(cfg.start_time)
    end_ts = parse_optional_timestamp(cfg.end_time)
    all_files = discover_omni_files(cfg.cache_dir, file_glob=cfg.file_glob, recursive=cfg.recursive)
    selected_files = [path for path in all_files if file_overlaps_time_range(path, start_ts, end_ts)]
    log(cfg, f"Discovered {len(all_files)} candidate files; selected {len(selected_files)} for loading")

    frames: list[pd.DataFrame] = []
    for idx, path in enumerate(selected_files, start=1):
        log(cfg, f"Loading file {idx}/{len(selected_files)}: {path.name}")
        cdf = CDF(str(path))
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(cdfepoch.to_datetime(cdf.varget("Epoch"))),
                "bz_gsm": cdf.varget(OMNI_COLUMNS["bz_gsm"]).astype(float),
                "flow_speed": cdf.varget(OMNI_COLUMNS["flow_speed"]).astype(float),
                "proton_density": cdf.varget(OMNI_COLUMNS["proton_density"]).astype(float),
                "sym_h": cdf.varget(OMNI_COLUMNS["sym_h"]).astype(float),
            }
        )
        frame["source_file"] = path.name
        frame["source_path"] = str(path)
        if start_ts is not None:
            frame = frame[frame["time"] >= start_ts].copy()
        if end_ts is not None:
            frame = frame[frame["time"] <= end_ts].copy()
        if frame.empty:
            continue
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No OMNI CDF rows found under {cfg.cache_dir} matching {cfg.file_glob} "
            f"for the requested time range"
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    for col, bounds in OMNI_BOUNDS.items():
        s = df[col]
        df[col] = s.where(
            (s != bounds["fill"]) & (s >= bounds["min"]) & (s <= bounds["max"])
        )

    log(cfg, f"Loaded {len(df)} minute rows after deduplication and bounds cleaning")
    return df


def filter_time_range(df: pd.DataFrame, cfg: SolarConfig) -> pd.DataFrame:
    start_ts = parse_optional_timestamp(cfg.start_time)
    end_ts = parse_optional_timestamp(cfg.end_time)
    out = df.copy()
    if start_ts is not None:
        out = out[out["time"] >= start_ts].copy()
    if end_ts is not None:
        out = out[out["time"] <= end_ts].copy()
    return out.reset_index(drop=True)


def fill_and_standardize(df: pd.DataFrame, cfg: SolarConfig) -> pd.DataFrame:
    filled = df.copy().set_index("time").asfreq("1min")
    for col in ["bz_gsm", "flow_speed", "proton_density", "sym_h"]:
        filled[f"{col}_raw_missing"] = filled[col].isna()
        filled[col] = filled[col].interpolate(
            method="time",
            limit=cfg.interpolation_limit_minutes,
            limit_direction="both",
        )

    for col in ["bz_gsm", "flow_speed", "proton_density"]:
        median = float(filled[col].median())
        mad = float((filled[col] - median).abs().median())
        scale = 1.4826 * mad if mad > 1e-12 else float(filled[col].std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        filled[f"{col}_z"] = (filled[col] - median) / scale
        filled[f"{col}_median"] = median
        filled[f"{col}_scale"] = scale

    filled = filled.reset_index().rename(columns={"index": "time"})
    return filled


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
    out = np.full(len(values), np.nan, dtype=float)
    if window <= 1:
        return np.asarray(values, dtype=float)
    for idx in range(len(values)):
        lo = max(0, idx - window + 1)
        seg = np.asarray(values[lo : idx + 1], dtype=float)
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


def ema_prior_states(x: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=float)
    if len(arr) == 0:
        return np.empty_like(arr), np.zeros(arr.shape[1] if arr.ndim == 2 else 0, dtype=float)

    alpha = 1.0 - float(beta)
    state = np.zeros(arr.shape[1], dtype=float)
    prior = np.zeros_like(arr, dtype=float)
    for idx in range(len(arr)):
        prior[idx] = state
        state = float(beta) * arr[idx] + alpha * state
    return prior, state


def compute_projective_state_features(
    x: np.ndarray,
    ms_prior: np.ndarray,
    ml_prior: np.ndarray,
    min_state_energy: float,
) -> dict[str, np.ndarray]:
    n_x = np.einsum("td,td->t", x, x)
    n_s = np.einsum("td,td->t", ms_prior, ms_prior)
    n_l = np.einsum("td,td->t", ml_prior, ml_prior)
    d_sl = np.einsum("td,td->t", ms_prior, ml_prior)
    d_xs = np.einsum("td,td->t", x, ms_prior)
    d_xl = np.einsum("td,td->t", x, ml_prior)

    proj_line_lock_sl = np.full(len(x), np.nan, dtype=float)
    proj_area_sl = np.full(len(x), np.nan, dtype=float)
    proj_line_lock_xl = np.full(len(x), np.nan, dtype=float)
    proj_transverse_xl = np.full(len(x), np.nan, dtype=float)
    proj_lock_barrier_sl = np.full(len(x), np.nan, dtype=float)
    proj_lock_barrier_xl = np.full(len(x), np.nan, dtype=float)
    proj_volume_xsl = np.full(len(x), np.nan, dtype=float)
    log_proj_volume_xsl = np.full(len(x), np.nan, dtype=float)
    short_long_explanation_imbalance = np.full(len(x), np.nan, dtype=float)
    memory_align = np.full(len(x), np.nan, dtype=float)
    linger = np.full(len(x), np.nan, dtype=float)
    novelty = np.full(len(x), np.nan, dtype=float)

    energy_mask = n_x >= float(min_state_energy)
    sl_mask = energy_mask & (n_s > EPS) & (n_l > EPS)
    xl_mask = energy_mask & (n_l > EPS)

    if np.any(sl_mask):
        proj_line_lock_sl[sl_mask] = np.clip((d_sl[sl_mask] ** 2) / (n_s[sl_mask] * n_l[sl_mask] + EPS), 0.0, 1.0)
        proj_area_sl[sl_mask] = np.clip(1.0 - proj_line_lock_sl[sl_mask], 0.0, 1.0)
        proj_lock_barrier_sl[sl_mask] = -np.log(proj_area_sl[sl_mask] + EPS)
        memory_align[sl_mask] = d_sl[sl_mask] / (np.sqrt(n_s[sl_mask] * n_l[sl_mask]) + EPS)
        short_long_explanation_imbalance[sl_mask] = (d_xs[sl_mask] ** 2) / (n_x[sl_mask] * n_s[sl_mask] + EPS) - (
            d_xl[sl_mask] ** 2
        ) / (n_x[sl_mask] * n_l[sl_mask] + EPS)

        det_raw = (
            n_x[sl_mask] * n_s[sl_mask] * n_l[sl_mask]
            + 2.0 * d_xs[sl_mask] * d_xl[sl_mask] * d_sl[sl_mask]
            - n_x[sl_mask] * (d_sl[sl_mask] ** 2)
            - n_s[sl_mask] * (d_xl[sl_mask] ** 2)
            - n_l[sl_mask] * (d_xs[sl_mask] ** 2)
        )
        det_raw = np.maximum(det_raw, 0.0)
        proj_volume_xsl[sl_mask] = np.clip(det_raw / (n_x[sl_mask] * n_s[sl_mask] * n_l[sl_mask] + EPS), 0.0, 1.0)
        log_proj_volume_xsl[sl_mask] = np.log(proj_volume_xsl[sl_mask] + EPS)

    if np.any(xl_mask):
        proj_line_lock_xl[xl_mask] = np.clip((d_xl[xl_mask] ** 2) / (n_x[xl_mask] * n_l[xl_mask] + EPS), 0.0, 1.0)
        proj_transverse_xl[xl_mask] = np.clip(1.0 - proj_line_lock_xl[xl_mask], 0.0, 1.0)
        proj_lock_barrier_xl[xl_mask] = -np.log(proj_transverse_xl[xl_mask] + EPS)
        linger[xl_mask] = d_xl[xl_mask] / (np.sqrt(n_x[xl_mask] * n_l[xl_mask]) + EPS)
        novelty[xl_mask] = 1.0 - linger[xl_mask]

    return {
        "proj_line_lock_sl": proj_line_lock_sl,
        "proj_area_sl": proj_area_sl,
        "proj_line_lock_xl": proj_line_lock_xl,
        "proj_transverse_xl": proj_transverse_xl,
        "proj_lock_barrier_sl": proj_lock_barrier_sl,
        "proj_lock_barrier_xl": proj_lock_barrier_xl,
        "proj_volume_xsl": proj_volume_xsl,
        "log_proj_volume_xsl": log_proj_volume_xsl,
        "short_long_explanation_imbalance": short_long_explanation_imbalance,
        "memory_align": memory_align,
        "linger": linger,
        "novelty": novelty,
    }


def solar_emission_logits(frame: pd.DataFrame) -> np.ndarray:
    score_z = frame["transition_score_z"].to_numpy(dtype=float)
    level_z = frame["projective_level_norm_z"].to_numpy(dtype=float)
    velocity_z = frame["projective_velocity_norm_z"].to_numpy(dtype=float)
    curvature_z = frame["projective_curvature_norm_z"].to_numpy(dtype=float)
    novelty_z = frame["novelty_z"].to_numpy(dtype=float)
    memory_align_z = frame["memory_align_z"].to_numpy(dtype=float)
    barrier_sl_z = frame["proj_lock_barrier_sl_z"].to_numpy(dtype=float)
    barrier_xl_z = frame["proj_lock_barrier_xl_z"].to_numpy(dtype=float)

    baseline_logit = -1.10 * score_z - 0.45 * novelty_z - 0.40 * barrier_sl_z - 0.30 * velocity_z
    transition_logit = 0.95 * score_z + 0.90 * velocity_z + 0.75 * curvature_z + 0.50 * novelty_z + 0.35 * barrier_sl_z - 0.20 * level_z
    active_logit = 0.80 * score_z + 0.85 * level_z + 0.30 * barrier_xl_z + 0.15 * memory_align_z - 0.25 * curvature_z
    return np.vstack([baseline_logit, transition_logit, active_logit]).T


def solar_viterbi_decode(logits: np.ndarray) -> np.ndarray:
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


def build_hmm_alert_mask(states: np.ndarray, cfg: SolarConfig) -> np.ndarray:
    transition_idx = SOLAR_HMM_STATE_TO_INDEX["transition"]
    active_idx = SOLAR_HMM_STATE_TO_INDEX["active"]
    active_tail = max(0, int(cfg.hmm_alert_active_tail_minutes))
    max_alert = max(1, int(cfg.hmm_alert_max_minutes))

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

        if alert_start is not None and (idx - alert_start + 1) <= max_alert:
            mask[idx] = True
        else:
            alert_start = None
            last_transition_idx = None
    return mask


def build_episode_table(active_mask: np.ndarray, time_values: pd.Series, merge_gap: int) -> pd.DataFrame:
    arr = np.asarray(active_mask, dtype=bool)
    if len(arr) == 0:
        return pd.DataFrame(columns=["episode_id", "start_idx", "end_idx", "start_time", "end_time", "duration_minutes"])

    rows: list[dict[str, object]] = []
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
        return pd.DataFrame(columns=["episode_id", "start_idx", "end_idx", "start_time", "end_time", "duration_minutes"])

    merged: list[dict[str, object]] = []
    current = rows[0]
    for row in rows[1:]:
        gap = int(row["start_idx"]) - int(current["end_idx"]) - 1
        if gap <= int(merge_gap):
            current["end_idx"] = row["end_idx"]
        else:
            merged.append(current)
            current = row
    merged.append(current)

    final_rows: list[dict[str, object]] = []
    times = pd.to_datetime(time_values).reset_index(drop=True)
    for episode_id, row in enumerate(merged):
        start_idx = int(row["start_idx"])
        end_idx = int(row["end_idx"])
        final_rows.append(
            {
                "episode_id": int(episode_id),
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": times.iloc[start_idx],
                "end_time": times.iloc[end_idx],
                "duration_minutes": int(end_idx - start_idx + 1),
            }
        )
    return pd.DataFrame(final_rows)


def apply_episode_cooldown(episodes_df: pd.DataFrame, cooldown_minutes: int) -> pd.DataFrame:
    if episodes_df.empty or cooldown_minutes <= 0:
        return episodes_df.copy()

    kept_rows: list[dict[str, object]] = []
    last_end_idx: int | None = None
    for row in episodes_df.sort_values("start_idx").to_dict("records"):
        start_idx = int(row["start_idx"])
        if last_end_idx is not None and start_idx <= last_end_idx + int(cooldown_minutes):
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


def enrich_projective_hmm_features(df: pd.DataFrame, cfg: SolarConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    state_arr = out[["bz_gsm_z", "flow_speed_z", "proton_density_z"]].to_numpy(dtype=float)
    ms_prior, _ = ema_prior_states(state_arr, beta=cfg.beta_short)
    ml_prior, _ = ema_prior_states(state_arr, beta=cfg.beta_long)
    out["short_memory_norm"] = np.sqrt(np.einsum("td,td->t", ms_prior, ms_prior))
    out["long_memory_norm"] = np.sqrt(np.einsum("td,td->t", ml_prior, ml_prior))

    for name, values in compute_projective_state_features(
        state_arr,
        ms_prior=ms_prior,
        ml_prior=ml_prior,
        min_state_energy=cfg.projective_min_state_energy,
    ).items():
        out[name] = values

    reference_cols = [
        "poincare_b",
        "gram_spread_sq",
        "energy_asym",
        "drift_norm",
        "memory_align",
        "novelty",
        "proj_lock_barrier_sl",
        "proj_lock_barrier_xl",
        "proj_volume_xsl",
    ]
    for column in reference_cols:
        center, scale = robust_center_scale(out[column].to_numpy(dtype=float))
        rel = (out[column].to_numpy(dtype=float) - center) / (scale + EPS)
        out[f"rel_{column}"] = np.where(np.isfinite(rel), rel, 0.0)
        out[f"vel_{column}"] = np.r_[0.0, np.diff(out[f"rel_{column}"].to_numpy(dtype=float))]
        out[f"acc_{column}"] = np.r_[0.0, np.diff(out[f"vel_{column}"].to_numpy(dtype=float))]

    rel_cols = [f"rel_{column}" for column in reference_cols]
    vel_cols = [f"vel_{column}" for column in reference_cols]
    acc_cols = [f"acc_{column}" for column in reference_cols]

    out["projective_level_norm_raw"] = np.sqrt(np.sum(np.square(out[rel_cols].to_numpy(dtype=float)), axis=1))
    out["projective_velocity_norm_raw"] = np.sqrt(np.sum(np.square(out[vel_cols].to_numpy(dtype=float)), axis=1))
    out["projective_curvature_norm_raw"] = np.sqrt(np.sum(np.square(out[acc_cols].to_numpy(dtype=float)), axis=1))

    window = max(1, int(cfg.hmm_rolling_window_minutes))
    for column in ["projective_level_norm_raw", "projective_velocity_norm_raw", "projective_curvature_norm_raw"]:
        out[column.replace("_raw", "")] = trailing_roll(out[column].to_numpy(dtype=float), window=window, reducer="mean")

    barrier_acc = np.nan_to_num(out["acc_proj_lock_barrier_sl"].to_numpy(dtype=float), nan=0.0)
    novelty_vel = np.nan_to_num(out["vel_novelty"].to_numpy(dtype=float), nan=0.0)
    out["transition_score_raw"] = np.sqrt(
        0.50 * np.square(out["projective_level_norm"].to_numpy(dtype=float))
        + 1.00 * np.square(out["projective_velocity_norm"].to_numpy(dtype=float))
        + 1.25 * np.square(out["projective_curvature_norm"].to_numpy(dtype=float))
        + 0.75 * np.square(barrier_acc)
        + 0.50 * np.square(novelty_vel)
    )
    out["transition_score"] = trailing_roll(out["transition_score_raw"].to_numpy(dtype=float), window=window, reducer="mean")

    for metric in [
        "projective_level_norm",
        "projective_velocity_norm",
        "projective_curvature_norm",
        "transition_score",
        "novelty",
        "memory_align",
        "proj_lock_barrier_sl",
        "proj_lock_barrier_xl",
    ]:
        center, scale = robust_center_scale(out[metric].to_numpy(dtype=float))
        z = (out[metric].to_numpy(dtype=float) - center) / (scale + EPS)
        out[f"{metric}_z"] = np.where(np.isfinite(z), z, 0.0)

    logits = solar_emission_logits(out)
    state_idx = solar_viterbi_decode(logits)
    out["hmm_state"] = [SOLAR_HMM_STATE_ORDER[int(idx)] for idx in state_idx]
    out["hmm_state_code"] = state_idx

    raw_alert_mask = build_hmm_alert_mask(state_idx, cfg)
    episodes_df = build_episode_table(raw_alert_mask, out["time"], merge_gap=cfg.hmm_merge_gap_minutes)
    episodes_df = apply_episode_cooldown(episodes_df, cooldown_minutes=cfg.hmm_alert_cooldown_minutes)
    out["hmm_alert"] = episode_table_to_mask(len(out), episodes_df)
    return out, episodes_df


def compute_router_features(df: pd.DataFrame, cfg: SolarConfig) -> pd.DataFrame:
    out = df.copy()
    arr = out[["bz_gsm_z", "flow_speed_z", "proton_density_z"]].to_numpy(dtype=float)
    lag = cfg.lag_minutes
    dim = 3
    inv_dim = 1.0 / float(dim)

    out["curr_energy"] = np.nan
    out["past_energy"] = np.nan
    out["dot_centered"] = np.nan
    out["twist_angle"] = np.nan
    out["twist_sq"] = np.nan
    out["phase_polarity"] = np.nan
    out["poincare_b"] = np.nan
    out["gram_spread_sq"] = np.nan
    out["energy_asym"] = np.nan
    out["drift_norm"] = np.nan

    if len(out) <= lag:
        return out

    x_curr = arr[lag:]
    x_past = arr[:-lag]

    sx = x_curr.sum(axis=1)
    sy = x_past.sum(axis=1)
    sxx = np.square(x_curr).sum(axis=1)
    syy = np.square(x_past).sum(axis=1)
    sxy = (x_curr * x_past).sum(axis=1)

    curr_energy = sxx - (sx * sx) * inv_dim
    past_energy = syy - (sy * sy) * inv_dim
    dot_centered = sxy - (sx * sy) * inv_dim

    prod = curr_energy * past_energy
    valid = (curr_energy > 1e-6) & (past_energy > 1e-6)

    cos2 = np.full_like(prod, np.nan)
    cos2[valid] = np.square(dot_centered[valid]) / prod[valid]
    cos2[valid] = np.clip(cos2[valid], 0.0, 1.0)

    twist_angle = np.full_like(prod, np.nan)
    twist_angle[valid] = 2.0 * (1.0 - cos2[valid])

    twist_sq = np.full_like(prod, np.nan)
    twist_sq[valid] = prod[valid] * twist_angle[valid]

    total_energy = curr_energy + past_energy
    energy_delta = curr_energy - past_energy

    phase_polarity = np.full_like(prod, np.nan)
    phase_polarity[valid] = np.where(dot_centered[valid] >= 0.0, 1.0, -1.0)

    poincare_b = np.full_like(prod, np.nan)
    poincare_b[valid] = dot_centered[valid] / curr_energy[valid]

    gram_spread_sq = np.full_like(prod, np.nan)
    gram_spread_sq[valid] = (
        np.square(energy_delta[valid]) + 4.0 * np.square(dot_centered[valid])
    )

    energy_asym = np.full_like(prod, np.nan)
    valid_total = valid & (total_energy > 1e-12)
    energy_asym[valid_total] = energy_delta[valid_total] / total_energy[valid_total]

    drift_centered_sq = np.full_like(prod, np.nan)
    drift_centered_sq[valid] = np.clip(
        curr_energy[valid] + past_energy[valid] - 2.0 * dot_centered[valid],
        0.0,
        None,
    )
    drift_norm = np.full_like(prod, np.nan)
    drift_norm[valid_total] = drift_centered_sq[valid_total] / total_energy[valid_total]

    out.loc[lag:, "curr_energy"] = curr_energy
    out.loc[lag:, "past_energy"] = past_energy
    out.loc[lag:, "dot_centered"] = dot_centered
    out.loc[lag:, "twist_angle"] = twist_angle
    out.loc[lag:, "twist_sq"] = twist_sq
    out.loc[lag:, "phase_polarity"] = phase_polarity
    out.loc[lag:, "poincare_b"] = poincare_b
    out.loc[lag:, "gram_spread_sq"] = gram_spread_sq
    out.loc[lag:, "energy_asym"] = energy_asym
    out.loc[lag:, "drift_norm"] = drift_norm
    return out


def detect_sustained_onsets(
    series: pd.Series,
    threshold: float,
    sustain_minutes: int,
    direction: str,
    min_gap_minutes: int,
) -> list[int]:
    if direction not in {"le", "ge"}:
        raise ValueError(f"Unsupported direction: {direction}")

    starts: list[int] = []
    run_length = 0
    last_start = -10**9
    flags = (
        series <= threshold if direction == "le" else series >= threshold
    ).fillna(False)

    for i, flag in enumerate(flags.tolist()):
        if flag:
            run_length += 1
            if run_length >= sustain_minutes:
                start = i - sustain_minutes + 1
                if start - last_start >= min_gap_minutes:
                    starts.append(start)
                    last_start = start
                run_length = 0
        else:
            run_length = 0

    return starts


def first_early_move(
    feature: pd.Series,
    event_idx: int,
    cfg: SolarConfig,
) -> tuple[pd.Timestamp | None, float | None]:
    baseline_start = max(0, event_idx - cfg.pre_event_minutes)
    baseline_end = max(baseline_start + 60, event_idx - cfg.baseline_minutes)
    detect_start = baseline_end
    detect_end = event_idx

    baseline = feature.iloc[baseline_start:baseline_end]
    detect = feature.iloc[detect_start:detect_end]

    median = float(baseline.median())
    mad = float((baseline - median).abs().median())
    scale = 1.4826 * mad if mad > 1e-12 else float(baseline.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        return None, None

    z = ((detect - median) / scale).abs()
    run_length = 0
    for timestamp, flag in zip(detect.index, (z > cfg.early_move_sigma).fillna(False)):
        run_length = run_length + 1 if flag else 0
        if run_length >= cfg.early_move_sustain_minutes:
            return timestamp, scale
    return None, scale


def minutes_between(later: pd.Timestamp | pd.NaT, earlier: pd.Timestamp | None) -> float | None:
    if earlier is None or pd.isna(earlier) or pd.isna(later):
        return None
    return float((later - earlier).total_seconds() / 60.0)


def first_pre_event_episode(
    event_idx: int,
    episodes_df: pd.DataFrame,
    pre_event_minutes: int,
) -> tuple[pd.Timestamp | None, float | None, int | None]:
    if episodes_df.empty:
        return None, None, None
    eligible = episodes_df.loc[
        (episodes_df["start_idx"] >= int(event_idx - pre_event_minutes))
        & (episodes_df["start_idx"] < int(event_idx))
    ].sort_values("start_idx")
    if eligible.empty:
        return None, None, None
    # For recurring alerts, the most recent pre-event onset is the actionable lead.
    best = eligible.iloc[-1]
    start_time = pd.Timestamp(best["start_time"])
    lead_minutes = float(event_idx - int(best["start_idx"]))
    return start_time, lead_minutes, int(best["episode_id"])


def build_event_table(
    df: pd.DataFrame,
    cfg: SolarConfig,
    hmm_episodes_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    indexed = df.set_index("time")
    if hmm_episodes_df is None:
        hmm_episodes_df = (
            build_episode_table(df["hmm_alert"].to_numpy(dtype=bool), df["time"], merge_gap=0)
            if "hmm_alert" in df.columns
            else pd.DataFrame(columns=["episode_id", "start_idx", "end_idx", "start_time", "end_time", "duration_minutes"])
        )

    storm_onsets = detect_sustained_onsets(
        indexed["sym_h"],
        threshold=cfg.storm_threshold,
        sustain_minutes=cfg.storm_sustain_minutes,
        direction="le",
        min_gap_minutes=cfg.storm_min_gap_minutes,
    )
    southward_onsets = detect_sustained_onsets(
        indexed["bz_gsm"],
        threshold=cfg.bz_south_threshold,
        sustain_minutes=cfg.bz_sustain_minutes,
        direction="le",
        min_gap_minutes=cfg.bz_sustain_minutes,
    )

    rows: list[dict[str, object]] = []
    for event_number, storm_idx in enumerate(storm_onsets, start=1):
        storm_time = indexed.index[storm_idx]
        prior_southward = [
            onset
            for onset in southward_onsets
            if onset < storm_idx and onset >= storm_idx - cfg.bz_lookback_minutes
        ]
        southward_idx = prior_southward[-1] if prior_southward else None
        southward_time = indexed.index[southward_idx] if southward_idx is not None else pd.NaT

        poincare_move_time, poincare_scale = first_early_move(
            indexed["poincare_b"], storm_idx, cfg
        )
        gram_move_time, gram_scale = first_early_move(
            indexed["gram_spread_sq"], storm_idx, cfg
        )
        hmm_move_time, hmm_lead_minutes, hmm_episode_id = first_pre_event_episode(
            event_idx=storm_idx,
            episodes_df=hmm_episodes_df,
            pre_event_minutes=cfg.pre_event_minutes,
        )

        feature_moves = {
            "poincare_b": poincare_move_time,
            "gram_spread_sq": gram_move_time,
        }
        valid_moves = {name: ts for name, ts in feature_moves.items() if ts is not None}
        lead_feature = min(valid_moves, key=valid_moves.get) if valid_moves else None
        lead_time = valid_moves[lead_feature] if lead_feature else pd.NaT

        combined_moves = {**feature_moves, "solar_hmm": hmm_move_time}
        valid_combined_moves = {name: ts for name, ts in combined_moves.items() if ts is not None}
        combined_lead_feature = min(valid_combined_moves, key=valid_combined_moves.get) if valid_combined_moves else None
        combined_lead_time = valid_combined_moves[combined_lead_feature] if combined_lead_feature else pd.NaT

        rows.append(
            {
                "record_id": storm_time.isoformat(),
                "domain": "solar",
                "entity_id": "geomagnetic_storm",
                "event_index": event_number,
                "regime": "magnetospheric_crumble",
                "angle_cross": np.nan,
                "long_cross": np.nan,
                "hybrid_cross": np.nan,
                "poincare_b": float(indexed.iloc[storm_idx]["poincare_b"])
                if pd.notna(indexed.iloc[storm_idx]["poincare_b"])
                else np.nan,
                "phase_polarity": float(indexed.iloc[storm_idx]["phase_polarity"])
                if pd.notna(indexed.iloc[storm_idx]["phase_polarity"])
                else np.nan,
                "gram_spread_sq": float(indexed.iloc[storm_idx]["gram_spread_sq"])
                if pd.notna(indexed.iloc[storm_idx]["gram_spread_sq"])
                else np.nan,
                "lead_feature": lead_feature,
                "lead_steps": minutes_between(storm_time, lead_time),
                "combined_lead_feature": combined_lead_feature,
                "combined_lead_steps": minutes_between(storm_time, combined_lead_time),
                "lead_time_units": "minutes",
                "storm_time": storm_time,
                "storm_threshold": cfg.storm_threshold,
                "storm_sustain_minutes": cfg.storm_sustain_minutes,
                "southward_bz_time": southward_time,
                "southward_bz_threshold": cfg.bz_south_threshold,
                "southward_bz_sustain_minutes": cfg.bz_sustain_minutes,
                "poincare_b_move_time": poincare_move_time,
                "gram_spread_sq_move_time": gram_move_time,
                "hmm_move_time": hmm_move_time,
                "hmm_episode_id": hmm_episode_id,
                "hmm_visible": int(hmm_move_time is not None),
                "poincare_b_scale": poincare_scale,
                "gram_spread_sq_scale": gram_scale,
                "poincare_b_lead_to_storm_min": minutes_between(storm_time, poincare_move_time),
                "gram_spread_sq_lead_to_storm_min": minutes_between(storm_time, gram_move_time),
                "hmm_lead_to_storm_min": hmm_lead_minutes,
                "poincare_b_lead_to_bz_min": minutes_between(southward_time, poincare_move_time),
                "gram_spread_sq_lead_to_bz_min": minutes_between(southward_time, gram_move_time),
                "hmm_lead_to_bz_min": minutes_between(southward_time, hmm_move_time),
                "bz_southward_lead_to_storm_min": minutes_between(storm_time, southward_time),
            }
        )

    return pd.DataFrame(rows)


def build_alignment_table(df: pd.DataFrame, events: pd.DataFrame, cfg: SolarConfig) -> pd.DataFrame:
    indexed = df.set_index("time")
    rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        storm_time = pd.Timestamp(event["storm_time"])
        start = storm_time - pd.Timedelta(minutes=cfg.pre_event_minutes)
        end = storm_time + pd.Timedelta(minutes=cfg.post_event_minutes)
        window = indexed.loc[start:end].copy()
        if window.empty:
            continue

        baseline = indexed.loc[
            storm_time - pd.Timedelta(minutes=cfg.pre_event_minutes) :
            storm_time - pd.Timedelta(minutes=cfg.baseline_minutes)
        ].copy()

        def normalize(feature_name: str) -> pd.Series:
            base = baseline[feature_name]
            med = base.median()
            mad = (base - med).abs().median()
            scale = 1.4826 * mad if pd.notna(mad) and mad > 1e-12 else base.std(ddof=0)
            if pd.isna(scale) or scale <= 1e-12:
                return pd.Series(np.nan, index=window.index)
            return (window[feature_name] - med) / scale

        aligned = pd.DataFrame(
            {
                "event_index": int(event["event_index"]),
                "storm_time": storm_time,
                "minutes_from_storm": (
                    (window.index - storm_time).total_seconds() / 60.0
                ).astype(int),
                "poincare_b_z": normalize("poincare_b").to_numpy(),
                "gram_spread_sq_z": normalize("gram_spread_sq").to_numpy(),
                "memory_align_z": normalize("memory_align").to_numpy() if "memory_align" in window.columns else np.nan,
                "novelty_z": normalize("novelty").to_numpy() if "novelty" in window.columns else np.nan,
                "proj_lock_barrier_sl_z": normalize("proj_lock_barrier_sl").to_numpy() if "proj_lock_barrier_sl" in window.columns else np.nan,
                "transition_score_z": normalize("transition_score").to_numpy() if "transition_score" in window.columns else np.nan,
                "bz_gsm_z": normalize("bz_gsm").to_numpy(),
                "sym_h_z": normalize("sym_h").to_numpy(),
                "bz_gsm": window["bz_gsm"].to_numpy(),
                "sym_h": window["sym_h"].to_numpy(),
                "hmm_state_code": window["hmm_state_code"].to_numpy() if "hmm_state_code" in window.columns else np.nan,
                "hmm_alert": window["hmm_alert"].astype(int).to_numpy() if "hmm_alert" in window.columns else 0,
            }
        )
        rows.extend(aligned.to_dict(orient="records"))

    return pd.DataFrame(rows)


def build_year_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "n_events",
                "n_poincare_before_storm",
                "n_gram_before_storm",
                "n_hmm_before_storm",
                "median_poincare_lead_to_storm_min",
                "median_gram_lead_to_storm_min",
                "median_hmm_lead_to_storm_min",
                "median_bz_southward_lead_to_storm_min",
                "top_lead_feature",
                "top_combined_lead_feature",
            ]
        )

    work = events_df.copy()
    work["year"] = pd.to_datetime(work["storm_time"]).dt.year
    summary = (
        work.groupby("year")
        .agg(
            n_events=("event_index", "count"),
            n_poincare_before_storm=("poincare_b_lead_to_storm_min", lambda s: int((pd.Series(s).fillna(-1) > 0).sum())),
            n_gram_before_storm=("gram_spread_sq_lead_to_storm_min", lambda s: int((pd.Series(s).fillna(-1) > 0).sum())),
            n_hmm_before_storm=("hmm_lead_to_storm_min", lambda s: int((pd.Series(s).fillna(-1) > 0).sum())),
            median_poincare_lead_to_storm_min=("poincare_b_lead_to_storm_min", "median"),
            median_gram_lead_to_storm_min=("gram_spread_sq_lead_to_storm_min", "median"),
            median_hmm_lead_to_storm_min=("hmm_lead_to_storm_min", "median"),
            median_bz_southward_lead_to_storm_min=("bz_southward_lead_to_storm_min", "median"),
        )
        .reset_index()
    )
    lead_mode = (
        work.groupby(["year", "lead_feature"])
        .size()
        .reset_index(name="count")
        .sort_values(["year", "count", "lead_feature"], ascending=[True, False, True])
        .drop_duplicates(subset=["year"])
        .rename(columns={"lead_feature": "top_lead_feature"})
    )
    combined_lead_mode = (
        work.groupby(["year", "combined_lead_feature"])
        .size()
        .reset_index(name="count")
        .sort_values(["year", "count", "combined_lead_feature"], ascending=[True, False, True])
        .drop_duplicates(subset=["year"])
        .rename(columns={"combined_lead_feature": "top_combined_lead_feature"})
    )
    return (
        summary
        .merge(lead_mode[["year", "top_lead_feature"]], on="year", how="left")
        .merge(combined_lead_mode[["year", "top_combined_lead_feature"]], on="year", how="left")
    )


def build_source_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=["source_file", "n_rows", "start_time", "end_time"])
    return (
        raw_df.groupby("source_file")
        .agg(
            n_rows=("time", "size"),
            start_time=("time", "min"),
            end_time=("time", "max"),
        )
        .reset_index()
        .sort_values("start_time")
        .reset_index(drop=True)
    )


def build_layered_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "n_events",
        "n_poincare_before_storm",
        "n_gram_before_storm",
        "n_hmm_before_storm",
        "n_poincare_before_bz",
        "n_gram_before_bz",
        "n_hmm_before_bz",
        "n_hmm_chosen_as_combined_lead",
        "median_poincare_lead_to_storm_min",
        "median_gram_lead_to_storm_min",
        "median_hmm_lead_to_storm_min",
        "median_bz_southward_lead_to_storm_min",
    ]
    if events_df.empty:
        return pd.DataFrame(columns=columns)

    row = {
        "n_events": int(len(events_df)),
        "n_poincare_before_storm": int((events_df["poincare_b_lead_to_storm_min"].fillna(-1) > 0).sum()),
        "n_gram_before_storm": int((events_df["gram_spread_sq_lead_to_storm_min"].fillna(-1) > 0).sum()),
        "n_hmm_before_storm": int((events_df["hmm_lead_to_storm_min"].fillna(-1) > 0).sum()),
        "n_poincare_before_bz": int((events_df["poincare_b_lead_to_bz_min"].fillna(-1) > 0).sum()),
        "n_gram_before_bz": int((events_df["gram_spread_sq_lead_to_bz_min"].fillna(-1) > 0).sum()),
        "n_hmm_before_bz": int((events_df["hmm_lead_to_bz_min"].fillna(-1) > 0).sum()),
        "n_hmm_chosen_as_combined_lead": int((events_df["combined_lead_feature"] == "solar_hmm").sum()),
        "median_poincare_lead_to_storm_min": float(events_df["poincare_b_lead_to_storm_min"].median()),
        "median_gram_lead_to_storm_min": float(events_df["gram_spread_sq_lead_to_storm_min"].median()),
        "median_hmm_lead_to_storm_min": float(events_df["hmm_lead_to_storm_min"].median()),
        "median_bz_southward_lead_to_storm_min": float(events_df["bz_southward_lead_to_storm_min"].median()),
    }
    return pd.DataFrame([row], columns=columns)


def build_hmm_confusion_outputs(
    feature_df: pd.DataFrame,
    events_df: pd.DataFrame,
    cfg: SolarConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix_columns = ["predicted_negative", "predicted_positive"]
    if feature_df.empty:
        empty_matrix = pd.DataFrame(
            [[0, 0], [0, 0]],
            index=["actual_negative", "actual_positive"],
            columns=matrix_columns,
        )
        empty_metrics = pd.DataFrame(
            [
                {
                    "tp_minutes": 0,
                    "fp_minutes": 0,
                    "tn_minutes": 0,
                    "fn_minutes": 0,
                    "sensitivity": np.nan,
                    "specificity": np.nan,
                    "precision": np.nan,
                    "alert_occupancy_fraction": np.nan,
                }
            ]
        )
        return empty_matrix, empty_metrics

    times = pd.to_datetime(feature_df["time"]).reset_index(drop=True)
    positive_mask = np.zeros(len(feature_df), dtype=bool)
    if not events_df.empty:
        for storm_time in pd.to_datetime(events_df["storm_time"]):
            start = storm_time - pd.Timedelta(minutes=cfg.pre_event_minutes)
            end = storm_time + pd.Timedelta(minutes=cfg.post_event_minutes)
            positive_mask |= (times >= start) & (times <= end)

    predicted_mask = feature_df["hmm_alert"].fillna(False).astype(bool).to_numpy()
    tp = int(np.sum(predicted_mask & positive_mask))
    fp = int(np.sum(predicted_mask & ~positive_mask))
    tn = int(np.sum(~predicted_mask & ~positive_mask))
    fn = int(np.sum(~predicted_mask & positive_mask))

    matrix = pd.DataFrame(
        [
            [tn, fp],
            [fn, tp],
        ],
        index=["actual_negative", "actual_positive"],
        columns=matrix_columns,
    )
    sens_denom = tp + fn
    spec_denom = tn + fp
    prec_denom = tp + fp
    metrics = pd.DataFrame(
        [
            {
                "tp_minutes": tp,
                "fp_minutes": fp,
                "tn_minutes": tn,
                "fn_minutes": fn,
                "sensitivity": float(tp / sens_denom) if sens_denom > 0 else np.nan,
                "specificity": float(tn / spec_denom) if spec_denom > 0 else np.nan,
                "precision": float(tp / prec_denom) if prec_denom > 0 else np.nan,
                "alert_occupancy_fraction": float(predicted_mask.mean()) if len(predicted_mask) else np.nan,
            }
        ]
    )
    return matrix, metrics


def make_pooled_plot(alignment_df: pd.DataFrame, output_path: Path) -> None:
    if alignment_df.empty:
        return

    pooled = (
        alignment_df.groupby("minutes_from_storm")
        .agg(
            poincare_b_z=("poincare_b_z", "median"),
            gram_spread_sq_z=("gram_spread_sq_z", "median"),
            transition_score_z=("transition_score_z", "median"),
            bz_gsm_z=("bz_gsm_z", "median"),
            sym_h_z=("sym_h_z", "median"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        pooled["minutes_from_storm"],
        pooled["poincare_b_z"],
        label="poincare_b (median z)",
        color="#9467bd",
    )
    ax.plot(
        pooled["minutes_from_storm"],
        pooled["gram_spread_sq_z"],
        label="gram_spread_sq (median z)",
        color="#2ca02c",
    )
    ax.plot(
        pooled["minutes_from_storm"],
        pooled["transition_score_z"],
        label="transition_score (median z)",
        color="#ff7f0e",
    )
    ax.plot(
        pooled["minutes_from_storm"],
        pooled["bz_gsm_z"],
        label="Bz_GSM (median z)",
        color="#1f77b4",
    )
    ax.plot(
        pooled["minutes_from_storm"],
        pooled["sym_h_z"],
        label="SYM_H (median z)",
        color="#d62728",
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Solar Wind Router Alignment Around Storm Onset")
    ax.set_xlabel("Minutes from storm onset")
    ax.set_ylabel("Median baseline-normalized deviation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def write_report(
    run_dir: Path,
    raw_df: pd.DataFrame,
    events_df: pd.DataFrame,
    year_summary_df: pd.DataFrame,
    hmm_confusion_df: pd.DataFrame,
    hmm_confusion_metrics_df: pd.DataFrame,
    source_summary_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    cfg: SolarConfig,
) -> None:
    time_min = pd.to_datetime(raw_df["time"]).min() if not raw_df.empty else pd.NaT
    time_max = pd.to_datetime(raw_df["time"]).max() if not raw_df.empty else pd.NaT
    lines = [
        "# Solar Wind Router Report",
        "",
        "## Setup",
        f"- Cache root: `{cfg.cache_dir}`",
        f"- File glob: `{cfg.file_glob}`",
        f"- Recursive search: `{cfg.recursive}`",
        f"- Time filter start: `{cfg.start_time}`",
        f"- Time filter end: `{cfg.end_time}`",
        f"- Write feature CSV: `{cfg.write_feature_csv}`",
        f"- Write alignment CSV: `{cfg.write_alignment_csv}`",
        f"- Source files loaded: `{source_summary_df['source_file'].nunique() if not source_summary_df.empty else 0}`",
        f"- Loaded interval start: `{time_min}`",
        f"- Loaded interval end: `{time_max}`",
        f"- Input state: `BZ_GSM`, `flow_speed`, `proton_density`",
        f"- Event index: `SYM_H <= {cfg.storm_threshold:g}` for {cfg.storm_sustain_minutes} consecutive minutes",
        f"- Southward Bz index: `BZ_GSM <= {cfg.bz_south_threshold:g}` for {cfg.bz_sustain_minutes} consecutive minutes",
        f"- Router lag: {cfg.lag_minutes} minutes",
        f"- Early move: baseline-normalized deviation > {cfg.early_move_sigma:g} sigma for {cfg.early_move_sustain_minutes} consecutive minutes",
        f"- Projective memory betas: short=`{cfg.beta_short:g}`, long=`{cfg.beta_long:g}`",
        f"- HMM alert extraction: transition onset + `{cfg.hmm_alert_active_tail_minutes}`-minute active tail, capped at `{cfg.hmm_alert_max_minutes}` minutes",
        f"- HMM merge/cooldown: merge gap `{cfg.hmm_merge_gap_minutes}` minutes, cooldown `{cfg.hmm_alert_cooldown_minutes}` minutes",
        f"- HMM confusion target: minutes inside `[storm_time - {cfg.pre_event_minutes} min, storm_time + {cfg.post_event_minutes} min]`",
        "",
    ]

    if events_df.empty:
        lines.extend(
            [
                "## Result",
                "- No storm onsets matched the configured thresholds in the cached OMNI interval.",
            ]
        )
    else:
        lines.extend(
            [
                "## Event Lead Table",
                events_df[
                    [
                        "event_index",
                        "storm_time",
                        "southward_bz_time",
                        "lead_feature",
                        "lead_steps",
                        "combined_lead_feature",
                        "combined_lead_steps",
                        "poincare_b_lead_to_storm_min",
                        "gram_spread_sq_lead_to_storm_min",
                        "hmm_lead_to_storm_min",
                        "poincare_b_lead_to_bz_min",
                        "gram_spread_sq_lead_to_bz_min",
                        "hmm_lead_to_bz_min",
                        "bz_southward_lead_to_storm_min",
                    ]
                ].to_string(index=False),
                "",
            ]
        )

        if not year_summary_df.empty:
            lines.extend(
                [
                    "## Annual Summary",
                    year_summary_df.to_string(index=False),
                    "",
                ]
            )

        if not hmm_confusion_df.empty and not hmm_confusion_metrics_df.empty:
            lines.extend(
                [
                    "## HMM Confusion Matrix",
                    hmm_confusion_df.to_string(),
                    "",
                    "## HMM Detection Metrics",
                    hmm_confusion_metrics_df.to_string(index=False),
                    "",
                ]
            )

        lead_counts = {
            "poincare_before_storm": int((events_df["poincare_b_lead_to_storm_min"].fillna(-1) > 0).sum()),
            "gram_before_storm": int((events_df["gram_spread_sq_lead_to_storm_min"].fillna(-1) > 0).sum()),
            "hmm_before_storm": int((events_df["hmm_lead_to_storm_min"].fillna(-1) > 0).sum()),
            "poincare_before_bz": int((events_df["poincare_b_lead_to_bz_min"].fillna(-1) > 0).sum()),
            "gram_before_bz": int((events_df["gram_spread_sq_lead_to_bz_min"].fillna(-1) > 0).sum()),
            "hmm_before_bz": int((events_df["hmm_lead_to_bz_min"].fillna(-1) > 0).sum()),
            "combined_hmm_first": int((events_df["combined_lead_feature"] == "solar_hmm").sum()),
            "n_events": int(len(events_df)),
        }
        lines.extend(
            [
                "## Summary",
                f"- `poincare_b` moved before storm onset on {lead_counts['poincare_before_storm']}/{lead_counts['n_events']} events.",
                f"- `gram_spread_sq` moved before storm onset on {lead_counts['gram_before_storm']}/{lead_counts['n_events']} events.",
                f"- `solar_hmm` moved before storm onset on {lead_counts['hmm_before_storm']}/{lead_counts['n_events']} events.",
                f"- `poincare_b` moved before the paired southward-Bz onset on {lead_counts['poincare_before_bz']}/{lead_counts['n_events']} events.",
                f"- `gram_spread_sq` moved before the paired southward-Bz onset on {lead_counts['gram_before_bz']}/{lead_counts['n_events']} events.",
                f"- `solar_hmm` moved before the paired southward-Bz onset on {lead_counts['hmm_before_bz']}/{lead_counts['n_events']} events.",
                f"- Combined layered detector chose `solar_hmm` as the earliest lead on {lead_counts['combined_hmm_first']}/{lead_counts['n_events']} events.",
                f"- Full feature CSV written: `{cfg.write_feature_csv}`",
                f"- Full alignment CSV written: `{cfg.write_alignment_csv}`",
            ]
        )

    (run_dir / "solar_report.md").write_text("\n".join(lines), encoding="utf-8")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(cfg).items()
        },
        "n_events": int(len(events_df)),
        "alignment_rows": int(len(alignment_df)),
        "n_hmm_visible_events": int((events_df["hmm_visible"].sum())) if "hmm_visible" in events_df.columns else 0,
        "hmm_tp_minutes": int(hmm_confusion_metrics_df.iloc[0]["tp_minutes"]) if not hmm_confusion_metrics_df.empty else 0,
        "hmm_fp_minutes": int(hmm_confusion_metrics_df.iloc[0]["fp_minutes"]) if not hmm_confusion_metrics_df.empty else 0,
        "n_source_files": int(source_summary_df["source_file"].nunique()) if not source_summary_df.empty else 0,
        "loaded_interval_start": None if pd.isna(time_min) else time_min.isoformat(),
        "loaded_interval_end": None if pd.isna(time_max) else time_max.isoformat(),
    }
    (run_dir / "solar_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def run_solar_router(cfg: SolarConfig) -> dict[str, object]:
    run_dir = prepare_run_directory(cfg.run_root)
    log(cfg, f"Run directory: {run_dir}")
    raw_df = load_omni_cache(cfg)
    clean_df = fill_and_standardize(raw_df, cfg)
    log(cfg, f"Filled and standardized {len(clean_df)} rows")
    feature_df = compute_router_features(clean_df, cfg)
    log(cfg, "Computed router features")
    feature_df, hmm_episodes_df = enrich_projective_hmm_features(feature_df, cfg)
    log(cfg, f"Annotated projective features and built {len(hmm_episodes_df)} HMM alert episodes")
    events_df = build_event_table(feature_df, cfg, hmm_episodes_df=hmm_episodes_df)
    log(cfg, f"Detected {len(events_df)} storm events")
    alignment_df = build_alignment_table(feature_df, events_df, cfg)
    log(cfg, f"Built alignment table with {len(alignment_df)} rows")
    year_summary_df = build_year_summary(events_df)
    layered_summary_df = build_layered_summary(events_df)
    hmm_confusion_df, hmm_confusion_metrics_df = build_hmm_confusion_outputs(feature_df, events_df, cfg)
    layered_summary_df = pd.concat(
        [layered_summary_df.reset_index(drop=True), hmm_confusion_metrics_df.reset_index(drop=True)],
        axis=1,
    )
    source_summary_df = build_source_summary(raw_df)

    feature_path = run_dir / "solar_timeseries_features.csv"
    events_path = run_dir / "solar_event_table.csv"
    hmm_episodes_path = run_dir / "solar_hmm_episode_table.csv"
    layered_summary_path = run_dir / "solar_layered_summary.csv"
    hmm_confusion_path = run_dir / "solar_hmm_confusion_matrix.csv"
    alignment_path = run_dir / "solar_alignment.csv"
    year_summary_path = run_dir / "solar_year_summary.csv"
    source_summary_path = run_dir / "solar_source_summary.csv"
    plot_path = run_dir / "solar_alignment_plot.png"

    if cfg.write_feature_csv:
        log(cfg, "Writing feature timeseries CSV")
        feature_df.to_csv(feature_path, index=False)
    events_df.to_csv(events_path, index=False)
    hmm_episodes_df.to_csv(hmm_episodes_path, index=False)
    layered_summary_df.to_csv(layered_summary_path, index=False)
    hmm_confusion_df.to_csv(hmm_confusion_path)
    if cfg.write_alignment_csv:
        log(cfg, "Writing alignment CSV")
        alignment_df.to_csv(alignment_path, index=False)
    year_summary_df.to_csv(year_summary_path, index=False)
    source_summary_df.to_csv(source_summary_path, index=False)
    make_pooled_plot(alignment_df, plot_path)
    write_report(
        run_dir,
        raw_df,
        events_df,
        year_summary_df,
        hmm_confusion_df,
        hmm_confusion_metrics_df,
        source_summary_df,
        alignment_df,
        cfg,
    )
    log(cfg, "Wrote summary outputs to disk")

    return {
        "run_dir": run_dir,
        "raw_df": raw_df,
        "feature_df": feature_df,
        "events_df": events_df,
        "hmm_episodes_df": hmm_episodes_df,
        "hmm_confusion_df": hmm_confusion_df,
        "hmm_confusion_metrics_df": hmm_confusion_metrics_df,
        "layered_summary_df": layered_summary_df,
        "year_summary_df": year_summary_df,
        "source_summary_df": source_summary_df,
        "alignment_df": alignment_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the solar-wind router on OMNI HRO2 cache files.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="OMNI cache directory.")
    parser.add_argument("--run-root", type=Path, default=None, help="Artifact run root directory.")
    parser.add_argument("--file-glob", type=str, default="omni_hro2_1min_*.cdf")
    parser.add_argument("--non-recursive", action="store_true", help="Disable recursive CDF search under the cache directory.")
    parser.add_argument("--start-time", type=str, default=None, help="Optional inclusive start timestamp.")
    parser.add_argument("--end-time", type=str, default=None, help="Optional inclusive end timestamp.")
    parser.add_argument("--write-feature-csv", action="store_true", help="Write the full minute-level feature CSV.")
    parser.add_argument("--write-alignment-csv", action="store_true", help="Write the full alignment CSV.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging.")
    parser.add_argument("--lag-minutes", type=int, default=60)
    parser.add_argument("--storm-threshold", type=float, default=-40.0)
    parser.add_argument("--storm-sustain-minutes", type=int, default=30)
    parser.add_argument("--storm-min-gap-minutes", type=int, default=720)
    parser.add_argument("--bz-south-threshold", type=float, default=-5.0)
    parser.add_argument("--bz-sustain-minutes", type=int, default=15)
    parser.add_argument("--bz-lookback-minutes", type=int, default=1440)
    parser.add_argument("--baseline-minutes", type=int, default=720)
    parser.add_argument("--pre-event-minutes", type=int, default=1440)
    parser.add_argument("--post-event-minutes", type=int, default=360)
    parser.add_argument("--early-move-sigma", type=float, default=1.5)
    parser.add_argument("--early-move-sustain-minutes", type=int, default=3)
    parser.add_argument("--interpolation-limit-minutes", type=int, default=30)
    parser.add_argument("--beta-short", type=float, default=0.10)
    parser.add_argument("--beta-long", type=float, default=0.01)
    parser.add_argument("--projective-min-state-energy", type=float, default=1e-8)
    parser.add_argument("--hmm-rolling-window-minutes", type=int, default=15)
    parser.add_argument("--hmm-merge-gap-minutes", type=int, default=15)
    parser.add_argument("--hmm-alert-active-tail-minutes", type=int, default=30)
    parser.add_argument("--hmm-alert-max-minutes", type=int, default=180)
    parser.add_argument("--hmm-alert-cooldown-minutes", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    defaults = default_config()
    args = parse_args()
    cfg = SolarConfig(
        cache_dir=args.cache_dir or defaults.cache_dir,
        run_root=args.run_root or defaults.run_root,
        file_glob=args.file_glob,
        recursive=not bool(args.non_recursive),
        start_time=args.start_time,
        end_time=args.end_time,
        verbose=not bool(args.quiet),
        write_feature_csv=bool(args.write_feature_csv),
        write_alignment_csv=bool(args.write_alignment_csv),
        lag_minutes=args.lag_minutes,
        interpolation_limit_minutes=args.interpolation_limit_minutes,
        storm_threshold=args.storm_threshold,
        storm_sustain_minutes=args.storm_sustain_minutes,
        storm_min_gap_minutes=args.storm_min_gap_minutes,
        bz_south_threshold=args.bz_south_threshold,
        bz_sustain_minutes=args.bz_sustain_minutes,
        bz_lookback_minutes=args.bz_lookback_minutes,
        baseline_minutes=args.baseline_minutes,
        pre_event_minutes=args.pre_event_minutes,
        post_event_minutes=args.post_event_minutes,
        early_move_sigma=args.early_move_sigma,
        early_move_sustain_minutes=args.early_move_sustain_minutes,
        beta_short=args.beta_short,
        beta_long=args.beta_long,
        projective_min_state_energy=args.projective_min_state_energy,
        hmm_rolling_window_minutes=args.hmm_rolling_window_minutes,
        hmm_merge_gap_minutes=args.hmm_merge_gap_minutes,
        hmm_alert_active_tail_minutes=args.hmm_alert_active_tail_minutes,
        hmm_alert_max_minutes=args.hmm_alert_max_minutes,
        hmm_alert_cooldown_minutes=args.hmm_alert_cooldown_minutes,
    )

    result = run_solar_router(cfg)
    run_dir = Path(result["run_dir"])
    events_df = result["events_df"]

    print(f"Run directory: {run_dir}")
    raw_df = result["raw_df"]
    print(f"Source rows loaded: {len(raw_df)}")
    if not raw_df.empty:
        print(f"Time span: {raw_df['time'].min()} -> {raw_df['time'].max()}")
        print(f"Source files: {raw_df['source_file'].nunique()}")
    print(f"Events detected: {len(events_df)}")
    if not events_df.empty:
        cols = [
            "event_index",
            "storm_time",
            "southward_bz_time",
            "lead_feature",
            "lead_steps",
            "combined_lead_feature",
            "combined_lead_steps",
            "poincare_b_lead_to_bz_min",
            "gram_spread_sq_lead_to_bz_min",
            "hmm_lead_to_bz_min",
        ]
        print(events_df[cols].to_string(index=False))
        year_summary_df = result["year_summary_df"]
        if not year_summary_df.empty:
            print()
            print(year_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
