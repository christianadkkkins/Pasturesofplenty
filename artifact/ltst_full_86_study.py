from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import wfdb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_lie import (
    CANONICAL_LIE_FEATURE_COLUMNS,
    compute_lie_state_features,
    compute_projective_state_features,
    ema_prior_states,
)


warnings.filterwarnings("ignore")

EPS = 1e-12
FLAG_OK = 0
FLAG_HISTORY_NOT_READY = 0x01
FLAG_RR_OUT_OF_RANGE = 0x02
FLAG_LOW_ENERGY = 0x04


@dataclass(frozen=True)
class LTSTConfig:
    db: str = "ltstdb"
    beat_ext: str = "atr"
    st_ext: str = "stc"
    records: tuple[str, ...] = ()
    horizons: tuple[int, ...] = (30, 60, 120, 250, 500, 750, 1000)
    top_n: int = 1102
    crossover_min_delta: float = 0.02
    dim: int = 9
    lag: int = 4
    max_beat_offset: int = 320
    min_energy_thresh: float = 1e-6
    alpha_slow: float = 0.2
    alpha_fast: float = 0.6
    beta_short: float = 0.10
    beta_long: float = 0.01
    beat_window_pre: int = 80
    beat_window_post: int = 120
    local_template_radius: int = 10
    normal_beat_symbols: tuple[str, ...] = ("N",)
    checkpoint_every: int = 5
    save_beat_level: bool = False
    run_simple_router: bool = True
    run_guarded_router: bool = True
    max_records: int | None = None
    run_root: Path = Path("artifact") / "runs"


def find_project_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "artifact").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root containing artifact/.")


def prepare_run_directory(base_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("ltst_full_86_%Y%m%dT%H%M%SZ")
    run_dir = base_root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def discover_records(db: str) -> list[str]:
    if hasattr(wfdb, "get_record_list"):
        records = wfdb.get_record_list(db)
        if records:
            return sorted(str(r) for r in records)
    if hasattr(wfdb, "io") and hasattr(wfdb.io, "download"):
        records = wfdb.io.download.get_record_list(db)
        if records:
            return sorted(str(r) for r in records)
    raise RuntimeError(f"Unable to discover records for PhysioNet database '{db}'.")


def parse_records_arg(records_arg: str | None, db: str) -> list[str]:
    if records_arg is None or records_arg.strip().lower() in {"", "all", "*"}:
        return discover_records(db)
    return [piece.strip() for piece in records_arg.split(",") if piece.strip()]


def ensure_remote_access(record: str, cfg: LTSTConfig) -> str:
    print(f"Preparing remote access for {record} [{cfg.st_ext}] via pn_dir='{cfg.db}'", flush=True)
    return cfg.db


def save_frame(df: pd.DataFrame, base_path: Path) -> None:
    df.to_csv(base_path.with_suffix(".csv"), index=False)
    try:
        df.to_parquet(base_path.with_suffix(".parquet"), index=False)
    except Exception:
        pass


def load_st_intervals(
    record: str,
    ext: str,
    pn_dir: str,
) -> tuple[list[tuple[int, int]] | None, dict[str, float] | None, str | None]:
    try:
        ann = wfdb.rdann(record, ext, pn_dir=pn_dir)
    except Exception as exc:
        return None, None, f"missing_annotation: {exc}"

    samples = np.asarray(ann.sample, dtype=int)
    aux = np.asarray(
        ann.aux_note if getattr(ann, "aux_note", None) is not None else [""] * len(samples),
        dtype=object,
    ).astype(str)

    intervals: list[tuple[int, int]] = []
    current_start: int | None = None
    found_explicit = False

    for sample, note in zip(samples, aux):
        text = (note or "").lower().strip()
        if any(tok in text for tok in ["begin", "start", "onset"]):
            current_start = int(sample)
            found_explicit = True
            continue
        if any(tok in text for tok in ["end", "stop", "offset"]):
            if current_start is not None and sample > current_start:
                intervals.append((current_start, int(sample)))
            current_start = None
            found_explicit = True
            continue
        if ("grst0" in text) or ("lrst0" in text) or ("(rtst0" in text) or ("(rtst1" in text):
            current_start = int(sample)
            found_explicit = True
            continue
        if ("grst1" in text) or ("lrst1" in text) or text == ")":
            if current_start is not None and sample > current_start:
                intervals.append((current_start, int(sample)))
            current_start = None
            found_explicit = True
            continue

    if not found_explicit:
        for i in range(0, len(samples) - 1, 2):
            start, end = int(samples[i]), int(samples[i + 1])
            if end > start:
                intervals.append((start, end))

    if not intervals:
        return None, None, "no_intervals_parsed"

    durations = [end - start for start, end in intervals]
    diag = {
        "n_intervals": float(len(intervals)),
        "mean_interval_samples": float(np.mean(durations)) if durations else np.nan,
    }
    return intervals, diag, None


def attach_st(df: pd.DataFrame, intervals: Sequence[tuple[int, int]]) -> pd.DataFrame:
    out = df.copy()
    inside_any = np.zeros(len(out), dtype=bool)
    episode_id = np.full(len(out), -1, dtype=int)
    samples = out["beat_sample"].to_numpy(dtype=int)

    for idx, (start, end) in enumerate(intervals):
        inside = (samples >= start) & (samples < end)
        inside_any |= inside
        episode_id[inside] = idx

    out["st_event"] = inside_any
    out["st_episode_id"] = episode_id
    return out


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) == 0:
        return np.nan
    aa = a.astype(float) - np.nanmean(a)
    bb = b.astype(float) - np.nanmean(b)
    da = np.sqrt(np.nansum(aa * aa))
    db = np.sqrt(np.nansum(bb * bb))
    if da < EPS or db < EPS:
        return np.nan
    return float(np.nansum(aa * bb) / (da * db))


def extract_window(signal: np.ndarray, center: int, cfg: LTSTConfig) -> np.ndarray:
    lo = max(0, int(center) - cfg.beat_window_pre)
    hi = min(len(signal), int(center) + cfg.beat_window_post)
    window = signal[lo:hi]
    target_len = cfg.beat_window_pre + cfg.beat_window_post
    if len(window) < target_len:
        window = np.concatenate([window, np.full(target_len - len(window), np.nan)])
    return window


def add_local_template_features(df: pd.DataFrame, signal: np.ndarray, cfg: LTSTConfig) -> pd.DataFrame:
    windows = [extract_window(signal, bs, cfg) for bs in df["beat_sample"].astype(int).tolist()]
    symbols = df["beat_symbol"].astype(str).to_numpy(dtype=object)
    template_corr: list[float] = []

    for i, window in enumerate(windows):
        window_fill = np.nan_to_num(window, nan=np.nanmean(window) if np.isfinite(np.nanmean(window)) else 0.0)
        lo = max(0, i - cfg.local_template_radius)
        hi = min(len(windows), i + cfg.local_template_radius + 1)
        pool_idx = [j for j in range(lo, hi) if j != i and symbols[j] in set(cfg.normal_beat_symbols)]
        if len(pool_idx) >= 3:
            template = np.nanmean(np.vstack([windows[j] for j in pool_idx]), axis=0)
            template_fill = np.nan_to_num(template, nan=np.nanmean(template) if np.isfinite(np.nanmean(template)) else 0.0)
            template_corr.append(safe_corr(window_fill, template_fill))
        else:
            template_corr.append(np.nan)

    out = df.copy()
    out["local_template_corr"] = template_corr
    return out


def append_shared_geometry_features(df: pd.DataFrame, x_center: np.ndarray, cfg: LTSTConfig) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    if len(out) == 0:
        return out
    if len(x_center) != len(out):
        raise ValueError("x_center rows must align with the beat-level frame.")

    ms_prior = ema_prior_states(x_center, beta=cfg.beta_short)
    ml_prior = ema_prior_states(x_center, beta=cfg.beta_long)
    projective = compute_projective_state_features(
        x_center,
        ms_prior=ms_prior,
        ml_prior=ml_prior,
        min_state_energy=cfg.min_energy_thresh,
    )
    lie = compute_lie_state_features(
        x_center,
        ms_prior=ms_prior,
        ml_prior=ml_prior,
    )
    for name, values in {**projective, **lie}.items():
        out[name] = values
    return out


def build_beat_level_with_invariants(record: str, cfg: LTSTConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    rec_dir = ensure_remote_access(record, cfg)
    rec = wfdb.rdrecord(record, pn_dir=rec_dir)
    ann = wfdb.rdann(record, cfg.beat_ext, pn_dir=rec_dir)
    signal = rec.p_signal[:, 0].astype(np.float64)

    beat_df = pd.DataFrame(
        {
            "beat_sample": np.asarray(ann.sample, dtype=int),
            "beat_symbol": np.asarray(ann.symbol, dtype=object).astype(str),
        }
    ).reset_index(drop=True)
    beat_df["rr_samples"] = beat_df["beat_sample"].diff()
    beat_df.loc[0, "rr_samples"] = np.nan

    state_buffer_size = 1 + cfg.max_beat_offset + (cfg.dim - 1) * cfg.lag
    history = np.full(state_buffer_size, np.nan, dtype=np.float64)
    head_idx = 0
    samples_seen = 0
    history_ready = False
    last_twist_sq = 0.0
    last_twist_angle = 0.0
    ema_slow = 0.0
    ema_fast = 0.0
    has_history = False

    def wrap_index(idx: int) -> int:
        idx %= state_buffer_size
        return idx if idx >= 0 else idx + state_buffer_size

    def push(value: float) -> None:
        nonlocal head_idx, samples_seen, history_ready
        history[head_idx] = float(value)
        head_idx = (head_idx + 1) % state_buffer_size
        samples_seen += 1
        if samples_seen >= state_buffer_size:
            history_ready = True

    def get_delayed(steps_back: int) -> float:
        return float(history[wrap_index(head_idx - 1 - steps_back)])

    beat_lookup = {int(s): i for i, s in enumerate(beat_df["beat_sample"].astype(int).tolist())}
    rows: list[dict[str, object]] = []
    x_center_rows: list[np.ndarray] = []

    for sample_idx, value in enumerate(signal):
        push(float(value))
        if sample_idx not in beat_lookup:
            continue

        row = beat_df.iloc[beat_lookup[sample_idx]]
        rr = row["rr_samples"]
        if (not history_ready) or pd.isna(rr) or rr <= 0 or (int(rr) + (cfg.dim - 1) * cfg.lag >= state_buffer_size):
            rows.append(
                {
                    "record": record,
                    "beat_sample": int(row["beat_sample"]),
                    "beat_symbol": str(row["beat_symbol"]),
                    "rr_samples": rr,
                    "status": FLAG_RR_OUT_OF_RANGE,
                }
            )
            continue

        x_curr = np.array([get_delayed(i * cfg.lag) for i in range(cfg.dim)], dtype=np.float64)
        x_past = np.array([get_delayed(int(rr) + i * cfg.lag) for i in range(cfg.dim)], dtype=np.float64)
        x_curr_center = x_curr - float(np.mean(x_curr))

        sx = sy = sxx = syy = sxy = 0.0
        for vx, vy in zip(x_curr, x_past):
            sx += vx
            sy += vy
            sxx += vx * vx
            syy += vy * vy
            sxy += vx * vy

        inv_d = 1.0 / float(cfg.dim)
        curr_energy = sxx - (sx * sx) * inv_d
        past_energy = syy - (sy * sy) * inv_d
        dot_centered = sxy - (sx * sy) * inv_d

        if curr_energy < cfg.min_energy_thresh or past_energy < cfg.min_energy_thresh:
            rows.append(
                {
                    "record": record,
                    "beat_sample": int(row["beat_sample"]),
                    "beat_symbol": str(row["beat_symbol"]),
                    "rr_samples": rr,
                    "status": FLAG_LOW_ENERGY,
                    "curr_energy": curr_energy,
                    "past_energy": past_energy,
                    "dot_centered": dot_centered,
                }
            )
            continue

        prod = curr_energy * past_energy
        cos2 = 0.0
        cos_signed = np.nan
        if prod > EPS:
            cos2 = (dot_centered * dot_centered) / prod
            cos2 = min(max(cos2, 0.0), 1.0)
            cos_signed = dot_centered / np.sqrt(prod)

        twist_angle = 2.0 * (1.0 - cos2)
        twist_sq = prod * twist_angle
        drift_centered_sq = max(curr_energy + past_energy - 2.0 * dot_centered, 0.0)
        drift_raw_sq = max(sxx + syy - 2.0 * sxy, 0.0)
        mean_dx = (sx - sy) * inv_d
        mean_shift_sq = float(cfg.dim) * mean_dx * mean_dx

        if has_history:
            raw_slow = twist_sq - last_twist_sq
            ema_slow = cfg.alpha_slow * raw_slow + (1.0 - cfg.alpha_slow) * ema_slow
            raw_fast = twist_angle - last_twist_angle
            ema_fast = cfg.alpha_fast * raw_fast + (1.0 - cfg.alpha_fast) * ema_fast
        else:
            ema_slow = 0.0
            ema_fast = 0.0
            has_history = True

        last_twist_sq = twist_sq
        last_twist_angle = twist_angle

        total_energy = curr_energy + past_energy
        energy_delta = curr_energy - past_energy
        energy_asym = energy_delta / total_energy if total_energy > EPS else np.nan
        drift_norm = drift_centered_sq / total_energy if total_energy > EPS else np.nan
        phase_polarity = 1.0 if dot_centered >= 0.0 else -1.0
        poincare_b = dot_centered / curr_energy if curr_energy > EPS else np.nan
        gram_spread_sq = energy_delta * energy_delta + 4.0 * dot_centered * dot_centered

        rows.append(
            {
                "record": record,
                "beat_sample": int(row["beat_sample"]),
                "beat_symbol": str(row["beat_symbol"]),
                "rr_samples": rr,
                "status": FLAG_OK,
                "curr_energy": curr_energy,
                "past_energy": past_energy,
                "dot_centered": dot_centered,
                "twist_angle": twist_angle,
                "twist_sq": twist_sq,
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "drift_centered_sq": drift_centered_sq,
                "drift_raw_sq": drift_raw_sq,
                "mean_shift_sq": mean_shift_sq,
                "phase_polarity": phase_polarity,
                "energy_delta": energy_delta,
                "energy_asym": energy_asym,
                "drift_norm": drift_norm,
                "poincare_b": poincare_b,
                "gram_spread_sq": gram_spread_sq,
                "cos_signed": cos_signed,
            }
        )
        x_center_rows.append(x_curr_center)

    df = pd.DataFrame(rows)
    df = df[df["status"] == FLAG_OK].copy().reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError(f"{record}: no valid beat-level rows")
    x_center = np.vstack(x_center_rows) if x_center_rows else np.empty((0, cfg.dim), dtype=np.float64)
    if len(x_center) != len(df):
        raise RuntimeError(f"{record}: shared geometry rows did not align with valid beat rows")

    intervals, diag, err = load_st_intervals(record, cfg.st_ext, rec_dir)
    if intervals is None or diag is None:
        raise RuntimeError(f"{record}: {err}")

    df = attach_st(df, intervals)
    df = add_local_template_features(df, signal, cfg)
    df = append_shared_geometry_features(df, x_center=x_center, cfg=cfg)
    df["baseline_score"] = 1.0 - df["local_template_corr"].astype(float)
    df["kernel_score_angle"] = df["twist_angle"].clip(lower=0) * df["ema_fast"].clip(lower=0)
    df["kernel_score_long"] = df["twist_sq"].clip(lower=0) * df["ema_slow"].clip(lower=0)
    angle_rank = df["kernel_score_angle"].rank(pct=True, method="average")
    long_rank = df["kernel_score_long"].rank(pct=True, method="average")
    df["kernel_score_hybrid"] = 0.5 * angle_rank + 0.5 * long_rank
    st_bool = df["st_event"].fillna(False).astype(bool)
    df["st_entry"] = st_bool & (~st_bool.shift(1).fillna(False))
    return df, diag


def top_n_indices(score_series: pd.Series, n: int) -> set[int]:
    return set(score_series.sort_values(ascending=False).head(n).index.tolist())


def future_hit_rate(idx_set: set[int], target_mask: np.ndarray, horizon_beats: int) -> tuple[int, float]:
    idx = np.array(sorted(idx_set), dtype=int)
    arr = np.asarray(target_mask, dtype=bool)
    hits = []
    for i in idx:
        lo = i + 1
        hi = min(len(arr), i + horizon_beats + 1)
        hits.append(bool(arr[lo:hi].any()) if lo < hi else False)
    hit_arr = np.array(hits, dtype=bool)
    return int(hit_arr.sum()), float(hit_arr.mean()) if len(hit_arr) else np.nan


def evaluate_record_from_beats(dfb: pd.DataFrame, diag: dict[str, float], cfg: LTSTConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    eval_mask = ~dfb["st_event"].fillna(False).astype(bool)
    if int(eval_mask.sum()) < cfg.top_n * 2:
        raise RuntimeError(f"{dfb['record'].iloc[0]}: insufficient non-ST beats")

    target = dfb["st_entry"].to_numpy(dtype=bool)
    methods = {
        "baseline": dfb.loc[eval_mask, "baseline_score"].astype(float),
        "angle": dfb.loc[eval_mask, "kernel_score_angle"].astype(float),
        "long": dfb.loc[eval_mask, "kernel_score_long"].astype(float),
        "hybrid": dfb.loc[eval_mask, "kernel_score_hybrid"].astype(float),
    }

    detail_rows: list[dict[str, object]] = []
    hit_map: dict[str, dict[int, float]] = {name: {} for name in methods}
    for method_name, score in methods.items():
        idx_set = top_n_indices(score, cfg.top_n)
        for horizon in cfg.horizons:
            n_hits, hit_rate = future_hit_rate(idx_set, target, horizon)
            hit_map[method_name][horizon] = hit_rate
            detail_rows.append(
                {
                    "record": dfb["record"].iloc[0],
                    "st_ext": cfg.st_ext,
                    "method": method_name,
                    "horizon_beats": horizon,
                    "hit_rate": hit_rate,
                    "n_hits": n_hits,
                    "n_events": cfg.top_n,
                    "n_st_intervals": int(diag["n_intervals"]),
                    "st_prevalence": float(dfb["st_event"].mean()),
                }
            )

    summary_row = {
        "record": dfb["record"].iloc[0],
        "st_ext": cfg.st_ext,
        "n_beats": int(len(dfb)),
        "n_non_st_beats": int(eval_mask.sum()),
        "n_st_intervals": int(diag["n_intervals"]),
        "st_prevalence": float(dfb["st_event"].mean()),
        "angle_cross": next((h for h in cfg.horizons if hit_map["angle"][h] - hit_map["baseline"][h] > cfg.crossover_min_delta), None),
        "long_cross": next((h for h in cfg.horizons if hit_map["long"][h] - hit_map["baseline"][h] > cfg.crossover_min_delta), None),
        "hybrid_cross": next((h for h in cfg.horizons if hit_map["hybrid"][h] - hit_map["baseline"][h] > cfg.crossover_min_delta), None),
        "delta_angle_120": hit_map["angle"].get(120, np.nan) - hit_map["baseline"].get(120, np.nan),
        "delta_long_500": hit_map["long"].get(500, np.nan) - hit_map["baseline"].get(500, np.nan),
        "delta_long_1000": hit_map["long"].get(1000, np.nan) - hit_map["baseline"].get(1000, np.nan),
        "delta_hybrid_250": hit_map["hybrid"].get(250, np.nan) - hit_map["baseline"].get(250, np.nan),
    }
    return pd.DataFrame(detail_rows), summary_row


def classify_regime(row: pd.Series) -> str:
    angle_cross = row["angle_cross"]
    long_cross = row["long_cross"]
    if pd.notna(angle_cross) and pd.isna(long_cross):
        return "angle_only"
    if pd.isna(angle_cross) and pd.notna(long_cross):
        return "long_only"
    if pd.notna(angle_cross) and pd.notna(long_cross):
        if angle_cross < long_cross:
            return "angle_first"
        if long_cross < angle_cross:
            return "long_first"
        return "both_same_horizon"
    return "neither"


def invariant_summary_row(dfb: pd.DataFrame) -> dict[str, object]:
    return {
        "record": dfb["record"].iloc[0],
        "frac_negative_phase": float((dfb["phase_polarity"] < 0).mean()),
        "poincare_b_p05": float(np.nanpercentile(dfb["poincare_b"], 5)),
        "poincare_b_p50": float(np.nanpercentile(dfb["poincare_b"], 50)),
        "poincare_b_p95": float(np.nanpercentile(dfb["poincare_b"], 95)),
        "median_energy_asym": float(np.nanmedian(dfb["energy_asym"])),
        "p10_energy_asym": float(np.nanpercentile(dfb["energy_asym"], 10)),
        "p90_energy_asym": float(np.nanpercentile(dfb["energy_asym"], 90)),
        "median_drift_norm": float(np.nanmedian(dfb["drift_norm"])),
        "p90_drift_norm": float(np.nanpercentile(dfb["drift_norm"], 90)),
        "median_gram_spread_sq": float(np.nanmedian(dfb["gram_spread_sq"])),
    }


def routed_analysis(dfb: pd.DataFrame, cfg: LTSTConfig) -> tuple[list[dict[str, object]], dict[str, object]]:
    eval_mask = ~dfb["st_event"].fillna(False).astype(bool)
    target = dfb["st_entry"].to_numpy(dtype=bool)
    baseline_score = dfb.loc[eval_mask, "baseline_score"].astype(float)
    angle_score = dfb.loc[eval_mask, "kernel_score_angle"].astype(float)
    long_score = dfb.loc[eval_mask, "kernel_score_long"].astype(float)
    route_flag = (dfb.loc[eval_mask, "phase_polarity"] < 0) | (dfb.loc[eval_mask, "poincare_b"] < 0.0)
    routed_score = long_score.copy()
    routed_score.loc[route_flag] = angle_score.loc[route_flag]

    methods = {"baseline": baseline_score, "angle": angle_score, "long": long_score, "routed": routed_score}
    detail_rows: list[dict[str, object]] = []
    hit_map: dict[str, dict[int, float]] = {name: {} for name in methods}

    for method_name, score in methods.items():
        idx_set = top_n_indices(score, cfg.top_n)
        for horizon in cfg.horizons:
            n_hits, hit_rate = future_hit_rate(idx_set, target, horizon)
            hit_map[method_name][horizon] = hit_rate
            detail_rows.append(
                {
                    "record": dfb["record"].iloc[0],
                    "method": method_name,
                    "horizon_beats": horizon,
                    "hit_rate": hit_rate,
                    "n_hits": n_hits,
                    "n_events": cfg.top_n,
                }
            )

    summary_row = {
        "record": dfb["record"].iloc[0],
        "routed_cross": next((h for h in cfg.horizons if hit_map["routed"][h] - hit_map["baseline"][h] > cfg.crossover_min_delta), None),
        "baseline_250": hit_map["baseline"].get(250, np.nan),
        "angle_250": hit_map["angle"].get(250, np.nan),
        "long_250": hit_map["long"].get(250, np.nan),
        "routed_250": hit_map["routed"].get(250, np.nan),
        "best_routed_gain_over_baseline": max(hit_map["routed"][h] - hit_map["baseline"][h] for h in cfg.horizons),
        "best_routed_gain_over_long": max(hit_map["routed"][h] - hit_map["long"][h] for h in cfg.horizons),
        "best_routed_gain_over_angle": max(hit_map["routed"][h] - hit_map["angle"][h] for h in cfg.horizons),
    }
    return detail_rows, summary_row


def guarded_analysis(dfb: pd.DataFrame, cfg: LTSTConfig) -> tuple[list[dict[str, object]], dict[str, object]]:
    chaos_neg_phase_thresh = 0.05
    chaos_poincare_p05_thresh = -0.10
    chaos_drift_p90_thresh = 0.60
    shear_drift_thresh = 0.20
    shear_poincare_thresh = 0.60
    use_strict_and = False

    eval_mask = ~dfb["st_event"].fillna(False).astype(bool)
    target = dfb["st_entry"].to_numpy(dtype=bool)
    frac_negative_phase = float((dfb["phase_polarity"] < 0).mean())
    poincare_b_p05 = float(np.nanpercentile(dfb["poincare_b"], 5))
    drift_p90 = float(np.nanpercentile(dfb["drift_norm"], 90))
    chaos_flag = (
        (frac_negative_phase > chaos_neg_phase_thresh)
        or (poincare_b_p05 < chaos_poincare_p05_thresh)
        or (drift_p90 > chaos_drift_p90_thresh)
    )

    baseline_score = dfb.loc[eval_mask, "baseline_score"].astype(float)
    angle_score = dfb.loc[eval_mask, "kernel_score_angle"].astype(float)
    long_score = dfb.loc[eval_mask, "kernel_score_long"].astype(float)
    drift_gate = dfb.loc[eval_mask, "drift_norm"].astype(float) > shear_drift_thresh
    poincare_gate = dfb.loc[eval_mask, "poincare_b"].astype(float) < shear_poincare_thresh
    shear_gate = drift_gate & poincare_gate if use_strict_and else (drift_gate | poincare_gate)

    guarded_score = long_score.copy()
    if not chaos_flag:
        guarded_score.loc[shear_gate] = angle_score.loc[shear_gate]

    methods = {
        "baseline": baseline_score,
        "angle": angle_score,
        "long": long_score,
        "guarded_router": guarded_score,
    }
    detail_rows: list[dict[str, object]] = []
    hit_map: dict[str, dict[int, float]] = {name: {} for name in methods}
    for method_name, score in methods.items():
        idx_set = top_n_indices(score, cfg.top_n)
        for horizon in cfg.horizons:
            n_hits, hit_rate = future_hit_rate(idx_set, target, horizon)
            hit_map[method_name][horizon] = hit_rate
            detail_rows.append(
                {
                    "record": dfb["record"].iloc[0],
                    "method": method_name,
                    "horizon_beats": horizon,
                    "hit_rate": hit_rate,
                    "n_hits": n_hits,
                    "n_events": cfg.top_n,
                }
            )

    summary_row = {
        "record": dfb["record"].iloc[0],
        "frac_negative_phase": frac_negative_phase,
        "poincare_b_p05": poincare_b_p05,
        "drift_norm_p90": drift_p90,
        "chaos_flag": bool(chaos_flag),
        "frac_shear_gated_beats": float(np.mean(shear_gate)),
        "guarded_cross": next((h for h in cfg.horizons if hit_map["guarded_router"][h] - hit_map["baseline"][h] > cfg.crossover_min_delta), None),
        "baseline_250": hit_map["baseline"].get(250, np.nan),
        "angle_250": hit_map["angle"].get(250, np.nan),
        "long_250": hit_map["long"].get(250, np.nan),
        "guarded_250": hit_map["guarded_router"].get(250, np.nan),
        "guarded_minus_baseline_250": hit_map["guarded_router"].get(250, np.nan) - hit_map["baseline"].get(250, np.nan),
        "guarded_minus_long_250": hit_map["guarded_router"].get(250, np.nan) - hit_map["long"].get(250, np.nan),
        "guarded_minus_angle_250": hit_map["guarded_router"].get(250, np.nan) - hit_map["angle"].get(250, np.nan),
        "best_guarded_gain_over_baseline": max(hit_map["guarded_router"][h] - hit_map["baseline"][h] for h in cfg.horizons),
        "best_guarded_gain_over_long": max(hit_map["guarded_router"][h] - hit_map["long"][h] for h in cfg.horizons),
        "best_guarded_gain_over_angle": max(hit_map["guarded_router"][h] - hit_map["angle"][h] for h in cfg.horizons),
    }
    return detail_rows, summary_row


def write_report(run_dir: Path, cfg: LTSTConfig, metadata: dict[str, object]) -> None:
    regime_counts = metadata.get("regime_counts", {})
    lines = [
        "# LTST Full Study Report",
        "",
        f"- Database: `{cfg.db}`",
        f"- ST annotation: `{cfg.st_ext}`",
        f"- Beat annotation: `{cfg.beat_ext}`",
        f"- Requested records: `{metadata.get('requested_records')}`",
        f"- Completed records: `{metadata.get('completed_records')}`",
        f"- Failed records: `{metadata.get('failed_records')}`",
        f"- Horizons: `{list(cfg.horizons)}`",
        f"- Top N alerts: `{cfg.top_n}`",
        "",
        "## Regime Counts",
    ]
    if regime_counts:
        for regime_name, count in regime_counts.items():
            lines.append(f"- `{regime_name}`: `{count}`")
    else:
        lines.append("- No completed records yet.")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def save_outputs(
    run_dir: Path,
    cfg: LTSTConfig,
    records: Sequence[str],
    summary_rows: list[dict[str, object]],
    detail_frames: list[pd.DataFrame],
    failed_rows: list[dict[str, object]],
    invariant_rows: list[dict[str, object]],
    route_detail_rows: list[dict[str, object]],
    route_summary_rows: list[dict[str, object]],
    guard_detail_rows: list[dict[str, object]],
    guard_summary_rows: list[dict[str, object]],
) -> None:
    summary_df = pd.DataFrame(summary_rows).sort_values("record").reset_index(drop=True) if summary_rows else pd.DataFrame()
    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    failed_df = pd.DataFrame(failed_rows)
    invariant_df = pd.DataFrame(invariant_rows).sort_values("record").reset_index(drop=True) if invariant_rows else pd.DataFrame()
    route_detail_df = pd.DataFrame(route_detail_rows)
    route_summary_df = pd.DataFrame(route_summary_rows).sort_values("record").reset_index(drop=True) if route_summary_rows else pd.DataFrame()
    guard_detail_df = pd.DataFrame(guard_detail_rows)
    guard_summary_df = pd.DataFrame(guard_summary_rows).sort_values("record").reset_index(drop=True) if guard_summary_rows else pd.DataFrame()

    regime_df = summary_df.copy()
    if len(regime_df):
        regime_df["regime"] = regime_df.apply(classify_regime, axis=1)
        regime_counts = regime_df["regime"].value_counts(dropna=False).to_dict()
    else:
        regime_counts = {}

    if len(summary_df):
        save_frame(summary_df, run_dir / "summary")
    if len(detail_df):
        save_frame(detail_df, run_dir / "detail")
    if len(failed_df):
        save_frame(failed_df, run_dir / "failed")
    if len(regime_df):
        save_frame(regime_df, run_dir / "regime")
    if len(invariant_df):
        save_frame(invariant_df, run_dir / "invariant_summary")
    if len(route_detail_df):
        save_frame(route_detail_df, run_dir / "routed_detail")
    if len(route_summary_df):
        save_frame(route_summary_df, run_dir / "routed_summary")
    if len(guard_detail_df):
        save_frame(guard_detail_df, run_dir / "guarded_detail")
    if len(guard_summary_df):
        save_frame(guard_summary_df, run_dir / "guarded_summary")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {k: str(v) if isinstance(v, Path) else (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
        "requested_records": len(records),
        "completed_records": len(summary_rows),
        "failed_records": len(failed_rows),
        "regime_counts": regime_counts,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(run_dir, cfg, metadata)


def run_full_study(cfg: LTSTConfig) -> dict[str, object]:
    records = list(cfg.records) if cfg.records else discover_records(cfg.db)
    if cfg.max_records is not None:
        records = records[: cfg.max_records]

    run_dir = prepare_run_directory(cfg.run_root)
    beat_level_dir = run_dir / "beat_level"
    if cfg.save_beat_level:
        beat_level_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir}", flush=True)
    print(f"Records to process: {len(records)}", flush=True)

    summary_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    failed_rows: list[dict[str, object]] = []
    invariant_rows: list[dict[str, object]] = []
    route_detail_rows: list[dict[str, object]] = []
    route_summary_rows: list[dict[str, object]] = []
    guard_detail_rows: list[dict[str, object]] = []
    guard_summary_rows: list[dict[str, object]] = []

    for idx, record in enumerate(records, start=1):
        print(f"[{idx}/{len(records)}] RUN {record} ...", flush=True)
        t0 = time.time()
        try:
            beat_df, diag = build_beat_level_with_invariants(record, cfg)
            detail_df, summary_row = evaluate_record_from_beats(beat_df, diag, cfg)
            detail_frames.append(detail_df)
            summary_rows.append(summary_row)
            invariant_rows.append(invariant_summary_row(beat_df))

            if cfg.run_simple_router:
                route_detail, route_summary = routed_analysis(beat_df, cfg)
                route_detail_rows.extend(route_detail)
                route_summary_rows.append(route_summary)

            if cfg.run_guarded_router:
                guard_detail, guard_summary = guarded_analysis(beat_df, cfg)
                guard_detail_rows.extend(guard_detail)
                guard_summary_rows.append(guard_summary)

            if cfg.save_beat_level:
                save_frame(beat_df, beat_level_dir / record)

            print(
                f"  OK {record} | angle_cross={summary_row['angle_cross']} "
                f"| long_cross={summary_row['long_cross']} "
                f"| hybrid_cross={summary_row['hybrid_cross']} "
                f"| {time.time() - t0:.2f}s",
                flush=True,
            )
        except Exception as exc:
            failed_rows.append({"record": record, "reason": str(exc)})
            print(f"  FAIL {record}: {exc}", flush=True)

        if idx % max(1, cfg.checkpoint_every) == 0 or idx == len(records):
            save_outputs(
                run_dir=run_dir,
                cfg=cfg,
                records=records,
                summary_rows=summary_rows,
                detail_frames=detail_frames,
                failed_rows=failed_rows,
                invariant_rows=invariant_rows,
                route_detail_rows=route_detail_rows,
                route_summary_rows=route_summary_rows,
                guard_detail_rows=guard_detail_rows,
                guard_summary_rows=guard_summary_rows,
            )
            print(f"  checkpoint saved after {idx} records", flush=True)

    return {
        "run_dir": run_dir,
        "requested_records": len(records),
        "completed_records": len(summary_rows),
        "failed_records": len(failed_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full LTST cohort study from the validated 5-record notebook logic.")
    parser.add_argument("--records", default="all", help="Comma-separated records or 'all'.")
    parser.add_argument("--db", default="ltstdb")
    parser.add_argument("--st-ext", default="stc")
    parser.add_argument("--beat-ext", default="atr")
    parser.add_argument("--top-n", type=int, default=1102)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--save-beat-level", action="store_true")
    parser.add_argument("--skip-simple-router", action="store_true")
    parser.add_argument("--skip-guarded-router", action="store_true")
    parser.add_argument("--run-root", default=None)
    return parser.parse_args()


def main() -> None:
    project_root = find_project_root()
    args = parse_args()
    cfg = LTSTConfig(
        db=args.db,
        beat_ext=args.beat_ext,
        st_ext=args.st_ext,
        records=tuple(parse_records_arg(args.records, args.db)),
        top_n=int(args.top_n),
        checkpoint_every=int(args.checkpoint_every),
        save_beat_level=bool(args.save_beat_level),
        run_simple_router=not bool(args.skip_simple_router),
        run_guarded_router=not bool(args.skip_guarded_router),
        max_records=args.max_records,
        run_root=Path(args.run_root).resolve() if args.run_root else (project_root / "artifact" / "runs"),
    )
    result = run_full_study(cfg)
    print(
        f"Completed LTST run: {result['completed_records']}/{result['requested_records']} records "
        f"(failed={result['failed_records']})",
        flush=True,
    )
    print(f"Output directory: {result['run_dir']}", flush=True)


if __name__ == "__main__":
    main()
