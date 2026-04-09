from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

try:
    from scipy.signal import lfilter
except Exception:
    lfilter = None


EPS = 1e-12
PRIMARY_FEATURE_COLUMNS = [
    "proj_area_sl",
    "proj_transverse_xl",
    "proj_lock_barrier_sl",
    "proj_lock_barrier_xl",
    "proj_volume_xsl",
    "log_proj_volume_xsl",
    "temporal_area_loss",
    "present_transverse_freedom",
    "memory_lock_barrier",
    "present_lock_barrier",
    "short_long_explanation_imbalance",
    "temporal_volume_3",
    "log_temporal_volume_3",
]


@dataclass(frozen=True)
class MemoryConfig:
    db: str = "ltstdb"
    beat_ext: str = "atr"
    st_ext: str = "stc"
    dim: int = 9
    lag: int = 4
    max_beat_offset: int = 320
    min_energy_thresh: float = 1e-6
    beta_short: float = 0.10
    beta_long: float = 0.01
    baseline_beats: int = 500
    records: tuple[str, ...] = ("s20021", "s20041", "s20151", "s30742")
    run_dir: Path = Path("artifact") / "runs" / "ltst_full_86_20260405T035505Z"


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def clamp_unit_interval(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def phenotype_from_regime(regime: str) -> str:
    if regime in {"angle_only", "angle_first"}:
        return "loose_orbit"
    if regime in {"long_only", "long_first", "both_same_horizon"}:
        return "constrained_orbit"
    return "rigid_orbit"


def state_buffer_size(cfg: MemoryConfig) -> int:
    return 1 + cfg.max_beat_offset + (cfg.dim - 1) * cfg.lag


def max_run(mask: np.ndarray) -> int:
    arr = np.asarray(mask, dtype=bool)
    if len(arr) == 0 or not arr.any():
        return 0
    padded = np.concatenate(([False], arr, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return int(np.max(ends - starts)) if len(starts) else 0


def load_st_intervals(record: str, ext: str, pn_dir: str) -> list[tuple[int, int]]:
    ann = wfdb.rdann(record, ext, pn_dir=pn_dir)
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
    return intervals


def finite_primary_mask(df: pd.DataFrame) -> pd.Series:
    return np.isfinite(df[PRIMARY_FEATURE_COLUMNS]).all(axis=1)


def ema_filter_states(x: np.ndarray, beta: float) -> np.ndarray:
    if len(x) == 0:
        return np.empty_like(x)
    if lfilter is not None:
        filtered = lfilter([beta], [1.0, -(1.0 - beta)], x, axis=0)
        return np.vstack([np.zeros((1, x.shape[1]), dtype=np.float64), filtered[:-1]])

    state = np.zeros(x.shape[1], dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    for idx in range(len(x)):
        out[idx] = state
        state = beta * x[idx] + (1.0 - beta) * state
    return out


def _interval_mask(samples: np.ndarray, intervals: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(len(samples), dtype=bool)
    for start, end in intervals:
        mask |= (samples >= int(start)) & (samples < int(end))
    return mask


def select_baseline_window(df: pd.DataFrame, baseline_beats: int) -> pd.DataFrame:
    non_st = ~df["st_event"].fillna(False).astype(bool)
    mask = non_st & finite_primary_mask(df)
    baseline = df.loc[mask].copy()
    if baseline_beats > 0:
        baseline = baseline.head(baseline_beats).copy()
    return baseline.reset_index(drop=True)


def _safe_quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.quantile(q)) if len(values.dropna()) else np.nan


def compute_local_record_summary(df: pd.DataFrame, baseline_beats: int = 500) -> dict[str, Any]:
    baseline = select_baseline_window(df, baseline_beats)
    st = df.loc[df["st_event"].fillna(False).astype(bool) & finite_primary_mask(df)].copy()
    if baseline.empty:
        raise RuntimeError(f"{df['record'].iloc[0]}: no valid baseline rows")

    def median(frame: pd.DataFrame, column: str) -> float:
        return float(frame[column].median()) if not frame.empty else np.nan

    def quant(frame: pd.DataFrame, column: str, q: float) -> float:
        return _safe_quantile(frame[column], q) if not frame.empty else np.nan

    def safe_ratio(num: float, den: float) -> float:
        if not np.isfinite(num) or not np.isfinite(den):
            return np.nan
        return float(num / (den + EPS))

    baseline_abs_delta = baseline["short_long_explanation_imbalance"].abs()
    st_abs_delta = st["short_long_explanation_imbalance"].abs() if not st.empty else pd.Series(dtype=float)

    return {
        "record": str(df["record"].iloc[0]),
        "regime": str(df["regime"].iloc[0]),
        "phenotype_target": str(df["phenotype_target"].iloc[0]),
        "baseline_n": int(len(baseline)),
        "st_n": int(len(st)),
        "baseline_median_proj_line_lock_sl": median(baseline, "proj_line_lock_sl"),
        "baseline_p95_proj_line_lock_sl": quant(baseline, "proj_line_lock_sl", 0.95),
        "baseline_median_temporal_area_loss": median(baseline, "temporal_area_loss"),
        "baseline_p95_temporal_area_loss": quant(baseline, "temporal_area_loss", 0.95),
        "baseline_median_proj_line_lock_xl": median(baseline, "proj_line_lock_xl"),
        "baseline_p95_proj_line_lock_xl": quant(baseline, "proj_line_lock_xl", 0.95),
        "baseline_median_present_transverse_freedom": median(baseline, "present_transverse_freedom"),
        "baseline_p05_present_transverse_freedom": quant(baseline, "present_transverse_freedom", 0.05),
        "baseline_median_memory_lock_barrier": median(baseline, "memory_lock_barrier"),
        "baseline_p95_memory_lock_barrier": quant(baseline, "memory_lock_barrier", 0.95),
        "baseline_median_present_lock_barrier": median(baseline, "present_lock_barrier"),
        "baseline_p95_present_lock_barrier": quant(baseline, "present_lock_barrier", 0.95),
        "baseline_median_abs_short_long_explanation_imbalance": float(baseline_abs_delta.median()),
        "baseline_p95_abs_short_long_explanation_imbalance": _safe_quantile(baseline_abs_delta, 0.95),
        "baseline_sign_fraction_short_long_explanation_imbalance_positive": float(
            (baseline["short_long_explanation_imbalance"] > 0.0).mean()
        ),
        "baseline_median_temporal_volume_3": median(baseline, "temporal_volume_3"),
        "baseline_p95_temporal_volume_3": quant(baseline, "temporal_volume_3", 0.95),
        "baseline_median_log_temporal_volume_3": median(baseline, "log_temporal_volume_3"),
        "baseline_median_temporal_volume_3_raw": median(baseline, "temporal_volume_3_raw"),
        "baseline_p95_temporal_volume_3_raw": quant(baseline, "temporal_volume_3_raw", 0.95),
        "baseline_median_memory_align": median(baseline, "memory_align"),
        "baseline_median_linger": median(baseline, "linger"),
        "baseline_median_memory_gap": median(baseline, "memory_gap"),
        "baseline_median_novelty": median(baseline, "novelty"),
        "st_median_proj_line_lock_sl": median(st, "proj_line_lock_sl"),
        "st_median_temporal_area_loss": median(st, "temporal_area_loss"),
        "st_median_proj_line_lock_xl": median(st, "proj_line_lock_xl"),
        "st_median_present_transverse_freedom": median(st, "present_transverse_freedom"),
        "st_median_memory_lock_barrier": median(st, "memory_lock_barrier"),
        "st_median_present_lock_barrier": median(st, "present_lock_barrier"),
        "st_median_abs_short_long_explanation_imbalance": float(st_abs_delta.median()) if len(st_abs_delta) else np.nan,
        "st_median_temporal_volume_3": median(st, "temporal_volume_3"),
        "st_median_log_temporal_volume_3": median(st, "log_temporal_volume_3"),
        "st_median_temporal_volume_3_raw": median(st, "temporal_volume_3_raw"),
        "memory_lock_barrier_st_over_baseline": safe_ratio(
            median(st, "memory_lock_barrier"),
            median(baseline, "memory_lock_barrier"),
        ),
        "present_lock_barrier_st_over_baseline": safe_ratio(
            median(st, "present_lock_barrier"),
            median(baseline, "present_lock_barrier"),
        ),
    }


def build_memory_probe(record: str, cfg: MemoryConfig, regime_lookup: dict[str, str]) -> pd.DataFrame:
    rec = wfdb.rdrecord(record, pn_dir=cfg.db)
    ann = wfdb.rdann(record, cfg.beat_ext, pn_dir=cfg.db)
    signal = rec.p_signal[:, 0].astype(np.float64)
    beat_df = pd.DataFrame(
        {
            "beat_sample": np.asarray(ann.sample, dtype=int),
            "beat_symbol": np.asarray(ann.symbol, dtype=object).astype(str),
        }
    ).reset_index(drop=True)
    beat_df["rr_samples"] = beat_df["beat_sample"].diff()
    beat_df.loc[0, "rr_samples"] = np.nan
    intervals = load_st_intervals(record, cfg.st_ext, cfg.db)
    beat_samples = beat_df["beat_sample"].to_numpy(dtype=int)
    rr_samples = beat_df["rr_samples"].to_numpy(dtype=float)
    lag_offsets = (cfg.lag * np.arange(cfg.dim, dtype=int))[None, :]
    state_indices = beat_samples[:, None] - lag_offsets

    valid_mask = np.isfinite(rr_samples) & (rr_samples > 0) & (state_indices.min(axis=1) >= 0)
    if not np.any(valid_mask):
        raise RuntimeError(f"{record}: no valid beats after state-index filtering")

    valid_beat_df = beat_df.loc[valid_mask].copy().reset_index(drop=True)
    valid_state_indices = state_indices[valid_mask]
    x_raw = signal[valid_state_indices]
    x_center = x_raw - np.mean(x_raw, axis=1, keepdims=True)
    n_x = np.einsum("td,td->t", x_center, x_center)

    energy_mask = n_x >= cfg.min_energy_thresh
    if not np.any(energy_mask):
        raise RuntimeError(f"{record}: no valid beats above minimum energy threshold")

    valid_beat_df = valid_beat_df.loc[energy_mask].copy().reset_index(drop=True)
    x_center = x_center[energy_mask]
    n_x = n_x[energy_mask]

    ms = ema_filter_states(x_center, cfg.beta_short)
    ml = ema_filter_states(x_center, cfg.beta_long)

    n_s = np.einsum("td,td->t", ms, ms)
    n_l = np.einsum("td,td->t", ml, ml)
    d_sl = np.einsum("td,td->t", ms, ml)
    d_xs = np.einsum("td,td->t", x_center, ms)
    d_xl = np.einsum("td,td->t", x_center, ml)

    memory_align = np.full(len(x_center), np.nan, dtype=np.float64)
    linger = np.full(len(x_center), np.nan, dtype=np.float64)
    proj_line_lock_sl = np.full(len(x_center), np.nan, dtype=np.float64)
    proj_area_sl = np.full(len(x_center), np.nan, dtype=np.float64)
    proj_line_lock_xl = np.full(len(x_center), np.nan, dtype=np.float64)
    proj_transverse_xl = np.full(len(x_center), np.nan, dtype=np.float64)
    proj_lock_barrier_sl = np.full(len(x_center), np.nan, dtype=np.float64)
    proj_lock_barrier_xl = np.full(len(x_center), np.nan, dtype=np.float64)
    delta_explain = np.full(len(x_center), np.nan, dtype=np.float64)
    temporal_volume_3 = np.full(len(x_center), np.nan, dtype=np.float64)
    temporal_volume_3_raw = np.full(len(x_center), np.nan, dtype=np.float64)
    log_temporal_volume_3 = np.full(len(x_center), np.nan, dtype=np.float64)

    sl_mask = (n_s > EPS) & (n_l > EPS)
    xl_mask = n_l > EPS

    if np.any(sl_mask):
        memory_align[sl_mask] = d_sl[sl_mask] / (np.sqrt(n_s[sl_mask] * n_l[sl_mask]) + EPS)
        proj_line_lock_sl[sl_mask] = np.clip(
            (d_sl[sl_mask] ** 2) / (n_s[sl_mask] * n_l[sl_mask] + EPS),
            0.0,
            1.0,
        )
        proj_area_sl[sl_mask] = np.clip(1.0 - proj_line_lock_sl[sl_mask], 0.0, 1.0)
        proj_lock_barrier_sl[sl_mask] = -np.log(proj_area_sl[sl_mask] + EPS)

    if np.any(xl_mask):
        linger[xl_mask] = d_xl[xl_mask] / (np.sqrt(n_x[xl_mask] * n_l[xl_mask]) + EPS)
        proj_line_lock_xl[xl_mask] = np.clip(
            (d_xl[xl_mask] ** 2) / (n_x[xl_mask] * n_l[xl_mask] + EPS),
            0.0,
            1.0,
        )
        proj_transverse_xl[xl_mask] = np.clip(1.0 - proj_line_lock_xl[xl_mask], 0.0, 1.0)
        proj_lock_barrier_xl[xl_mask] = -np.log(proj_transverse_xl[xl_mask] + EPS)

    if np.any(sl_mask):
        delta_explain[sl_mask] = (d_xs[sl_mask] ** 2) / (n_x[sl_mask] * n_s[sl_mask] + EPS) - (
            d_xl[sl_mask] ** 2
        ) / (n_x[sl_mask] * n_l[sl_mask] + EPS)
        temporal_volume_raw = (
            n_x[sl_mask] * n_s[sl_mask] * n_l[sl_mask]
            + 2.0 * d_xs[sl_mask] * d_xl[sl_mask] * d_sl[sl_mask]
            - n_x[sl_mask] * (d_sl[sl_mask] ** 2)
            - n_s[sl_mask] * (d_xl[sl_mask] ** 2)
            - n_l[sl_mask] * (d_xs[sl_mask] ** 2)
        )
        temporal_volume_raw = np.maximum(temporal_volume_raw, 0.0)
        temporal_volume_3_raw[sl_mask] = temporal_volume_raw
        temporal_volume_3[sl_mask] = np.clip(
            temporal_volume_raw / (n_x[sl_mask] * n_s[sl_mask] * n_l[sl_mask] + EPS),
            0.0,
            1.0,
        )
        log_temporal_volume_3[sl_mask] = np.log(temporal_volume_3[sl_mask] + EPS)

    memory_gap = n_s + n_l - 2.0 * d_sl
    novelty = np.where(np.isfinite(linger), 1.0 - linger, np.nan)

    beat_sample_vec = valid_beat_df["beat_sample"].to_numpy(dtype=int)
    st_event = _interval_mask(beat_sample_vec, intervals)

    df = pd.DataFrame(
        {
            "record": record,
            "regime": regime_lookup.get(record, "unknown"),
            "phenotype_target": phenotype_from_regime(regime_lookup.get(record, "unknown")),
            "beat_sample": beat_sample_vec,
            "beat_symbol": valid_beat_df["beat_symbol"].astype(str).to_numpy(dtype=object),
            "rr_samples": valid_beat_df["rr_samples"].to_numpy(dtype=float),
            "st_event": st_event,
            "n_x": n_x,
            "n_s": n_s,
            "n_l": n_l,
            "d_sl": d_sl,
            "d_xs": d_xs,
            "d_xl": d_xl,
            "proj_line_lock_sl": proj_line_lock_sl,
            "proj_area_sl": proj_area_sl,
            "proj_line_lock_xl": proj_line_lock_xl,
            "proj_transverse_xl": proj_transverse_xl,
            "proj_lock_barrier_sl": proj_lock_barrier_sl,
            "proj_lock_barrier_xl": proj_lock_barrier_xl,
            "proj_volume_xsl": temporal_volume_3,
            "log_proj_volume_xsl": log_temporal_volume_3,
            "temporal_area_loss": proj_area_sl,
            "present_transverse_freedom": proj_transverse_xl,
            "memory_lock_barrier": proj_lock_barrier_sl,
            "present_lock_barrier": proj_lock_barrier_xl,
            "short_long_explanation_imbalance": delta_explain,
            "temporal_volume_3": temporal_volume_3,
            "temporal_volume_3_raw": temporal_volume_3_raw,
            "log_temporal_volume_3": log_temporal_volume_3,
            "memory_align": memory_align,
            "linger": linger,
            "memory_gap": memory_gap,
            "novelty": novelty,
        }
    )
    if len(df) == 0:
        raise RuntimeError(f"{record}: no valid memory-probe rows")
    st_bool = df["st_event"].fillna(False).astype(bool)
    df["st_entry"] = st_bool & (~st_bool.shift(1).fillna(False))
    df["beat_index"] = np.arange(len(df), dtype=int)
    return df


def summarize_probe(df: pd.DataFrame, baseline_beats: int = 500) -> dict[str, Any]:
    summary = compute_local_record_summary(df, baseline_beats=baseline_beats)
    baseline = select_baseline_window(df, baseline_beats)
    if not baseline.empty:
        thresh_align = float(baseline["memory_align"].quantile(0.95))
        active_align = baseline["memory_align"] >= thresh_align
        summary["baseline_p95_memory_align"] = thresh_align
        summary["baseline_max_memory_align_run"] = max_run(active_align.to_numpy(dtype=bool))
    else:
        summary["baseline_p95_memory_align"] = np.nan
        summary["baseline_max_memory_align_run"] = 0
    return summary


def plot_probe(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(13, 13), sharex=True)
    title = f"{df['record'].iloc[0]} | regime={df['regime'].iloc[0]} | Gram-memory probe"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    panels = [
        ("memory_align", "#6a3d9a", "memory_align"),
        ("temporal_area_loss", "#1f78b4", "temporal_area_loss"),
        ("present_transverse_freedom", "#33a02c", "present_transverse_freedom"),
        ("memory_lock_barrier", "#e31a1c", "memory_lock_barrier"),
        ("present_lock_barrier", "#ff7f00", "present_lock_barrier"),
        ("log_temporal_volume_3", "#b15928", "log_temporal_volume_3"),
    ]

    st_entry_idx = np.flatnonzero(df["st_entry"].fillna(False).to_numpy(dtype=bool))
    for ax, (column, color, label) in zip(axes, panels):
        ax.plot(df["beat_index"], df[column], color=color, linewidth=0.9, label=label)
        for idx in st_entry_idx[:60]:
            ax.axvline(int(idx), color="black", alpha=0.08, linewidth=0.8)
        ax.set_ylabel(label)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Beat index")
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Gram-memory O(d) features on selected LTST records.")
    parser.add_argument("--run-dir", default=str(Path("artifact") / "runs" / "ltst_full_86_20260405T035505Z"))
    parser.add_argument("--records", default="s20021,s20041,s20151,s30742")
    parser.add_argument("--beta-short", type=float, default=0.10)
    parser.add_argument("--beta-long", type=float, default=0.01)
    parser.add_argument("--baseline-beats", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    cfg = MemoryConfig(
        run_dir=run_dir,
        records=tuple(piece.strip() for piece in args.records.split(",") if piece.strip()),
        beta_short=args.beta_short,
        beta_long=args.beta_long,
        baseline_beats=args.baseline_beats,
    )
    regime_lookup = dict(pd.read_csv(run_dir / "regime.csv")[["record", "regime"]].itertuples(index=False, name=None))
    out_dir = run_dir / "memory_compression_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for record in cfg.records:
        print(f"[MEMORY] {record} ...", flush=True)
        df = build_memory_probe(record, cfg, regime_lookup)
        df.to_csv(out_dir / f"{record}_memory_probe.csv", index=False)
        plot_probe(df, out_dir / f"{record}_memory_probe.png")
        summary_rows.append(summarize_probe(df, baseline_beats=cfg.baseline_beats))

    summary_df = pd.DataFrame(summary_rows).sort_values(["phenotype_target", "record"]).reset_index(drop=True)
    summary_df.to_csv(out_dir / "memory_probe_summary.csv", index=False)

    metadata = {"config": json_safe(asdict(cfg)), "output_dir": str(out_dir)}
    with (out_dir / "memory_probe_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(json_safe(metadata), fp, indent=2)

    report = [
        "# LTST Gram-Memory Probe",
        "",
        "Two-timescale O(d) Gram-memory geometry on selected LTST records.",
        "",
        summary_df.to_markdown(index=False),
        "",
    ]
    (out_dir / "memory_probe_report.md").write_text("\n".join(report), encoding="utf-8")

    print(summary_df.to_string(index=False), flush=True)
    print(f"[MEMORY] Output directory: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
