from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ltst_memory_compression_probe as mem


EPS = 1e-9
PHENOTYPE_ORDER = ["loose_orbit", "constrained_orbit", "rigid_orbit"]
PHENOTYPE_LABELS = {
    "loose_orbit": "Loose Orbit",
    "constrained_orbit": "Constrained Orbit",
    "rigid_orbit": "Rigid Orbit",
}
PHENOTYPE_COLORS = {
    "loose_orbit": "#d95f02",
    "constrained_orbit": "#1b9e77",
    "rigid_orbit": "#7570b3",
}
LOW_THRESHOLD_COLUMNS = {
    "temporal_area_loss": "baseline_max_temporal_area_loss_low_run",
    "present_transverse_freedom": "baseline_max_present_transverse_freedom_low_run",
    "log_temporal_volume_3": "baseline_max_log_temporal_volume_3_low_run",
}
HIGH_THRESHOLD_COLUMNS = {
    "memory_lock_barrier": "baseline_max_memory_lock_barrier_high_run",
    "present_lock_barrier": "baseline_max_present_lock_barrier_high_run",
    "abs_short_long_explanation_imbalance": "baseline_max_abs_short_long_explanation_imbalance_high_run",
}
BASELINE_CACHE_COLUMNS = [
    "record",
    "beat_index",
    "beat_sample",
    "proj_line_lock_sl",
    "temporal_area_loss",
    "proj_line_lock_xl",
    "present_transverse_freedom",
    "memory_lock_barrier",
    "present_lock_barrier",
    "short_long_explanation_imbalance",
    "temporal_volume_3",
    "temporal_volume_3_raw",
    "log_temporal_volume_3",
    "memory_align",
    "linger",
    "memory_gap",
    "novelty",
]
PRIMARY_BASELINE_VECTOR = [
    "baseline_median_proj_line_lock_sl",
    "baseline_p95_proj_line_lock_sl",
    "baseline_median_temporal_area_loss",
    "baseline_p95_temporal_area_loss",
    "baseline_median_proj_line_lock_xl",
    "baseline_p95_proj_line_lock_xl",
    "baseline_median_present_transverse_freedom",
    "baseline_p05_present_transverse_freedom",
    "baseline_median_memory_lock_barrier",
    "baseline_p95_memory_lock_barrier",
    "baseline_median_present_lock_barrier",
    "baseline_p95_present_lock_barrier",
    "baseline_median_abs_short_long_explanation_imbalance",
    "baseline_p95_abs_short_long_explanation_imbalance",
    "baseline_sign_fraction_short_long_explanation_imbalance_positive",
    "baseline_median_temporal_volume_3",
    "baseline_p95_temporal_volume_3",
    "baseline_median_log_temporal_volume_3",
    "baseline_max_temporal_area_loss_low_run",
    "baseline_max_present_transverse_freedom_low_run",
    "baseline_max_memory_lock_barrier_high_run",
    "baseline_max_present_lock_barrier_high_run",
    "baseline_max_abs_short_long_explanation_imbalance_high_run",
    "baseline_max_log_temporal_volume_3_low_run",
]
BENCHMARK_COLUMNS = PRIMARY_BASELINE_VECTOR + [
    "baseline_median_memory_align",
    "baseline_median_linger",
    "baseline_median_memory_gap",
    "baseline_median_novelty",
    "baseline_median_temporal_volume_3_raw",
    "baseline_p95_temporal_volume_3_raw",
    "memory_lock_barrier_st_over_baseline",
    "present_lock_barrier_st_over_baseline",
    "proj_line_lock_sl_st_abs_relative_change",
    "proj_line_lock_xl_st_abs_relative_change",
    "temporal_area_loss_st_abs_relative_change",
    "present_transverse_freedom_st_abs_relative_change",
    "memory_lock_barrier_st_abs_relative_change",
    "present_lock_barrier_st_abs_relative_change",
    "abs_short_long_explanation_imbalance_st_abs_relative_change",
    "temporal_volume_3_st_abs_relative_change",
]
REQUIRED_LOCAL_COLUMNS = set(
    [
        "record",
        "regime",
        "phenotype_target",
        "baseline_median_proj_line_lock_sl",
        "baseline_median_proj_line_lock_xl",
        "baseline_median_temporal_area_loss",
        "baseline_median_present_transverse_freedom",
        "baseline_median_memory_lock_barrier",
        "baseline_median_present_lock_barrier",
        "baseline_median_abs_short_long_explanation_imbalance",
        "baseline_median_temporal_volume_3",
        "baseline_median_log_temporal_volume_3",
        "st_median_proj_line_lock_sl",
        "st_median_proj_line_lock_xl",
        "st_median_temporal_area_loss",
        "st_median_present_transverse_freedom",
        "st_median_memory_lock_barrier",
        "st_median_present_lock_barrier",
    ]
)


@dataclass(frozen=True)
class MemoryTaxonomyConfig:
    run_dir: Path
    out_dir_name: str = "memory_taxonomy_v2"
    beta_short: float = 0.10
    beta_long: float = 0.01
    baseline_beats: int = 500
    checkpoint_every: int = 5
    exemplar_records: tuple[str, ...] = ("s20021", "s20041", "s20151", "s30742")
    records: tuple[str, ...] = ()


def json_safe(value: object) -> object:
    return mem.json_safe(value)


def ensure_output_dir(run_dir: Path, out_dir_name: str) -> Path:
    out_dir = run_dir / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def parse_records_arg(records_arg: str | None, regime_df: pd.DataFrame) -> list[str]:
    if records_arg is None or records_arg.strip().lower() in {"", "all", "*"}:
        return regime_df["record"].astype(str).sort_values().tolist()
    requested = [piece.strip() for piece in records_arg.split(",") if piece.strip()]
    known = set(regime_df["record"].astype(str))
    missing = sorted(set(requested) - known)
    if missing:
        raise ValueError(f"Unknown records requested: {', '.join(missing)}")
    return requested


def safe_relative_change(baseline_value: float, st_value: float) -> float:
    if not np.isfinite(baseline_value) or not np.isfinite(st_value):
        return np.nan
    return float(abs(st_value - baseline_value) / (abs(baseline_value) + EPS))


def auc_one_vs_rest(scores: np.ndarray, labels: np.ndarray, positive_label: str) -> float:
    mask = np.isfinite(scores)
    x = scores[mask]
    y = labels[mask] == positive_label
    n_pos = int(np.sum(y))
    n_neg = int(np.sum(~y))
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    pos_ranks = float(np.sum(ranks[y]))
    return float((pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def best_oriented_auc(scores: np.ndarray, labels: np.ndarray, positive_label: str) -> tuple[float, str]:
    auc_high = auc_one_vs_rest(scores, labels, positive_label)
    auc_low = auc_one_vs_rest(-scores, labels, positive_label)
    if not np.isfinite(auc_high) and not np.isfinite(auc_low):
        return np.nan, "higher"
    if not np.isfinite(auc_low) or (np.isfinite(auc_high) and auc_high >= auc_low):
        return auc_high, "higher"
    return auc_low, "lower"


def build_probe_cfg(cfg: MemoryTaxonomyConfig, record: str) -> mem.MemoryConfig:
    return mem.MemoryConfig(
        beta_short=cfg.beta_short,
        beta_long=cfg.beta_long,
        baseline_beats=cfg.baseline_beats,
        records=(record,),
        run_dir=cfg.run_dir,
    )


def load_cached_local_records(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def baseline_cache_path(out_dir: Path, record: str) -> Path:
    cache_dir = out_dir / "baseline_window_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{record}_baseline500.csv"


def load_baseline_cache(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def local_row_is_compatible(row: dict[str, Any]) -> bool:
    return REQUIRED_LOCAL_COLUMNS.issubset(set(row.keys()))


def baseline_cache_is_compatible(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return False
    return set(BASELINE_CACHE_COLUMNS).issubset(set(header.columns))


def compute_or_load_local_rows(
    regime_df: pd.DataFrame,
    cfg: MemoryTaxonomyConfig,
    out_dir: Path,
) -> pd.DataFrame:
    local_cache_path = out_dir / "memory_taxonomy_local_records.csv"
    cached = load_cached_local_records(local_cache_path)
    requested = list(cfg.records)
    regime_lookup = dict(regime_df[["record", "regime"]].itertuples(index=False, name=None))

    cached_rows: list[dict[str, Any]] = []
    cached_lookup: dict[str, dict[str, Any]] = {}
    if not cached.empty:
        for row in cached.to_dict(orient="records"):
            record = str(row.get("record", ""))
            if record:
                cached_lookup[record] = row

    reusable_records = [
        record
        for record in requested
        if record in cached_lookup
        and local_row_is_compatible(cached_lookup[record])
        and baseline_cache_is_compatible(baseline_cache_path(out_dir, record))
    ]
    if reusable_records:
        print(f"[MEM-TAX-V2] Reusing cached local summaries for {len(reusable_records)} record(s)", flush=True)
        cached_rows.extend([cached_lookup[record] for record in reusable_records])

    missing_records = [record for record in requested if record not in set(reusable_records)]
    exemplar_set = set(cfg.exemplar_records)

    for idx, record in enumerate(missing_records, start=1):
        regime = regime_lookup[record]
        print(
            f"[MEM-TAX-V2] {idx}/{len(missing_records)} {record} ({regime}) "
            f"| beta_short={cfg.beta_short:g} beta_long={cfg.beta_long:g}",
            flush=True,
        )
        t0 = time.time()
        probe_cfg = build_probe_cfg(cfg, record)
        probe_df = mem.build_memory_probe(record, probe_cfg, regime_lookup)
        baseline_df = mem.select_baseline_window(probe_df, cfg.baseline_beats)
        if baseline_df.empty:
            raise RuntimeError(f"{record}: no valid baseline rows after Gram-memory filtering")

        local_row = mem.compute_local_record_summary(probe_df, baseline_beats=cfg.baseline_beats)
        cached_rows.append(local_row)

        baseline_df.loc[:, BASELINE_CACHE_COLUMNS].to_csv(baseline_cache_path(out_dir, record), index=False)

        if record in exemplar_set:
            probe_df.to_csv(out_dir / f"{record}_memory_probe.csv", index=False)
            mem.plot_probe(probe_df, out_dir / f"{record}_memory_probe.png")

        print(
            f"[MEM-TAX-V2]   OK area_xl={local_row['baseline_median_present_transverse_freedom']:.6f} "
            f"barrier_xl={local_row['baseline_median_present_lock_barrier']:.6f} "
            f"logV3={local_row['baseline_median_log_temporal_volume_3']:.6f} "
            f"| {time.time() - t0:.1f}s",
            flush=True,
        )

        if idx % max(1, cfg.checkpoint_every) == 0 or idx == len(missing_records):
            local_df = pd.DataFrame(cached_rows).drop_duplicates(subset=["record"], keep="last")
            local_df = local_df.sort_values(["phenotype_target", "record"]).reset_index(drop=True)
            save_table(local_df, local_cache_path)
            print(f"[MEM-TAX-V2] Checkpoint saved: {local_cache_path}", flush=True)

    local_df = pd.DataFrame(cached_rows).drop_duplicates(subset=["record"], keep="last")
    local_df = local_df.sort_values(["phenotype_target", "record"]).reset_index(drop=True)
    save_table(local_df, local_cache_path)
    return local_df


def compute_thresholds(records: list[str], out_dir: Path) -> dict[str, float]:
    baseline_frames = [load_baseline_cache(baseline_cache_path(out_dir, record)) for record in records]
    pooled = pd.concat(baseline_frames, ignore_index=True)
    thresholds = {
        "temporal_area_loss_low": float(pooled["temporal_area_loss"].quantile(0.10)),
        "present_transverse_freedom_low": float(pooled["present_transverse_freedom"].quantile(0.10)),
        "log_temporal_volume_3_low": float(pooled["log_temporal_volume_3"].quantile(0.10)),
        "memory_lock_barrier_high": float(pooled["memory_lock_barrier"].quantile(0.90)),
        "present_lock_barrier_high": float(pooled["present_lock_barrier"].quantile(0.90)),
        "abs_short_long_explanation_imbalance_high": float(
            pooled["short_long_explanation_imbalance"].abs().quantile(0.90)
        ),
    }
    return thresholds


def apply_threshold_runs(summary_df: pd.DataFrame, out_dir: Path, thresholds: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in summary_df.to_dict(orient="records"):
        baseline_df = load_baseline_cache(baseline_cache_path(out_dir, str(row["record"])))
        baseline_df["abs_short_long_explanation_imbalance"] = baseline_df["short_long_explanation_imbalance"].abs()

        row["baseline_max_temporal_area_loss_low_run"] = mem.max_run(
            (baseline_df["temporal_area_loss"].to_numpy(dtype=float) <= thresholds["temporal_area_loss_low"])
        )
        row["baseline_max_present_transverse_freedom_low_run"] = mem.max_run(
            (
                baseline_df["present_transverse_freedom"].to_numpy(dtype=float)
                <= thresholds["present_transverse_freedom_low"]
            )
        )
        row["baseline_max_log_temporal_volume_3_low_run"] = mem.max_run(
            (baseline_df["log_temporal_volume_3"].to_numpy(dtype=float) <= thresholds["log_temporal_volume_3_low"])
        )
        row["baseline_max_memory_lock_barrier_high_run"] = mem.max_run(
            (baseline_df["memory_lock_barrier"].to_numpy(dtype=float) >= thresholds["memory_lock_barrier_high"])
        )
        row["baseline_max_present_lock_barrier_high_run"] = mem.max_run(
            (
                baseline_df["present_lock_barrier"].to_numpy(dtype=float)
                >= thresholds["present_lock_barrier_high"]
            )
        )
        row["baseline_max_abs_short_long_explanation_imbalance_high_run"] = mem.max_run(
            (
                baseline_df["abs_short_long_explanation_imbalance"].to_numpy(dtype=float)
                >= thresholds["abs_short_long_explanation_imbalance_high"]
            )
        )
        row["proj_line_lock_sl_st_abs_relative_change"] = safe_relative_change(
            float(row["baseline_median_proj_line_lock_sl"]),
            float(row["st_median_proj_line_lock_sl"]),
        )
        row["proj_line_lock_xl_st_abs_relative_change"] = safe_relative_change(
            float(row["baseline_median_proj_line_lock_xl"]),
            float(row["st_median_proj_line_lock_xl"]),
        )
        row["temporal_area_loss_st_abs_relative_change"] = safe_relative_change(
            float(row["baseline_median_temporal_area_loss"]),
            float(row["st_median_temporal_area_loss"]),
        )
        row["present_transverse_freedom_st_abs_relative_change"] = safe_relative_change(
            float(row["baseline_median_present_transverse_freedom"]),
            float(row["st_median_present_transverse_freedom"]),
        )
        row["memory_lock_barrier_st_abs_relative_change"] = safe_relative_change(
            float(row["baseline_median_memory_lock_barrier"]),
            float(row["st_median_memory_lock_barrier"]),
        )
        row["present_lock_barrier_st_abs_relative_change"] = safe_relative_change(
            float(row["baseline_median_present_lock_barrier"]),
            float(row["st_median_present_lock_barrier"]),
        )
        row["abs_short_long_explanation_imbalance_st_abs_relative_change"] = safe_relative_change(
            float(row["baseline_median_abs_short_long_explanation_imbalance"]),
            float(row["st_median_abs_short_long_explanation_imbalance"]),
        )
        row["temporal_volume_3_st_abs_relative_change"] = safe_relative_change(
            float(row["baseline_median_temporal_volume_3"]),
            float(row["st_median_temporal_volume_3"]),
        )
        row["log_temporal_volume_3_st_abs_relative_change"] = safe_relative_change(
            float(row["baseline_median_log_temporal_volume_3"]),
            float(row["st_median_log_temporal_volume_3"]),
        )
        rows.append(row)

    enriched = pd.DataFrame(rows).sort_values(["phenotype_target", "record"]).reset_index(drop=True)
    return enriched


def build_group_tables(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    phenotype_df = (
        summary_df.groupby("phenotype_target")
        .agg(
            n_records=("record", "count"),
            median_proj_line_lock_sl=("baseline_median_proj_line_lock_sl", "median"),
            median_area_sl=("baseline_median_temporal_area_loss", "median"),
            median_proj_line_lock_xl=("baseline_median_proj_line_lock_xl", "median"),
            median_area_xl=("baseline_median_present_transverse_freedom", "median"),
            median_barrier_sl=("baseline_median_memory_lock_barrier", "median"),
            median_barrier_xl=("baseline_median_present_lock_barrier", "median"),
            median_log_volume=("baseline_median_log_temporal_volume_3", "median"),
            median_abs_delta_explain=("baseline_median_abs_short_long_explanation_imbalance", "median"),
            median_memory_align=("baseline_median_memory_align", "median"),
            median_linger=("baseline_median_linger", "median"),
            median_lock_barrier_st_change=("memory_lock_barrier_st_abs_relative_change", "median"),
            median_present_lock_barrier_st_change=("present_lock_barrier_st_abs_relative_change", "median"),
        )
        .reindex(PHENOTYPE_ORDER)
        .reset_index()
    )

    regime_df = (
        summary_df.groupby("regime")
        .agg(
            n_records=("record", "count"),
            median_proj_line_lock_xl=("baseline_median_proj_line_lock_xl", "median"),
            median_area_xl=("baseline_median_present_transverse_freedom", "median"),
            median_barrier_xl=("baseline_median_present_lock_barrier", "median"),
            median_log_volume=("baseline_median_log_temporal_volume_3", "median"),
            median_memory_align=("baseline_median_memory_align", "median"),
            median_linger=("baseline_median_linger", "median"),
        )
        .reset_index()
        .sort_values("median_barrier_xl", ascending=False)
        .reset_index(drop=True)
    )
    return phenotype_df, regime_df


def build_benchmark_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    labels = summary_df["phenotype_target"].to_numpy(dtype=object)
    rows: list[dict[str, Any]] = []
    for metric in BENCHMARK_COLUMNS:
        scores = summary_df[metric].to_numpy(dtype=float)
        auroc, direction = best_oriented_auc(scores, labels, "loose_orbit")
        rows.append(
            {
                "task": "loose_vs_rest",
                "metric": metric,
                "positive_class": "loose_orbit",
                "direction": direction,
                "auroc": auroc,
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "auroc", "metric"], ascending=[True, False, True]).reset_index(drop=True)


def build_high_stiffness_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    subset = summary_df[summary_df["phenotype_target"].isin(["constrained_orbit", "rigid_orbit"])].copy()
    if subset.empty:
        return pd.DataFrame(columns=["task", "metric", "positive_class", "direction", "auroc"])
    labels = subset["phenotype_target"].to_numpy(dtype=object)
    rows: list[dict[str, Any]] = []
    for metric in BENCHMARK_COLUMNS:
        scores = subset[metric].to_numpy(dtype=float)
        auroc, direction = best_oriented_auc(scores, labels, "rigid_orbit")
        rows.append(
            {
                "task": "rigid_vs_constrained",
                "metric": metric,
                "positive_class": "rigid_orbit",
                "direction": direction,
                "auroc": auroc,
            }
        )
    return pd.DataFrame(rows).sort_values(["auroc", "metric"], ascending=[False, True]).reset_index(drop=True)


def nearest_centroid_3way(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    df = summary_df[["record", "phenotype_target"] + PRIMARY_BASELINE_VECTOR].copy()
    predictions: list[dict[str, Any]] = []
    for idx in range(len(df)):
        test = df.iloc[[idx]].copy()
        train = df.drop(df.index[idx]).copy()
        feature_cols = [col for col in PRIMARY_BASELINE_VECTOR if train[col].notna().sum() >= 3]
        if not feature_cols:
            predictions.append(
                {
                    "record": str(test["record"].iloc[0]),
                    "phenotype_target": str(test["phenotype_target"].iloc[0]),
                    "predicted_phenotype": np.nan,
                }
            )
            continue

        mu = train[feature_cols].mean(axis=0)
        sigma = train[feature_cols].std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        z_train = (train[feature_cols] - mu) / sigma
        z_test = ((test[feature_cols] - mu) / sigma).iloc[0]

        centroid_rows = []
        for phenotype in PHENOTYPE_ORDER:
            sdf = z_train[train["phenotype_target"] == phenotype]
            if sdf.empty:
                continue
            centroid = sdf.mean(axis=0)
            dist = float(np.sqrt(np.nansum((z_test.to_numpy(dtype=float) - centroid.to_numpy(dtype=float)) ** 2)))
            centroid_rows.append((phenotype, dist))
        centroid_rows.sort(key=lambda item: item[1])
        predicted = centroid_rows[0][0] if centroid_rows else np.nan
        predictions.append(
            {
                "record": str(test["record"].iloc[0]),
                "phenotype_target": str(test["phenotype_target"].iloc[0]),
                "predicted_phenotype": predicted,
            }
        )

    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.merge(summary_df[["record", "regime"]], on="record", how="left")

    confusion = pd.crosstab(
        pred_df["phenotype_target"],
        pred_df["predicted_phenotype"],
        rownames=["actual"],
        colnames=["predicted"],
        dropna=False,
    ).reindex(index=PHENOTYPE_ORDER, columns=PHENOTYPE_ORDER, fill_value=0)

    f1_values = []
    recall_values = []
    for phenotype in PHENOTYPE_ORDER:
        tp = float(confusion.loc[phenotype, phenotype])
        fp = float(confusion[phenotype].sum() - tp)
        fn = float(confusion.loc[phenotype].sum() - tp)
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        f1 = 2.0 * precision * recall / (precision + recall + EPS)
        f1_values.append(f1)
        recall_values.append(recall)
    macro_f1 = float(np.mean(f1_values))
    macro_balanced_accuracy = float(np.mean(recall_values))

    return pred_df, confusion.reset_index(), macro_f1, macro_balanced_accuracy


def _scatter_by_phenotype(
    summary_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for phenotype, sdf in summary_df.groupby("phenotype_target"):
        ax.scatter(
            sdf[x_col],
            sdf[y_col],
            alpha=0.8,
            color=PHENOTYPE_COLORS[phenotype],
            label=PHENOTYPE_LABELS[phenotype],
            s=46,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_metric_strips(summary_df: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("baseline_median_temporal_area_loss", "Area Loss (sl)"),
        ("baseline_median_present_transverse_freedom", "Present Freedom (xl)"),
        ("baseline_median_memory_lock_barrier", "Lock Barrier (sl)"),
        ("baseline_median_present_lock_barrier", "Present Lock (xl)"),
        ("baseline_median_abs_short_long_explanation_imbalance", "|Delta Explain|"),
        ("baseline_median_log_temporal_volume_3", "log(V3)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    rng = np.random.default_rng(42)
    for ax, (metric, title) in zip(axes.flat, metrics):
        for idx, phenotype in enumerate(PHENOTYPE_ORDER):
            sdf = summary_df[summary_df["phenotype_target"] == phenotype]
            jitter = rng.normal(0.0, 0.04, size=len(sdf))
            ax.scatter(
                np.full(len(sdf), idx) + jitter,
                sdf[metric],
                color=PHENOTYPE_COLORS[phenotype],
                alpha=0.8,
                s=28,
            )
            med = float(np.nanmedian(sdf[metric])) if len(sdf) else np.nan
            ax.hlines(med, idx - 0.22, idx + 0.22, color="black", linewidth=2)
        ax.set_xticks(range(len(PHENOTYPE_ORDER)))
        ax.set_xticklabels([PHENOTYPE_LABELS[p] for p in PHENOTYPE_ORDER], rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_baseline_vs_st_lock_barrier(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for phenotype, sdf in summary_df.groupby("phenotype_target"):
        ax.scatter(
            sdf["baseline_median_memory_lock_barrier"],
            sdf["st_median_memory_lock_barrier"],
            alpha=0.8,
            color=PHENOTYPE_COLORS[phenotype],
            label=PHENOTYPE_LABELS[phenotype],
        )
    min_val = float(
        np.nanmin(
            summary_df[["baseline_median_memory_lock_barrier", "st_median_memory_lock_barrier"]].to_numpy(dtype=float)
        )
    )
    max_val = float(
        np.nanmax(
            summary_df[["baseline_median_memory_lock_barrier", "st_median_memory_lock_barrier"]].to_numpy(dtype=float)
        )
    )
    ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Baseline median memory_lock_barrier")
    ax.set_ylabel("ST median memory_lock_barrier")
    ax.set_title("Baseline vs ST memory lock barrier")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(
    out_dir: Path,
    cfg: MemoryTaxonomyConfig,
    summary_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    regime_group_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    high_stiff_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    macro_f1: float,
    macro_balanced_accuracy: float,
    thresholds: dict[str, float],
) -> None:
    top_loose = benchmark_df.head(5)
    top_rigid = high_stiff_df.head(5)
    lines = [
        "# LTST O(d) Gram-Memory Taxonomy v2",
        "",
        "Full-cohort baseline-first Gram-memory analysis using two EMA memory horizons and quadratic-form summaries only.",
        "",
        "## Setup",
        f"- Run directory: `{cfg.run_dir}`",
        f"- Output directory: `{out_dir}`",
        f"- Short beta: `{cfg.beta_short}`",
        f"- Long beta: `{cfg.beta_long}`",
        f"- Baseline beats per record: `{cfg.baseline_beats}`",
        f"- Records analyzed: `{len(summary_df)}`",
        "",
        "## Cohort Thresholds",
        "\n".join(f"- `{key}` = `{value:.6g}`" for key, value in thresholds.items()),
        "",
        "## Phenotype Summary",
        phenotype_df.to_markdown(index=False),
        "",
        "## Top Loose-vs-Rest Metrics",
        top_loose.to_markdown(index=False),
        "",
        "## High-Stiffness Split (Rigid vs Constrained)",
        top_rigid.to_markdown(index=False),
        "",
        "## 3-Way Nearest-Centroid",
        f"- Macro-F1: `{macro_f1:.4f}`",
        f"- Macro balanced accuracy: `{macro_balanced_accuracy:.4f}`",
        "",
        confusion_df.to_markdown(index=False),
        "",
        "## Regime Summary",
        regime_group_df.to_markdown(index=False),
        "",
        "## Per-Record Predictions",
        pred_df.to_markdown(index=False),
        "",
    ]
    (out_dir / "memory_taxonomy_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LTST O(d) Gram-memory taxonomy from raw LTST records.")
    parser.add_argument("--run-dir", required=True, help="Completed ltst_full_86 run directory for regime labels and outputs.")
    parser.add_argument("--records", default="all", help="Comma-separated record list or 'all'.")
    parser.add_argument("--beta-short", type=float, default=0.10)
    parser.add_argument("--beta-long", type=float, default=0.01)
    parser.add_argument("--baseline-beats", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    regime_df = pd.read_csv(run_dir / "regime.csv")
    records = tuple(parse_records_arg(args.records, regime_df))
    cfg = MemoryTaxonomyConfig(
        run_dir=run_dir,
        beta_short=args.beta_short,
        beta_long=args.beta_long,
        baseline_beats=args.baseline_beats,
        checkpoint_every=args.checkpoint_every,
        records=records,
    )
    out_dir = ensure_output_dir(run_dir, cfg.out_dir_name)

    local_df = compute_or_load_local_rows(regime_df, cfg, out_dir)
    local_df = local_df[local_df["record"].isin(records)].copy().reset_index(drop=True)

    thresholds = compute_thresholds(records=list(records), out_dir=out_dir)
    summary_df = apply_threshold_runs(local_df, out_dir, thresholds)

    phenotype_df, regime_group_df = build_group_tables(summary_df)
    benchmark_df = build_benchmark_table(summary_df)
    high_stiff_df = build_high_stiffness_table(summary_df)
    pred_df, confusion_df, macro_f1, macro_balanced_accuracy = nearest_centroid_3way(summary_df)

    save_table(summary_df, out_dir / "memory_taxonomy_records.csv")
    save_table(phenotype_df, out_dir / "memory_taxonomy_phenotype_summary.csv")
    save_table(regime_group_df, out_dir / "memory_taxonomy_regime_summary.csv")
    save_table(benchmark_df, out_dir / "memory_taxonomy_benchmarks.csv")
    save_table(high_stiff_df, out_dir / "memory_taxonomy_high_stiffness.csv")
    save_table(pred_df, out_dir / "memory_taxonomy_3way.csv")
    save_table(confusion_df, out_dir / "memory_taxonomy_3way_confusion.csv")

    _scatter_by_phenotype(
        summary_df,
        "baseline_median_temporal_area_loss",
        "baseline_median_present_transverse_freedom",
        out_dir / "figure_area_sl_vs_area_xl.png",
        "Temporal area loss vs present transverse freedom",
        "Baseline median temporal_area_loss",
        "Baseline median present_transverse_freedom",
    )
    _scatter_by_phenotype(
        summary_df,
        "baseline_median_memory_lock_barrier",
        "baseline_median_present_lock_barrier",
        out_dir / "figure_barrier_sl_vs_barrier_xl.png",
        "Memory lock barrier vs present lock barrier",
        "Baseline median memory_lock_barrier",
        "Baseline median present_lock_barrier",
    )
    _scatter_by_phenotype(
        summary_df,
        "baseline_median_log_temporal_volume_3",
        "baseline_median_abs_short_long_explanation_imbalance",
        out_dir / "figure_log_volume_vs_delta_explain.png",
        "Temporal 3-volume vs short-long explanation imbalance",
        "Baseline median log_temporal_volume_3",
        "Baseline median abs(short_long_explanation_imbalance)",
    )
    plot_metric_strips(summary_df, out_dir / "figure_memory_metric_strips.png")
    plot_baseline_vs_st_lock_barrier(summary_df, out_dir / "figure_baseline_vs_st_lock_barrier.png")

    metadata = {
        "config": json_safe(asdict(cfg)),
        "output_dir": str(out_dir),
        "n_records": int(len(summary_df)),
        "thresholds": json_safe(thresholds),
        "three_way_macro_f1": float(macro_f1),
        "three_way_macro_balanced_accuracy": float(macro_balanced_accuracy),
    }
    with (out_dir / "memory_taxonomy_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(json_safe(metadata), fp, indent=2)

    write_report(
        out_dir=out_dir,
        cfg=cfg,
        summary_df=summary_df,
        phenotype_df=phenotype_df,
        regime_group_df=regime_group_df,
        benchmark_df=benchmark_df,
        high_stiff_df=high_stiff_df,
        pred_df=pred_df,
        confusion_df=confusion_df,
        macro_f1=macro_f1,
        macro_balanced_accuracy=macro_balanced_accuracy,
        thresholds=thresholds,
    )

    print(f"[MEM-TAX-V2] Output directory: {out_dir}", flush=True)
    print(phenotype_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
