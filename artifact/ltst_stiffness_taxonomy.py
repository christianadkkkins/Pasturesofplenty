from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-9
PHENOTYPE_ORDER = ["loose_orbit", "constrained_orbit", "rigid_orbit"]
PHENOTYPE_TO_CODE = {label: idx for idx, label in enumerate(PHENOTYPE_ORDER)}
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
REGIME_TO_PHENOTYPE = {
    "angle_only": "loose_orbit",
    "angle_first": "loose_orbit",
    "long_only": "constrained_orbit",
    "long_first": "constrained_orbit",
    "both_same_horizon": "constrained_orbit",
    "neither": "rigid_orbit",
}
BEAT_COLUMNS = [
    "beat_sample",
    "st_event",
    "st_entry",
    "poincare_b",
    "drift_norm",
    "kernel_score_angle",
    "kernel_score_long",
]
EXEMPLAR_RECORDS = ("s20021", "s20041", "s30742")


@dataclass(frozen=True)
class TaxonomyConfig:
    run_dir: Path
    out_dir_name: str = "stiffness_taxonomy"
    primary_window: int = 500
    baseline_windows: tuple[int, ...] = (250, 500, 1000)
    exemplar_records: tuple[str, ...] = EXEMPLAR_RECORDS


def find_project_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "artifact").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root containing artifact/.")


def find_latest_ltst_run(project_root: Path) -> Path:
    runs_dir = project_root / "artifact" / "runs"
    candidates = sorted(runs_dir.glob("ltst_full_86_*"))
    if not candidates:
        raise FileNotFoundError("No ltst_full_86_* run directory found under artifact/runs.")
    return candidates[-1]


def ensure_output_dir(run_dir: Path, out_dir_name: str) -> Path:
    out_dir = run_dir / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def safe_median(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    return float(clean.median())


def safe_quantile(series: pd.Series, q: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    return float(clean.quantile(q))


def safe_relative_change(baseline_value: float, st_value: float) -> float:
    if not np.isfinite(baseline_value) or not np.isfinite(st_value):
        return np.nan
    return float(abs(st_value - baseline_value) / (abs(baseline_value) + EPS))


def json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def phenotype_from_regime(regime: str) -> str:
    if regime not in REGIME_TO_PHENOTYPE:
        raise KeyError(f"Unknown regime: {regime}")
    return REGIME_TO_PHENOTYPE[regime]


def read_summary_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    regime_df = pd.read_csv(run_dir / "regime.csv")
    invariant_df = pd.read_csv(run_dir / "invariant_summary.csv")
    return regime_df, invariant_df


def read_beat_frame(beat_level_dir: Path, record: str) -> pd.DataFrame:
    parquet_path = beat_level_dir / f"{record}.parquet"
    csv_path = beat_level_dir / f"{record}.csv"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path, columns=BEAT_COLUMNS)
        except Exception:
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path, usecols=BEAT_COLUMNS)
    raise FileNotFoundError(f"Missing beat-level file for {record}.")


def summarize_slice(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_median_si": safe_median(df["stiffness_index"]),
        f"{prefix}_p05_si": safe_quantile(df["stiffness_index"], 0.05),
        f"{prefix}_median_stiffness_margin": safe_median(df["stiffness_margin"]),
        f"{prefix}_median_poincare_b": safe_median(df["poincare_b"]),
        f"{prefix}_median_drift_norm": safe_median(df["drift_norm"]),
        f"{prefix}_median_angle_score": safe_median(df["kernel_score_angle"]),
        f"{prefix}_median_long_score": safe_median(df["kernel_score_long"]),
    }


def build_record_windows(
    regime_df: pd.DataFrame,
    beat_level_dir: Path,
    baseline_windows: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    windows = tuple(sorted(set(int(v) for v in baseline_windows)))

    for idx, row in regime_df.sort_values("record").iterrows():
        record = str(row["record"])
        regime = str(row["regime"])
        phenotype_target = phenotype_from_regime(regime)
        print(f"[STIFFNESS] {idx + 1}/{len(regime_df)} {record} ({regime})", flush=True)

        beat_df = read_beat_frame(beat_level_dir, record)
        beat_df = beat_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["poincare_b", "drift_norm", "kernel_score_angle", "kernel_score_long"]
        )
        beat_df = beat_df.sort_values("beat_sample").copy()
        beat_df["st_event"] = beat_df["st_event"].astype(bool)
        beat_df["stiffness_index"] = beat_df["poincare_b"] / (beat_df["drift_norm"] + EPS)
        beat_df["stiffness_margin"] = beat_df["poincare_b"] - beat_df["drift_norm"]

        non_st = beat_df.loc[~beat_df["st_event"]].copy()
        st = beat_df.loc[beat_df["st_event"]].copy()
        all_non_st_summary = summarize_slice(non_st, "all_non_st")
        st_summary = summarize_slice(st, "st")

        for baseline_window in windows:
            baseline = non_st.head(int(baseline_window)).copy()
            baseline_summary = summarize_slice(baseline, "baseline")
            row_out: dict[str, object] = {
                "record": record,
                "regime": regime,
                "phenotype_target": phenotype_target,
                "baseline_window": int(baseline_window),
                "baseline_non_st_count": int(len(baseline)),
                "all_non_st_count": int(len(non_st)),
                "st_count": int(len(st)),
                "baseline_short": bool(len(baseline) < int(baseline_window)),
                **baseline_summary,
                **st_summary,
                **all_non_st_summary,
            }
            row_out["si_abs_relative_change"] = safe_relative_change(
                float(row_out["baseline_median_si"]),
                float(row_out["st_median_si"]),
            )
            row_out["angle_abs_relative_change"] = safe_relative_change(
                float(row_out["baseline_median_angle_score"]),
                float(row_out["st_median_angle_score"]),
            )
            row_out["long_abs_relative_change"] = safe_relative_change(
                float(row_out["baseline_median_long_score"]),
                float(row_out["st_median_long_score"]),
            )
            rows.append(row_out)

    return pd.DataFrame(rows)


def midpoint_candidates(values: pd.Series) -> list[float]:
    clean = np.sort(pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().unique())
    if len(clean) < 3:
        raise ValueError("Need at least three unique values to search two cutpoints.")
    return [float((clean[i] + clean[i + 1]) / 2.0) for i in range(len(clean) - 1)]


def predict_scalar(values: pd.Series | np.ndarray, t1: float, t2: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.empty(len(arr), dtype=object)
    out[arr < t1] = "loose_orbit"
    out[(arr >= t1) & (arr < t2)] = "constrained_orbit"
    out[arr >= t2] = "rigid_orbit"
    return out


def encode_labels(labels: list[str] | pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray([PHENOTYPE_TO_CODE[str(label)] for label in list(labels)], dtype=np.int64)


def predict_scalar_codes(values: pd.Series | np.ndarray, t1: float, t2: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.empty(len(arr), dtype=np.int64)
    out[arr < t1] = 0
    out[(arr >= t1) & (arr < t2)] = 1
    out[arr >= t2] = 2
    return out


def confusion_counts(y_true_codes: np.ndarray, y_pred_codes: np.ndarray) -> np.ndarray:
    bins = np.bincount((y_true_codes * 3) + y_pred_codes, minlength=9)
    return bins.reshape(3, 3)


def confusion_dataframe(y_true: list[str] | pd.Series, y_pred: list[str] | pd.Series) -> pd.DataFrame:
    y_true_codes = encode_labels(y_true)
    y_pred_codes = encode_labels(y_pred)
    cm = confusion_counts(y_true_codes, y_pred_codes)
    return pd.DataFrame(cm, index=PHENOTYPE_ORDER, columns=PHENOTYPE_ORDER)


def macro_balanced_accuracy_from_counts(cm: np.ndarray) -> float:
    row_sums = cm.sum(axis=1).astype(float)
    recalls = np.divide(np.diag(cm).astype(float), row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)
    return float(np.mean(recalls))


def macro_f1_from_counts(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    precision_denom = cm.sum(axis=0).astype(float)
    recall_denom = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, precision_denom, out=np.zeros_like(tp), where=precision_denom > 0)
    recall = np.divide(tp, recall_denom, out=np.zeros_like(tp), where=recall_denom > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    return float(np.mean(f1))


def macro_balanced_accuracy(cm: pd.DataFrame) -> float:
    return macro_balanced_accuracy_from_counts(cm.to_numpy(dtype=float))


def macro_f1(cm: pd.DataFrame) -> float:
    return macro_f1_from_counts(cm.to_numpy(dtype=float))


def minimum_threshold_margin(values: pd.Series | np.ndarray, t1: float, t2: float) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.min(np.minimum(np.abs(arr - t1), np.abs(arr - t2))))


def search_best_thresholds(df: pd.DataFrame, value_col: str, target_col: str = "phenotype_target") -> dict[str, object]:
    work = df[[value_col, target_col]].dropna().copy()
    candidates = midpoint_candidates(work[value_col])
    values = np.asarray(work[value_col], dtype=float)
    y_true_codes = encode_labels(work[target_col])
    best: dict[str, object] | None = None

    for i, t1 in enumerate(candidates[:-1]):
        for t2 in candidates[i + 1 :]:
            pred_codes = predict_scalar_codes(values, t1, t2)
            cm_counts = confusion_counts(y_true_codes, pred_codes)
            f1 = macro_f1_from_counts(cm_counts)
            bal_acc = macro_balanced_accuracy_from_counts(cm_counts)
            margin = minimum_threshold_margin(values, t1, t2)
            result = {
                "t1": float(t1),
                "t2": float(t2),
                "macro_f1": float(f1),
                "macro_balanced_accuracy": float(bal_acc),
                "minimum_margin": float(margin),
                "confusion_matrix": pd.DataFrame(cm_counts, index=PHENOTYPE_ORDER, columns=PHENOTYPE_ORDER),
                "predicted_codes": pred_codes,
                "predicted": np.asarray([PHENOTYPE_ORDER[idx] for idx in pred_codes], dtype=object),
            }
            if best is None:
                best = result
                continue
            if result["macro_f1"] > best["macro_f1"]:
                best = result
                continue
            if result["macro_f1"] < best["macro_f1"]:
                continue
            if result["macro_balanced_accuracy"] > best["macro_balanced_accuracy"]:
                best = result
                continue
            if result["macro_balanced_accuracy"] < best["macro_balanced_accuracy"]:
                continue
            if result["minimum_margin"] > best["minimum_margin"]:
                best = result
                continue
            if result["minimum_margin"] < best["minimum_margin"]:
                continue
            if result["t1"] < best["t1"] or (
                np.isclose(result["t1"], best["t1"]) and result["t2"] < best["t2"]
            ):
                best = result

    if best is None:
        raise RuntimeError(f"Could not identify threshold pair for {value_col}.")
    return best


def leave_one_out_predictions(df: pd.DataFrame, value_col: str, target_col: str = "phenotype_target") -> tuple[pd.DataFrame, dict[str, object]]:
    work = df[["record", value_col, target_col]].dropna().copy()
    rows: list[dict[str, object]] = []

    for _, row in work.iterrows():
        train = work.loc[work["record"] != row["record"]].copy()
        model = search_best_thresholds(train, value_col, target_col)
        pred = predict_scalar(np.asarray([float(row[value_col])]), float(model["t1"]), float(model["t2"]))[0]
        rows.append(
            {
                "record": str(row["record"]),
                "loo_predicted_phenotype": str(pred),
                "loo_t1": float(model["t1"]),
                "loo_t2": float(model["t2"]),
            }
        )

    pred_df = pd.DataFrame(rows)
    merged = work.merge(pred_df, on="record", how="left")
    cm = confusion_dataframe(merged[target_col].tolist(), merged["loo_predicted_phenotype"].tolist())
    metrics = {
        "confusion_matrix": cm,
        "macro_f1": macro_f1(cm),
        "macro_balanced_accuracy": macro_balanced_accuracy(cm),
    }
    return pred_df, metrics


def auc_one_vs_rest(labels: pd.Series, raw_scores: pd.Series, positive_label: str, invert: bool = False) -> float:
    work = pd.DataFrame({"label": labels, "score": raw_scores}).dropna().copy()
    if invert:
        work["score"] = -pd.to_numeric(work["score"], errors="coerce")
    work["is_positive"] = work["label"] == positive_label
    work = work.dropna(subset=["score"])
    n_pos = int(work["is_positive"].sum())
    n_neg = int((~work["is_positive"]).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.to_numeric(work["score"], errors="coerce").rank(method="average")
    sum_ranks_pos = float(ranks[work["is_positive"]].sum())
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def benchmark_scalar(df: pd.DataFrame, value_col: str, label: str) -> dict[str, object]:
    work = df[["record", "phenotype_target", value_col]].dropna().copy()
    full_model = search_best_thresholds(work, value_col, "phenotype_target")
    full_pred = predict_scalar(work[value_col], float(full_model["t1"]), float(full_model["t2"]))
    apparent_cm = confusion_dataframe(work["phenotype_target"].tolist(), full_pred.tolist())
    loo_pred_df, loo_metrics = leave_one_out_predictions(work, value_col, "phenotype_target")
    return {
        "feature": label,
        "value_col": value_col,
        "global_t1": float(full_model["t1"]),
        "global_t2": float(full_model["t2"]),
        "apparent_macro_f1": float(macro_f1(apparent_cm)),
        "apparent_macro_balanced_accuracy": float(macro_balanced_accuracy(apparent_cm)),
        "loo_macro_f1": float(loo_metrics["macro_f1"]),
        "loo_macro_balanced_accuracy": float(loo_metrics["macro_balanced_accuracy"]),
        "rigid_vs_rest_auc": float(
            auc_one_vs_rest(work["phenotype_target"], work[value_col], positive_label="rigid_orbit", invert=False)
        ),
        "loose_vs_rest_auc": float(
            auc_one_vs_rest(work["phenotype_target"], work[value_col], positive_label="loose_orbit", invert=True)
        ),
        "loo_predictions": loo_pred_df,
        "loo_confusion_matrix": loo_metrics["confusion_matrix"],
        "apparent_confusion_matrix": apparent_cm,
    }


def add_summary_ratio(primary_df: pd.DataFrame, invariant_df: pd.DataFrame) -> pd.DataFrame:
    inv = invariant_df.copy()
    inv["summary_ratio"] = pd.to_numeric(inv["poincare_b_p50"], errors="coerce") / (
        pd.to_numeric(inv["p90_drift_norm"], errors="coerce") + EPS
    )
    return primary_df.merge(inv[["record", "summary_ratio"]], on="record", how="left")


def build_counts_table(primary_df: pd.DataFrame) -> pd.DataFrame:
    counts = primary_df.groupby(["phenotype_target", "regime"]).size().rename("count").reset_index()
    counts["phenotype_label"] = counts["phenotype_target"].map(PHENOTYPE_LABELS)
    return counts.sort_values(["phenotype_target", "regime"]).reset_index(drop=True)


def build_results_table(primary_df: pd.DataFrame, benchmark: dict[str, object]) -> pd.DataFrame:
    counts = primary_df["phenotype_target"].value_counts()
    return pd.DataFrame(
        [
            {
                "n_records": int(len(primary_df)),
                "n_loose_orbit": int(counts.get("loose_orbit", 0)),
                "n_constrained_orbit": int(counts.get("constrained_orbit", 0)),
                "n_rigid_orbit": int(counts.get("rigid_orbit", 0)),
                "global_t1": float(benchmark["global_t1"]),
                "global_t2": float(benchmark["global_t2"]),
                "apparent_macro_f1": float(benchmark["apparent_macro_f1"]),
                "apparent_macro_balanced_accuracy": float(benchmark["apparent_macro_balanced_accuracy"]),
                "loo_macro_f1": float(benchmark["loo_macro_f1"]),
                "loo_macro_balanced_accuracy": float(benchmark["loo_macro_balanced_accuracy"]),
            }
        ]
    )


def build_stability_table(primary_df: pd.DataFrame) -> pd.DataFrame:
    def summarize(col: str) -> dict[str, object]:
        vals = pd.to_numeric(primary_df[col], errors="coerce").dropna()
        return {
            "metric": col,
            "median_abs_relative_change": float(vals.median()) if not vals.empty else np.nan,
            "frac_abs_relative_change_le_0_10": float((vals <= 0.10).mean()) if not vals.empty else np.nan,
            "frac_abs_relative_change_le_0_20": float((vals <= 0.20).mean()) if not vals.empty else np.nan,
        }

    return pd.DataFrame(
        [
            summarize("si_abs_relative_change"),
            summarize("angle_abs_relative_change"),
            summarize("long_abs_relative_change"),
        ]
    )


def build_sensitivity_table(
    window_df: pd.DataFrame,
    invariant_df: pd.DataFrame,
    baseline_windows: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    phenotype_lookup = (
        window_df.loc[window_df["baseline_window"] == int(min(baseline_windows)), ["record", "phenotype_target"]]
        .drop_duplicates()
        .set_index("record")["phenotype_target"]
    )

    for baseline_window in baseline_windows:
        subset = window_df.loc[window_df["baseline_window"] == int(baseline_window)].copy()
        result = benchmark_scalar(subset, "baseline_median_si", f"baseline_si_{baseline_window}")
        rows.append(
            {
                "analysis": f"baseline_si_{baseline_window}",
                "global_t1": float(result["global_t1"]),
                "global_t2": float(result["global_t2"]),
                "apparent_macro_f1": float(result["apparent_macro_f1"]),
                "apparent_macro_balanced_accuracy": float(result["apparent_macro_balanced_accuracy"]),
                "loo_macro_f1": float(result["loo_macro_f1"]),
                "loo_macro_balanced_accuracy": float(result["loo_macro_balanced_accuracy"]),
            }
        )

    inv = invariant_df.copy()
    inv["phenotype_target"] = inv["record"].map(phenotype_lookup)
    inv["summary_ratio"] = pd.to_numeric(inv["poincare_b_p50"], errors="coerce") / (
        pd.to_numeric(inv["p90_drift_norm"], errors="coerce") + EPS
    )
    inv = inv.dropna(subset=["summary_ratio", "phenotype_target"]).copy()
    summary_ratio_result = benchmark_scalar(inv, "summary_ratio", "summary_ratio")
    rows.append(
        {
            "analysis": "summary_ratio",
            "global_t1": float(summary_ratio_result["global_t1"]),
            "global_t2": float(summary_ratio_result["global_t2"]),
            "apparent_macro_f1": float(summary_ratio_result["apparent_macro_f1"]),
            "apparent_macro_balanced_accuracy": float(summary_ratio_result["apparent_macro_balanced_accuracy"]),
            "loo_macro_f1": float(summary_ratio_result["loo_macro_f1"]),
            "loo_macro_balanced_accuracy": float(summary_ratio_result["loo_macro_balanced_accuracy"]),
        }
    )

    return pd.DataFrame(rows)


def plot_violin_strip(primary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = [primary_df.loc[primary_df["phenotype_target"] == label, "baseline_median_si"].dropna().to_numpy() for label in PHENOTYPE_ORDER]
    positions = np.arange(1, len(PHENOTYPE_ORDER) + 1)
    violins = ax.violinplot(groups, positions=positions, widths=0.8, showmeans=False, showmedians=True, showextrema=False)
    for body, label in zip(violins["bodies"], PHENOTYPE_ORDER):
        body.set_facecolor(PHENOTYPE_COLORS[label])
        body.set_alpha(0.35)
        body.set_edgecolor("black")
    if "cmedians" in violins:
        violins["cmedians"].set_color("black")
        violins["cmedians"].set_linewidth(1.5)

    for pos, label in zip(positions, PHENOTYPE_ORDER):
        values = primary_df.loc[primary_df["phenotype_target"] == label, "baseline_median_si"].dropna().to_numpy()
        jitter = np.linspace(-0.16, 0.16, len(values)) if len(values) > 1 else np.zeros(len(values))
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            color=PHENOTYPE_COLORS[label],
            edgecolor="black",
            linewidth=0.4,
            s=36,
            alpha=0.9,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([PHENOTYPE_LABELS[label] for label in PHENOTYPE_ORDER])
    ax.set_ylabel("Baseline Stiffness Index")
    ax.set_title("Baseline Stiffness Index by Cardiac Phenotype")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_threshold_histogram(primary_df: pd.DataFrame, t1: float, t2: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(
        float(primary_df["baseline_median_si"].min()),
        float(primary_df["baseline_median_si"].max()),
        18,
    )
    for label in PHENOTYPE_ORDER:
        vals = primary_df.loc[primary_df["phenotype_target"] == label, "baseline_median_si"].dropna().to_numpy()
        ax.hist(vals, bins=bins, alpha=0.45, label=PHENOTYPE_LABELS[label], color=PHENOTYPE_COLORS[label])
    ax.axvline(float(t1), color="black", linestyle="--", linewidth=1.5, label=f"t1 = {t1:.2f}")
    ax.axvline(float(t2), color="black", linestyle=":", linewidth=1.5, label=f"t2 = {t2:.2f}")
    ax.set_xlabel("Baseline Stiffness Index")
    ax.set_ylabel("Record Count")
    ax.set_title("Baseline Stiffness Thresholds for Phenotype Taxonomy")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_baseline_st_scatter(primary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    x = pd.to_numeric(primary_df["baseline_median_si"], errors="coerce")
    y = pd.to_numeric(primary_df["st_median_si"], errors="coerce")
    finite = np.isfinite(x) & np.isfinite(y)
    plot_df = primary_df.loc[finite].copy()

    for label in PHENOTYPE_ORDER:
        sub = plot_df.loc[plot_df["phenotype_target"] == label]
        ax.scatter(
            sub["baseline_median_si"],
            sub["st_median_si"],
            label=PHENOTYPE_LABELS[label],
            color=PHENOTYPE_COLORS[label],
            edgecolor="black",
            linewidth=0.4,
            s=42,
            alpha=0.9,
        )

    lo = float(min(plot_df["baseline_median_si"].min(), plot_df["st_median_si"].min()))
    hi = float(max(plot_df["baseline_median_si"].max(), plot_df["st_median_si"].max()))
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.2, label="Identity")
    ax.set_xlabel("Baseline SI (first 500 non-ST beats)")
    ax.set_ylabel("ST SI (all ST beats)")
    ax.set_title("Baseline vs ST Stiffness Index")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_exemplar_panel(run_dir: Path, out_path: Path, exemplar_records: tuple[str, ...]) -> None:
    fig, axs = plt.subplots(len(exemplar_records), 3, figsize=(16, 4.5 * len(exemplar_records)), sharex=False)
    if len(exemplar_records) == 1:
        axs = np.asarray([axs])

    for row_idx, record in enumerate(exemplar_records):
        beat_df = read_beat_frame(run_dir / "beat_level", record)
        beat_df = beat_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["poincare_b", "drift_norm", "kernel_score_angle", "kernel_score_long"]
        )
        beat_df = beat_df.sort_values("beat_sample").copy()
        beat_df["st_event"] = beat_df["st_event"].astype(bool)
        beat_df["stiffness_index"] = beat_df["poincare_b"] / (beat_df["drift_norm"] + EPS)
        plot_df = beat_df[
            ["beat_sample", "st_event", "poincare_b", "drift_norm", "stiffness_index", "kernel_score_angle", "kernel_score_long"]
        ].copy()
        plot_df["poincare_b_roll"] = plot_df["poincare_b"].rolling(500, min_periods=50).median()
        plot_df["drift_norm_roll"] = plot_df["drift_norm"].rolling(500, min_periods=50).median()
        plot_df["stiffness_index_roll"] = plot_df["stiffness_index"].rolling(500, min_periods=50).median()
        plot_df["angle_roll"] = plot_df["kernel_score_angle"].rolling(500, min_periods=50).median()
        plot_df["long_roll"] = plot_df["kernel_score_long"].rolling(500, min_periods=50).median()

        ax0, ax1, ax2 = axs[row_idx]
        ax0.plot(plot_df["beat_sample"], plot_df["poincare_b_roll"], color="#6a3d9a", label="poincare_b")
        ax0.plot(plot_df["beat_sample"], plot_df["drift_norm_roll"], color="#1f78b4", label="drift_norm")
        ax0.set_ylabel(f"{record}\nInvariant")
        ax0.legend(loc="upper right")
        ax0.grid(True, alpha=0.25)

        ax1.plot(plot_df["beat_sample"], plot_df["stiffness_index_roll"], color="#2ca02c", label="stiffness_index")
        ax1.set_ylabel("Stiffness")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.25)

        ax2.plot(plot_df["beat_sample"], plot_df["angle_roll"], color="#d62728", label="angle")
        ax2.plot(plot_df["beat_sample"], plot_df["long_roll"], color="#ff9896", label="long")
        ax2.set_ylabel("Kernel score")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.25)

        st_mask = plot_df["st_event"].to_numpy(dtype=bool)
        samples = plot_df["beat_sample"].to_numpy()
        start = None
        for i, inside in enumerate(st_mask):
            if inside and start is None:
                start = samples[i]
            elif not inside and start is not None:
                end = samples[i]
                for ax in (ax0, ax1, ax2):
                    ax.axvspan(start, end, color="gray", alpha=0.08)
                start = None
        if start is not None:
            end = samples[-1]
            for ax in (ax0, ax1, ax2):
                ax.axvspan(start, end, color="gray", alpha=0.08)

        ax2.set_xlabel("Beat sample")

    axs[0][0].set_title("Rolling Median poincare_b vs drift_norm")
    axs[0][1].set_title("Rolling Median Stiffness Index")
    axs[0][2].set_title("Rolling Median Angle vs Long Scores")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_report(
    cfg: TaxonomyConfig,
    out_dir: Path,
    results_table: pd.DataFrame,
    stability_table: pd.DataFrame,
) -> str:
    result = results_table.iloc[0]
    stable_10 = stability_table.loc[stability_table["metric"] == "si_abs_relative_change", "frac_abs_relative_change_le_0_10"].iloc[0]
    stable_20 = stability_table.loc[stability_table["metric"] == "si_abs_relative_change", "frac_abs_relative_change_le_0_20"].iloc[0]
    si_change = stability_table.loc[stability_table["metric"] == "si_abs_relative_change", "median_abs_relative_change"].iloc[0]
    angle_change = stability_table.loc[stability_table["metric"] == "angle_abs_relative_change", "median_abs_relative_change"].iloc[0]
    long_change = stability_table.loc[stability_table["metric"] == "long_abs_relative_change", "median_abs_relative_change"].iloc[0]
    return f"""# LTST Stiffness Taxonomy Report

## Setup
- Source run: `{cfg.run_dir}`
- Output directory: `{out_dir}`
- Canonical stiffness variable: baseline non-ST median of per-beat `poincare_b / (drift_norm + eps)`
- Primary baseline window: `{cfg.primary_window}` non-ST beats
- Sensitivity windows: `{list(cfg.baseline_windows)}`

## Primary Phenotype Taxonomy
- Records analyzed: `{int(result['n_records'])}`
- Loose orbit count: `{int(result['n_loose_orbit'])}`
- Constrained orbit count: `{int(result['n_constrained_orbit'])}`
- Rigid orbit count: `{int(result['n_rigid_orbit'])}`
- Chosen full-cohort thresholds: `t1 = {float(result['global_t1']):.6f}`, `t2 = {float(result['global_t2']):.6f}`
- Apparent macro-F1: `{float(result['apparent_macro_f1']):.6f}`
- Apparent macro balanced accuracy: `{float(result['apparent_macro_balanced_accuracy']):.6f}`
- Leave-one-record-out macro-F1: `{float(result['loo_macro_f1']):.6f}`
- Leave-one-record-out macro balanced accuracy: `{float(result['loo_macro_balanced_accuracy']):.6f}`

## Stability Check
- Median absolute SI change from baseline to ST: `{float(si_change):.6f}`
- Fraction of records with SI change <= 10%: `{float(stable_10):.6f}`
- Fraction of records with SI change <= 20%: `{float(stable_20):.6f}`
- Median absolute angle-score change: `{float(angle_change):.6f}`
- Median absolute long-score change: `{float(long_change):.6f}`

## Outputs
- `stiffness_taxonomy_records.csv`
- `stiffness_taxonomy_results.csv`
- `stiffness_taxonomy_confusion_matrix.csv`
- `stiffness_taxonomy_stability.csv`
- `stiffness_taxonomy_sensitivity.csv`
- `stiffness_taxonomy_benchmarks.csv`
- `figure_stiffness_by_phenotype.png`
- `figure_stiffness_histogram.png`
- `figure_stiffness_baseline_vs_st.png`
- `figure_stiffness_exemplar_panel.png`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the LTST stiffness phenotype taxonomy from a completed ltst_full_86 run.")
    parser.add_argument("--run-dir", default=None, help="Path to a completed ltst_full_86_* run directory.")
    parser.add_argument("--out-dir-name", default="stiffness_taxonomy")
    parser.add_argument("--primary-window", type=int, default=500)
    parser.add_argument("--baseline-windows", default="250,500,1000")
    return parser.parse_args()


def main() -> None:
    project_root = find_project_root()
    args = parse_args()
    baseline_windows = tuple(int(part.strip()) for part in str(args.baseline_windows).split(",") if part.strip())
    run_dir = Path(args.run_dir).resolve() if args.run_dir else find_latest_ltst_run(project_root)
    cfg = TaxonomyConfig(
        run_dir=run_dir,
        out_dir_name=str(args.out_dir_name),
        primary_window=int(args.primary_window),
        baseline_windows=baseline_windows,
    )
    out_dir = ensure_output_dir(cfg.run_dir, cfg.out_dir_name)

    regime_df, invariant_df = read_summary_tables(cfg.run_dir)
    windows_path = out_dir / "stiffness_taxonomy_windows.csv"
    if windows_path.exists():
        print(f"[STIFFNESS] Reusing cached window table: {windows_path}", flush=True)
        window_df = pd.read_csv(windows_path)
    else:
        window_df = build_record_windows(regime_df, cfg.run_dir / "beat_level", cfg.baseline_windows)
        window_df.to_csv(windows_path, index=False)

    primary_df = window_df.loc[window_df["baseline_window"] == int(cfg.primary_window)].copy()
    primary_df = add_summary_ratio(primary_df, invariant_df)

    primary_benchmark = benchmark_scalar(primary_df, "baseline_median_si", "baseline_si")
    record_table = primary_df.merge(primary_benchmark["loo_predictions"], on="record", how="left")
    record_table["predicted_phenotype"] = predict_scalar(
        record_table["baseline_median_si"],
        float(primary_benchmark["global_t1"]),
        float(primary_benchmark["global_t2"]),
    )
    record_table["global_t1"] = float(primary_benchmark["global_t1"])
    record_table["global_t2"] = float(primary_benchmark["global_t2"])

    counts_table = build_counts_table(primary_df)
    results_table = build_results_table(primary_df, primary_benchmark)
    stability_table = build_stability_table(primary_df)
    sensitivity_table = build_sensitivity_table(window_df, invariant_df, cfg.baseline_windows)

    benchmark_rows: list[dict[str, object]] = []
    benchmark_results = [
        primary_benchmark,
        benchmark_scalar(primary_df, "baseline_median_poincare_b", "baseline_median_poincare_b"),
        benchmark_scalar(primary_df, "baseline_median_stiffness_margin", "baseline_median_stiffness_margin"),
        benchmark_scalar(primary_df, "summary_ratio", "summary_ratio"),
    ]
    for result in benchmark_results:
        benchmark_rows.append(
            {
                "feature": str(result["feature"]),
                "global_t1": float(result["global_t1"]),
                "global_t2": float(result["global_t2"]),
                "apparent_macro_f1": float(result["apparent_macro_f1"]),
                "apparent_macro_balanced_accuracy": float(result["apparent_macro_balanced_accuracy"]),
                "loo_macro_f1": float(result["loo_macro_f1"]),
                "loo_macro_balanced_accuracy": float(result["loo_macro_balanced_accuracy"]),
                "rigid_vs_rest_auc": float(result["rigid_vs_rest_auc"]),
                "loose_vs_rest_auc": float(result["loose_vs_rest_auc"]),
            }
        )
    benchmark_table = pd.DataFrame(benchmark_rows)

    record_table.to_csv(out_dir / "stiffness_taxonomy_records.csv", index=False)
    counts_table.to_csv(out_dir / "stiffness_taxonomy_counts.csv", index=False)
    results_table.to_csv(out_dir / "stiffness_taxonomy_results.csv", index=False)
    stability_table.to_csv(out_dir / "stiffness_taxonomy_stability.csv", index=False)
    sensitivity_table.to_csv(out_dir / "stiffness_taxonomy_sensitivity.csv", index=False)
    benchmark_table.to_csv(out_dir / "stiffness_taxonomy_benchmarks.csv", index=False)
    primary_benchmark["loo_confusion_matrix"].to_csv(out_dir / "stiffness_taxonomy_confusion_matrix.csv")

    plot_violin_strip(primary_df, out_dir / "figure_stiffness_by_phenotype.png")
    plot_threshold_histogram(
        primary_df,
        float(primary_benchmark["global_t1"]),
        float(primary_benchmark["global_t2"]),
        out_dir / "figure_stiffness_histogram.png",
    )
    plot_baseline_st_scatter(primary_df, out_dir / "figure_stiffness_baseline_vs_st.png")
    plot_exemplar_panel(cfg.run_dir, out_dir / "figure_stiffness_exemplar_panel.png", cfg.exemplar_records)

    metadata = {
        "generated_from_run_dir": str(cfg.run_dir),
        "config": json_safe(asdict(cfg)),
        "n_records": int(len(primary_df)),
        "global_thresholds": {
            "t1": float(primary_benchmark["global_t1"]),
            "t2": float(primary_benchmark["global_t2"]),
        },
        "loo_metrics": {
            "macro_f1": float(primary_benchmark["loo_macro_f1"]),
            "macro_balanced_accuracy": float(primary_benchmark["loo_macro_balanced_accuracy"]),
        },
    }
    with (out_dir / "stiffness_taxonomy_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    report = build_report(cfg, out_dir, results_table, stability_table)
    (out_dir / "stiffness_taxonomy_report.md").write_text(report, encoding="utf-8")

    print(f"Stiffness taxonomy output directory: {out_dir}", flush=True)
    print(results_table.to_string(index=False), flush=True)
    print(stability_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
