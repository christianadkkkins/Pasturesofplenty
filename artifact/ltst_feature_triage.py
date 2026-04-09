from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-12
SEED = 7
TASKS = {
    "detected_any": "regime != neither",
    "loose_orbit": "angle_only or angle_first",
    "rigid_orbit": "regime == neither",
    "guarded_hit": "guarded_cross is present",
}


@dataclass(frozen=True)
class FeatureTriageConfig:
    run_dir: Path
    output_dir: Path
    n_folds: int = 5
    seed: int = SEED


def safe_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score LTST invariant summaries for predictive value, stability, and analog friendliness.")
    parser.add_argument("--run-dir", default=str(Path("artifact") / "runs" / "ltst_full_86_20260405T035505Z"))
    parser.add_argument("--n-folds", type=int, default=5)
    return parser.parse_args()


def collapse_phenotype(regime: str) -> str:
    if regime in {"angle_only", "angle_first"}:
        return "loose_orbit"
    if regime in {"long_only", "long_first", "both_same_horizon"}:
        return "constrained_orbit"
    return "rigid_orbit"


def load_tables(cfg: FeatureTriageConfig) -> pd.DataFrame:
    invariant_df = pd.read_csv(cfg.run_dir / "invariant_summary.csv")
    regime_df = pd.read_csv(cfg.run_dir / "regime.csv")[["record", "regime"]]
    guarded_df = pd.read_csv(cfg.run_dir / "guarded_summary.csv")[["record", "guarded_cross"]]

    df = invariant_df.merge(regime_df, on="record", how="inner").merge(guarded_df, on="record", how="left")
    df["phenotype_target"] = df["regime"].map(collapse_phenotype)
    df["detected_any"] = (df["regime"] != "neither").astype(int)
    df["loose_orbit"] = (df["phenotype_target"] == "loose_orbit").astype(int)
    df["rigid_orbit"] = (df["phenotype_target"] == "rigid_orbit").astype(int)
    df["guarded_hit"] = df["guarded_cross"].notna().astype(int)
    return df


def binary_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isfinite(scores) & np.isfinite(y)
    y = y[mask].astype(int)
    scores = scores[mask].astype(float)
    if len(y) == 0:
        return np.nan
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return np.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=float)
    rank_pos = float(ranks[y == 1].sum())
    auc = (rank_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    y = y.astype(int)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    max_folds = min(n_folds, len(pos_idx), len(neg_idx))
    if max_folds < 2:
        return [np.arange(len(y), dtype=int)]

    rng = np.random.default_rng(seed)
    pos_idx = rng.permutation(pos_idx)
    neg_idx = rng.permutation(neg_idx)
    buckets: list[list[int]] = [[] for _ in range(max_folds)]
    for i, idx in enumerate(pos_idx):
        buckets[i % max_folds].append(int(idx))
    for i, idx in enumerate(neg_idx):
        buckets[i % max_folds].append(int(idx))
    return [np.asarray(sorted(bucket), dtype=int) for bucket in buckets]


def cross_validated_auc(y: np.ndarray, x: np.ndarray, n_folds: int, seed: int) -> tuple[float, float, str]:
    mask = np.isfinite(x)
    y = y[mask].astype(int)
    x = x[mask].astype(float)
    if len(y) == 0:
        return np.nan, np.nan, "higher"

    folds = stratified_folds(y, n_folds=n_folds, seed=seed)
    if len(folds) == 1:
        auc = binary_auroc(y, x)
        direction = "higher" if (np.isnan(auc) or auc >= 0.5) else "lower"
        return max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan, 0.0, direction

    aucs: list[float] = []
    directions: list[str] = []
    all_idx = np.arange(len(y), dtype=int)
    for fold in folds:
        test_idx = fold
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        train_auc = binary_auroc(y[train_idx], x[train_idx])
        direction = "higher" if (np.isnan(train_auc) or train_auc >= 0.5) else "lower"
        signed_scores = x[test_idx] if direction == "higher" else -x[test_idx]
        fold_auc = binary_auroc(y[test_idx], signed_scores)
        if np.isfinite(fold_auc):
            aucs.append(float(fold_auc))
            directions.append(direction)

    if not aucs:
        return np.nan, np.nan, "higher"
    direction = max(set(directions), key=directions.count)
    return float(np.mean(aucs)), float(np.std(aucs, ddof=0)), direction


def analog_profile(feature: str) -> tuple[float, str, str]:
    family = "other"
    score = 1.5
    reasons: list[str] = []

    if feature.startswith("poincare_b"):
        family = "poincare_b"
        score = 3.0
        reasons.append("bounded return-ratio style feature")
    elif "drift_norm" in feature:
        family = "drift_norm"
        score = 3.0
        reasons.append("positive norm with simple arithmetic")
    elif "energy_asym" in feature:
        family = "energy_asym"
        score = 2.5
        reasons.append("signed asymmetry feature with manageable dynamic range")
    elif "gram_spread_sq" in feature:
        family = "gram_spread_sq"
        score = 1.5
        reasons.append("quadratic Gram spread with wide dynamic range")
    elif feature.startswith("frac_negative_phase"):
        family = "frac_negative_phase"
        score = 1.75
        reasons.append("counting/statistical feature over many beats")

    if any(tag in feature for tag in ["p05", "p95", "p10", "p90"]):
        score -= 0.5
        reasons.append("needs streaming quantile estimation")
    elif any(tag in feature for tag in ["p50", "median"]):
        reasons.append("central tendency summary is hardware-friendlier")

    if feature.startswith("frac_"):
        score -= 0.25
        reasons.append("requires accumulation over a long window")

    score = float(np.clip(score, 1.0, 3.0))
    label = "high" if score >= 2.75 else "medium" if score >= 2.0 else "low"
    return score, label, "; ".join(reasons)


def recommend_row(best_auc: float, best_std: float, analog_score: float) -> str:
    if np.isfinite(best_auc) and best_auc >= 0.70 and best_std <= 0.08 and analog_score >= 2.0:
        return "keep"
    if np.isfinite(best_auc) and best_auc >= 0.60 and best_std <= 0.15 and analog_score >= 1.5:
        return "defer"
    return "drop"


def build_task_metrics(df: pd.DataFrame, cfg: FeatureTriageConfig, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in feature_cols:
        x = df[feature].to_numpy(dtype=float)
        for task_name in TASKS:
            y = df[task_name].to_numpy(dtype=int)
            apparent_auc = binary_auroc(y, x)
            mean_auc, std_auc, direction = cross_validated_auc(y, x, n_folds=cfg.n_folds, seed=cfg.seed)
            rows.append(
                {
                    "feature": feature,
                    "task": task_name,
                    "task_definition": TASKS[task_name],
                    "apparent_auc": max(apparent_auc, 1.0 - apparent_auc) if np.isfinite(apparent_auc) else np.nan,
                    "cv_mean_auc": mean_auc,
                    "cv_std_auc": std_auc,
                    "direction": direction,
                    "n_positive": int(y.sum()),
                    "n_negative": int(len(y) - y.sum()),
                }
            )
    return pd.DataFrame(rows)


def build_triage_table(task_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature, group in task_df.groupby("feature", sort=True):
        g = group.sort_values(["cv_mean_auc", "apparent_auc"], ascending=False).reset_index(drop=True)
        best = g.iloc[0]
        analog_score, analog_label, analog_reason = analog_profile(feature)
        mean_cv = float(g["cv_mean_auc"].mean()) if g["cv_mean_auc"].notna().any() else np.nan
        triage_score = float(
            0.50 * (best["cv_mean_auc"] if np.isfinite(best["cv_mean_auc"]) else 0.5)
            + 0.25 * ((best["cv_mean_auc"] - best["cv_std_auc"]) if np.isfinite(best["cv_mean_auc"]) and np.isfinite(best["cv_std_auc"]) else 0.5)
            + 0.25 * (analog_score / 3.0)
        )
        rows.append(
            {
                "feature": feature,
                "best_task": best["task"],
                "best_task_definition": best["task_definition"],
                "best_cv_mean_auc": float(best["cv_mean_auc"]),
                "best_cv_std_auc": float(best["cv_std_auc"]),
                "best_apparent_auc": float(best["apparent_auc"]),
                "best_direction": best["direction"],
                "mean_cv_auc_across_tasks": mean_cv,
                "analog_score": analog_score,
                "analog_label": analog_label,
                "analog_reason": analog_reason,
                "triage_score": triage_score,
                "recommendation": recommend_row(float(best["cv_mean_auc"]), float(best["cv_std_auc"]), analog_score),
            }
        )
    triage_df = pd.DataFrame(rows).sort_values(
        ["recommendation", "triage_score", "best_cv_mean_auc", "analog_score"],
        ascending=[True, False, False, False],
    )
    triage_df["rank"] = np.arange(1, len(triage_df) + 1, dtype=int)
    return triage_df


def write_report(
    cfg: FeatureTriageConfig,
    triage_df: pd.DataFrame,
    task_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    keep_df = triage_df[triage_df["recommendation"] == "keep"].head(10)
    defer_df = triage_df[triage_df["recommendation"] == "defer"].head(10)
    drop_df = triage_df[triage_df["recommendation"] == "drop"].head(10)

    lines = [
        "# LTST Cardiac Feature Triage",
        "",
        "Feature triage over the completed LTST full-cohort run, scored on predictive value, cross-fold stability, and analog friendliness.",
        "",
        "## Assumptions",
        "- Cohort source: the completed 86-record LTST run at the supplied `run_dir`.",
        "- Predictive value is the best single-feature cross-validated AUROC across four record-level tasks.",
        "- Cross-fold stability is the standard deviation of the held-out AUROC across deterministic stratified folds.",
        "- Analog friendliness is a transparent rubric based on feature family and summary-stat complexity, not a measured hardware benchmark.",
        "",
        "## Tasks",
        "\n".join(f"- `{name}`: `{desc}`" for name, desc in TASKS.items()),
        "",
        "## Keep",
        keep_df.to_markdown(index=False) if not keep_df.empty else "No keep candidates under the current rubric.",
        "",
        "## Defer",
        defer_df.to_markdown(index=False) if not defer_df.empty else "No defer candidates under the current rubric.",
        "",
        "## Drop",
        drop_df.to_markdown(index=False) if not drop_df.empty else "No drop candidates under the current rubric.",
        "",
        "## Top Task Metrics",
        task_df.sort_values(["cv_mean_auc", "feature"], ascending=[False, True]).head(20).to_markdown(index=False),
        "",
    ]
    (output_dir / "ltst_feature_triage_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = run_dir / "feature_triage"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = FeatureTriageConfig(run_dir=run_dir, output_dir=output_dir, n_folds=int(args.n_folds))
    df = load_tables(cfg)
    feature_cols = [col for col in df.columns if col not in {"record", "regime", "guarded_cross", "phenotype_target", *TASKS.keys()}]
    task_df = build_task_metrics(df, cfg=cfg, feature_cols=feature_cols)
    triage_df = build_triage_table(task_df)

    triage_df.to_csv(output_dir / "ltst_feature_triage.csv", index=False)
    task_df.to_csv(output_dir / "ltst_feature_task_metrics.csv", index=False)
    write_report(cfg, triage_df, task_df, output_dir)

    metadata = {
        "config": safe_json(asdict(cfg)),
        "n_records": int(len(df)),
        "feature_count": int(len(feature_cols)),
        "tasks": TASKS,
    }
    with (output_dir / "ltst_feature_triage_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"[LTST-TRIAGE] Output directory: {output_dir}", flush=True)
    print(triage_df.head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
