from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


EPS = 1e-9


@dataclass(frozen=True)
class GateProbeConfig:
    bg_rows: int = 240
    thresholds: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    selective_margin_threshold: float = 0.0
    fast_rows: int = 3
    short_rows: int = 9
    long_rows: int = 100
    out_prefix: str = "ltst_three_record_online_dmd_gate_probe"


def robust_z_from_bg(values: pd.Series, bg_values: pd.Series, clip: float = 8.0) -> np.ndarray:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    bg = pd.to_numeric(bg_values, errors="coerce").to_numpy(dtype=float)
    med = float(np.nanmedian(bg))
    mad = float(np.nanmedian(np.abs(bg - med)))
    if not np.isfinite(mad) or mad < EPS:
        sd = float(np.nanstd(bg))
        if not np.isfinite(sd) or sd < EPS:
            z = np.zeros_like(x, dtype=float)
        else:
            z = (x - float(np.nanmean(bg))) / (sd + EPS)
    else:
        z = 0.6745 * (x - med) / (mad + EPS)
    z = np.nan_to_num(z, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(z, -clip, clip)


def prepare_run_directory(base_root: Path, prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = base_root / f"{prefix}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def add_gate_scores(panel_df: pd.DataFrame, bg_rows: int) -> pd.DataFrame:
    required = [
        "dmd_observer_lie_metric_drift",
        "dmd_observer_pair_angle_sl",
        "dmd_observer_memory_align",
        "dmd_observer_lie_transition_score",
        "dmd_observer_lie_orbit_norm",
        "dmd_observer_lie_strain_norm",
        "dmd_observer_combined_transition_score",
    ]
    missing = [col for col in required if col not in panel_df.columns]
    if missing:
        raise KeyError(f"Panel is missing gate columns: {missing}")
    rows: list[pd.DataFrame] = []
    for patient_id, group in panel_df.groupby("patient_id", sort=False):
        g = group.sort_values("time_seconds").reset_index(drop=True).copy()
        bg = g.iloc[: min(bg_rows, len(g))].copy()
        g["gate_metric_drift_z"] = robust_z_from_bg(g["dmd_observer_lie_metric_drift"], bg["dmd_observer_lie_metric_drift"])
        g["gate_pair_angle_sl_z"] = robust_z_from_bg(g["dmd_observer_pair_angle_sl"], bg["dmd_observer_pair_angle_sl"])
        g["gate_memory_align_z"] = robust_z_from_bg(g["dmd_observer_memory_align"], bg["dmd_observer_memory_align"])
        g["gate_lie_transition_score_z"] = robust_z_from_bg(g["dmd_observer_lie_transition_score"], bg["dmd_observer_lie_transition_score"])
        g["gate_lie_orbit_norm_z"] = robust_z_from_bg(g["dmd_observer_lie_orbit_norm"], bg["dmd_observer_lie_orbit_norm"])
        g["gate_lie_strain_norm_z"] = robust_z_from_bg(g["dmd_observer_lie_strain_norm"], bg["dmd_observer_lie_strain_norm"])
        g["gate_combined_transition_score_z"] = robust_z_from_bg(
            g["dmd_observer_combined_transition_score"],
            bg["dmd_observer_combined_transition_score"],
        )
        g["observer_gate_score"] = (
            g["gate_metric_drift_z"] + g["gate_pair_angle_sl_z"] - g["gate_memory_align_z"]
        ) / 3.0
        g["observer_magnitude_score"] = (
            g["gate_lie_transition_score_z"]
            + g["gate_lie_orbit_norm_z"]
            + g["gate_lie_strain_norm_z"]
            + g["gate_combined_transition_score_z"]
        ) / 4.0
        g["observer_selective_margin"] = g["observer_gate_score"] - g["observer_magnitude_score"]
        g["observer_gate_relu"] = g["observer_gate_score"].clip(lower=0.0)
        g["observer_gate_active_0"] = (g["observer_gate_score"] > 0.0).astype(int)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def load_predictions(run_dir: Path) -> pd.DataFrame:
    pred_files = sorted(run_dir.glob("*_hazard_predictions.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No hazard prediction file found in {run_dir}")
    pred_df = pd.read_csv(pred_files[0])
    required = {"patient_id", "time_seconds", "horizon", "model_name", "prediction"}
    if not required.issubset(pred_df.columns):
        raise KeyError(f"Prediction file missing columns: {sorted(required - set(pred_df.columns))}")
    return pred_df


def load_panel(run_dir: Path) -> pd.DataFrame:
    panel_files = sorted(run_dir.glob("*_panel_1s.csv"))
    if not panel_files:
        raise FileNotFoundError(f"No 1-second panel file found in {run_dir}")
    return pd.read_csv(panel_files[0])


def add_multiscale_gate_scores(df: pd.DataFrame, fast_rows: int, short_rows: int, long_rows: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for patient_id, group in df.groupby("patient_id", sort=False):
        g = group.sort_values("time_seconds").reset_index(drop=True).copy()
        g["fast_gate_score"] = g["observer_gate_score"].rolling(window=max(1, fast_rows), min_periods=1).mean()
        g["short_gate_score"] = g["observer_gate_score"].rolling(window=max(1, short_rows), min_periods=1).mean()
        g["long_gate_score"] = g["observer_gate_score"].rolling(window=max(1, long_rows), min_periods=1).mean()
        g["fast_selective_margin"] = g["observer_selective_margin"].rolling(window=max(1, fast_rows), min_periods=1).mean()
        g["short_selective_margin"] = g["observer_selective_margin"].rolling(window=max(1, short_rows), min_periods=1).mean()
        g["long_selective_margin"] = g["observer_selective_margin"].rolling(window=max(1, long_rows), min_periods=1).mean()
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def build_probe_frame(panel_df: pd.DataFrame, pred_df: pd.DataFrame, cfg: GateProbeConfig) -> pd.DataFrame:
    panel_with_gate = add_gate_scores(panel_df, bg_rows=cfg.bg_rows)
    panel_with_gate = add_multiscale_gate_scores(
        panel_with_gate,
        fast_rows=cfg.fast_rows,
        short_rows=cfg.short_rows,
        long_rows=cfg.long_rows,
    )
    pred_base = (
        pred_df.pivot_table(
            index=["patient_id", "time_seconds", "horizon"],
            columns="model_name",
            values="prediction",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    score_cols = [
        "dmd_substrate_score",
        "dmd_observer_pressure_score",
        "dmd_hybrid_score",
        "future_onset_within_5",
        "future_onset_within_10",
        "future_onset_within_20",
        "future_onset_within_30",
        "observer_gate_score",
        "observer_gate_relu",
        "observer_gate_active_0",
        "observer_magnitude_score",
        "observer_selective_margin",
        "fast_gate_score",
        "short_gate_score",
        "long_gate_score",
        "fast_selective_margin",
        "short_selective_margin",
        "long_selective_margin",
        "gate_metric_drift_z",
        "gate_pair_angle_sl_z",
        "gate_memory_align_z",
        "gate_lie_transition_score_z",
        "gate_lie_orbit_norm_z",
        "gate_lie_strain_norm_z",
        "gate_combined_transition_score_z",
        "dmd_observer_lie_metric_drift",
        "dmd_observer_pair_angle_sl",
        "dmd_observer_memory_align",
        "dmd_observer_lie_transition_score",
        "dmd_observer_lie_orbit_norm",
        "dmd_observer_lie_strain_norm",
        "dmd_observer_combined_transition_score",
    ]
    available_cols = [col for col in score_cols if col in panel_with_gate.columns]
    merged = pred_base.merge(
        panel_with_gate[["patient_id", "time_seconds", *available_cols]],
        on=["patient_id", "time_seconds"],
        how="left",
    )
    return merged.sort_values(["horizon", "patient_id", "time_seconds"]).reset_index(drop=True)


def add_routed_models(
    df: pd.DataFrame,
    thresholds: tuple[float, ...],
    selective_margin_threshold: float = 0.0,
) -> pd.DataFrame:
    out = df.copy()
    out["substrate_only"] = pd.to_numeric(out.get("substrate_only"), errors="coerce")
    out["observer_only"] = pd.to_numeric(out.get("observer_only"), errors="coerce")
    out["hybrid_only"] = pd.to_numeric(out.get("hybrid_only"), errors="coerce")
    gate_score = pd.to_numeric(out["observer_gate_score"], errors="coerce").fillna(0.0)
    gate_relu = pd.to_numeric(out["observer_gate_relu"], errors="coerce").fillna(0.0)
    fast_gate = pd.to_numeric(out.get("fast_gate_score"), errors="coerce").fillna(0.0)
    short_gate = pd.to_numeric(out.get("short_gate_score"), errors="coerce").fillna(0.0)
    long_gate = pd.to_numeric(out.get("long_gate_score"), errors="coerce").fillna(0.0)
    if "observer_selective_margin" in out.columns:
        selective_margin = pd.to_numeric(out["observer_selective_margin"], errors="coerce").fillna(-np.inf)
    else:
        selective_margin = pd.Series(-np.inf, index=out.index, dtype=float)
    fast_margin = pd.to_numeric(out.get("fast_selective_margin"), errors="coerce").fillna(-np.inf)
    short_margin = pd.to_numeric(out.get("short_selective_margin"), errors="coerce").fillna(-np.inf)
    long_margin = pd.to_numeric(out.get("long_selective_margin"), errors="coerce").fillna(-np.inf)
    substrate = out["substrate_only"].fillna(0.0)
    observer = out["observer_only"].fillna(0.0)
    for thr in thresholds:
        thr_label = str(thr).replace(".", "p")
        active = gate_score >= thr
        selective_active = active & (selective_margin >= selective_margin_threshold)
        multiscale_active = (
            (fast_gate >= thr)
            & (short_gate >= thr)
            & (fast_margin >= selective_margin_threshold)
            & (short_margin >= selective_margin_threshold)
        )
        vhr_active = (
            (fast_gate >= thr)
            & (short_gate >= thr)
            & (long_gate >= thr)
            & (fast_margin >= selective_margin_threshold)
            & (short_margin >= selective_margin_threshold)
            & (long_margin >= selective_margin_threshold)
        )
        out[f"route_max_t{thr_label}"] = np.where(active, np.maximum(substrate, observer), substrate)
        out[f"route_switch_t{thr_label}"] = np.where(active, observer, substrate)
        scaled = np.clip((gate_relu - thr) / max(1.0, 2.0 - thr), 0.0, 1.0)
        out[f"route_blend_t{thr_label}"] = (1.0 - scaled) * substrate + scaled * np.maximum(substrate, observer)
        out[f"route_selective_max_t{thr_label}"] = np.where(selective_active, np.maximum(substrate, observer), substrate)
        out[f"route_selective_switch_t{thr_label}"] = np.where(selective_active, observer, substrate)
        selective_scaled = np.where(selective_active, scaled, 0.0)
        out[f"route_selective_blend_t{thr_label}"] = (1.0 - selective_scaled) * substrate + selective_scaled * np.maximum(substrate, observer)
        out[f"route_multiscale_max_t{thr_label}"] = np.where(multiscale_active, np.maximum(substrate, observer), substrate)
        out[f"route_multiscale_switch_t{thr_label}"] = np.where(multiscale_active, observer, substrate)
        multiscale_scaled = np.where(multiscale_active, scaled, 0.0)
        out[f"route_multiscale_blend_t{thr_label}"] = (1.0 - multiscale_scaled) * substrate + multiscale_scaled * np.maximum(substrate, observer)
        out[f"route_vhr_max_t{thr_label}"] = np.where(vhr_active, np.maximum(substrate, observer), substrate)
        out[f"route_vhr_switch_t{thr_label}"] = np.where(vhr_active, observer, substrate)
        vhr_scaled = np.where(vhr_active, scaled, 0.0)
        out[f"route_vhr_blend_t{thr_label}"] = (1.0 - vhr_scaled) * substrate + vhr_scaled * np.maximum(substrate, observer)
    return out


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    s = np.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)
    if y.size == 0 or np.unique(y).size < 2:
        return np.nan, np.nan
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def evaluate_models(probe_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_cols = [
        col
        for col in probe_df.columns
        if col in {"substrate_only", "observer_only", "hybrid_only"} or col.startswith("route_")
    ]
    rows: list[dict[str, object]] = []
    overall_rows: list[dict[str, object]] = []
    for horizon, horizon_df in probe_df.groupby("horizon", sort=True):
        label_col = f"future_onset_within_{int(horizon)}"
        if label_col not in horizon_df.columns:
            continue
        for patient_id, patient_df in horizon_df.groupby("patient_id", sort=False):
            y = patient_df[label_col].to_numpy(dtype=int)
            for model_name in model_cols:
                s = patient_df[model_name].to_numpy(dtype=float)
                auroc, ap = compute_binary_metrics(y, s)
                rows.append(
                    {
                        "horizon": int(horizon),
                        "patient_id": patient_id,
                        "model_name": model_name,
                        "auroc": auroc,
                        "ap": ap,
                        "n_rows": int(len(patient_df)),
                        "n_positive": int(y.sum()),
                    }
                )
        y_all = horizon_df[label_col].to_numpy(dtype=int)
        for model_name in model_cols:
            s_all = horizon_df[model_name].to_numpy(dtype=float)
            auroc, ap = compute_binary_metrics(y_all, s_all)
            overall_rows.append(
                {
                    "horizon": int(horizon),
                    "model_name": model_name,
                    "auroc": auroc,
                    "ap": ap,
                    "n_rows": int(len(horizon_df)),
                    "n_positive": int(y_all.sum()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(overall_rows)


def summarize_gate_windows(probe_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for patient_id, group in probe_df.groupby("patient_id", sort=False):
        g = group.sort_values("time_seconds").reset_index(drop=True)
        bg = g.loc[(g["time_seconds"] >= -40) & (g["time_seconds"] <= -20)]
        pre = g.loc[(g["time_seconds"] >= -20) & (g["time_seconds"] <= -1)]
        onset = g.loc[(g["time_seconds"] >= -5) & (g["time_seconds"] <= 5)]
        for col in [
            "observer_gate_score",
            "observer_magnitude_score",
            "observer_selective_margin",
            "fast_gate_score",
            "short_gate_score",
            "long_gate_score",
            "fast_selective_margin",
            "short_selective_margin",
            "long_selective_margin",
            "gate_metric_drift_z",
            "gate_pair_angle_sl_z",
            "gate_memory_align_z",
        ]:
            rows.append(
                {
                    "patient_id": patient_id,
                    "feature": col,
                    "background_mean": float(pd.to_numeric(bg[col], errors="coerce").mean()),
                    "pre_mean": float(pd.to_numeric(pre[col], errors="coerce").mean()),
                    "onset_mean": float(pd.to_numeric(onset[col], errors="coerce").mean()),
                    "onset_minus_background": float(pd.to_numeric(onset[col], errors="coerce").mean() - pd.to_numeric(bg[col], errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows)


def render_report(
    patient_metrics: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    gate_windows: pd.DataFrame,
    out_dir: Path,
    cfg: GateProbeConfig,
) -> Path:
    best_rows = (
        overall_metrics.sort_values(["horizon", "auroc", "ap"], ascending=[True, False, False])
        .groupby("horizon", as_index=False)
        .head(3)
        .reset_index(drop=True)
    )
    lines = [
        "# LTST Three-Record DMD Gate Probe",
        "",
        "This probe keeps `substrate_only` as the default lane and evaluates whether a simple",
        "`metric_drift + pair_angle_sl - memory_align` gate can help route observer predictions",
        "only inside `s30742`-style geometry pockets, with a second-stage veto against broad Lie-magnitude activation.",
        "",
        "## Gate",
        "",
        "`observer_gate_score = (z(metric_drift) + z(pair_angle_sl) - z(memory_align)) / 3`",
        "",
        "`observer_magnitude_score = mean(z(lie_transition), z(lie_orbit), z(lie_strain), z(combined_transition))`",
        "",
        "`observer_selective_margin = observer_gate_score - observer_magnitude_score`",
        "",
        "`fast_gate_score`, `short_gate_score`, and `long_gate_score` are trailing rolling means of `observer_gate_score`",
        f"over {cfg.fast_rows}, {cfg.short_rows}, and {cfg.long_rows} rows respectively, with matching rolling selective margins.",
        "",
        f"Thresholds tested: {', '.join(str(x) for x in cfg.thresholds)}",
        "",
        f"Selective margin threshold: {cfg.selective_margin_threshold}",
        "",
        "## Top Overall Models By Horizon",
        "",
        best_rows.to_markdown(index=False),
        "",
        "## Gate Window Means",
        "",
        gate_windows.to_markdown(index=False),
        "",
    ]
    report_path = out_dir / f"{out_dir.name}_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_thresholds(raw: str) -> tuple[float, ...]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe pocket-conditioned observer routes on a nested-causal LTST DMD run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing *_panel_1s.csv and *_hazard_predictions.csv.")
    parser.add_argument("--bg-rows", type=int, default=240, help="Number of early rows per record used as the gate baseline.")
    parser.add_argument("--thresholds", type=str, default="0.0,0.25,0.5,0.75,1.0", help="Comma-separated gate thresholds.")
    parser.add_argument("--selective-margin-threshold", type=float, default=0.0, help="Minimum structured-minus-magnitude margin required for selective routes.")
    parser.add_argument("--fast-rows", type=int, default=3, help="Trailing rows for the fast-timescale gate average.")
    parser.add_argument("--short-rows", type=int, default=9, help="Trailing rows for the short-timescale gate average.")
    parser.add_argument("--long-rows", type=int, default=100, help="Trailing rows for the long-timescale gate average.")
    parser.add_argument("--out-prefix", type=str, default="ltst_three_record_online_dmd_gate_probe", help="Output directory prefix.")
    args = parser.parse_args()

    cfg = GateProbeConfig(
        bg_rows=args.bg_rows,
        thresholds=parse_thresholds(args.thresholds),
        selective_margin_threshold=args.selective_margin_threshold,
        fast_rows=args.fast_rows,
        short_rows=args.short_rows,
        long_rows=args.long_rows,
        out_prefix=args.out_prefix,
    )
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    out_dir = prepare_run_directory(run_dir, cfg.out_prefix)

    panel_df = load_panel(run_dir)
    pred_df = load_predictions(run_dir)
    probe_df = build_probe_frame(panel_df, pred_df, cfg)
    probe_df = add_routed_models(probe_df, cfg.thresholds, selective_margin_threshold=cfg.selective_margin_threshold)
    patient_metrics, overall_metrics = evaluate_models(probe_df)
    gate_windows = summarize_gate_windows(probe_df)
    report_path = render_report(patient_metrics, overall_metrics, gate_windows, out_dir, cfg)

    probe_df.to_csv(out_dir / f"{out_dir.name}_probe_frame.csv", index=False)
    patient_metrics.to_csv(out_dir / f"{out_dir.name}_patient_metrics.csv", index=False)
    overall_metrics.to_csv(out_dir / f"{out_dir.name}_overall_metrics.csv", index=False)
    gate_windows.to_csv(out_dir / f"{out_dir.name}_gate_windows.csv", index=False)
    (out_dir / f"{out_dir.name}_manifest.json").write_text(
        json.dumps(
            {
                "source_run_dir": str(run_dir),
                "bg_rows": cfg.bg_rows,
                "thresholds": list(cfg.thresholds),
                "selective_margin_threshold": cfg.selective_margin_threshold,
                "fast_rows": cfg.fast_rows,
                "short_rows": cfg.short_rows,
                "long_rows": cfg.long_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
