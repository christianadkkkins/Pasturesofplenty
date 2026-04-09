from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class ExcursionConfig:
    run_dir: Path
    baseline_non_st_beats: int = 500
    taus: tuple[float, ...] = (0.90, 0.95, 0.98)
    alpha: float = 0.05
    active_eps: float = 1e-6


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


def phenotype_target(regime: str) -> str:
    if regime in {"angle_only", "angle_first"}:
        return "loose_orbit"
    if regime in {"long_only", "long_first", "both_same_horizon"}:
        return "constrained_orbit"
    return "rigid_orbit"


def beat_path(beat_dir: Path, record: str) -> Path:
    parquet = beat_dir / f"{record}.parquet"
    if parquet.exists():
        return parquet
    csv = beat_dir / f"{record}.csv"
    if csv.exists():
        return csv
    raise FileNotFoundError(f"Missing beat-level file for {record}.")


def read_beat_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def ema_positive(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty(len(values), dtype=float)
    acc = 0.0
    for i, value in enumerate(values):
        acc = alpha * float(value) + (1.0 - alpha) * acc
        out[i] = acc
    return out


def max_run(mask: np.ndarray) -> int:
    arr = np.asarray(mask, dtype=bool)
    if len(arr) == 0 or not arr.any():
        return 0
    padded = np.concatenate(([False], arr, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return int(np.max(ends - starts)) if len(starts) else 0


def analyze_record(record: str, regime: str, beat_dir: Path, cfg: ExcursionConfig) -> pd.DataFrame:
    df = read_beat_frame(beat_path(beat_dir, record)).sort_values("beat_sample").reset_index(drop=True)
    pb = df["poincare_b"].to_numpy(dtype=float)
    non_st = ~df["st_event"].fillna(False).astype(bool).to_numpy()
    st = ~non_st
    baseline_idx = np.flatnonzero(non_st)[: cfg.baseline_non_st_beats]

    if len(baseline_idx) == 0:
        raise RuntimeError(f"{record}: no non-ST baseline beats")

    rows: list[dict[str, Any]] = []
    for tau in cfg.taus:
        excursion = np.maximum(tau - pb, 0.0)
        p_tau = ema_positive(excursion, cfg.alpha)
        active = p_tau > cfg.active_eps

        base_exc = excursion[baseline_idx]
        base_pt = p_tau[baseline_idx]
        base_occ = base_exc > 0
        base_active = active[baseline_idx]

        st_exc = excursion[st]
        st_pt = p_tau[st]
        st_occ = st_exc > 0

        rows.append(
            {
                "record": record,
                "regime": regime,
                "phenotype_target": phenotype_target(regime),
                "tau": tau,
                "baseline_n_beats": int(len(baseline_idx)),
                "baseline_median_pb": float(np.nanmedian(pb[baseline_idx])),
                "baseline_median_excursion": float(np.nanmedian(base_exc)),
                "baseline_p95_excursion": float(np.nanpercentile(base_exc, 95)),
                "baseline_excursion_occupancy": float(np.mean(base_occ)),
                "baseline_median_ptau": float(np.nanmedian(base_pt)),
                "baseline_p95_ptau": float(np.nanpercentile(base_pt, 95)),
                "baseline_max_excursion_run": max_run(base_occ),
                "baseline_max_ptau_active_run": max_run(base_active),
                "st_n_beats": int(np.sum(st)),
                "st_median_ptau": float(np.nanmedian(st_pt)) if len(st_pt) else np.nan,
                "st_p95_ptau": float(np.nanpercentile(st_pt, 95)) if len(st_pt) else np.nan,
                "st_excursion_occupancy": float(np.mean(st_occ)) if len(st_occ) else np.nan,
                "st_max_excursion_run": max_run(st_occ) if len(st_occ) else 0,
                "ptau_st_vs_baseline_ratio": float(np.nanmedian(st_pt) / np.nanmedian(base_pt))
                if len(st_pt) and np.nanmedian(base_pt) > EPS
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def group_summary(per_record_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        per_record_df.groupby(["tau", group_col])
        .agg(
            n_records=("record", "count"),
            median_baseline_occupancy=("baseline_excursion_occupancy", "median"),
            median_baseline_p95_ptau=("baseline_p95_ptau", "median"),
            median_baseline_max_exc_run=("baseline_max_excursion_run", "median"),
            median_baseline_max_ptau_run=("baseline_max_ptau_active_run", "median"),
            median_st_ratio=("ptau_st_vs_baseline_ratio", "median"),
        )
        .reset_index()
        .sort_values(["tau", "median_baseline_p95_ptau", "median_baseline_occupancy"])
        .reset_index(drop=True)
    )


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


def build_benchmark_table(per_record_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tau, sdf in per_record_df.groupby("tau"):
        labels = sdf["phenotype_target"].to_numpy(dtype=object)
        for metric in ["baseline_excursion_occupancy", "baseline_p95_ptau", "baseline_max_excursion_run"]:
            scores = sdf[metric].to_numpy(dtype=float)
            for positive in ["loose_orbit", "constrained_orbit", "rigid_orbit"]:
                rows.append(
                    {
                        "tau": tau,
                        "metric": metric,
                        "positive_class": positive,
                        "auroc": auc_one_vs_rest(scores, labels, positive),
                    }
                )
    return pd.DataFrame(rows).sort_values(["tau", "metric", "positive_class"]).reset_index(drop=True)


def plot_tau_plane(per_record_df: pd.DataFrame, output_path: Path, tau: float = 0.95) -> None:
    sdf = per_record_df[np.isclose(per_record_df["tau"], tau)].copy()
    if len(sdf) == 0:
        return
    colors = {
        "angle_only": "#d62728",
        "angle_first": "#ff9896",
        "long_only": "#1f77b4",
        "long_first": "#6baed6",
        "both_same_horizon": "#2ca02c",
        "neither": "#9467bd",
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    for regime, rdf in sdf.groupby("regime"):
        ax.scatter(
            rdf["baseline_excursion_occupancy"],
            rdf["baseline_p95_ptau"],
            s=30 + 2.5 * rdf["baseline_max_excursion_run"].to_numpy(dtype=float),
            alpha=0.8,
            color=colors.get(regime, "#444444"),
            label=regime,
        )
    ax.set_xlabel("Baseline excursion occupancy")
    ax.set_ylabel("Baseline p95 P_tau")
    ax.set_title("Baseline lower-tail excursion plane (tau=0.95)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze lower-tail poincare_b excursion persistence on a completed LTST run.")
    parser.add_argument("--run-dir", required=True, help="Path to a completed ltst_full_86 run directory.")
    parser.add_argument("--baseline-non-st-beats", type=int, default=500)
    parser.add_argument("--taus", default="0.90,0.95,0.98", help="Comma-separated excursion thresholds.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--active-eps", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExcursionConfig(
        run_dir=Path(args.run_dir).resolve(),
        baseline_non_st_beats=args.baseline_non_st_beats,
        taus=tuple(float(piece.strip()) for piece in args.taus.split(",") if piece.strip()),
        alpha=args.alpha,
        active_eps=args.active_eps,
    )
    beat_dir = cfg.run_dir / "beat_level"
    regime_df = pd.read_csv(cfg.run_dir / "regime.csv")
    out_dir = cfg.run_dir / "pb_excursion_taxonomy"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [analyze_record(row.record, row.regime, beat_dir, cfg) for row in regime_df.itertuples(index=False)]
    per_record_df = pd.concat(frames, ignore_index=True).sort_values(["tau", "regime", "record"]).reset_index(drop=True)
    regime_summary_df = group_summary(per_record_df, "regime")
    phenotype_summary_df = group_summary(per_record_df, "phenotype_target")
    benchmark_df = build_benchmark_table(per_record_df)

    per_record_df.to_csv(out_dir / "pb_excursion_per_record.csv", index=False)
    regime_summary_df.to_csv(out_dir / "pb_excursion_regime_summary.csv", index=False)
    phenotype_summary_df.to_csv(out_dir / "pb_excursion_phenotype_summary.csv", index=False)
    benchmark_df.to_csv(out_dir / "pb_excursion_benchmarks.csv", index=False)
    plot_tau_plane(per_record_df, out_dir / "figure_tau095_excursion_plane.png", tau=0.95)

    metadata = {
        "config": json_safe(asdict(cfg)),
        "n_records": int(per_record_df["record"].nunique()),
        "output_dir": str(out_dir),
    }
    with (out_dir / "pb_excursion_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    report_lines = [
        "# LTST Poincare_b Excursion Taxonomy",
        "",
        "This analysis treats phenotype as a lower-tail excursion process of the beatwise return coefficient rather than a static stiffness scalar.",
        "",
        "## Regime Summary",
        regime_summary_df.to_markdown(index=False),
        "",
        "## Phenotype Summary",
        phenotype_summary_df.to_markdown(index=False),
        "",
        "## One-vs-rest AUROC",
        benchmark_df.to_markdown(index=False),
        "",
    ]
    (out_dir / "pb_excursion_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[PB-EXCURSION] Output directory: {out_dir}", flush=True)
    print(regime_summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
