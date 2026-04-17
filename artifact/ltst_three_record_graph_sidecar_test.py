from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import wfdb


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_lie import (
    compute_lie_state_features,
    compute_oriented_state_features,
    compute_projective_state_features,
    ema_prior_states,
)


EPS = 1e-9


@dataclass(frozen=True)
class SidecarTestConfig:
    records: tuple[str, ...] = ("s20011", "s20041", "s30742")
    pn_dir: str = "ltstdb"
    pre_event_seconds: int = 1200
    post_event_seconds: int = 60
    downsample: int = 5
    grid_step_seconds: float = 1.0
    graph_window_seconds: int = 10
    adapt_beta: float = 0.90
    bg_rows: int = 240
    gap_rows: int = 30
    pre_rows: int = 120
    post_rows: int = 60
    hazard_horizons: tuple[int, ...] = (5, 10, 20, 30)
    out_prefix: str = "ltst_three_record_graph_updated_sidecar"
    min_state_energy: float = 1e-9
    short_beta: float = 0.65
    long_beta: float = 0.92
    run_root: Path = Path("artifact") / "runs"


def find_project_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "artifact").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root containing artifact/.")


def prepare_run_directory(base_root: Path, prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base_root / f"{prefix}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_unique(names: Iterable[str]) -> list[str]:
    counts: dict[str, int] = {}
    out: list[str] = []
    for raw in names:
        name = str(raw)
        counts[name] = counts.get(name, 0) + 1
        out.append(name if counts[name] == 1 else f"{name}_{counts[name] - 1}")
    return out


def robust_z(values: np.ndarray, clip: float = 8.0) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    if not np.isfinite(mad) or mad < EPS:
        z = np.zeros_like(x, dtype=float)
    else:
        z = 0.6745 * (x - med) / (mad + EPS)
    z = np.nan_to_num(z, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(z, -clip, clip)


def robust_z_from_bg(values: pd.Series, bg_values: pd.Series, clip: float = 8.0) -> np.ndarray:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    bg = pd.to_numeric(bg_values, errors="coerce").to_numpy(dtype=float)
    med = float(np.nanmedian(bg))
    mad = float(np.nanmedian(np.abs(bg - med)))
    if not np.isfinite(mad) or mad < EPS:
        z = np.zeros_like(x, dtype=float)
    else:
        z = 0.6745 * (x - med) / (mad + EPS)
    z = np.nan_to_num(z, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(z, -clip, clip)


def summarize_event_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    cfg: SidecarTestConfig,
    event_col: str = "event_onset",
    time_col: str = "time_seconds",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy().sort_values(time_col).reset_index(drop=True)
    event_idx_arr = np.where(work[event_col].to_numpy() > 0)[0]
    if len(event_idx_arr) < 1:
        raise ValueError("No event_onset rows found.")
    event_idx = int(event_idx_arr[0])

    bg_end = max(0, event_idx - cfg.gap_rows)
    bg_start = max(0, bg_end - cfg.bg_rows)
    pre_start = max(0, event_idx - cfg.pre_rows)
    pre_end = event_idx
    post_start = event_idx
    post_end = min(len(work), event_idx + cfg.post_rows)

    bg_df = work.iloc[bg_start:bg_end].copy()
    pre_df = work.iloc[pre_start:pre_end].copy()
    post_df = work.iloc[post_start:post_end].copy()
    if len(bg_df) < 20 or len(pre_df) < 20:
        raise ValueError("Not enough rows in background or pre-event window.")

    rows = []
    for c in feature_cols:
        bg_vals = pd.to_numeric(bg_df[c], errors="coerce").to_numpy(dtype=float)
        pre_vals = pd.to_numeric(pre_df[c], errors="coerce").to_numpy(dtype=float)
        post_vals = pd.to_numeric(post_df[c], errors="coerce").to_numpy(dtype=float)
        bg_mean = float(np.nanmean(bg_vals))
        pre_mean = float(np.nanmean(pre_vals))
        post_mean = float(np.nanmean(post_vals))
        bg_std = float(np.nanstd(bg_vals) + EPS)
        pre_shift = float((pre_mean - bg_mean) / bg_std)
        post_shift = float((post_mean - bg_mean) / bg_std)
        peak_abs_pre = float(np.nanmax(np.abs(pre_vals))) if len(pre_vals) else np.nan
        rows.append(
            {
                "feature": c,
                "bg_mean": bg_mean,
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "pre_shift_sd": pre_shift,
                "post_shift_sd": post_shift,
                "peak_abs_pre_z": peak_abs_pre,
                "abs_pre_shift": abs(pre_shift),
            }
        )

    rank = pd.DataFrame(rows).sort_values(["abs_pre_shift", "peak_abs_pre_z"], ascending=False).reset_index(drop=True)
    summary = pd.DataFrame(
        [
            {
                "event_idx": event_idx,
                "background_rows": len(bg_df),
                "pre_rows": len(pre_df),
                "post_rows": len(post_df),
            }
        ]
    )
    return rank, summary, bg_df, pre_df, post_df


def build_pressure_score(df: pd.DataFrame, rank_df: pd.DataFrame, bg_df: pd.DataFrame, topn: int = 4) -> tuple[pd.DataFrame, list[str]]:
    out_df = df.copy().sort_values("time_seconds").reset_index(drop=True)
    top_feats: list[str] = []
    for feat in rank_df["feature"].tolist():
        if feat in out_df.columns and pd.api.types.is_numeric_dtype(out_df[feat]):
            top_feats.append(feat)
        if len(top_feats) >= topn:
            break
    if not top_feats:
        raise ValueError("No usable ranked features found.")
    for feat in top_feats:
        out_df[feat + "_zbg"] = robust_z_from_bg(out_df[feat], bg_df[feat])
    zcols = [feat + "_zbg" for feat in top_feats]
    out_df["event_pressure_score"] = out_df[zcols].abs().mean(axis=1)
    return out_df, top_feats


def graph_stats(adj: np.ndarray, prefix: str, drift: float = np.nan) -> dict[str, float]:
    matrix = np.asarray(adj, dtype=float)
    d = matrix.shape[0]
    work = matrix.copy()
    np.fill_diagonal(work, 0.0)
    off = work[~np.eye(d, dtype=bool)]
    off_abs = np.abs(off)
    density = float(np.mean(off_abs > 1e-9)) if off_abs.size else 0.0
    frob = float(np.linalg.norm(work, ord="fro"))
    try:
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(work))))
    except Exception:
        spectral_radius = np.nan
    asymmetry = float(np.linalg.norm(work - work.T, ord="fro"))
    if off_abs.sum() > EPS:
        p = off_abs / off_abs.sum()
        edge_entropy = float(-np.sum(p * np.log(p + EPS)))
    else:
        edge_entropy = 0.0
    out_strength = np.sum(np.abs(work), axis=1)
    in_strength = np.sum(np.abs(work), axis=0)
    out = {
        f"{prefix}_graph_density": density,
        f"{prefix}_graph_frob": frob,
        f"{prefix}_graph_spectral_radius": spectral_radius,
        f"{prefix}_graph_asymmetry": asymmetry,
        f"{prefix}_graph_edge_entropy": edge_entropy,
        f"{prefix}_graph_out_strength_mean": float(np.mean(out_strength)),
        f"{prefix}_graph_out_strength_std": float(np.std(out_strength)),
        f"{prefix}_graph_in_strength_mean": float(np.mean(in_strength)),
        f"{prefix}_graph_in_strength_std": float(np.std(in_strength)),
        f"{prefix}_graph_drift": float(drift) if np.isfinite(drift) else np.nan,
        f"{prefix}_max_edge": float(np.max(off_abs)) if off_abs.size else 0.0,
        f"{prefix}_pair_asym": float(np.mean(np.abs(work - work.T))) if off_abs.size else 0.0,
    }
    if d >= 2:
        out[f"{prefix}_ab"] = float(work[0, 1])
        out[f"{prefix}_ba"] = float(work[1, 0])
    return out


def corr_graph(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.shape[1] < 2:
        return np.zeros((x.shape[1], x.shape[1]), dtype=float)
    corr = np.corrcoef(x, rowvar=False)
    return np.nan_to_num(corr, nan=0.0)


def lag_graph(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    d = x.shape[1]
    if len(x) < 3:
        return np.zeros((d, d), dtype=float)
    out = np.zeros((d, d), dtype=float)
    for i in range(d):
        xi = x[:-1, i]
        for j in range(d):
            yj = x[1:, j]
            if np.std(xi) < EPS or np.std(yj) < EPS:
                out[i, j] = 0.0
            else:
                out[i, j] = float(np.corrcoef(xi, yj)[0, 1])
    return np.nan_to_num(out, nan=0.0)


def fit_logistic_predict(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> np.ndarray:
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipe.fit(train_x, train_y)
    return pipe.predict_proba(test_x)[:, 1]


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score))


def compute_a_l_from_pair_angles(pair_angle_xs: np.ndarray, pair_angle_xl: np.ndarray, pair_angle_sl: np.ndarray) -> np.ndarray:
    a = np.asarray(pair_angle_xs, dtype=float)
    b = np.asarray(pair_angle_xl, dtype=float)
    c = np.asarray(pair_angle_sl, dtype=float)
    out = np.full(len(a), np.nan, dtype=float)
    denom = np.sin(b) * np.sin(c)
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & (np.abs(denom) > 1e-6)
    if np.any(valid):
        cos_a_l = (np.cos(a[valid]) - np.cos(b[valid]) * np.cos(c[valid])) / denom[valid]
        cos_a_l = np.clip(cos_a_l, -1.0, 1.0)
        out[valid] = np.arccos(cos_a_l)
    return out


def compute_updated_observer_features(x_state: np.ndarray, cfg: SidecarTestConfig) -> pd.DataFrame:
    ms = ema_prior_states(x_state, beta=cfg.short_beta)
    ml = ema_prior_states(x_state, beta=cfg.long_beta)
    projective = compute_projective_state_features(x_state, ms, ml, min_state_energy=cfg.min_state_energy)
    lie = compute_lie_state_features(x_state, ms, ml, sfreq=1.0)
    oriented = compute_oriented_state_features(x_state, ms, ml, min_state_energy=cfg.min_state_energy, sfreq=1.0)

    vel = np.zeros(len(x_state), dtype=float)
    curv = np.zeros(len(x_state), dtype=float)
    if len(x_state) > 1:
        vel[1:] = np.linalg.norm(np.diff(x_state, axis=0), axis=1)
    if len(x_state) > 2:
        curv[2:] = np.linalg.norm(x_state[2:] - 2.0 * x_state[1:-1] + x_state[:-2], axis=1)

    observer = pd.DataFrame(
        {
            "observer_memory_align": projective["memory_align"],
            "observer_novelty": projective["novelty"],
            "observer_proj_lock_barrier_sl": projective["proj_lock_barrier_sl"],
            "observer_proj_lock_barrier_xl": projective["proj_lock_barrier_xl"],
            "observer_proj_volume_xsl": projective["proj_volume_xsl"],
            "observer_velocity_norm": vel,
            "observer_curvature_norm": curv,
            "observer_lie_orbit_norm": lie["lie_orbit_norm"],
            "observer_lie_strain_norm": lie["lie_strain_norm"],
            "observer_lie_commutator_norm": lie["lie_commutator_norm"],
            "observer_lie_metric_drift": lie["lie_metric_drift"],
            "observer_gram_logdet": lie["gram_logdet"],
            "observer_omega_xs": oriented["lie_orbit_xs"],
            "observer_omega_xl": oriented["lie_orbit_xl"],
            "observer_omega_sl": oriented["lie_orbit_sl"],
            "observer_pair_angle_xs": oriented["pair_angle_xs"],
            "observer_pair_angle_xl": oriented["pair_angle_xl"],
            "observer_pair_angle_sl": oriented["pair_angle_sl"],
            "observer_oriented_pair_angle_sl": oriented["oriented_pair_angle_sl"],
            "observer_oriented_volume_xsl": oriented["oriented_volume_xsl"],
            "observer_negative_volume_excursion_xsl": oriented["negative_volume_excursion_xsl"],
            "observer_phase_coherence_residual": oriented["phase_coherence_residual_xs_xl_sl"],
            "observer_sheaf_defect_log_ratio": oriented["sheaf_defect_log_ratio"],
            "observer_spectral_orbit_divergence_proxy": oriented["spectral_orbit_divergence_proxy"],
            "observer_spectral_strain_divergence_proxy": oriented["spectral_strain_divergence_proxy"],
        }
    )

    observer["observer_A_l"] = compute_a_l_from_pair_angles(
        oriented["pair_angle_xs"],
        oriented["pair_angle_xl"],
        oriented["pair_angle_sl"],
    )
    observer["observer_theta_sl"] = oriented["oriented_pair_angle_sl"]

    sin_b = np.sin(observer["observer_pair_angle_xl"].to_numpy(dtype=float))
    cos_b = np.cos(observer["observer_pair_angle_xl"].to_numpy(dtype=float))
    cot_b = np.divide(cos_b, sin_b, out=np.full(len(observer), np.nan, dtype=float), where=np.abs(sin_b) > 1e-6)
    theta_sl = observer["observer_theta_sl"].to_numpy(dtype=float)
    a_l = observer["observer_A_l"].to_numpy(dtype=float)

    phi_first_order = np.full(len(observer), np.nan, dtype=float)
    phi_second_order = np.full(len(observer), np.nan, dtype=float)
    phi_remainder_bound = np.full(len(observer), np.nan, dtype=float)
    valid_basic = np.isfinite(theta_sl) & np.isfinite(a_l)
    phi_first_order[valid_basic] = -2.0 * theta_sl[valid_basic] * np.square(np.cos(a_l[valid_basic] / 2.0))
    valid_second = valid_basic & np.isfinite(cot_b)
    phi_second_order[valid_second] = phi_first_order[valid_second] + 0.5 * theta_sl[valid_second] * np.abs(theta_sl[valid_second]) * cot_b[valid_second] * np.square(np.sin(a_l[valid_second]))
    angle_b = observer["observer_pair_angle_xl"].to_numpy(dtype=float)
    valid_bound = valid_second & np.isfinite(angle_b)
    phi_remainder_bound[valid_bound] = (
        0.5 * np.square(theta_sl[valid_bound]) * np.abs(cot_b[valid_bound]) * np.square(np.sin(a_l[valid_bound]))
        + (np.abs(theta_sl[valid_bound]) ** 3 / 6.0)
        * ((1.0 + np.square(np.cos(angle_b[valid_bound]))) / np.maximum(np.abs(np.sin(angle_b[valid_bound])) ** 3, 1e-6))
    )
    phi_conf = np.divide(1.0, 1.0 + phi_remainder_bound, out=np.full(len(observer), np.nan, dtype=float), where=np.isfinite(phi_remainder_bound))
    phi_second_order_gated = phi_second_order * phi_conf
    observer["observer_phi_first_order"] = phi_first_order
    observer["observer_phi_second_order"] = phi_second_order
    observer["observer_phi_remainder_bound"] = phi_remainder_bound
    observer["observer_phi_confidence"] = phi_conf
    observer["observer_phi_second_order_gated"] = phi_second_order_gated

    projective_proxy = np.sqrt(
        np.square(robust_z(observer["observer_novelty"].to_numpy(dtype=float)))
        + np.square(robust_z(observer["observer_proj_lock_barrier_sl"].to_numpy(dtype=float)))
        + np.square(robust_z(observer["observer_pair_angle_sl"].to_numpy(dtype=float)))
        + np.square(robust_z(observer["observer_negative_volume_excursion_xsl"].to_numpy(dtype=float)))
    )
    lie_proxy = np.sqrt(
        np.square(robust_z(observer["observer_lie_orbit_norm"].to_numpy(dtype=float)))
        + np.square(robust_z(observer["observer_lie_strain_norm"].to_numpy(dtype=float)))
        + np.square(robust_z(observer["observer_lie_commutator_norm"].to_numpy(dtype=float)))
        + 0.5 * np.square(robust_z(observer["observer_lie_metric_drift"].to_numpy(dtype=float)))
        + 0.25 * np.square(robust_z(observer["observer_gram_logdet"].to_numpy(dtype=float)))
    )
    observer["observer_projective_transition_score"] = projective_proxy
    observer["observer_lie_transition_score"] = lie_proxy
    observer["observer_combined_transition_score"] = 0.70 * projective_proxy + 0.30 * lie_proxy

    observer_num_cols = [c for c in observer.columns if pd.api.types.is_numeric_dtype(observer[c])]
    for col in observer_num_cols:
        observer[col + "_z"] = robust_z(observer[col].to_numpy(dtype=float))

    return observer


def load_record_excerpts(cfg: SidecarTestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    record_frames: list[pd.DataFrame] = []
    record_meta: list[dict[str, object]] = []
    for rec in cfg.records:
        ann = wfdb.rdann(rec, "sta", pn_dir=cfg.pn_dir)
        if len(ann.sample) == 0:
            continue
        event_sample = int(ann.sample[0])
        hdr = wfdb.rdheader(rec, pn_dir=cfg.pn_dir)
        fs = float(hdr.fs)
        sampfrom = max(0, int(event_sample - cfg.pre_event_seconds * fs))
        sampto = int(event_sample + cfg.post_event_seconds * fs)
        rec_obj = wfdb.rdrecord(rec, pn_dir=cfg.pn_dir, sampfrom=sampfrom, sampto=sampto)
        lead_cols = make_unique(rec_obj.sig_name)
        df_rec = pd.DataFrame(rec_obj.p_signal, columns=lead_cols)
        df_rec["sample"] = np.arange(sampfrom, sampto)
        df_rec["time_seconds"] = (df_rec["sample"] - event_sample) / fs
        df_rec["patient_id"] = rec
        df_rec["event_onset"] = 0
        nearest_idx = int(np.argmin(np.abs(df_rec["sample"].to_numpy() - event_sample)))
        df_rec.loc[nearest_idx, "event_onset"] = 1
        df_rec["event_active"] = (df_rec["time_seconds"] >= 0).astype(int)
        df_rec["event"] = df_rec["event_active"]
        df_rec["event_sample"] = event_sample
        df_rec["annotation_ext"] = "sta"
        df_rec["fs"] = fs
        if cfg.downsample > 1:
            df_rec = df_rec.iloc[:: cfg.downsample].reset_index(drop=True)
        df_rec["row_idx"] = np.arange(len(df_rec))
        record_frames.append(df_rec)
        record_meta.append(
            {
                "patient_id": rec,
                "rows": len(df_rec),
                "event_sample": event_sample,
                "fs_original": fs,
                "fs_effective": fs / cfg.downsample,
                "lead_cols": lead_cols,
            }
        )
    return pd.concat(record_frames, ignore_index=True, sort=False), pd.DataFrame(record_meta)


def build_record_panel(df_rec: pd.DataFrame, meta_row: pd.Series, cfg: SidecarTestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = df_rec.sort_values("time_seconds").reset_index(drop=True).copy()
    fs_eff = float(meta_row["fs_effective"])
    meta_cols = {
        "sample",
        "time_seconds",
        "patient_id",
        "event",
        "event_active",
        "event_onset",
        "event_sample",
        "annotation_ext",
        "fs",
        "row_idx",
    }
    lead_cols = [c for c in g.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(g[c])]
    lead_cols = [c for c in lead_cols if g[c].notna().sum() > 0]
    for col in lead_cols:
        g[col] = robust_z(g[col].to_numpy(dtype=float))
        g[f"d_{col}"] = np.r_[0.0, np.diff(g[col].to_numpy(dtype=float))]

    state_cols = lead_cols + [f"d_{col}" for col in lead_cols]
    if len(lead_cols) >= 2:
        g["lead_diff"] = g[lead_cols[0]] - g[lead_cols[1]]
        g["lead_sum"] = g[lead_cols[0]] + g[lead_cols[1]]
        state_cols += ["lead_diff", "lead_sum"]

    step = max(1, int(round(cfg.grid_step_seconds * fs_eff)))
    win = max(step * 2, int(round(cfg.graph_window_seconds * fs_eff)))
    sample_idx = np.arange(win, len(g), step)
    if len(sample_idx) < 50:
        raise ValueError(f"{meta_row['patient_id']}: too few rows after analysis windowing.")

    state_matrix = []
    time_grid = []
    onset_grid = []
    active_grid = []
    for idx in sample_idx:
        cur = g.iloc[idx]
        state_matrix.append(cur[state_cols].to_numpy(dtype=float))
        time_grid.append(float(cur["time_seconds"]))
        onset_grid.append(int(cur["event_onset"]))
        active_grid.append(int(cur["event_active"]))
    x_state = np.vstack(state_matrix)
    observer = compute_updated_observer_features(x_state, cfg)

    prev_adapt = None
    rows: list[dict[str, object]] = []
    for k, idx in enumerate(sample_idx):
        window = g.iloc[idx - win : idx].copy()
        x_lead = window[lead_cols].to_numpy(dtype=float)
        corr = corr_graph(x_lead)
        lag = lag_graph(x_lead)
        if prev_adapt is None:
            adapt = lag.copy()
            adapt_drift = 0.0
        else:
            adapt = cfg.adapt_beta * prev_adapt + (1.0 - cfg.adapt_beta) * lag
            adapt_drift = float(np.linalg.norm(adapt - prev_adapt, ord="fro"))
        prev_adapt = adapt.copy()

        row = {
            "patient_id": meta_row["patient_id"],
            "time_seconds": time_grid[k],
            "event_onset": onset_grid[k],
            "event_active": active_grid[k],
            "event": active_grid[k],
        }
        row.update(graph_stats(corr, "corr"))
        row.update(graph_stats(lag, "granger"))
        row.update(graph_stats(adapt, "adapt", drift=adapt_drift))
        row.update(observer.iloc[k].to_dict())
        rows.append(row)

    panel = pd.DataFrame(rows).sort_values("time_seconds").reset_index(drop=True)
    onset_idx = np.where(panel["event_onset"].to_numpy() > 0)[0]
    if len(onset_idx) > 0:
        first_onset = int(onset_idx[0])
        at_risk = np.zeros(len(panel), dtype=int)
        at_risk[: first_onset + 1] = 1
        panel["at_risk"] = at_risk
        panel["time_from_event"] = panel["time_seconds"]
        for horizon in cfg.hazard_horizons:
            col = f"future_onset_within_{horizon}"
            future = np.zeros(len(panel), dtype=int)
            lo = max(0, first_onset - horizon)
            future[lo:first_onset] = 1
            panel[col] = future
    else:
        panel["at_risk"] = 0
        panel["time_from_event"] = panel["time_seconds"]
        for horizon in cfg.hazard_horizons:
            panel[f"future_onset_within_{horizon}"] = 0

    graph_feature_cols = [c for c in panel.columns if c.startswith(("adapt_", "corr_", "granger_"))]
    observer_feature_cols = [c for c in panel.columns if c.startswith("observer_") and not c.endswith("_zbg")]
    graph_rank, graph_summary, bg_df, _, _ = summarize_event_windows(panel, graph_feature_cols, cfg)
    observer_rank, _, _, _, _ = summarize_event_windows(panel, observer_feature_cols, cfg)

    graph_scored, graph_feats = build_pressure_score(panel, graph_rank, bg_df, topn=4)
    observer_scored, observer_feats = build_pressure_score(panel, observer_rank, bg_df, topn=4)
    panel = graph_scored.copy()
    for feat in observer_feats:
        panel[feat + "_zbg"] = observer_scored[feat + "_zbg"]

    panel["graph_pressure_score"] = panel["event_pressure_score"]
    panel["observer_pressure_score"] = panel[[feat + "_zbg" for feat in observer_feats]].abs().mean(axis=1)
    panel["hybrid_pressure_score"] = 0.75 * panel["graph_pressure_score"] + 0.25 * panel["observer_pressure_score"]

    summary = graph_summary.copy()
    summary["patient_id"] = meta_row["patient_id"]
    summary["n_rows"] = len(panel)
    summary["event_onset_rows"] = int(panel["event_onset"].sum())
    summary["event_active_rows"] = int(panel["event_active"].sum())
    summary["graph_best_feature"] = graph_feats[0] if graph_feats else np.nan
    summary["observer_best_feature"] = observer_feats[0] if observer_feats else np.nan
    summary["graph_feats"] = ", ".join(graph_feats)
    summary["observer_feats"] = ", ".join(observer_feats)
    return panel, summary


def run_hazard(panel: pd.DataFrame, cfg: SidecarTestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    risk_df = panel.loc[panel["at_risk"] == 1].copy().reset_index(drop=True)
    model_sets = {
        "graph_only": ["graph_pressure_score"],
        "observer_only": ["observer_pressure_score"],
        "hybrid_only": ["hybrid_pressure_score"],
        "graph_plus_observer": ["graph_pressure_score", "observer_pressure_score"],
    }
    hazard_rows: list[dict[str, object]] = []
    hazard_pred_rows: list[pd.DataFrame] = []
    for horizon in cfg.hazard_horizons:
        target_col = f"future_onset_within_{horizon}"
        for model_name, feat_cols in model_sets.items():
            fold_preds: list[pd.DataFrame] = []
            for held_out in sorted(risk_df["patient_id"].unique()):
                train = risk_df.loc[risk_df["patient_id"] != held_out].copy()
                test = risk_df.loc[risk_df["patient_id"] == held_out].copy()
                if train.empty or test.empty or train[target_col].nunique() < 2:
                    continue
                pred = fit_logistic_predict(train[feat_cols], train[target_col], test[feat_cols])
                temp = test[["patient_id", "time_seconds", target_col]].copy()
                temp["horizon"] = horizon
                temp["model_name"] = model_name
                temp["prediction"] = pred
                fold_preds.append(temp)
                hazard_rows.append(
                    {
                        "horizon": horizon,
                        "held_out_patient": held_out,
                        "model_name": model_name,
                        "auroc": safe_auc(test[target_col].to_numpy(dtype=int), pred),
                        "ap": safe_ap(test[target_col].to_numpy(dtype=int), pred),
                        "n_test_rows": len(test),
                        "n_positive_test": int(test[target_col].sum()),
                    }
                )
            if fold_preds:
                hazard_pred_rows.extend(fold_preds)
    hazard_results = pd.DataFrame(hazard_rows).sort_values(["horizon", "ap"], ascending=[True, False]).reset_index(drop=True)
    hazard_predictions = pd.concat(hazard_pred_rows, ignore_index=True) if hazard_pred_rows else pd.DataFrame()
    return hazard_results, hazard_predictions


def write_report(run_dir: Path, cfg: SidecarTestConfig, panel: pd.DataFrame, summaries: pd.DataFrame, hazard_results: pd.DataFrame) -> None:
    lines = [
        "# LTST Three-Record Graph + Updated Geometry Sidecar Test",
        "",
        "This run preserves the remote notebook's graph branch but replaces the older hand-built observer with the newer projective/Lie/oriented sidecar from `geometry_lie.py`.",
        "",
        "## Records",
        "",
    ]
    for rec in cfg.records:
        lines.append(f"- `{rec}`")
    lines.extend(
        [
            "",
            "## Per-Record Summary",
            "",
            summaries.to_markdown(index=False) if not summaries.empty else "_No summaries generated._",
            "",
            "## Hazard Results",
            "",
            hazard_results.to_markdown(index=False) if not hazard_results.empty else "_No hazard results generated._",
            "",
            "## Output Files",
            "",
            f"- `{run_dir.name}_panel.csv`",
            f"- `{run_dir.name}_record_summary.csv`",
            f"- `{run_dir.name}_hazard_results.csv`",
            f"- `{run_dir.name}_hazard_predictions.csv`",
        ]
    )
    (run_dir / f"{run_dir.name}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-record LTST graph + updated geometry sidecar test.")
    parser.add_argument("--records", default="s20011,s20041,s30742", help="Comma-separated LTST records.")
    parser.add_argument("--pn-dir", default="ltstdb", help="WFDB PhysioNet directory or local database root.")
    parser.add_argument("--out-prefix", default="ltst_three_record_graph_updated_sidecar", help="Run directory prefix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SidecarTestConfig(records=tuple(piece.strip() for piece in args.records.split(",") if piece.strip()), pn_dir=args.pn_dir, out_prefix=args.out_prefix)
    project_root = find_project_root()
    run_dir = prepare_run_directory(project_root / cfg.run_root, cfg.out_prefix)

    raw_all, record_meta_df = load_record_excerpts(cfg)
    panels: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    for rec, df_rec in raw_all.groupby("patient_id"):
        meta_row = record_meta_df.loc[record_meta_df["patient_id"] == rec].iloc[0]
        panel, summary = build_record_panel(df_rec, meta_row, cfg)
        panels.append(panel)
        summaries.append(summary)

    model_df = pd.concat(panels, ignore_index=True, sort=False)
    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    hazard_results, hazard_predictions = run_hazard(model_df, cfg)

    panel_path = run_dir / f"{run_dir.name}_panel.csv"
    summary_path = run_dir / f"{run_dir.name}_record_summary.csv"
    hazard_results_path = run_dir / f"{run_dir.name}_hazard_results.csv"
    hazard_predictions_path = run_dir / f"{run_dir.name}_hazard_predictions.csv"
    model_df.to_csv(panel_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    hazard_results.to_csv(hazard_results_path, index=False)
    hazard_predictions.to_csv(hazard_predictions_path, index=False)
    write_report(run_dir, cfg, model_df, summary_df, hazard_results)

    manifest = {
        "config": asdict(cfg),
        "run_dir": str(run_dir),
        "panel_csv": str(panel_path),
        "record_summary_csv": str(summary_path),
        "hazard_results_csv": str(hazard_results_path),
        "hazard_predictions_csv": str(hazard_predictions_path),
        "n_rows": int(len(model_df)),
        "n_records": int(model_df["patient_id"].nunique()) if not model_df.empty else 0,
    }
    (run_dir / f"{run_dir.name}_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
