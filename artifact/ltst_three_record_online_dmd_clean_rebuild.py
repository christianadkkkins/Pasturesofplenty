from __future__ import annotations

import argparse
import json
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
LEAKAGE_PREFIXES = ("future_onset_within_",)
EXCLUDED_RANK_COLUMNS = {
    "event",
    "event_onset",
    "event_active",
    "at_risk",
    "time_seconds",
    "time_sec_bin",
    "sample",
    "event_sample",
    "fs",
    "row_idx",
}


@dataclass(frozen=True)
class CleanDMDConfig:
    records: tuple[str, ...] = ("s20011", "s20041", "s30742")
    pn_dir: str = "ltstdb"
    pre_event_seconds: int = 1200
    post_event_seconds: int = 60
    downsample: int = 5
    dmd_init_batch: int = 40
    use_forgetting: bool = False
    rho: float = 0.9995
    top_k_eigs: int = 3
    short_beta: float = 0.65
    long_beta: float = 0.92
    min_state_energy: float = 1e-9
    modal_clip: float = 6.0
    bg_rows: int = 240
    gap_rows: int = 30
    pre_rows: int = 120
    post_rows: int = 60
    hazard_horizons: tuple[int, ...] = (5, 10, 20, 30)
    topn_substrate_feats: int = 6
    topn_observer_feats: int = 6
    min_baseline_valid_fraction: float = 0.6
    min_baseline_nonconstant_records: int = 2
    out_prefix: str = "ltst_three_record_online_dmd_clean_rebuild"
    run_root: Path = Path("artifact") / "runs"


def find_project_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "artifact").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root containing artifact/.")


def prepare_run_directory(base_root: Path, prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = base_root / f"{prefix}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def make_unique(names: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw in names:
        name = str(raw)
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name] - 1}")
    return out


def robust_z(values: np.ndarray, clip: float = 8.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    med = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - med)))
    if not np.isfinite(mad) or mad < EPS:
        sd = float(np.nanstd(arr))
        if not np.isfinite(sd) or sd < EPS:
            z = np.zeros_like(arr, dtype=float)
        else:
            z = (arr - float(np.nanmean(arr))) / (sd + EPS)
    else:
        z = 0.6745 * (arr - med) / (mad + EPS)
    z = np.nan_to_num(z, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(z, -clip, clip)


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


def ensure_event_columns(df: pd.DataFrame, time_col: str = "time_seconds", event_col: str = "event", onset_col: str = "event_onset") -> pd.DataFrame:
    g = df.copy().sort_values(time_col).reset_index(drop=True)
    if event_col not in g.columns:
        g[event_col] = 0
    g[event_col] = pd.to_numeric(g[event_col], errors="coerce").fillna(0).astype(int)
    if onset_col not in g.columns:
        g[onset_col] = 0
    g[onset_col] = pd.to_numeric(g[onset_col], errors="coerce").fillna(0).astype(int)
    if g[onset_col].sum() < 1 and g[event_col].sum() > 0:
        idx = np.where(g[event_col].to_numpy() > 0)[0]
        if len(idx) > 0:
            g.loc[idx[0], onset_col] = 1
    return g


def add_hazard_labels(df: pd.DataFrame, horizons: tuple[int, ...], time_col: str = "time_seconds", onset_col: str = "event_onset") -> pd.DataFrame:
    g = ensure_event_columns(df, time_col=time_col, onset_col=onset_col)
    onset_idx = np.where(g[onset_col].to_numpy() > 0)[0]
    first_onset = int(onset_idx[0]) if len(onset_idx) > 0 else None
    g["event_active"] = g["event"].astype(int)
    g["at_risk"] = 1
    if first_onset is not None:
        g.loc[first_onset:, "at_risk"] = 0
    for horizon in horizons:
        future = np.zeros(len(g), dtype=int)
        if first_onset is not None:
            start = max(0, first_onset - horizon)
            future[start:first_onset] = 1
        g[f"future_onset_within_{horizon}"] = future
    return g


def load_record_excerpts(cfg: CleanDMDConfig) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    record_data: dict[str, dict[str, object]] = {}
    frames: list[pd.DataFrame] = []
    for rec_name in cfg.records:
        ann = wfdb.rdann(rec_name, "sta", pn_dir=cfg.pn_dir)
        if len(ann.sample) == 0:
            continue
        event_sample = int(ann.sample[0])
        hdr = wfdb.rdheader(rec_name, pn_dir=cfg.pn_dir)
        fs = float(hdr.fs)
        sampfrom = max(0, int(event_sample - cfg.pre_event_seconds * fs))
        sampto = int(event_sample + cfg.post_event_seconds * fs)
        rec_obj = wfdb.rdrecord(rec_name, pn_dir=cfg.pn_dir, sampfrom=sampfrom, sampto=sampto)
        lead_cols = make_unique(rec_obj.sig_name)
        df_rec = pd.DataFrame(rec_obj.p_signal, columns=lead_cols)
        df_rec["sample"] = np.arange(sampfrom, sampto)
        df_rec["time_seconds"] = (df_rec["sample"] - event_sample) / fs
        df_rec["patient_id"] = rec_name
        df_rec["event"] = 0
        nearest_idx = int(np.argmin(np.abs(df_rec["sample"].to_numpy() - event_sample)))
        df_rec.loc[nearest_idx, "event"] = 1
        df_rec["event_sample"] = event_sample
        df_rec["annotation_ext"] = "sta"
        df_rec["fs"] = fs
        if cfg.downsample > 1:
            df_rec = df_rec.iloc[:: cfg.downsample].reset_index(drop=True)
        df_rec["row_idx"] = np.arange(len(df_rec))
        df_rec = ensure_event_columns(df_rec, time_col="time_seconds")
        record_data[rec_name] = {
            "df": df_rec.copy(),
            "lead_cols": [col for col in lead_cols if col in df_rec.columns],
            "fs_effective": fs / cfg.downsample,
        }
        frames.append(df_rec.copy())
    return record_data, pd.concat(frames, ignore_index=True, sort=False)


def build_state_frames(record_data: dict[str, dict[str, object]]) -> list[tuple[str, pd.DataFrame, list[str]]]:
    state_frames: list[tuple[str, pd.DataFrame, list[str]]] = []
    for pid, payload in record_data.items():
        g = payload["df"].copy().sort_values("time_seconds").reset_index(drop=True)
        lead_cols = [str(col) for col in payload["lead_cols"] if col in g.columns]
        cleaned_leads: list[str] = []
        for col in lead_cols:
            g[col] = pd.to_numeric(g[col], errors="coerce")
            if g[col].notna().sum() >= 20:
                cleaned_leads.append(col)
        lead_cols = cleaned_leads
        if not lead_cols:
            continue
        for col in lead_cols:
            vals = g[col].to_numpy(dtype=float)
            g[col] = robust_z(vals)
            g[f"d_{col}"] = np.r_[0.0, np.diff(g[col].to_numpy(dtype=float))]
        if len(lead_cols) >= 2:
            g["lead_diff"] = g[lead_cols[0]] - g[lead_cols[1]]
            g["lead_sum"] = g[lead_cols[0]] + g[lead_cols[1]]
            state_cols = lead_cols + [f"d_{col}" for col in lead_cols] + ["lead_diff", "lead_sum"]
        else:
            state_cols = lead_cols + [f"d_{col}" for col in lead_cols]
        for col in state_cols:
            g[col] = pd.to_numeric(g[col], errors="coerce").fillna(0.0)
        state_frames.append((pid, g, state_cols))
    return state_frames


def build_snapshot_pairs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=float)
    return arr[:-1].T, arr[1:].T


def initialize_dmd_from_batch(x_batch: np.ndarray, ridge: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    xk, yk = build_snapshot_pairs(x_batch)
    dim = xk.shape[0]
    gram = xk @ xk.T + ridge * np.eye(dim)
    p0 = np.linalg.pinv(gram)
    a0 = (yk @ xk.T) @ p0
    return a0, p0


def _scalar_quad(x: np.ndarray, p: np.ndarray) -> float:
    val = x.T @ p @ x
    return float(np.asarray(val).item())


def online_dmd_rank1_update(a: np.ndarray, p: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_col = np.asarray(x, dtype=float).reshape(-1, 1)
    y_col = np.asarray(y, dtype=float).reshape(-1, 1)
    quad = _scalar_quad(x_col, p)
    denom = 1.0 + quad
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return a, p
    px = p @ x_col
    p_next = p - (px @ px.T) / denom
    a_next = a + ((y_col - a @ x_col) @ (x_col.T @ p)) / denom
    p_next = 0.5 * (p_next + p_next.T)
    if not np.isfinite(p_next).all() or not np.isfinite(a_next).all():
        return a, p
    return a_next, p_next


def online_dmd_weighted_update(a: np.ndarray, p: np.ndarray, x: np.ndarray, y: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    x_col = np.asarray(x, dtype=float).reshape(-1, 1)
    y_col = np.asarray(y, dtype=float).reshape(-1, 1)
    quad = _scalar_quad(x_col, p)
    denom = rho + quad
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return a, p
    px = p @ x_col
    p_next = (p - (px @ px.T) / denom) / rho
    a_next = a + ((y_col - a @ x_col) @ (x_col.T @ p)) / denom
    p_next = 0.5 * (p_next + p_next.T)
    if not np.isfinite(p_next).all() or not np.isfinite(a_next).all():
        return a, p
    return a_next, p_next


def run_online_dmd(x: np.ndarray, init_batch: int, weighted: bool, rho: float) -> list[dict[str, object]]:
    arr = np.asarray(x, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape[0] <= init_batch + 2:
        return []
    a, p = initialize_dmd_from_batch(arr[: init_batch + 1], ridge=1e-6)
    rows: list[dict[str, object]] = []
    for t in range(init_batch, len(arr) - 1):
        x_t = arr[t]
        y_t = arr[t + 1]
        if weighted:
            a, p = online_dmd_weighted_update(a, p, x_t, y_t, rho=rho)
        else:
            a, p = online_dmd_rank1_update(a, p, x_t, y_t)
        if not np.isfinite(a).all() or not np.isfinite(p).all():
            break
        try:
            eigvals, eigvecs = np.linalg.eig(a)
        except np.linalg.LinAlgError:
            break
        rows.append({"row_idx": t, "A_t": a.copy(), "P_t": p.copy(), "eigvals": eigvals.copy(), "eigvecs": eigvecs.copy()})
    return rows


def extract_dmd_frame(state_frames: list[tuple[str, pd.DataFrame, list[str]]], cfg: CleanDMDConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pid, g, state_cols in state_frames:
        x = g[state_cols].to_numpy(dtype=float)
        dmd_out = run_online_dmd(x, init_batch=cfg.dmd_init_batch, weighted=cfg.use_forgetting, rho=cfg.rho)
        for item in dmd_out:
            t = int(item["row_idx"])
            out_idx = min(t + 1, len(g) - 1)
            eigvals = np.asarray(item["eigvals"])
            eigvecs = np.asarray(item["eigvecs"])
            order = np.argsort(np.abs(eigvals))[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            row: dict[str, object] = {
                "patient_id": pid,
                "row_idx": out_idx,
                "time_seconds": float(g.loc[out_idx, "time_seconds"]),
                "event": int(g.loc[out_idx, "event"]),
                "event_onset": int(g.loc[out_idx, "event_onset"]),
                "dominant_radius": float(np.max(np.abs(eigvals))),
                "dominant_real": float(np.max(np.real(eigvals))),
                "dominant_imag_abs": float(np.max(np.abs(np.imag(eigvals)))),
            }
            top_k = min(cfg.top_k_eigs, eigvals.shape[0])
            for k in range(top_k):
                row[f"lambda_{k}_real"] = float(np.real(eigvals[k]))
                row[f"lambda_{k}_imag"] = float(np.imag(eigvals[k]))
                row[f"lambda_{k}_abs"] = float(np.abs(eigvals[k]))
                mode_vec = eigvecs[:, k]
                for j, val in enumerate(mode_vec):
                    row[f"mode_{k}_real_{j}"] = float(np.real(val))
                    row[f"mode_{k}_imag_{j}"] = float(np.imag(val))
            rows.append(row)
    dmd_df = pd.DataFrame(rows).sort_values(["patient_id", "time_seconds"]).reset_index(drop=True)
    if dmd_df.empty:
        return dmd_df
    for col in [
        "dominant_radius",
        "dominant_real",
        "dominant_imag_abs",
        "lambda_0_abs",
        "lambda_0_real",
        "lambda_0_imag",
        "lambda_1_abs",
        "lambda_1_real",
        "lambda_1_imag",
        "lambda_2_abs",
        "lambda_2_real",
        "lambda_2_imag",
    ]:
        if col in dmd_df.columns:
            dmd_df[f"{col}_drift"] = dmd_df.groupby("patient_id")[col].diff().fillna(0.0)
    if "lambda_0_abs" in dmd_df.columns:
        dmd_df["unit_circle_gap"] = np.abs(dmd_df["lambda_0_abs"] - 1.0)
    if {"lambda_0_real", "lambda_1_real", "lambda_2_real"} <= set(dmd_df.columns):
        dmd_df["top3_real_sum"] = dmd_df["lambda_0_real"] + dmd_df["lambda_1_real"] + dmd_df["lambda_2_real"]
    if {"lambda_0_abs", "lambda_1_abs", "lambda_2_abs"} <= set(dmd_df.columns):
        dmd_df["top3_abs_sum"] = dmd_df["lambda_0_abs"] + dmd_df["lambda_1_abs"] + dmd_df["lambda_2_abs"]
    if {"lambda_0_imag", "lambda_1_imag", "lambda_2_imag"} <= set(dmd_df.columns):
        dmd_df["top3_imag_abs_sum"] = np.abs(dmd_df["lambda_0_imag"]) + np.abs(dmd_df["lambda_1_imag"]) + np.abs(dmd_df["lambda_2_imag"])
    if "mode_0_real_0" in dmd_df.columns:
        dmd_df["mode0_real_drift"] = dmd_df.groupby("patient_id")["mode_0_real_0"].diff().fillna(0.0)
    return dmd_df


def candidate_modal_cols(df: pd.DataFrame) -> list[str]:
    prefixes = (
        "dominant_",
        "lambda_0_",
        "lambda_1_",
        "lambda_2_",
        "unit_circle_gap",
        "top3_real_sum",
        "top3_abs_sum",
        "top3_imag_abs_sum",
        "mode0_real_drift",
    )
    cols: list[str] = []
    for col in df.columns:
        if col == "row_idx":
            continue
        if any(col.startswith(prefix) for prefix in prefixes) and pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def stabilize_modal_frame(df: pd.DataFrame, state_cols: list[str], clip: float) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    stable_cols: list[str] = []
    for col in state_cols:
        vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        z = robust_z(vals, clip=clip)
        stable_col = f"stable_{col}"
        out[stable_col] = z
        stable_cols.append(stable_col)
    return out, stable_cols


def compute_a_l_from_pair_angles(pair_angle_xs: np.ndarray, pair_angle_xl: np.ndarray, pair_angle_sl: np.ndarray) -> np.ndarray:
    a = np.asarray(pair_angle_xs, dtype=float)
    b = np.asarray(pair_angle_xl, dtype=float)
    c = np.asarray(pair_angle_sl, dtype=float)
    out = np.full(len(a), np.nan, dtype=float)
    denom = np.sin(b) * np.sin(c)
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & (np.abs(denom) > 1e-6)
    if np.any(valid):
        cos_a_l = (np.cos(a[valid]) - np.cos(b[valid]) * np.cos(c[valid])) / denom[valid]
        out[valid] = np.arccos(np.clip(cos_a_l, -1.0, 1.0))
    return out


def compute_updated_dmd_observer(x_state: np.ndarray, cfg: CleanDMDConfig) -> pd.DataFrame:
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
            "dmd_observer_memory_align": projective["memory_align"],
            "dmd_observer_novelty": projective["novelty"],
            "dmd_observer_proj_lock_barrier_sl": projective["proj_lock_barrier_sl"],
            "dmd_observer_proj_lock_barrier_xl": projective["proj_lock_barrier_xl"],
            "dmd_observer_proj_volume_xsl": projective["proj_volume_xsl"],
            "dmd_observer_velocity_norm": vel,
            "dmd_observer_curvature_norm": curv,
            "dmd_observer_lie_orbit_norm": lie["lie_orbit_norm"],
            "dmd_observer_lie_strain_norm": lie["lie_strain_norm"],
            "dmd_observer_lie_commutator_norm": lie["lie_commutator_norm"],
            "dmd_observer_lie_metric_drift": lie["lie_metric_drift"],
            "dmd_observer_gram_logdet": lie["gram_logdet"],
            "dmd_observer_pair_angle_xs": oriented["pair_angle_xs"],
            "dmd_observer_pair_angle_xl": oriented["pair_angle_xl"],
            "dmd_observer_pair_angle_sl": oriented["pair_angle_sl"],
            "dmd_observer_oriented_pair_angle_sl": oriented["oriented_pair_angle_sl"],
            "dmd_observer_oriented_volume_xsl": oriented["oriented_volume_xsl"],
            "dmd_observer_negative_volume_excursion_xsl": oriented["negative_volume_excursion_xsl"],
            "dmd_observer_phase_coherence_residual": oriented["phase_coherence_residual_xs_xl_sl"],
            "dmd_observer_sheaf_defect_log_ratio": oriented["sheaf_defect_log_ratio"],
        }
    )
    observer["dmd_observer_A_l"] = compute_a_l_from_pair_angles(
        oriented["pair_angle_xs"],
        oriented["pair_angle_xl"],
        oriented["pair_angle_sl"],
    )
    proj_proxy = np.sqrt(
        np.square(robust_z(observer["dmd_observer_novelty"].to_numpy(dtype=float)))
        + np.square(robust_z(observer["dmd_observer_proj_lock_barrier_sl"].to_numpy(dtype=float)))
        + np.square(robust_z(observer["dmd_observer_pair_angle_sl"].to_numpy(dtype=float)))
        + np.square(robust_z(observer["dmd_observer_negative_volume_excursion_xsl"].to_numpy(dtype=float)))
    )
    lie_proxy = np.sqrt(
        np.square(robust_z(observer["dmd_observer_lie_orbit_norm"].to_numpy(dtype=float)))
        + np.square(robust_z(observer["dmd_observer_lie_strain_norm"].to_numpy(dtype=float)))
        + np.square(robust_z(observer["dmd_observer_lie_commutator_norm"].to_numpy(dtype=float)))
        + 0.5 * np.square(robust_z(observer["dmd_observer_lie_metric_drift"].to_numpy(dtype=float)))
        + 0.25 * np.square(robust_z(observer["dmd_observer_gram_logdet"].to_numpy(dtype=float)))
    )
    observer["dmd_observer_projective_transition_score"] = proj_proxy
    observer["dmd_observer_lie_transition_score"] = lie_proxy
    observer["dmd_observer_combined_transition_score"] = 0.70 * proj_proxy + 0.30 * lie_proxy
    for col in [c for c in observer.columns if pd.api.types.is_numeric_dtype(observer[c])]:
        observer[col + "_z"] = robust_z(observer[col].to_numpy(dtype=float))
    return observer


def build_master_panel(dmd_feat_df: pd.DataFrame, cfg: CleanDMDConfig) -> pd.DataFrame:
    observer_frames: list[pd.DataFrame] = []
    for pid, g in dmd_feat_df.groupby("patient_id", sort=False):
        g = g.sort_values("time_seconds").reset_index(drop=True).copy()
        usable_cols = [col for col in candidate_modal_cols(g) if g[col].notna().sum() >= 20]
        if len(usable_cols) < 3:
            continue
        stable_frame, stable_cols = stabilize_modal_frame(g, usable_cols, clip=cfg.modal_clip)
        observer = compute_updated_dmd_observer(stable_frame[stable_cols].to_numpy(dtype=float), cfg)
        observer.insert(0, "event_onset", g["event_onset"].to_numpy(dtype=int))
        observer.insert(0, "event", g["event"].to_numpy(dtype=int))
        observer.insert(0, "time_seconds", g["time_seconds"].to_numpy(dtype=float))
        observer.insert(0, "patient_id", g["patient_id"].to_numpy())
        observer_frames.append(observer)
    observer_df = pd.concat(observer_frames, ignore_index=True) if observer_frames else pd.DataFrame()
    return dmd_feat_df.merge(observer_df, on=["patient_id", "time_seconds", "event", "event_onset"], how="left")


def resample_to_1s(panel: pd.DataFrame, horizons: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = panel.copy()
    df["time_sec_bin"] = np.round(df["time_seconds"]).astype(int)
    meta_binary_cols = [col for col in ("event", "event_onset") if col in df.columns]
    num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != "time_sec_bin"]
    feature_num_cols = [col for col in num_cols if col not in meta_binary_cols + ["time_seconds"]]
    agg: dict[str, str] = {col: "mean" for col in feature_num_cols}
    for col in meta_binary_cols:
        agg[col] = "max"
    agg["time_seconds"] = "mean"
    sec_panel = (
        df.groupby(["patient_id", "time_sec_bin"], as_index=False)
        .agg(agg)
        .rename(columns={"time_sec_bin": "time_seconds_1s"})
    )
    sec_panel["time_seconds"] = sec_panel["time_seconds_1s"].astype(float)
    sec_panel = sec_panel.drop(columns=["time_seconds_1s"]).sort_values(["patient_id", "time_seconds"]).reset_index(drop=True)
    hazard_frames = [add_hazard_labels(g.copy(), horizons) for _, g in sec_panel.groupby("patient_id", sort=False)]
    hazard_df = pd.concat(hazard_frames, ignore_index=True) if hazard_frames else sec_panel.iloc[0:0].copy()
    risk_df = hazard_df.loc[hazard_df["at_risk"] == 1].copy().reset_index(drop=True)
    horizon_rows = []
    for horizon in horizons:
        col = f"future_onset_within_{horizon}"
        horizon_rows.append(
            {
                "horizon_seconds": horizon,
                "n_rows": len(risk_df),
                "n_positive": int(risk_df[col].sum()),
                "base_rate": float(risk_df[col].mean()) if len(risk_df) > 0 else np.nan,
            }
        )
    return hazard_df, risk_df, pd.DataFrame(horizon_rows)


def feature_is_rankable(name: str) -> bool:
    if name in EXCLUDED_RANK_COLUMNS:
        return False
    if any(name.startswith(prefix) for prefix in LEAKAGE_PREFIXES):
        return False
    return True


def summarize_event_windows(df: pd.DataFrame, feature_cols: list[str], cfg: CleanDMDConfig, event_col: str = "event_onset", time_col: str = "time_seconds") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy().sort_values(time_col).reset_index(drop=True)
    onset_idx_arr = np.where(work[event_col].to_numpy() > 0)[0]
    if len(onset_idx_arr) < 1:
        raise ValueError("No event_onset rows found.")
    event_idx = int(onset_idx_arr[0])
    bg_end = max(0, event_idx - cfg.gap_rows)
    bg_start = max(0, bg_end - cfg.bg_rows)
    pre_start = max(0, event_idx - cfg.pre_rows)
    pre_end = event_idx
    post_start = event_idx
    post_end = min(len(work), event_idx + cfg.post_rows)
    bg_df = work.iloc[bg_start:bg_end].copy()
    pre_df = work.iloc[pre_start:pre_end].copy()
    post_df = work.iloc[post_start:post_end].copy()
    rows = []
    for col in feature_cols:
        bg_vals = pd.to_numeric(bg_df[col], errors="coerce").to_numpy(dtype=float)
        pre_vals = pd.to_numeric(pre_df[col], errors="coerce").to_numpy(dtype=float)
        post_vals = pd.to_numeric(post_df[col], errors="coerce").to_numpy(dtype=float)
        if len(bg_vals) < 5 or len(pre_vals) < 5:
            continue
        bg_mean = float(np.nanmean(bg_vals))
        pre_mean = float(np.nanmean(pre_vals))
        post_mean = float(np.nanmean(post_vals))
        bg_std = float(np.nanstd(bg_vals) + EPS)
        rows.append(
            {
                "feature": col,
                "bg_mean": bg_mean,
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "pre_shift_sd": float((pre_mean - bg_mean) / bg_std),
                "post_shift_sd": float((post_mean - bg_mean) / bg_std),
                "peak_abs_pre_z": float(np.nanmax(np.abs(pre_vals))) if len(pre_vals) else np.nan,
                "abs_pre_shift": abs(float((pre_mean - bg_mean) / bg_std)),
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
    return rank, summary, bg_df


def build_pressure_score(df: pd.DataFrame, rank_df: pd.DataFrame, bg_df: pd.DataFrame, topn: int, score_name: str) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy().sort_values("time_seconds").reset_index(drop=True)
    top_feats: list[str] = []
    for feat in rank_df["feature"].tolist():
        if feat in out.columns and pd.api.types.is_numeric_dtype(out[feat]):
            top_feats.append(feat)
        if len(top_feats) >= topn:
            break
    if not top_feats:
        out[score_name] = np.nan
        return out, []
    for feat in top_feats:
        out[f"{feat}_zbg"] = robust_z_from_bg(out[feat], bg_df[feat])
    zcols = [f"{feat}_zbg" for feat in top_feats]
    out[score_name] = out[zcols].abs().mean(axis=1)
    return out, top_feats


def build_ranked_scores(hazard_1s_df: pd.DataFrame, cfg: CleanDMDConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panels: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    ranks: list[pd.DataFrame] = []
    for pid, g in hazard_1s_df.groupby("patient_id", sort=False):
        g = g.sort_values("time_seconds").reset_index(drop=True).copy()
        feature_cols = [
            col
            for col in g.columns
            if feature_is_rankable(col) and pd.api.types.is_numeric_dtype(g[col]) and g[col].notna().sum() >= 20
        ]
        substrate_cols = [col for col in feature_cols if not col.startswith("dmd_observer_")]
        observer_cols = [col for col in feature_cols if col.startswith("dmd_observer_")]
        substrate_rank, summary, bg_df = summarize_event_windows(g, substrate_cols, cfg)
        observer_rank, _, _ = summarize_event_windows(g, observer_cols, cfg)
        substrate_rank["patient_id"] = pid
        substrate_rank["family"] = "dmd_substrate"
        observer_rank["patient_id"] = pid
        observer_rank["family"] = "dmd_observer"
        substrate_scored, substrate_feats = build_pressure_score(g, substrate_rank, bg_df, cfg.topn_substrate_feats, "dmd_substrate_score")
        observer_scored, observer_feats = build_pressure_score(g, observer_rank, bg_df, cfg.topn_observer_feats, "dmd_observer_pressure_score")
        panel = substrate_scored.copy()
        for feat in observer_feats:
            zcol = f"{feat}_zbg"
            if zcol in observer_scored.columns:
                panel[zcol] = observer_scored[zcol]
        panel["dmd_observer_pressure_score"] = observer_scored["dmd_observer_pressure_score"] if observer_feats else np.nan
        panel["dmd_hybrid_score"] = panel[["dmd_substrate_score", "dmd_observer_pressure_score"]].mean(axis=1)
        summary["patient_id"] = pid
        summary["n_rows"] = len(g)
        summary["event_onset_rows"] = int(g["event_onset"].sum())
        summary["event_active_rows"] = int(g["event_active"].sum()) if "event_active" in g.columns else int(g["event"].sum())
        summary["substrate_best_feature"] = substrate_feats[0] if substrate_feats else np.nan
        summary["observer_best_feature"] = observer_feats[0] if observer_feats else np.nan
        summary["substrate_feats"] = ", ".join(substrate_feats)
        summary["observer_feats"] = ", ".join(observer_feats)
        panels.append(panel)
        summaries.append(summary)
        if not substrate_rank.empty:
            ranks.append(substrate_rank)
        if not observer_rank.empty:
            ranks.append(observer_rank)
    scored = pd.concat(panels, ignore_index=True, sort=False) if panels else pd.DataFrame()
    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    rank_df = pd.concat(ranks, ignore_index=True) if ranks else pd.DataFrame()
    return scored, summary_df, rank_df


def aggregate_train_feature_ranks(train_df: pd.DataFrame, cfg: CleanDMDConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_frames: list[pd.DataFrame] = []
    for pid, g in train_df.groupby("patient_id", sort=False):
        g = g.sort_values("time_seconds").reset_index(drop=True).copy()
        feature_cols = [
            col
            for col in g.columns
            if feature_is_rankable(col) and pd.api.types.is_numeric_dtype(g[col]) and g[col].notna().sum() >= 20
        ]
        substrate_cols = [col for col in feature_cols if not col.startswith("dmd_observer_")]
        observer_cols = [col for col in feature_cols if col.startswith("dmd_observer_")]
        if substrate_cols:
            substrate_rank, _, _ = summarize_event_windows(g, substrate_cols, cfg)
            if not substrate_rank.empty:
                substrate_rank["patient_id"] = pid
                substrate_rank["family"] = "dmd_substrate"
                rank_frames.append(substrate_rank)
        if observer_cols:
            observer_rank, _, _ = summarize_event_windows(g, observer_cols, cfg)
            if not observer_rank.empty:
                observer_rank["patient_id"] = pid
                observer_rank["family"] = "dmd_observer"
                rank_frames.append(observer_rank)
    if not rank_frames:
        return pd.DataFrame(), pd.DataFrame()
    rank_df = pd.concat(rank_frames, ignore_index=True)
    agg = (
        rank_df.groupby(["family", "feature"], as_index=False)
        .agg(
            mean_abs_pre_shift=("abs_pre_shift", "mean"),
            median_abs_pre_shift=("abs_pre_shift", "median"),
            mean_peak_abs_pre_z=("peak_abs_pre_z", "mean"),
            records=("patient_id", "nunique"),
        )
        .sort_values(["family", "mean_abs_pre_shift", "mean_peak_abs_pre_z"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    return rank_df, agg


def filter_fold_features_by_baseline(
    train_df: pd.DataFrame,
    features: list[str],
    bg_rows: int,
    min_valid_fraction: float,
    min_nonconstant_records: int,
) -> tuple[list[str], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    kept: list[str] = []
    for feat in features:
        valid_records = 0
        nonconstant_records = 0
        for _, g in train_df.groupby("patient_id", sort=False):
            g = g.sort_values("time_seconds").reset_index(drop=True)
            baseline = pd.to_numeric(g[feat].head(min(bg_rows, len(g))), errors="coerce")
            if len(baseline) == 0:
                continue
            valid_fraction = float(baseline.notna().mean())
            if valid_fraction >= min_valid_fraction:
                valid_records += 1
                valid_vals = baseline.dropna().to_numpy(dtype=float)
                if len(valid_vals) >= 5 and np.nanstd(valid_vals) > EPS:
                    nonconstant_records += 1
        row = {
            "feature": feat,
            "valid_records": valid_records,
            "nonconstant_records": nonconstant_records,
            "kept": int(nonconstant_records >= min_nonconstant_records),
        }
        rows.append(row)
        if row["kept"]:
            kept.append(feat)
    return kept, pd.DataFrame(rows)


def apply_fold_score(df: pd.DataFrame, features: list[str], score_name: str, bg_rows: int) -> pd.Series:
    if not features:
        return pd.Series(np.nan, index=df.index, dtype=float)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df.groupby("patient_id", sort=False):
        g = g.sort_values("time_seconds").reset_index()
        baseline = g.head(min(bg_rows, len(g)))
        zcols: list[np.ndarray] = []
        for feat in features:
            if feat not in g.columns:
                continue
            zvals = robust_z_from_bg(g[feat], baseline[feat])
            zcols.append(np.abs(zvals))
        if zcols:
            score = np.nanmean(np.vstack(zcols), axis=0)
            out.loc[g["index"].to_numpy()] = score
    return out


def fit_predict_logistic(train_x: pd.DataFrame, train_y: np.ndarray, test_x: pd.DataFrame) -> np.ndarray:
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


def run_hazard_nested_causal(hazard_1s_df: pd.DataFrame, cfg: CleanDMDConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    preds: list[pd.DataFrame] = []
    fold_meta: list[dict[str, object]] = []
    baseline_diag_frames: list[pd.DataFrame] = []
    for horizon in cfg.hazard_horizons:
        target_col = f"future_onset_within_{horizon}"
        for held_out in sorted(hazard_1s_df["patient_id"].unique()):
            train_full = hazard_1s_df.loc[hazard_1s_df["patient_id"] != held_out].copy()
            test_full = hazard_1s_df.loc[hazard_1s_df["patient_id"] == held_out].copy()
            train = train_full.loc[train_full["at_risk"] == 1].copy()
            test = test_full.loc[test_full["at_risk"] == 1].copy()
            if train.empty or test.empty or train_full.empty or test_full.empty:
                continue
            train_rank_df, train_rank_summary = aggregate_train_feature_ranks(train_full, cfg)
            substrate_feats = (
                train_rank_summary.loc[train_rank_summary["family"] == "dmd_substrate", "feature"]
                .head(cfg.topn_substrate_feats)
                .tolist()
            )
            observer_feats = (
                train_rank_summary.loc[train_rank_summary["family"] == "dmd_observer", "feature"]
                .head(cfg.topn_observer_feats)
                .tolist()
            )
            substrate_feats, substrate_diag = filter_fold_features_by_baseline(
                train_full,
                substrate_feats,
                cfg.bg_rows,
                cfg.min_baseline_valid_fraction,
                cfg.min_baseline_nonconstant_records,
            )
            observer_feats, observer_diag = filter_fold_features_by_baseline(
                train_full,
                observer_feats,
                cfg.bg_rows,
                cfg.min_baseline_valid_fraction,
                cfg.min_baseline_nonconstant_records,
            )
            if not substrate_diag.empty:
                substrate_diag["horizon"] = horizon
                substrate_diag["held_out_patient"] = held_out
                substrate_diag["family"] = "dmd_substrate"
                baseline_diag_frames.append(substrate_diag)
            if not observer_diag.empty:
                observer_diag["horizon"] = horizon
                observer_diag["held_out_patient"] = held_out
                observer_diag["family"] = "dmd_observer"
                baseline_diag_frames.append(observer_diag)

            train_full["dmd_substrate_score"] = apply_fold_score(train_full, substrate_feats, "dmd_substrate_score", cfg.bg_rows)
            test_full["dmd_substrate_score"] = apply_fold_score(test_full, substrate_feats, "dmd_substrate_score", cfg.bg_rows)
            train_full["dmd_observer_pressure_score"] = apply_fold_score(train_full, observer_feats, "dmd_observer_pressure_score", cfg.bg_rows)
            test_full["dmd_observer_pressure_score"] = apply_fold_score(test_full, observer_feats, "dmd_observer_pressure_score", cfg.bg_rows)
            train_full["dmd_hybrid_score"] = train_full[["dmd_substrate_score", "dmd_observer_pressure_score"]].mean(axis=1)
            test_full["dmd_hybrid_score"] = test_full[["dmd_substrate_score", "dmd_observer_pressure_score"]].mean(axis=1)
            train_scored = train_full.loc[train_full["at_risk"] == 1].copy()
            test_scored = test_full.loc[test_full["at_risk"] == 1].copy()

            fold_meta.append(
                {
                    "horizon": horizon,
                    "held_out_patient": held_out,
                    "substrate_features": ", ".join(substrate_feats),
                    "observer_features": ", ".join(observer_feats),
                    "n_train_rows": len(train_scored),
                    "n_test_rows": len(test_scored),
                    "n_train_rank_rows": len(train_rank_df),
                    "n_substrate_features": len(substrate_feats),
                    "n_observer_features": len(observer_feats),
                }
            )

            feature_sets = {
                "substrate_only": ["dmd_substrate_score"],
                "observer_only": ["dmd_observer_pressure_score"],
                "hybrid_only": ["dmd_hybrid_score"],
            }
            for model_name, feat_cols in feature_sets.items():
                y_train = pd.to_numeric(train_scored[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
                y_test = pd.to_numeric(test_scored[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
                if len(np.unique(y_train)) < 2:
                    continue
                pred = fit_predict_logistic(train_scored[feat_cols], y_train, test_scored[feat_cols])
                temp = test_scored[["patient_id", "time_seconds", target_col, "dmd_substrate_score", "dmd_observer_pressure_score", "dmd_hybrid_score"]].copy()
                temp["horizon"] = horizon
                temp["model_name"] = model_name
                temp["prediction"] = pred
                preds.append(temp)
                rows.append(
                    {
                        "horizon": horizon,
                        "held_out_patient": held_out,
                        "model_name": model_name,
                        "auroc": safe_auc(y_test, pred),
                        "ap": safe_ap(y_test, pred),
                        "n_test_rows": len(test_scored),
                        "n_positive_test": int(y_test.sum()),
                    }
                )
    return (
        pd.DataFrame(rows).sort_values(["horizon", "ap"], ascending=[True, False]).reset_index(drop=True),
        pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(),
        pd.DataFrame(fold_meta),
        pd.concat(baseline_diag_frames, ignore_index=True) if baseline_diag_frames else pd.DataFrame(),
    )


def write_report(run_dir: Path, cfg: CleanDMDConfig, raw_panel: pd.DataFrame, hazard_panel: pd.DataFrame, summary_df: pd.DataFrame, hazard_results: pd.DataFrame, horizon_summary: pd.DataFrame) -> None:
    lines = [
        "# LTST Three-Record Online DMD Clean Rebuild",
        "",
        "## Scope",
        f"- Records: {', '.join(cfg.records)}",
        "- Hazard evaluation is 1-second panel only.",
        "- Feature selection is nested inside each leave-one-record-out training fold.",
        "- Ranking and score construction exclude `future_onset_within_*`, `event*`, and `at_risk` columns.",
        "- The observer branch uses the updated geometry sidecar from `geometry_lie.py` on stabilized modal state.",
        "- DMD rows are shifted so a feature at time `t` uses data through `t`, not `t+1`.",
        "",
        "## Panel sizes",
        f"- Raw DMD panel rows: {len(raw_panel)}",
        f"- 1-second panel rows: {len(hazard_panel)}",
        "",
        "## 1-second horizons",
    ]
    for row in horizon_summary.to_dict("records"):
        lines.append(f"- {row['horizon_seconds']}s: {row['n_positive']} positives / {row['n_rows']} rows ({row['base_rate']:.4f})")
    if not summary_df.empty:
        lines.extend(["", "## Record summary"])
        for row in summary_df.to_dict("records"):
            lines.append(
                f"- {row['patient_id']}: substrate fold features `{row.get('substrate_features', '')}`, observer fold features `{row.get('observer_features', '')}`"
            )
    if not hazard_results.empty:
        lines.extend(["", "## Best hazard rows"])
        best = hazard_results.sort_values(["horizon", "ap"], ascending=[True, False]).groupby("horizon", as_index=False).head(1)
        for row in best.to_dict("records"):
            lines.append(f"- horizon {row['horizon']}: `{row['model_name']}` AUROC `{row['auroc']:.3f}`, AP `{row['ap']:.4f}`")
    (run_dir / f"{run_dir.name}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean online-DMD replication rebuild with updated geometry sidecar.")
    parser.add_argument("--records", default="s20011,s20041,s30742")
    parser.add_argument("--pn-dir", default="ltstdb")
    parser.add_argument("--out-prefix", default="ltst_three_record_online_dmd_clean_rebuild")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CleanDMDConfig(
        records=tuple(piece.strip() for piece in args.records.split(",") if piece.strip()),
        pn_dir=args.pn_dir,
        out_prefix=args.out_prefix,
    )
    project_root = find_project_root()
    run_dir = prepare_run_directory(project_root / cfg.run_root, cfg.out_prefix)

    record_data, _ = load_record_excerpts(cfg)
    state_frames = build_state_frames(record_data)
    dmd_feat_df = extract_dmd_frame(state_frames, cfg)
    master_panel = build_master_panel(dmd_feat_df, cfg)
    hazard_1s_df, _, horizon_summary = resample_to_1s(master_panel, cfg.hazard_horizons)
    hazard_results, hazard_predictions, fold_meta, baseline_diag = run_hazard_nested_causal(hazard_1s_df, cfg)
    summary_df = fold_meta.drop_duplicates(subset=["held_out_patient"]).rename(columns={"held_out_patient": "patient_id"})
    rank_df = fold_meta.copy()

    panel_path = run_dir / f"{run_dir.name}_panel_1s.csv"
    summary_path = run_dir / f"{run_dir.name}_record_summary.csv"
    rank_path = run_dir / f"{run_dir.name}_feature_rank.csv"
    hazard_results_path = run_dir / f"{run_dir.name}_hazard_results.csv"
    hazard_predictions_path = run_dir / f"{run_dir.name}_hazard_predictions.csv"
    horizon_summary_path = run_dir / f"{run_dir.name}_hazard_horizon_summary.csv"
    raw_dmd_path = run_dir / f"{run_dir.name}_dmd_features.csv"
    fold_meta_path = run_dir / f"{run_dir.name}_fold_meta.csv"
    baseline_diag_path = run_dir / f"{run_dir.name}_baseline_feature_diag.csv"

    hazard_1s_df.to_csv(panel_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    rank_df.to_csv(rank_path, index=False)
    hazard_results.to_csv(hazard_results_path, index=False)
    hazard_predictions.to_csv(hazard_predictions_path, index=False)
    horizon_summary.to_csv(horizon_summary_path, index=False)
    dmd_feat_df.to_csv(raw_dmd_path, index=False)
    fold_meta.to_csv(fold_meta_path, index=False)
    baseline_diag.to_csv(baseline_diag_path, index=False)

    write_report(run_dir, cfg, master_panel, hazard_1s_df, summary_df, hazard_results, horizon_summary)

    manifest = {
        "config": asdict(cfg),
        "paths": {
            "panel_1s_csv": str(panel_path),
            "record_summary_csv": str(summary_path),
            "feature_rank_csv": str(rank_path),
            "hazard_results_csv": str(hazard_results_path),
            "hazard_predictions_csv": str(hazard_predictions_path),
            "hazard_horizon_summary_csv": str(horizon_summary_path),
            "dmd_features_csv": str(raw_dmd_path),
            "fold_meta_csv": str(fold_meta_path),
            "baseline_feature_diag_csv": str(baseline_diag_path),
        },
        "n_panel_rows": int(len(hazard_1s_df)),
        "n_records": int(hazard_1s_df["patient_id"].nunique()) if not hazard_1s_df.empty else 0,
    }
    manifest["config"]["run_root"] = str(cfg.run_root)
    (run_dir / f"{run_dir.name}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
