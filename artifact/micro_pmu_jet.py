from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-12
A = np.exp(2j * np.pi / 3.0)
USECOLS = ["Time", "VL1", "VL2", "VL3", "AL1", "AL2", "AL3", "IC1", "IC2", "IC3", "AC1", "AC2", "AC3", "Events"]

LEAD_METRICS: dict[str, str] = {
    "jet_area_zd": "absdev",
    "jet_area_za": "absdev",
    "jet_area_da": "absdev",
    "jet_delta": "higher",
    "jet_volume": "absdev",
    "jet_twist_norm": "higher",
    "jet_twist_delta": "absdev",
    "jet_freq_proxy": "absdev",
    "jet_rocof_proxy": "absdev",
    "phase_current_spread": "higher",
}

BENCHMARK_DIRECTIONS: dict[str, str] = {
    **LEAD_METRICS,
    "jet_g22": "higher",
    "jet_g33": "higher",
    "jet_q12": "absdev",
    "jet_q13": "absdev",
    "jet_q23": "absdev",
    "voltage_angle_spread": "higher",
    "current_angle_spread": "higher",
}

PLOT_METRICS = ["jet_twist_norm", "jet_area_zd", "jet_volume", "phase_current_spread"]
ALIGNMENT_WINDOWS = [(-120, -1), (-60, -1), (-30, -1), (0, 60)]


@dataclass(frozen=True)
class MicroPMUJetConfig:
    data_path: Path = Path("data") / "Micro PMU October 1 Dataset" / "_LBNL_a6_bus1_2015-10-01.csv"
    run_root: Path = Path("artifact") / "runs"
    chunk_size: int = 200000
    min_voltage_norm: float = 1e-9
    pre_event_steps: int = 600
    post_event_steps: int = 240
    baseline_steps: int = 240
    threshold_sigma: float = 1.5
    sustain_steps: int = 6
    rearm_sigma: float = 0.5
    rearm_steps: int = 12
    non_event_stride: int = 200


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


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_run_dir(run_root: Path) -> Path:
    run_dir = run_root / f"micro_pmu_jet_{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def sequence_basis(orientation: str) -> np.ndarray:
    q_zero = np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128) / np.sqrt(3.0)
    if orientation == "ABC":
        q_pos = np.array([1.0 + 0.0j, A**2, A], dtype=np.complex128) / np.sqrt(3.0)
        q_neg = np.array([1.0 + 0.0j, A, A**2], dtype=np.complex128) / np.sqrt(3.0)
    elif orientation == "ACB":
        q_pos = np.array([1.0 + 0.0j, A, A**2], dtype=np.complex128) / np.sqrt(3.0)
        q_neg = np.array([1.0 + 0.0j, A**2, A], dtype=np.complex128) / np.sqrt(3.0)
    else:
        raise ValueError(f"Unknown orientation: {orientation}")
    return np.vstack([q_zero, q_pos, q_neg])


def voltage_phasors_from_chunk(chunk: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    vl = chunk[["VL1", "VL2", "VL3"]].to_numpy(dtype=np.float64)
    al = np.deg2rad(chunk[["AL1", "AL2", "AL3"]].to_numpy(dtype=np.float64))
    ic = chunk[["IC1", "IC2", "IC3"]].to_numpy(dtype=np.float64)
    ac = np.deg2rad(chunk[["AC1", "AC2", "AC3"]].to_numpy(dtype=np.float64))

    v = vl * np.exp(1j * al)
    aux = {
        "mean_voltage": vl.mean(axis=1),
        "mean_current": ic.mean(axis=1),
        "phase_voltage_spread": vl.std(axis=1, ddof=0),
        "phase_current_spread": ic.std(axis=1, ddof=0),
        "voltage_angle_spread": np.rad2deg(al).std(axis=1, ddof=0),
        "current_angle_spread": np.rad2deg(ac).std(axis=1, ddof=0),
    }
    return v.astype(np.complex128), aux


def normalize_voltage_rows(v: np.ndarray, cfg: MicroPMUJetConfig) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(v, axis=1)
    valid = norms > cfg.min_voltage_norm
    u = np.zeros_like(v, dtype=np.complex128)
    u[valid] = v[valid] / norms[valid, None]
    return u, valid


def compute_sequence_components(u: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coeff = u @ basis.conj().T
    zero = np.abs(coeff[:, 0]) ** 2
    pos = np.abs(coeff[:, 1]) ** 2
    neg = np.abs(coeff[:, 2]) ** 2
    return zero, pos, neg


def infer_orientation_and_stats(cfg: MicroPMUJetConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    row_count = 0
    event_rows = 0
    n_onsets = 0
    first_time = None
    last_time = None
    prev_event = False

    pos_abc_parts: list[np.ndarray] = []
    pos_acb_parts: list[np.ndarray] = []
    sample_rows = 0

    basis_abc = sequence_basis("ABC")
    basis_acb = sequence_basis("ACB")

    for chunk in pd.read_csv(cfg.data_path, usecols=USECOLS, chunksize=cfg.chunk_size):
        v, _ = voltage_phasors_from_chunk(chunk)
        u, valid = normalize_voltage_rows(v, cfg)
        events = chunk["Events"].to_numpy(dtype=np.float64) != 0
        times = chunk["Time"].to_numpy(dtype=np.float64)

        row_count += len(chunk)
        event_rows += int(events.sum())
        if len(times):
            if first_time is None:
                first_time = float(times[0])
            last_time = float(times[-1])
            onset_mask = events & ~np.concatenate(([prev_event], events[:-1]))
            n_onsets += int(onset_mask.sum())
            prev_event = bool(events[-1])

        sample_mask = (~events) & valid
        if cfg.non_event_stride > 1:
            sample_mask &= (np.arange(len(chunk)) % cfg.non_event_stride) == 0
        if np.any(sample_mask):
            u_sample = u[sample_mask]
            _, pos_abc, _ = compute_sequence_components(u_sample, basis_abc)
            _, pos_acb, _ = compute_sequence_components(u_sample, basis_acb)
            pos_abc_parts.append(pos_abc.astype(np.float64))
            pos_acb_parts.append(pos_acb.astype(np.float64))
            sample_rows += int(len(u_sample))

    pos_abc = np.concatenate(pos_abc_parts) if pos_abc_parts else np.array([], dtype=np.float64)
    pos_acb = np.concatenate(pos_acb_parts) if pos_acb_parts else np.array([], dtype=np.float64)
    abc_median = float(np.nanmedian(pos_abc)) if len(pos_abc) else np.nan
    acb_median = float(np.nanmedian(pos_acb)) if len(pos_acb) else np.nan
    chosen = "ABC" if (np.isnan(acb_median) or abc_median >= acb_median) else "ACB"

    orientation_df = pd.DataFrame(
        [
            {
                "orientation": "ABC",
                "sample_median_seq_pos_frac": abc_median,
                "sample_mean_seq_pos_frac": float(np.nanmean(pos_abc)) if len(pos_abc) else np.nan,
                "sample_count": sample_rows,
                "chosen": chosen == "ABC",
            },
            {
                "orientation": "ACB",
                "sample_median_seq_pos_frac": acb_median,
                "sample_mean_seq_pos_frac": float(np.nanmean(pos_acb)) if len(pos_acb) else np.nan,
                "sample_count": sample_rows,
                "chosen": chosen == "ACB",
            },
        ]
    )
    stats = {
        "row_count": row_count,
        "event_rows": event_rows,
        "n_onsets": n_onsets,
        "first_time": first_time,
        "last_time": last_time,
        "chosen_orientation": chosen,
        "abc_median_seq_pos_frac": abc_median,
        "acb_median_seq_pos_frac": acb_median,
        "orientation_sample_count": sample_rows,
    }
    return stats, orientation_df


def positive_sequence_phasor(v: np.ndarray, basis: np.ndarray, cfg: MicroPMUJetConfig) -> tuple[np.ndarray, np.ndarray]:
    u, valid = normalize_voltage_rows(v, cfg)
    coeff = u @ basis.conj().T
    z = coeff[:, 1]
    return z.astype(np.complex128), valid


def compute_jet_chunk_features(
    z: np.ndarray,
    valid: np.ndarray,
    prev_z: complex | None,
    prev_dz: complex | None,
) -> tuple[dict[str, np.ndarray], complex | None, complex | None]:
    prev_z_arr = np.empty(len(z), dtype=np.complex128)
    prev_z_arr[:] = np.nan + 1j * np.nan
    if len(z) > 0:
        prev_z_arr[1:] = z[:-1]
        if prev_z is not None:
            prev_z_arr[0] = prev_z

    dz = z - prev_z_arr

    prev_dz_arr = np.empty(len(z), dtype=np.complex128)
    prev_dz_arr[:] = np.nan + 1j * np.nan
    if len(z) > 0:
        prev_dz_arr[1:] = dz[:-1]
        if prev_dz is not None:
            prev_dz_arr[0] = prev_dz

    ddz = dz - prev_dz_arr

    valid_jet = valid & np.isfinite(dz.real) & np.isfinite(dz.imag) & np.isfinite(ddz.real) & np.isfinite(ddz.imag)
    valid_jet &= (np.abs(z) > EPS) & (np.abs(dz) > EPS) & (np.abs(ddz) > EPS)

    g11 = np.full(len(z), np.nan, dtype=np.float64)
    g22 = np.full(len(z), np.nan, dtype=np.float64)
    g33 = np.full(len(z), np.nan, dtype=np.float64)
    g12 = np.full(len(z), np.nan, dtype=np.float64)
    g13 = np.full(len(z), np.nan, dtype=np.float64)
    g23 = np.full(len(z), np.nan, dtype=np.float64)
    q12 = np.full(len(z), np.nan, dtype=np.float64)
    q13 = np.full(len(z), np.nan, dtype=np.float64)
    q23 = np.full(len(z), np.nan, dtype=np.float64)
    area_zd = np.full(len(z), np.nan, dtype=np.float64)
    area_za = np.full(len(z), np.nan, dtype=np.float64)
    area_da = np.full(len(z), np.nan, dtype=np.float64)
    delta = np.full(len(z), np.nan, dtype=np.float64)
    volume = np.full(len(z), np.nan, dtype=np.float64)
    twist_norm = np.full(len(z), np.nan, dtype=np.float64)
    twist_delta = np.full(len(z), np.nan, dtype=np.float64)
    freq_proxy = np.full(len(z), np.nan, dtype=np.float64)
    rocof_proxy = np.full(len(z), np.nan, dtype=np.float64)

    if np.any(valid_jet):
        z_v = z[valid_jet]
        dz_v = dz[valid_jet]
        ddz_v = ddz[valid_jet]

        g11_v = np.abs(z_v) ** 2
        g22_v = np.abs(dz_v) ** 2
        g33_v = np.abs(ddz_v) ** 2
        c12_v = z_v * np.conj(dz_v)
        c13_v = z_v * np.conj(ddz_v)
        c23_v = dz_v * np.conj(ddz_v)
        g12_v = np.real(c12_v)
        g13_v = np.real(c13_v)
        g23_v = np.real(c23_v)
        q12_v = np.imag(c12_v)
        q13_v = np.imag(c13_v)
        q23_v = np.imag(c23_v)

        l12 = np.clip((g12_v**2) / (g11_v * g22_v + EPS), 0.0, 1.0)
        l13 = np.clip((g13_v**2) / (g11_v * g33_v + EPS), 0.0, 1.0)
        l23 = np.clip((g23_v**2) / (g22_v * g33_v + EPS), 0.0, 1.0)
        area12 = np.clip(1.0 - l12, 0.0, 1.0)
        area13 = np.clip(1.0 - l13, 0.0, 1.0)
        area23 = np.clip(1.0 - l23, 0.0, 1.0)
        det_raw = g11_v * (g22_v * g33_v - g23_v**2) - g12_v * (g12_v * g33_v - g23_v * g13_v) + g13_v * (g12_v * g23_v - g22_v * g13_v)
        det_raw = np.maximum(det_raw, 0.0)
        vol_v = np.clip(det_raw / (g11_v * g22_v * g33_v + EPS), 0.0, 1.0)

        g11[valid_jet] = g11_v
        g22[valid_jet] = g22_v
        g33[valid_jet] = g33_v
        g12[valid_jet] = g12_v
        g13[valid_jet] = g13_v
        g23[valid_jet] = g23_v
        q12[valid_jet] = q12_v
        q13[valid_jet] = q13_v
        q23[valid_jet] = q23_v
        area_zd[valid_jet] = area12
        area_za[valid_jet] = area13
        area_da[valid_jet] = area23
        delta[valid_jet] = np.abs(l12 - l13)
        volume[valid_jet] = vol_v
        twist_norm[valid_jet] = np.sqrt(q12_v**2 + q13_v**2 + q23_v**2)
        twist_delta[valid_jet] = np.abs(q12_v - q13_v)
        freq_proxy[valid_jet] = q12_v / (g11_v + EPS)
        rocof_proxy[valid_jet] = q23_v / (g22_v + EPS)

    next_prev_z = z[-1] if len(z) else prev_z
    next_prev_dz = dz[-1] if len(dz) and np.isfinite(dz[-1].real) and np.isfinite(dz[-1].imag) else prev_dz
    features = {
        "jet_g11": g11,
        "jet_g22": g22,
        "jet_g33": g33,
        "jet_g12": g12,
        "jet_g13": g13,
        "jet_g23": g23,
        "jet_q12": q12,
        "jet_q13": q13,
        "jet_q23": q23,
        "jet_area_zd": area_zd,
        "jet_area_za": area_za,
        "jet_area_da": area_da,
        "jet_delta": delta,
        "jet_volume": volume,
        "jet_twist_norm": twist_norm,
        "jet_twist_delta": twist_delta,
        "jet_freq_proxy": freq_proxy,
        "jet_rocof_proxy": rocof_proxy,
    }
    return features, next_prev_z, next_prev_dz


def finalize_event(col: dict[str, Any], event_windows: list[pd.DataFrame], cfg: MicroPMUJetConfig) -> None:
    if col["remaining"] > 0:
        return
    event_df = pd.concat(col["parts"], ignore_index=True)
    if len(event_df) < cfg.pre_event_steps + 1 + cfg.post_event_steps:
        return
    event_df = event_df.iloc[: cfg.pre_event_steps + 1 + cfg.post_event_steps].copy().reset_index(drop=True)
    event_df["event_index"] = int(col["event_index"])
    event_df["event_onset_step"] = int(col["onset_step"])
    event_df["event_onset_time_ns"] = float(col["onset_time_ns"])
    event_df["steps_from_onset"] = np.arange(-cfg.pre_event_steps, cfg.post_event_steps + 1, dtype=int)
    event_windows.append(event_df)


def second_pass_analyze(
    cfg: MicroPMUJetConfig,
    basis: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prev_event = False
    event_index = 0
    active_collectors: list[dict[str, Any]] = []
    recent_tail = pd.DataFrame()
    event_windows: list[pd.DataFrame] = []
    event_sample_parts: list[pd.DataFrame] = []
    non_event_sample_parts: list[pd.DataFrame] = []
    global_row = 0
    prev_z: complex | None = None
    prev_dz: complex | None = None

    for chunk_id, chunk in enumerate(pd.read_csv(cfg.data_path, usecols=USECOLS, chunksize=cfg.chunk_size)):
        v, aux = voltage_phasors_from_chunk(chunk)
        z, valid = positive_sequence_phasor(v, basis, cfg)
        feat, prev_z, prev_dz = compute_jet_chunk_features(z, valid, prev_z, prev_dz)

        times = chunk["Time"].to_numpy(dtype=np.float64)
        events = chunk["Events"].to_numpy(dtype=np.float64)
        event_flag = events != 0
        index = np.arange(global_row, global_row + len(chunk), dtype=int)
        global_row += len(chunk)

        chunk_df = pd.DataFrame(
            {
                "row_index": index,
                "time_ns": times,
                "event_code": events,
                "event_flag": event_flag,
                "mean_voltage": aux["mean_voltage"],
                "mean_current": aux["mean_current"],
                "phase_voltage_spread": aux["phase_voltage_spread"],
                "phase_current_spread": aux["phase_current_spread"],
                "voltage_angle_spread": aux["voltage_angle_spread"],
                "current_angle_spread": aux["current_angle_spread"],
                **feat,
            }
        )

        event_sample_parts.append(chunk_df.loc[event_flag].copy())
        non_event_sample_parts.append(chunk_df.loc[~event_flag].iloc[:: cfg.non_event_stride].copy())

        if active_collectors:
            carry_active: list[dict[str, Any]] = []
            for col in active_collectors:
                take = min(col["remaining"], len(chunk_df))
                if take > 0:
                    col["parts"].append(chunk_df.iloc[:take].copy())
                    col["remaining"] -= take
                if col["remaining"] <= 0:
                    finalize_event(col, event_windows, cfg)
                else:
                    carry_active.append(col)
            active_collectors = carry_active

        onset_mask = event_flag & ~np.concatenate(([prev_event], event_flag[:-1]))
        onset_positions = np.flatnonzero(onset_mask)
        tail_plus_chunk = pd.concat([recent_tail, chunk_df], ignore_index=True)
        tail_len = len(recent_tail)

        for pos in onset_positions:
            onset_in_combined = tail_len + int(pos)
            pre_df = tail_plus_chunk.iloc[max(0, onset_in_combined - cfg.pre_event_steps) : onset_in_combined].copy()
            if len(pre_df) < cfg.pre_event_steps:
                continue
            current_df = chunk_df.iloc[pos : pos + 1].copy()
            collector = {
                "event_index": event_index,
                "onset_step": int(chunk_df.iloc[pos]["row_index"]),
                "onset_time_ns": float(chunk_df.iloc[pos]["time_ns"]),
                "parts": [pre_df, current_df],
                "remaining": cfg.post_event_steps,
            }
            event_index += 1
            available = min(collector["remaining"], len(chunk_df) - pos - 1)
            if available > 0:
                collector["parts"].append(chunk_df.iloc[pos + 1 : pos + 1 + available].copy())
                collector["remaining"] -= available
            if collector["remaining"] <= 0:
                finalize_event(collector, event_windows, cfg)
            else:
                active_collectors.append(collector)

        recent_tail = tail_plus_chunk.tail(cfg.pre_event_steps).copy().reset_index(drop=True)
        prev_event = bool(event_flag[-1]) if len(event_flag) else prev_event

        if (chunk_id + 1) % 10 == 0:
            print(
                f"[MICRO-PMU-JET] chunk {chunk_id + 1} | rows {global_row} | events {event_index} | active {len(active_collectors)}",
                flush=True,
            )

    sampled_df = pd.concat(event_sample_parts + non_event_sample_parts, ignore_index=True)
    alignment_df = pd.concat(event_windows, ignore_index=True) if event_windows else pd.DataFrame()
    return sampled_df, alignment_df


def first_rearmed_sustained_crossing(
    z: np.ndarray,
    threshold_sigma: float,
    sustain_steps: int,
    rearm_sigma: float,
    rearm_steps: int,
) -> int | None:
    low_run = 0
    high_run = 0
    armed = False
    high_start = -1
    for idx, value in enumerate(z):
        if not np.isfinite(value):
            low_run = 0
            high_run = 0
            continue
        if value < rearm_sigma:
            low_run += 1
            if low_run >= rearm_steps:
                armed = True
        else:
            low_run = 0

        if armed and value > threshold_sigma:
            if high_run == 0:
                high_start = idx
            high_run += 1
            if high_run >= sustain_steps:
                return high_start
        else:
            high_run = 0
    return None


def z_score_search(search_vals: np.ndarray, mu: float, sigma: float, direction: str) -> np.ndarray:
    if direction == "higher":
        return (search_vals - mu) / sigma
    if direction == "lower":
        return (mu - search_vals) / sigma
    return np.abs(search_vals - mu) / sigma


def build_event_table(alignment_df: pd.DataFrame, cfg: MicroPMUJetConfig) -> pd.DataFrame:
    rows = []
    if alignment_df.empty:
        return pd.DataFrame()

    for event_id, g in alignment_df.groupby("event_index", sort=True):
        g = g.sort_values("steps_from_onset").reset_index(drop=True)
        baseline = g[
            (g["steps_from_onset"] >= -cfg.pre_event_steps)
            & (g["steps_from_onset"] < -cfg.pre_event_steps + cfg.baseline_steps)
        ]
        search = g[
            (g["steps_from_onset"] >= -cfg.pre_event_steps + cfg.baseline_steps)
            & (g["steps_from_onset"] < 0)
        ]
        if len(baseline) < 30 or len(search) < cfg.sustain_steps:
            continue

        result = {
            "event_index": int(event_id),
            "event_onset_step": int(g["event_onset_step"].iloc[0]),
            "event_onset_time_ns": float(g["event_onset_time_ns"].iloc[0]),
        }
        best_feature = np.nan
        best_lead = np.nan
        for metric, direction in LEAD_METRICS.items():
            base_vals = baseline[metric].to_numpy(dtype=float)
            mu = float(np.nanmedian(base_vals))
            sigma = float(np.nanstd(base_vals, ddof=0))
            sigma = sigma if sigma > EPS else 1.0
            search_vals = search[metric].to_numpy(dtype=float)
            z = z_score_search(search_vals, mu, sigma, direction)
            cross = first_rearmed_sustained_crossing(
                z=z,
                threshold_sigma=cfg.threshold_sigma,
                sustain_steps=cfg.sustain_steps,
                rearm_sigma=cfg.rearm_sigma,
                rearm_steps=cfg.rearm_steps,
            )
            lead = float(-search["steps_from_onset"].iloc[cross]) if cross is not None else np.nan
            result[f"{metric}_lead_steps"] = lead
            if np.isfinite(lead) and (not np.isfinite(best_lead) or lead > best_lead):
                best_lead = lead
                best_feature = metric

        result["lead_feature"] = best_feature
        result["lead_steps"] = best_lead
        rows.append(result)

    event_df = pd.DataFrame(rows).sort_values("event_index").reset_index(drop=True)
    if not event_df.empty:
        event_df["event_onset_time_utc"] = pd.to_datetime(event_df["event_onset_time_ns"], unit="ns", utc=True)
    return event_df


def auc_one_vs_rest(scores: np.ndarray, labels: np.ndarray, positive_label: int = 1) -> float:
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


def best_oriented_auc(scores: np.ndarray, labels: np.ndarray, positive_label: int = 1) -> tuple[float, str]:
    auc_high = auc_one_vs_rest(scores, labels, positive_label)
    auc_low = auc_one_vs_rest(-scores, labels, positive_label)
    if not np.isfinite(auc_high) and not np.isfinite(auc_low):
        return np.nan, "higher"
    if not np.isfinite(auc_low) or (np.isfinite(auc_high) and auc_high >= auc_low):
        return auc_high, "higher"
    return auc_low, "lower"


def build_benchmark_table(sampled_df: pd.DataFrame) -> pd.DataFrame:
    labels = sampled_df["event_flag"].to_numpy(dtype=int)
    metrics = [
        "jet_area_zd",
        "jet_area_za",
        "jet_area_da",
        "jet_delta",
        "jet_volume",
        "jet_twist_norm",
        "jet_twist_delta",
        "jet_freq_proxy",
        "jet_rocof_proxy",
        "jet_g22",
        "jet_g33",
        "jet_q12",
        "jet_q13",
        "jet_q23",
        "phase_current_spread",
        "voltage_angle_spread",
        "current_angle_spread",
    ]
    rows = []
    for metric in metrics:
        direction = BENCHMARK_DIRECTIONS.get(metric, "higher")
        scores = sampled_df[metric].to_numpy(dtype=float)
        if direction == "lower":
            scores = -scores
        elif direction == "absdev":
            center = float(np.nanmedian(scores))
            scores = np.abs(scores - center)
        auroc = auc_one_vs_rest(scores, labels, 1)
        if metric.startswith("jet_"):
            family = "jet"
        else:
            family = "raw"
        rows.append({"metric": metric, "direction": direction, "auroc": auroc, "family": family})
    return pd.DataFrame(rows).sort_values(["auroc", "metric"], ascending=[False, True]).reset_index(drop=True)


def build_alignment_table(alignment_df: pd.DataFrame, cfg: MicroPMUJetConfig) -> pd.DataFrame:
    if alignment_df.empty:
        return pd.DataFrame()
    parts = []
    for _, g in alignment_df.groupby("event_index", sort=True):
        g = g.sort_values("steps_from_onset").copy()
        baseline = g[
            (g["steps_from_onset"] >= -cfg.pre_event_steps)
            & (g["steps_from_onset"] < -cfg.pre_event_steps + cfg.baseline_steps)
        ]
        if len(baseline) < 30:
            continue
        for metric in PLOT_METRICS:
            direction = LEAD_METRICS[metric]
            base = baseline[metric].to_numpy(dtype=float)
            mu = float(np.nanmedian(base))
            sigma = float(np.nanstd(base, ddof=0))
            sigma = sigma if sigma > EPS else 1.0
            g[f"{metric}_z"] = z_score_search(g[metric].to_numpy(dtype=float), mu, sigma, direction)
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summarize_alignment_windows(aligned_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if aligned_df.empty:
        return pd.DataFrame(), "Alignment windows unavailable."

    grouped = aligned_df.groupby("steps_from_onset")[[f"{metric}_z" for metric in PLOT_METRICS]].median().reset_index()
    rows = []
    for start, end in ALIGNMENT_WINDOWS:
        window = grouped[(grouped["steps_from_onset"] >= start) & (grouped["steps_from_onset"] <= end)]
        if window.empty:
            continue
        row = {"window": f"[{start},{end}]"}
        for metric in PLOT_METRICS:
            row[f"{metric}_median_z"] = float(window[f"{metric}_z"].median())
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    if len(summary_df) >= 3:
        early = summary_df.loc[summary_df["window"] == "[-120,-1]"]
        late = summary_df.loc[summary_df["window"] == "[-30,-1]"]
        if not early.empty and not late.empty:
            deltas = {metric: float(late[f"{metric}_median_z"].iloc[0] - early[f"{metric}_median_z"].iloc[0]) for metric in PLOT_METRICS[:-1]}
            best_metric = max(deltas, key=lambda k: abs(deltas[k]))
            best_delta = deltas[best_metric]
            if abs(best_delta) > 0.10:
                note = f"Jet-space alignment shows some approach-to-onset; largest median-window shift is `{best_metric}` = `{best_delta:.3f}` from `[-120,-1]` to `[-30,-1]`."
            else:
                note = f"Jet-space alignment still looks plateau-like; largest median-window shift is only `{best_metric}` = `{best_delta:.3f}` from `[-120,-1]` to `[-30,-1]`."
            return summary_df, note
    return summary_df, "Alignment windows computed, but the jet-space approach-to-onset heuristic was inconclusive."


def plot_alignment(aligned_df: pd.DataFrame, output_path: Path) -> None:
    if aligned_df.empty:
        return
    grouped = aligned_df.groupby("steps_from_onset")[[f"{metric}_z" for metric in PLOT_METRICS]].median()
    time_s = grouped.index.to_numpy(dtype=float) / 120.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_s, grouped["jet_twist_norm_z"], color="#6a3d9a", label="jet_twist_norm (median z)")
    ax.plot(time_s, grouped["jet_area_zd_z"], color="#1b9e77", label="jet_area_zd (median z)")
    ax.plot(time_s, grouped["jet_volume_z"], color="#d95f02", label="jet_volume (median z)")
    ax.plot(time_s, grouped["phase_current_spread_z"], color="#4c78a8", label="phase_current_spread (median z)")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Seconds from event onset")
    ax.set_ylabel("Median baseline-normalized deviation")
    ax.set_title("Micro PMU phasor jet alignment around real event onset")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def latest_baseline(run_root: Path, prefix: str, benchmark_name: str, event_name: str) -> dict[str, Any] | None:
    dirs = sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.startswith(prefix)], key=lambda p: p.stat().st_mtime)
    if not dirs:
        return None
    run_dir = dirs[-1]
    bench_path = run_dir / benchmark_name
    event_path = run_dir / event_name
    if not bench_path.exists() or not event_path.exists():
        return None

    bench_df = pd.read_csv(bench_path)
    event_df = pd.read_csv(event_path)
    best_auroc = float(bench_df["auroc"].max()) if not bench_df.empty else np.nan
    event_count = int(event_df["lead_steps"].notna().sum()) if "lead_steps" in event_df.columns else np.nan
    return {"run_dir": str(run_dir), "best_auroc": best_auroc, "lead_events": event_count}


def synthetic_sanity_checks() -> pd.DataFrame:
    basis = sequence_basis("ABC")
    positive = np.array([[1.0 + 0.0j, A**2, A]], dtype=np.complex128)
    reversed_order = np.array([[1.0 + 0.0j, A, A**2]], dtype=np.complex128)
    common_mode = np.array([[1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j]], dtype=np.complex128)

    rows = []
    for scenario, v in [
        ("balanced_positive_sequence", positive),
        ("reversed_order_under_ABC", reversed_order),
        ("common_mode", common_mode),
    ]:
        u, valid = normalize_voltage_rows(v, MicroPMUJetConfig())
        coeff = u @ basis.conj().T
        z = coeff[:, 1]
        feat, _, _ = compute_jet_chunk_features(z, valid, prev_z=None, prev_dz=None)
        rows.append(
            {
                "scenario": scenario,
                "jet_g11": float(feat["jet_g11"][0]) if len(v) else np.nan,
                "jet_area_zd": float(feat["jet_area_zd"][0]) if len(v) else np.nan,
                "jet_twist_norm": float(feat["jet_twist_norm"][0]) if len(v) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    run_dir: Path,
    cfg: MicroPMUJetConfig,
    stats: dict[str, Any],
    orientation_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    event_df: pd.DataFrame,
    alignment_summary_df: pd.DataFrame,
    alignment_note: str,
    cross_phase_baseline: dict[str, Any] | None,
    sequence_baseline: dict[str, Any] | None,
    sanity_df: pd.DataFrame,
) -> None:
    lead_counts = event_df["lead_feature"].value_counts().to_dict() if not event_df.empty else {}
    best_jet = benchmark_df.loc[benchmark_df["family"] == "jet"].head(1)

    comparison_lines = []
    if cross_phase_baseline is not None:
        comparison_lines.append(
            f"- Best jet AUROC vs cross-phase baseline: `{float(best_jet['auroc'].iloc[0]) if not best_jet.empty else np.nan:.6f}` vs `{cross_phase_baseline['best_auroc']:.6f}`"
        )
        comparison_lines.append(
            f"- Lead-detected events vs cross-phase baseline: `{int(event_df['lead_steps'].notna().sum()) if not event_df.empty else 0}` vs `{cross_phase_baseline['lead_events']}`"
        )
    if sequence_baseline is not None:
        comparison_lines.append(
            f"- Best jet AUROC vs sequence baseline: `{float(best_jet['auroc'].iloc[0]) if not best_jet.empty else np.nan:.6f}` vs `{sequence_baseline['best_auroc']:.6f}`"
        )
        comparison_lines.append(
            f"- Lead-detected events vs sequence baseline: `{int(event_df['lead_steps'].notna().sum()) if not event_df.empty else 0}` vs `{sequence_baseline['lead_events']}`"
        )
    if not comparison_lines:
        comparison_lines.append("- Prior baselines unavailable.")

    lines = [
        "# Micro PMU October 1 Phasor Jet Experiment",
        "",
        "Ray-space / jet-space analysis on the full October 1 micro-PMU day file using the positive-sequence phasor and its first two local differences.",
        "",
        "## Setup",
        f"- Data path: `{cfg.data_path}`",
        f"- Rows scanned: `{stats['row_count']}`",
        f"- Event rows (`Events != 0`): `{stats['event_rows']}`",
        f"- Event onsets: `{stats['n_onsets']}`",
        f"- Chosen healthy orientation: `{stats['chosen_orientation']}`",
        f"- Orientation sample count: `{stats['orientation_sample_count']}`",
        "- Local jet state: `z = V+`, `dz = z[t]-z[t-1]`, `ddz = dz[t]-dz[t-1]`",
        "- Raw jet channels: `g11 g22 g33 g12 g13 g23 q12 q13 q23`",
        f"- Pre-event window: `{cfg.pre_event_steps}` samples",
        f"- Post-event window: `{cfg.post_event_steps}` samples",
        f"- Baseline window: `{cfg.baseline_steps}` samples",
        f"- Lead rule: `z > {cfg.threshold_sigma}` for `{cfg.sustain_steps}` samples after rearming below `{cfg.rearm_sigma}` for `{cfg.rearm_steps}` samples",
        "",
        "## Orientation Inference",
        orientation_df.to_markdown(index=False),
        "",
        "## Synthetic Sanity Checks",
        sanity_df.to_markdown(index=False),
        "",
        "## Top Metrics",
        benchmark_df.head(15).to_markdown(index=False),
        "",
        "## Comparison To Prior PMU Baselines",
        *comparison_lines,
        "",
        "## Event Lead Summary",
        f"- Events with a detected lead: `{int(event_df['lead_steps'].notna().sum()) if not event_df.empty else 0}` / `{len(event_df)}`",
        "\n".join(f"- `{k}`: `{v}` events" for k, v in lead_counts.items()) if lead_counts else "- No lead metrics crossed under the current rule set.",
        "",
        "## Alignment Windows",
        alignment_summary_df.to_markdown(index=False) if not alignment_summary_df.empty else "Alignment summary unavailable.",
        "",
        f"- Interpretation: {alignment_note}",
        "",
        event_df.head(20).to_markdown(index=False) if not event_df.empty else "No event windows completed.",
        "",
    ]
    (run_dir / "micro_pmu_jet_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phasor jet-space analysis on the Micro PMU October 1 dataset.")
    parser.add_argument("--data-path", default=str(Path("data") / "Micro PMU October 1 Dataset" / "_LBNL_a6_bus1_2015-10-01.csv"))
    parser.add_argument("--run-root", default=str(Path("artifact") / "runs"))
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--threshold-sigma", type=float, default=1.5)
    parser.add_argument("--sustain-steps", type=int, default=6)
    parser.add_argument("--rearm-sigma", type=float, default=0.5)
    parser.add_argument("--rearm-steps", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MicroPMUJetConfig(
        data_path=Path(args.data_path).resolve(),
        run_root=Path(args.run_root).resolve(),
        chunk_size=args.chunk_size,
        threshold_sigma=args.threshold_sigma,
        sustain_steps=args.sustain_steps,
        rearm_sigma=args.rearm_sigma,
        rearm_steps=args.rearm_steps,
    )
    run_dir = ensure_run_dir(cfg.run_root)

    stats, orientation_df = infer_orientation_and_stats(cfg)
    basis = sequence_basis(stats["chosen_orientation"])
    sampled_df, event_windows_df = second_pass_analyze(cfg, basis)
    event_df = build_event_table(event_windows_df, cfg)
    benchmark_df = build_benchmark_table(sampled_df)
    aligned_df = build_alignment_table(event_windows_df, cfg)
    alignment_summary_df, alignment_note = summarize_alignment_windows(aligned_df)
    cross_phase_baseline = latest_baseline(cfg.run_root, "micro_pmu_cross_phase_", "micro_pmu_cross_phase_benchmarks.csv", "micro_pmu_cross_phase_event_table.csv")
    sequence_baseline = latest_baseline(cfg.run_root, "micro_pmu_sequence_", "micro_pmu_sequence_benchmarks.csv", "micro_pmu_sequence_event_table.csv")
    sanity_df = synthetic_sanity_checks()

    sampled_df.to_csv(run_dir / "micro_pmu_jet_sampled_features.csv", index=False)
    event_df.to_csv(run_dir / "micro_pmu_jet_event_table.csv", index=False)
    aligned_df.to_csv(run_dir / "micro_pmu_jet_alignment.csv", index=False)
    benchmark_df.to_csv(run_dir / "micro_pmu_jet_benchmarks.csv", index=False)
    orientation_df.to_csv(run_dir / "micro_pmu_jet_orientation_summary.csv", index=False)

    plot_alignment(aligned_df, run_dir / "micro_pmu_jet_alignment_plot.png")

    metadata = {
        "config": json_safe(asdict(cfg)),
        "output_dir": str(run_dir),
        "row_count": int(stats["row_count"]),
        "event_rows": int(stats["event_rows"]),
        "n_onsets": int(stats["n_onsets"]),
        "first_time_ns": stats["first_time"],
        "last_time_ns": stats["last_time"],
        "chosen_orientation": stats["chosen_orientation"],
        "abc_median_seq_pos_frac": stats["abc_median_seq_pos_frac"],
        "acb_median_seq_pos_frac": stats["acb_median_seq_pos_frac"],
        "orientation_sample_count": stats["orientation_sample_count"],
        "cross_phase_baseline": json_safe(cross_phase_baseline),
        "sequence_baseline": json_safe(sequence_baseline),
        "synthetic_sanity": sanity_df.to_dict(orient="records"),
    }
    with (run_dir / "micro_pmu_jet_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(json_safe(metadata), fp, indent=2)

    write_report(
        run_dir=run_dir,
        cfg=cfg,
        stats=stats,
        orientation_df=orientation_df,
        benchmark_df=benchmark_df,
        event_df=event_df,
        alignment_summary_df=alignment_summary_df,
        alignment_note=alignment_note,
        cross_phase_baseline=cross_phase_baseline,
        sequence_baseline=sequence_baseline,
        sanity_df=sanity_df,
    )

    print(f"[MICRO-PMU-JET] Output directory: {run_dir}", flush=True)
    print(benchmark_df.head(12).to_string(index=False), flush=True)
    if not event_df.empty:
        print(event_df.head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
