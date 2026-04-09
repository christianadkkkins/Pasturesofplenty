from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.signal import lfilter
except Exception:
    lfilter = None


EPS = 1e-12
USECOLS = ["Time", "VL1", "VL2", "VL3", "AL1", "AL2", "AL3", "IC1", "IC2", "IC3", "AC1", "AC2", "AC3", "Events"]
STATE_BASE_COLS = ["voltage_re_l1", "voltage_im_l1", "voltage_re_l2", "voltage_im_l2", "voltage_re_l3", "voltage_im_l3",
                   "current_re_l1", "current_im_l1", "current_re_l2", "current_im_l2", "current_re_l3", "current_im_l3"]
LEAD_METRICS = {
    "proj_lock_barrier_xl": "higher",
    "proj_transverse_xl": "lower",
    "proj_volume_xsl": "lower",
    "proj_area_sl": "lower",
    "phase_current_spread": "higher",
}
PROJECTIVE_LEAD_METRICS = [
    "proj_lock_barrier_xl",
    "proj_transverse_xl",
    "proj_volume_xsl",
    "proj_area_sl",
]
RAW_LEAD_METRIC = "phase_current_spread"


@dataclass(frozen=True)
class MicroPMUConfig:
    data_path: Path = Path("data") / "Micro PMU October 1 Dataset" / "_LBNL_a6_bus1_2015-10-01.csv"
    run_root: Path = Path("artifact") / "runs"
    beta_short: float = 0.10
    beta_long: float = 0.01
    chunk_size: int = 200000
    min_state_energy: float = 1e-8
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
    run_dir = run_root / f"micro_pmu_oct1_{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def batch_update_welford(count: int, mean: np.ndarray, m2: np.ndarray, x: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    if len(x) == 0:
        return count, mean, m2
    batch_n = x.shape[0]
    batch_mean = x.mean(axis=0)
    batch_m2 = ((x - batch_mean) ** 2).sum(axis=0)
    if count == 0:
        return batch_n, batch_mean, batch_m2
    delta = batch_mean - mean
    total = count + batch_n
    new_mean = mean + delta * (batch_n / total)
    new_m2 = m2 + batch_m2 + (delta ** 2) * count * batch_n / total
    return total, new_mean, new_m2


def phasor_state_from_chunk(chunk: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    vl = chunk[["VL1", "VL2", "VL3"]].to_numpy(dtype=np.float64)
    al = np.deg2rad(chunk[["AL1", "AL2", "AL3"]].to_numpy(dtype=np.float64))
    ic = chunk[["IC1", "IC2", "IC3"]].to_numpy(dtype=np.float64)
    ac = np.deg2rad(chunk[["AC1", "AC2", "AC3"]].to_numpy(dtype=np.float64))

    v_re = vl * np.cos(al)
    v_im = vl * np.sin(al)
    i_re = ic * np.cos(ac)
    i_im = ic * np.sin(ac)

    state = np.column_stack(
        [
            v_re[:, 0], v_im[:, 0],
            v_re[:, 1], v_im[:, 1],
            v_re[:, 2], v_im[:, 2],
            i_re[:, 0], i_im[:, 0],
            i_re[:, 1], i_im[:, 1],
            i_re[:, 2], i_im[:, 2],
        ]
    )
    aux = {
        "mean_voltage": vl.mean(axis=1),
        "mean_current": ic.mean(axis=1),
        "phase_voltage_spread": vl.std(axis=1, ddof=0),
        "phase_current_spread": ic.std(axis=1, ddof=0),
        "voltage_angle_spread": np.rad2deg(al).std(axis=1, ddof=0),
        "current_angle_spread": np.rad2deg(ac).std(axis=1, ddof=0),
    }
    return state, aux


def ema_prior_chunk(x: np.ndarray, beta: float, prev_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        return np.empty_like(x), prev_state
    alpha = 1.0 - beta
    if lfilter is not None:
        zi = (alpha * prev_state)[None, :]
        filtered, zf = lfilter([beta], [1.0, -alpha], x, axis=0, zi=zi)
        prior = np.vstack([prev_state[None, :], filtered[:-1]])
        next_state = filtered[-1]
        return prior, next_state

    state = prev_state.copy()
    prior = np.zeros_like(x)
    for idx in range(len(x)):
        prior[idx] = state
        state = beta * x[idx] + alpha * state
    return prior, state


def compute_projective_chunk_features(x: np.ndarray, ms_prior: np.ndarray, ml_prior: np.ndarray, cfg: MicroPMUConfig) -> dict[str, np.ndarray]:
    n_x = np.einsum("td,td->t", x, x)
    n_s = np.einsum("td,td->t", ms_prior, ms_prior)
    n_l = np.einsum("td,td->t", ml_prior, ml_prior)
    d_sl = np.einsum("td,td->t", ms_prior, ml_prior)
    d_xs = np.einsum("td,td->t", x, ms_prior)
    d_xl = np.einsum("td,td->t", x, ml_prior)

    proj_line_lock_sl = np.full(len(x), np.nan, dtype=np.float64)
    proj_area_sl = np.full(len(x), np.nan, dtype=np.float64)
    proj_line_lock_xl = np.full(len(x), np.nan, dtype=np.float64)
    proj_transverse_xl = np.full(len(x), np.nan, dtype=np.float64)
    proj_lock_barrier_sl = np.full(len(x), np.nan, dtype=np.float64)
    proj_lock_barrier_xl = np.full(len(x), np.nan, dtype=np.float64)
    proj_volume_xsl = np.full(len(x), np.nan, dtype=np.float64)
    log_proj_volume_xsl = np.full(len(x), np.nan, dtype=np.float64)
    short_long_explanation_imbalance = np.full(len(x), np.nan, dtype=np.float64)
    memory_align = np.full(len(x), np.nan, dtype=np.float64)
    linger = np.full(len(x), np.nan, dtype=np.float64)
    novelty = np.full(len(x), np.nan, dtype=np.float64)

    energy_mask = n_x >= cfg.min_state_energy
    sl_mask = energy_mask & (n_s > EPS) & (n_l > EPS)
    xl_mask = energy_mask & (n_l > EPS)

    if np.any(sl_mask):
        proj_line_lock_sl[sl_mask] = np.clip((d_sl[sl_mask] ** 2) / (n_s[sl_mask] * n_l[sl_mask] + EPS), 0.0, 1.0)
        proj_area_sl[sl_mask] = np.clip(1.0 - proj_line_lock_sl[sl_mask], 0.0, 1.0)
        proj_lock_barrier_sl[sl_mask] = -np.log(proj_area_sl[sl_mask] + EPS)
        memory_align[sl_mask] = d_sl[sl_mask] / (np.sqrt(n_s[sl_mask] * n_l[sl_mask]) + EPS)
        short_long_explanation_imbalance[sl_mask] = (d_xs[sl_mask] ** 2) / (n_x[sl_mask] * n_s[sl_mask] + EPS) - (
            d_xl[sl_mask] ** 2
        ) / (n_x[sl_mask] * n_l[sl_mask] + EPS)

        det_raw = (
            n_x[sl_mask] * n_s[sl_mask] * n_l[sl_mask]
            + 2.0 * d_xs[sl_mask] * d_xl[sl_mask] * d_sl[sl_mask]
            - n_x[sl_mask] * (d_sl[sl_mask] ** 2)
            - n_s[sl_mask] * (d_xl[sl_mask] ** 2)
            - n_l[sl_mask] * (d_xs[sl_mask] ** 2)
        )
        det_raw = np.maximum(det_raw, 0.0)
        proj_volume_xsl[sl_mask] = np.clip(det_raw / (n_x[sl_mask] * n_s[sl_mask] * n_l[sl_mask] + EPS), 0.0, 1.0)
        log_proj_volume_xsl[sl_mask] = np.log(proj_volume_xsl[sl_mask] + EPS)

    if np.any(xl_mask):
        proj_line_lock_xl[xl_mask] = np.clip((d_xl[xl_mask] ** 2) / (n_x[xl_mask] * n_l[xl_mask] + EPS), 0.0, 1.0)
        proj_transverse_xl[xl_mask] = np.clip(1.0 - proj_line_lock_xl[xl_mask], 0.0, 1.0)
        proj_lock_barrier_xl[xl_mask] = -np.log(proj_transverse_xl[xl_mask] + EPS)
        linger[xl_mask] = d_xl[xl_mask] / (np.sqrt(n_x[xl_mask] * n_l[xl_mask]) + EPS)
        novelty[xl_mask] = 1.0 - linger[xl_mask]

    return {
        "proj_line_lock_sl": proj_line_lock_sl,
        "proj_area_sl": proj_area_sl,
        "proj_line_lock_xl": proj_line_lock_xl,
        "proj_transverse_xl": proj_transverse_xl,
        "proj_lock_barrier_sl": proj_lock_barrier_sl,
        "proj_lock_barrier_xl": proj_lock_barrier_xl,
        "proj_volume_xsl": proj_volume_xsl,
        "log_proj_volume_xsl": log_proj_volume_xsl,
        "short_long_explanation_imbalance": short_long_explanation_imbalance,
        "memory_align": memory_align,
        "linger": linger,
        "novelty": novelty,
    }


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


def first_pass_stats(cfg: MicroPMUConfig) -> dict[str, Any]:
    count = 0
    mean = np.zeros(len(STATE_BASE_COLS), dtype=np.float64)
    m2 = np.zeros(len(STATE_BASE_COLS), dtype=np.float64)
    row_count = 0
    event_rows = 0
    n_onsets = 0
    first_time = None
    last_time = None
    prev_event = False

    for chunk in pd.read_csv(cfg.data_path, usecols=USECOLS, chunksize=cfg.chunk_size):
        state, _ = phasor_state_from_chunk(chunk)
        count, mean, m2 = batch_update_welford(count, mean, m2, state)

        times = chunk["Time"].to_numpy(dtype=np.float64)
        events = chunk["Events"].to_numpy(dtype=np.float64) != 0
        row_count += len(chunk)
        event_rows += int(events.sum())
        if len(times):
            if first_time is None:
                first_time = float(times[0])
            last_time = float(times[-1])
            onset_mask = events & ~np.concatenate(([prev_event], events[:-1]))
            n_onsets += int(onset_mask.sum())
            prev_event = bool(events[-1])

    variance = m2 / max(count - 1, 1)
    std = np.sqrt(np.maximum(variance, EPS))
    return {
        "row_count": row_count,
        "event_rows": event_rows,
        "n_onsets": n_onsets,
        "state_mean": mean,
        "state_std": std,
        "first_time": first_time,
        "last_time": last_time,
    }


def finalize_event(col: dict[str, Any], event_windows: list[pd.DataFrame], cfg: MicroPMUConfig) -> None:
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


def second_pass_analyze(cfg: MicroPMUConfig, stats: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean = stats["state_mean"]
    std = stats["state_std"]
    ms_state = np.zeros(len(STATE_BASE_COLS), dtype=np.float64)
    ml_state = np.zeros(len(STATE_BASE_COLS), dtype=np.float64)
    prev_event = False
    event_index = 0

    active_collectors: list[dict[str, Any]] = []
    recent_tail = pd.DataFrame()
    event_windows: list[pd.DataFrame] = []
    event_sample_parts: list[pd.DataFrame] = []
    non_event_sample_parts: list[pd.DataFrame] = []

    global_row = 0
    for chunk_id, chunk in enumerate(pd.read_csv(cfg.data_path, usecols=USECOLS, chunksize=cfg.chunk_size)):
        state_raw, aux = phasor_state_from_chunk(chunk)
        x = (state_raw - mean) / std
        ms_prior, ms_state = ema_prior_chunk(x, cfg.beta_short, ms_state)
        ml_prior, ml_state = ema_prior_chunk(x, cfg.beta_long, ml_state)
        feat = compute_projective_chunk_features(x, ms_prior, ml_prior, cfg)

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
                f"[MICRO-PMU] chunk {chunk_id + 1} | rows {global_row} | events {event_index} | active {len(active_collectors)}",
                flush=True,
            )

    sampled_df = pd.concat(event_sample_parts + non_event_sample_parts, ignore_index=True)
    alignment_df = pd.concat(event_windows, ignore_index=True) if event_windows else pd.DataFrame()
    return sampled_df, alignment_df, pd.DataFrame(event_windows[0:0]) if False else alignment_df


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


def build_event_table(alignment_df: pd.DataFrame, cfg: MicroPMUConfig) -> pd.DataFrame:
    rows = []
    if alignment_df.empty:
        return pd.DataFrame()

    for event_id, g in alignment_df.groupby("event_index", sort=True):
        g = g.sort_values("steps_from_onset").reset_index(drop=True)
        baseline = g[(g["steps_from_onset"] >= -cfg.pre_event_steps) & (g["steps_from_onset"] < -cfg.pre_event_steps + cfg.baseline_steps)]
        search = g[(g["steps_from_onset"] >= -cfg.pre_event_steps + cfg.baseline_steps) & (g["steps_from_onset"] < 0)]
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
            mu = float(np.nanmean(base_vals))
            sigma = float(np.nanstd(base_vals, ddof=0))
            sigma = sigma if sigma > EPS else 1.0
            search_vals = search[metric].to_numpy(dtype=float)
            z = (search_vals - mu) / sigma if direction == "higher" else (mu - search_vals) / sigma
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


def build_benchmark_table(sampled_df: pd.DataFrame) -> pd.DataFrame:
    labels = sampled_df["event_flag"].to_numpy(dtype=int)
    metrics = [
        "proj_line_lock_sl",
        "proj_area_sl",
        "proj_line_lock_xl",
        "proj_transverse_xl",
        "proj_lock_barrier_sl",
        "proj_lock_barrier_xl",
        "proj_volume_xsl",
        "log_proj_volume_xsl",
        "short_long_explanation_imbalance",
        "memory_align",
        "linger",
        "novelty",
        "mean_voltage",
        "mean_current",
        "phase_voltage_spread",
        "phase_current_spread",
    ]
    rows = []
    for metric in metrics:
        auroc, direction = best_oriented_auc(sampled_df[metric].to_numpy(dtype=float), labels, 1)
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "auroc": auroc,
                "family": "projective"
                if metric.startswith("proj_")
                or metric.startswith("log_proj_")
                or metric in {"memory_align", "linger", "novelty", "short_long_explanation_imbalance"}
                else "raw",
            }
        )
    return pd.DataFrame(rows).sort_values(["auroc", "metric"], ascending=[False, True]).reset_index(drop=True)


def build_projective_event_outputs(
    event_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if event_df.empty:
        empty = event_df.copy()
        summary_df = pd.DataFrame(columns=["metric", "value"])
        return empty, empty, empty, summary_df

    enriched_df = event_df.copy()
    projective_features: list[Any] = []
    projective_leads: list[float] = []

    for _, row in enriched_df.iterrows():
        best_feature = np.nan
        best_lead = np.nan
        for metric in PROJECTIVE_LEAD_METRICS:
            lead = row.get(f"{metric}_lead_steps", np.nan)
            if np.isfinite(lead) and (not np.isfinite(best_lead) or lead > best_lead):
                best_lead = float(lead)
                best_feature = metric
        projective_features.append(best_feature)
        projective_leads.append(best_lead)

    raw_leads = enriched_df[f"{RAW_LEAD_METRIC}_lead_steps"].to_numpy(dtype=float)
    projective_leads_arr = np.asarray(projective_leads, dtype=float)
    projective_visible = np.isfinite(projective_leads_arr)
    raw_visible = np.isfinite(raw_leads)

    enriched_df["projective_lead_feature"] = projective_features
    enriched_df["projective_lead_steps"] = projective_leads_arr
    enriched_df["projective_visible"] = projective_visible
    enriched_df["raw_visible"] = raw_visible
    enriched_df["geometry_only"] = projective_visible & ~raw_visible
    enriched_df["raw_only"] = raw_visible & ~projective_visible
    enriched_df["geometry_first"] = projective_visible & (~raw_visible | (projective_leads_arr > raw_leads))
    enriched_df["raw_first"] = raw_visible & (~projective_visible | (raw_leads > projective_leads_arr))
    enriched_df["lead_tie"] = projective_visible & raw_visible & np.isclose(projective_leads_arr, raw_leads, equal_nan=False)
    enriched_df["projective_margin_vs_phase_current_spread"] = np.where(
        projective_visible & raw_visible,
        projective_leads_arr - raw_leads,
        np.nan,
    )

    relations = np.full(len(enriched_df), "no_lead", dtype=object)
    relations[enriched_df["geometry_only"].to_numpy(dtype=bool)] = "geometry_only"
    relations[enriched_df["raw_only"].to_numpy(dtype=bool)] = "raw_only"
    relations[enriched_df["geometry_first"].to_numpy(dtype=bool)] = "geometry_first"
    relations[enriched_df["raw_first"].to_numpy(dtype=bool)] = "raw_first"
    relations[enriched_df["lead_tie"].to_numpy(dtype=bool)] = "tie"
    enriched_df["lead_relation"] = relations

    projective_only_df = enriched_df.loc[enriched_df["projective_visible"]].copy().reset_index(drop=True)
    geometry_first_df = enriched_df.loc[enriched_df["geometry_first"]].copy().reset_index(drop=True)

    summary_rows = [
        {"metric": "total_events", "value": int(len(enriched_df))},
        {"metric": "projective_visible_events", "value": int(projective_visible.sum())},
        {"metric": "geometry_first_events", "value": int(enriched_df["geometry_first"].sum())},
        {"metric": "geometry_only_events", "value": int(enriched_df["geometry_only"].sum())},
        {"metric": "raw_first_events", "value": int(enriched_df["raw_first"].sum())},
        {"metric": "raw_only_events", "value": int(enriched_df["raw_only"].sum())},
        {"metric": "tied_events", "value": int(enriched_df["lead_tie"].sum())},
    ]
    for feature, count in projective_only_df["projective_lead_feature"].value_counts().items():
        summary_rows.append({"metric": f"projective_winner_{feature}", "value": int(count)})
    summary_df = pd.DataFrame(summary_rows)
    return enriched_df, projective_only_df, geometry_first_df, summary_df


def write_projective_outputs(
    run_dir: Path,
    event_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enriched_df, projective_only_df, geometry_first_df, summary_df = build_projective_event_outputs(event_df)
    enriched_df.to_csv(run_dir / "micro_pmu_projective_event_table.csv", index=False)
    projective_only_df.to_csv(run_dir / "micro_pmu_projective_only_event_table.csv", index=False)
    geometry_first_df.to_csv(run_dir / "micro_pmu_geometry_first_event_table.csv", index=False)
    summary_df.to_csv(run_dir / "micro_pmu_projective_summary.csv", index=False)
    return enriched_df, projective_only_df, geometry_first_df, summary_df


def build_alignment_table(alignment_df: pd.DataFrame, cfg: MicroPMUConfig) -> pd.DataFrame:
    if alignment_df.empty:
        return pd.DataFrame()
    parts = []
    for event_id, g in alignment_df.groupby("event_index", sort=True):
        g = g.sort_values("steps_from_onset").copy()
        baseline = g[(g["steps_from_onset"] >= -cfg.pre_event_steps) & (g["steps_from_onset"] < -cfg.pre_event_steps + cfg.baseline_steps)]
        if len(baseline) < 30:
            continue
        for metric in ["proj_lock_barrier_xl", "proj_transverse_xl", "proj_volume_xsl", "phase_current_spread"]:
            base = baseline[metric].to_numpy(dtype=float)
            mu = float(np.nanmedian(base))
            sigma = float(np.nanstd(base, ddof=0))
            sigma = sigma if sigma > EPS else 1.0
            g[f"{metric}_z"] = (g[metric] - mu) / sigma
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def plot_alignment(aligned_df: pd.DataFrame, output_path: Path) -> None:
    if aligned_df.empty:
        return
    grouped = aligned_df.groupby("steps_from_onset")[
        ["proj_lock_barrier_xl_z", "proj_transverse_xl_z", "proj_volume_xsl_z", "phase_current_spread_z"]
    ].median()
    time_s = grouped.index.to_numpy(dtype=float) / 120.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_s, grouped["proj_lock_barrier_xl_z"], color="#6a3d9a", label="proj_lock_barrier_xl (median z)")
    ax.plot(time_s, grouped["proj_transverse_xl_z"], color="#1b9e77", label="proj_transverse_xl (median z)")
    ax.plot(time_s, grouped["proj_volume_xsl_z"], color="#d95f02", label="proj_volume_xsl (median z)")
    ax.plot(time_s, grouped["phase_current_spread_z"], color="#4c78a8", label="phase_current_spread (median z)")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Seconds from event onset")
    ax.set_ylabel("Median baseline-normalized deviation")
    ax.set_title("Micro PMU October 1 alignment around real event onset")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(
    run_dir: Path,
    cfg: MicroPMUConfig,
    stats: dict[str, Any],
    benchmark_df: pd.DataFrame,
    event_df: pd.DataFrame,
    projective_event_df: pd.DataFrame,
) -> None:
    lead_counts = event_df["lead_feature"].value_counts().to_dict() if not event_df.empty else {}
    projective_counts = (
        projective_event_df["projective_lead_feature"].value_counts().to_dict()
        if not projective_event_df.empty
        else {}
    )
    lines = [
        "# Micro PMU October 1 Real-Data Experiment",
        "",
        "Chunked two-pass projective analysis of the full October 1 micro-PMU day file.",
        "",
        "## Setup",
        f"- Data path: `{cfg.data_path}`",
        f"- Rows scanned: `{stats['row_count']}`",
        f"- Event rows (`Events != 0`): `{stats['event_rows']}`",
        f"- Event onsets: `{stats['n_onsets']}`",
        f"- Short beta: `{cfg.beta_short}`",
        f"- Long beta: `{cfg.beta_long}`",
        f"- Pre-event window: `{cfg.pre_event_steps}` samples",
        f"- Post-event window: `{cfg.post_event_steps}` samples",
        f"- Baseline window: `{cfg.baseline_steps}` samples",
        f"- Lead rule: `z > {cfg.threshold_sigma}` for `{cfg.sustain_steps}` samples after rearming below `{cfg.rearm_sigma}` for `{cfg.rearm_steps}` samples",
        "",
        "## Top Metrics",
        benchmark_df.head(12).to_markdown(index=False),
        "",
        "## Event Lead Summary",
        f"- Events with a detected lead: `{int(event_df['lead_steps'].notna().sum()) if not event_df.empty else 0}` / `{len(event_df)}`",
        "\n".join(f"- `{k}`: `{v}` events" for k, v in lead_counts.items()) if lead_counts else "- No lead metrics crossed under the current rule set.",
        "",
        event_df.head(20).to_markdown(index=False) if not event_df.empty else "No event windows completed.",
        "",
        "## Projective-Only Summary",
        f"- Events with any projective lead: `{int(projective_event_df['projective_visible'].sum()) if not projective_event_df.empty else 0}` / `{len(projective_event_df)}`",
        f"- Geometry-first events: `{int(projective_event_df['geometry_first'].sum()) if not projective_event_df.empty else 0}`",
        f"- Geometry-only events: `{int(projective_event_df['geometry_only'].sum()) if not projective_event_df.empty else 0}`",
        f"- Raw-only events: `{int(projective_event_df['raw_only'].sum()) if not projective_event_df.empty else 0}`",
        "\n".join(f"- `{k}`: `{v}` projective wins" for k, v in projective_counts.items())
        if projective_counts
        else "- No projective metric crossed under the current rule set.",
        "",
        projective_event_df.loc[projective_event_df["geometry_first"]].head(20).to_markdown(index=False)
        if not projective_event_df.empty and projective_event_df["geometry_first"].any()
        else "No geometry-first events under the current rule set.",
        "",
        "## Geometry-First Artifacts",
        f"- Full enriched event table: `{run_dir / 'micro_pmu_projective_event_table.csv'}`",
        f"- Projective-visible subset: `{run_dir / 'micro_pmu_projective_only_event_table.csv'}`",
        f"- Geometry-first subset: `{run_dir / 'micro_pmu_geometry_first_event_table.csv'}`",
        f"- Projective summary: `{run_dir / 'micro_pmu_projective_summary.csv'}`",
        "",
    ]
    (run_dir / "micro_pmu_report.md").write_text("\n".join(lines), encoding="utf-8")


def load_existing_run_context(run_dir: Path) -> tuple[MicroPMUConfig, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    metadata = json.loads((run_dir / "micro_pmu_metadata.json").read_text(encoding="utf-8"))
    cfg_dict = dict(metadata["config"])
    cfg_dict["data_path"] = Path(cfg_dict["data_path"])
    cfg_dict["run_root"] = Path(cfg_dict["run_root"])
    cfg = MicroPMUConfig(**cfg_dict)
    stats = {
        "row_count": int(metadata["row_count"]),
        "event_rows": int(metadata["event_rows"]),
        "n_onsets": int(metadata["n_onsets"]),
        "first_time": metadata["first_time_ns"],
        "last_time": metadata["last_time_ns"],
    }
    benchmark_df = pd.read_csv(run_dir / "micro_pmu_benchmarks.csv")
    event_df = pd.read_csv(run_dir / "micro_pmu_event_table.csv")
    return cfg, stats, benchmark_df, event_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run projective event analysis on the Micro PMU October 1 dataset.")
    parser.add_argument("--data-path", default=str(Path("data") / "Micro PMU October 1 Dataset" / "_LBNL_a6_bus1_2015-10-01.csv"))
    parser.add_argument("--run-root", default=str(Path("artifact") / "runs"))
    parser.add_argument("--postprocess-run-dir", default=None)
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--beta-short", type=float, default=0.10)
    parser.add_argument("--beta-long", type=float, default=0.01)
    parser.add_argument("--threshold-sigma", type=float, default=1.5)
    parser.add_argument("--sustain-steps", type=int, default=6)
    parser.add_argument("--rearm-sigma", type=float, default=0.5)
    parser.add_argument("--rearm-steps", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.postprocess_run_dir:
        run_dir = Path(args.postprocess_run_dir).resolve()
        cfg, stats, benchmark_df, event_df = load_existing_run_context(run_dir)
        projective_event_df, _, _, _ = write_projective_outputs(run_dir, event_df)
        write_report(run_dir, cfg, stats, benchmark_df, event_df, projective_event_df)
        print(f"[MICRO-PMU] Postprocessed existing run: {run_dir}", flush=True)
        if not projective_event_df.empty:
            print(
                projective_event_df.loc[projective_event_df["geometry_first"]].head(12).to_string(index=False),
                flush=True,
            )
        return

    cfg = MicroPMUConfig(
        data_path=Path(args.data_path).resolve(),
        run_root=Path(args.run_root).resolve(),
        chunk_size=args.chunk_size,
        beta_short=args.beta_short,
        beta_long=args.beta_long,
        threshold_sigma=args.threshold_sigma,
        sustain_steps=args.sustain_steps,
        rearm_sigma=args.rearm_sigma,
        rearm_steps=args.rearm_steps,
    )
    run_dir = ensure_run_dir(cfg.run_root)

    stats = first_pass_stats(cfg)
    sampled_df, event_windows_df, _ = second_pass_analyze(cfg, stats)
    event_df = build_event_table(event_windows_df, cfg)
    benchmark_df = build_benchmark_table(sampled_df)
    aligned_df = build_alignment_table(event_windows_df, cfg)

    sampled_df.to_csv(run_dir / "micro_pmu_sampled_features.csv", index=False)
    event_df.to_csv(run_dir / "micro_pmu_event_table.csv", index=False)
    aligned_df.to_csv(run_dir / "micro_pmu_alignment.csv", index=False)
    benchmark_df.to_csv(run_dir / "micro_pmu_benchmarks.csv", index=False)
    projective_event_df, _, _, _ = write_projective_outputs(run_dir, event_df)

    plot_alignment(aligned_df, run_dir / "micro_pmu_alignment_plot.png")

    metadata = {
        "config": json_safe(asdict(cfg)),
        "output_dir": str(run_dir),
        "row_count": int(stats["row_count"]),
        "event_rows": int(stats["event_rows"]),
        "n_onsets": int(stats["n_onsets"]),
        "first_time_ns": stats["first_time"],
        "last_time_ns": stats["last_time"],
    }
    with (run_dir / "micro_pmu_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(json_safe(metadata), fp, indent=2)

    write_report(run_dir, cfg, stats, benchmark_df, event_df, projective_event_df)

    print(f"[MICRO-PMU] Output directory: {run_dir}", flush=True)
    print(benchmark_df.head(12).to_string(index=False), flush=True)
    if not event_df.empty:
        print(event_df.head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
