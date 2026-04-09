from __future__ import annotations

from typing import Any

import numpy as np


EPS = 1e-9
PROJECTIVE_FEATURE_COLUMNS = (
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
)
CANONICAL_PROJECTIVE_REFERENCE_COLUMNS = (
    "memory_align",
    "novelty",
    "proj_lock_barrier_sl",
    "proj_lock_barrier_xl",
    "proj_volume_xsl",
)
CANONICAL_LIE_FEATURE_COLUMNS = (
    "gram_trace",
    "gram_logdet",
    "lie_orbit_norm",
    "lie_strain_norm",
    "lie_commutator_norm",
    "lie_metric_drift",
    "gram_drift_energy",
    "gram_divergence_ratio",
    "gram_divergence_proxy",
)
CANONICAL_PROJECTIVE_TRANSITION_WEIGHTS = {
    "level": 0.50,
    "velocity": 1.00,
    "curvature": 1.25,
    "barrier_acc": 0.75,
    "novelty_vel": 0.50,
}
CANONICAL_LIE_TRANSITION_WEIGHTS = {
    "orbit": 1.00,
    "strain": 1.00,
    "commutator": 0.75,
    "metric_drift": 0.50,
    "gram_logdet": 0.25,
}


def ema_prior_states(x: np.ndarray, beta: float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if len(arr) == 0:
        return np.empty_like(arr)
    alpha = 1.0 - float(beta)
    state = np.zeros(arr.shape[1], dtype=float)
    prior = np.zeros_like(arr, dtype=float)
    for idx in range(len(arr)):
        prior[idx] = state
        state = float(beta) * arr[idx] + alpha * state
    return prior


def compute_projective_state_features(
    x: np.ndarray,
    ms_prior: np.ndarray,
    ml_prior: np.ndarray,
    min_state_energy: float,
) -> dict[str, np.ndarray]:
    n_x = np.einsum("td,td->t", x, x)
    n_s = np.einsum("td,td->t", ms_prior, ms_prior)
    n_l = np.einsum("td,td->t", ml_prior, ml_prior)
    d_sl = np.einsum("td,td->t", ms_prior, ml_prior)
    d_xs = np.einsum("td,td->t", x, ms_prior)
    d_xl = np.einsum("td,td->t", x, ml_prior)

    proj_line_lock_sl = np.full(len(x), np.nan, dtype=float)
    proj_area_sl = np.full(len(x), np.nan, dtype=float)
    proj_line_lock_xl = np.full(len(x), np.nan, dtype=float)
    proj_transverse_xl = np.full(len(x), np.nan, dtype=float)
    proj_lock_barrier_sl = np.full(len(x), np.nan, dtype=float)
    proj_lock_barrier_xl = np.full(len(x), np.nan, dtype=float)
    proj_volume_xsl = np.full(len(x), np.nan, dtype=float)
    log_proj_volume_xsl = np.full(len(x), np.nan, dtype=float)
    short_long_explanation_imbalance = np.full(len(x), np.nan, dtype=float)
    memory_align = np.full(len(x), np.nan, dtype=float)
    linger = np.full(len(x), np.nan, dtype=float)
    novelty = np.full(len(x), np.nan, dtype=float)

    energy_mask = n_x >= float(min_state_energy)
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


def compute_lie_state_features(
    x: np.ndarray,
    ms_prior: np.ndarray,
    ml_prior: np.ndarray,
    sfreq: float,
    chunk_size: int = 50000,
) -> dict[str, np.ndarray]:
    n_rows = len(x)
    gram_trace = np.zeros(n_rows, dtype=float)
    gram_logdet = np.zeros(n_rows, dtype=float)
    lie_orbit_norm = np.zeros(n_rows, dtype=float)
    lie_strain_norm = np.zeros(n_rows, dtype=float)
    lie_commutator_norm = np.zeros(n_rows, dtype=float)
    lie_metric_drift = np.zeros(n_rows, dtype=float)
    gram_drift_energy = np.zeros(n_rows, dtype=float)
    gram_divergence_ratio = np.ones(n_rows, dtype=float)
    gram_divergence_proxy = np.zeros(n_rows, dtype=float)

    eye3 = np.eye(3, dtype=float)
    prev_k: np.ndarray | None = None
    prev_d2: float | None = None
    dt_seconds = 1.0 / float(sfreq) if float(sfreq) > 0 else 1.0
    for start in range(0, n_rows, chunk_size):
        stop = min(n_rows, start + chunk_size)
        x_chunk = x[start:stop]
        ms_chunk = ms_prior[start:stop]
        ml_chunk = ml_prior[start:stop]

        if start == 0:
            prev_x = np.zeros_like(x_chunk)
            prev_ms = np.zeros_like(ms_chunk)
            prev_ml = np.zeros_like(ml_chunk)
            if len(x_chunk) > 1:
                prev_x[1:] = x_chunk[:-1]
                prev_ms[1:] = ms_chunk[:-1]
                prev_ml[1:] = ml_chunk[:-1]
        else:
            prev_x = x[start - 1 : stop - 1]
            prev_ms = ms_prior[start - 1 : stop - 1]
            prev_ml = ml_prior[start - 1 : stop - 1]

        dx_chunk = x_chunk - prev_x
        dms_chunk = ms_chunk - prev_ms
        dml_chunk = ml_chunk - prev_ml

        q_chunk = np.stack([x_chunk, ms_chunk, ml_chunk], axis=2)
        dq_chunk = np.stack([dx_chunk, dms_chunk, dml_chunk], axis=2)

        k_raw = np.matmul(np.swapaxes(q_chunk, 1, 2), q_chunk)
        k_trace_raw = np.trace(k_raw, axis1=1, axis2=2)
        lam = 1e-6 * (k_trace_raw / 3.0 + EPS)
        k_t = k_raw + lam[:, None, None] * eye3[None, :, :]
        c_t = np.matmul(np.swapaxes(q_chunk, 1, 2), dq_chunk)
        inv_k = np.linalg.inv(k_t)
        b_t = np.matmul(inv_k, c_t)
        k_inv_bt_k = np.matmul(inv_k, np.matmul(np.swapaxes(b_t, 1, 2), k_t))
        a_t = 0.5 * (b_t - k_inv_bt_k)
        s_t = 0.5 * (b_t + k_inv_bt_k)
        commutator = np.matmul(s_t, a_t) - np.matmul(a_t, s_t)

        gram_trace[start:stop] = np.trace(k_t, axis1=1, axis2=2)
        gram_logdet[start:stop] = np.log(np.maximum(np.linalg.det(k_t), EPS))
        lie_orbit_norm[start:stop] = np.sqrt(np.sum(np.square(a_t), axis=(1, 2)))
        lie_strain_norm[start:stop] = np.sqrt(np.sum(np.square(s_t), axis=(1, 2)))
        lie_commutator_norm[start:stop] = np.sqrt(np.sum(np.square(commutator), axis=(1, 2)))
        d2_chunk = np.sum(np.square(dq_chunk), axis=(1, 2))
        gram_drift_energy[start:stop] = d2_chunk

        k_norm = np.sqrt(np.sum(np.square(k_t), axis=(1, 2)))
        drift_chunk = np.zeros(stop - start, dtype=float)
        if len(k_t) > 1:
            drift_chunk[1:] = np.sqrt(np.sum(np.square(k_t[1:] - k_t[:-1]), axis=(1, 2))) / (k_norm[1:] + EPS)
        if prev_k is not None and len(k_t):
            drift_chunk[0] = np.sqrt(np.sum(np.square(k_t[0] - prev_k))) / (k_norm[0] + EPS)
        lie_metric_drift[start:stop] = drift_chunk

        ratio_chunk = np.ones(stop - start, dtype=float)
        if len(d2_chunk) > 1:
            ratio_chunk[1:] = d2_chunk[1:] / (d2_chunk[:-1] + EPS)
        if prev_d2 is not None and len(d2_chunk):
            ratio_chunk[0] = d2_chunk[0] / (prev_d2 + EPS)
        gram_divergence_ratio[start:stop] = ratio_chunk
        gram_divergence_proxy[start:stop] = (ratio_chunk - 1.0) / (2.0 * dt_seconds)

        if len(k_t):
            prev_k = k_t[-1].copy()
            prev_d2 = float(d2_chunk[-1])

    return {
        "gram_trace": gram_trace,
        "gram_logdet": gram_logdet,
        "lie_orbit_norm": lie_orbit_norm,
        "lie_strain_norm": lie_strain_norm,
        "lie_commutator_norm": lie_commutator_norm,
        "lie_metric_drift": lie_metric_drift,
        "gram_drift_energy": gram_drift_energy,
        "gram_divergence_ratio": gram_divergence_ratio,
        "gram_divergence_proxy": gram_divergence_proxy,
    }


def canonical_projective_transition_score(
    level: np.ndarray,
    velocity: np.ndarray,
    curvature: np.ndarray,
    barrier_acc: np.ndarray,
    novelty_vel: np.ndarray,
) -> np.ndarray:
    return np.sqrt(
        CANONICAL_PROJECTIVE_TRANSITION_WEIGHTS["level"] * np.square(level)
        + CANONICAL_PROJECTIVE_TRANSITION_WEIGHTS["velocity"] * np.square(velocity)
        + CANONICAL_PROJECTIVE_TRANSITION_WEIGHTS["curvature"] * np.square(curvature)
        + CANONICAL_PROJECTIVE_TRANSITION_WEIGHTS["barrier_acc"] * np.square(barrier_acc)
        + CANONICAL_PROJECTIVE_TRANSITION_WEIGHTS["novelty_vel"] * np.square(novelty_vel)
    )


def canonical_lie_transition_score(
    orbit_z: np.ndarray,
    strain_z: np.ndarray,
    commutator_z: np.ndarray,
    metric_drift_z: np.ndarray,
    gram_logdet_z: np.ndarray,
) -> np.ndarray:
    return np.sqrt(
        CANONICAL_LIE_TRANSITION_WEIGHTS["orbit"] * np.square(orbit_z)
        + CANONICAL_LIE_TRANSITION_WEIGHTS["strain"] * np.square(strain_z)
        + CANONICAL_LIE_TRANSITION_WEIGHTS["commutator"] * np.square(commutator_z)
        + CANONICAL_LIE_TRANSITION_WEIGHTS["metric_drift"] * np.square(metric_drift_z)
        + CANONICAL_LIE_TRANSITION_WEIGHTS["gram_logdet"] * np.square(gram_logdet_z)
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
