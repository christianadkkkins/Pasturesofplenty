from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


EPS = 1e-9
ORIENTATION_EPS = 1e-7
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
ORIENTED_STATE_FEATURE_COLUMNS = (
    "pair_align_xs",
    "pair_align_xl",
    "pair_align_sl",
    "pair_angle_xs",
    "pair_angle_xl",
    "pair_angle_sl",
    "pair_quadrature_xs",
    "pair_quadrature_xl",
    "pair_quadrature_sl",
    "pair_phase_ratio_xs",
    "pair_phase_ratio_xl",
    "pair_phase_ratio_sl",
    "oriented_pair_angle_xs",
    "oriented_pair_angle_xl",
    "oriented_pair_angle_sl",
    "oriented_angle_contrast_xs_xl",
    "lie_orbit_xs",
    "lie_orbit_xl",
    "lie_orbit_sl",
    "lie_orbit_strain_ratio",
    "lie_orbit_dominance",
    "lie_commutator_coupling_ratio",
    "elevation_angle_x_over_sl",
    "oriented_volume_xsl",
    "structural_branch_sign_xsl",
    "negative_volume_excursion_xsl",
    "phase_coherence_residual_xs_xl_sl",
    "sheaf_defect_log_ratio",
    "spectral_orbit_drift_energy",
    "spectral_strain_drift_energy",
    "spectral_orbit_divergence_ratio",
    "spectral_strain_divergence_ratio",
    "spectral_orbit_divergence_proxy",
    "spectral_strain_divergence_proxy",
)


@dataclass
class AdaptiveMetricConfig:
    burn_in_length: int = 500
    alpha: float = 2.0
    beta: float = 0.05
    z_thresh: float = 0.0
    feature_names: tuple[str, ...] = ("proj_lock_barrier_sl", "memory_align")
    branch_feature_name: str | None = None
    branch_threshold: float = 0.0
    min_branch_burn_in_count: int = 8


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


def _safe_unit_vector(vec: np.ndarray, floor: float = EPS) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= float(floor):
        return None
    return np.asarray(vec, dtype=float) / norm


def _row_unit_columns(q_row: np.ndarray, min_state_energy: float) -> np.ndarray | None:
    cols = []
    for idx in range(q_row.shape[1]):
        col = np.asarray(q_row[:, idx], dtype=float)
        energy = float(np.dot(col, col))
        if not np.isfinite(energy) or energy < float(min_state_energy):
            return None
        unit_col = _safe_unit_vector(col, floor=np.sqrt(float(min_state_energy)))
        if unit_col is None:
            return None
        cols.append(unit_col)
    return np.column_stack(cols)


def _positive_diag_qr(frame: np.ndarray) -> np.ndarray | None:
    if frame.ndim != 2 or frame.shape[1] != 3 or frame.shape[0] < 3:
        return None
    if not np.all(np.isfinite(frame)):
        return None
    if int(np.linalg.matrix_rank(frame)) < 3:
        return None
    try:
        basis, upper = np.linalg.qr(frame, mode="reduced")
    except np.linalg.LinAlgError:
        return None
    if basis.shape != (frame.shape[0], 3) or upper.shape != (3, 3):
        return None
    signs = np.where(np.diag(upper) < 0.0, -1.0, 1.0)
    basis = basis @ np.diag(signs)
    return basis if np.all(np.isfinite(basis)) else None


def _reference_basis_from_row(frame: np.ndarray, prev_basis: np.ndarray | None) -> np.ndarray | None:
    basis = _positive_diag_qr(frame)
    if basis is None:
        return None
    if prev_basis is not None:
        transition = prev_basis.T @ basis
        det_transition = float(np.linalg.det(transition))
        if np.isfinite(det_transition) and det_transition < 0.0:
            basis[:, 2] *= -1.0
    return basis


def _pair_normal_from_coords(a_vec: np.ndarray, b_vec: np.ndarray) -> np.ndarray | None:
    return _safe_unit_vector(np.cross(a_vec, b_vec), floor=EPS)


def _pair_orientation_from_prev_normal(
    pair_align: float,
    curr_a_vec: np.ndarray,
    curr_b_vec: np.ndarray,
    prev_normal: np.ndarray | None,
) -> tuple[float, float, float, np.ndarray | None]:
    pair_align_clipped = float(np.clip(pair_align, -1.0, 1.0))
    pair_angle = float(np.arccos(pair_align_clipped))
    if prev_normal is None:
        return pair_angle, np.nan, np.nan, None

    curr_normal = _pair_normal_from_coords(curr_a_vec, curr_b_vec)
    if curr_normal is None:
        oriented_pair_angle = float(np.arctan2(0.0, pair_align_clipped))
        return pair_angle, 0.0, oriented_pair_angle, prev_normal.copy()

    orientation_projection = float(np.dot(curr_normal, prev_normal))
    if not np.isfinite(orientation_projection):
        return pair_angle, np.nan, np.nan, None

    pair_quadrature = float(np.sign(orientation_projection) * np.sqrt(max(1.0 - pair_align_clipped**2, 0.0)))
    if abs(orientation_projection) <= ORIENTATION_EPS:
        pair_quadrature = float(np.sqrt(max(1.0 - pair_align_clipped**2, 0.0)))
    oriented_pair_angle = float(np.arctan2(pair_quadrature, pair_align_clipped))
    if orientation_projection < 0.0:
        curr_normal = -curr_normal
    return pair_angle, pair_quadrature, oriented_pair_angle, curr_normal


def _oriented_volume_from_prev_normal(
    coord_x_vec: np.ndarray,
    coord_s_vec: np.ndarray,
    coord_l_vec: np.ndarray,
    pair_angle_sl: float,
    prev_normal: np.ndarray | None,
) -> tuple[float, float]:
    if prev_normal is None:
        return np.nan, np.nan

    coord_x_unit = _safe_unit_vector(coord_x_vec, floor=EPS)
    if coord_x_unit is None:
        return np.nan, np.nan

    curr_normal = _pair_normal_from_coords(coord_s_vec, coord_l_vec)
    if curr_normal is None:
        curr_normal = prev_normal.copy()
    else:
        orientation_projection = float(np.dot(curr_normal, prev_normal))
        if not np.isfinite(orientation_projection):
            return np.nan, np.nan
        if orientation_projection < 0.0:
            curr_normal = -curr_normal

    lift = float(np.clip(np.dot(coord_x_unit, curr_normal), -1.0, 1.0))
    elevation_angle = float(np.arctan2(lift, np.sqrt(max(1.0 - lift * lift, 0.0))))
    oriented_volume = float(np.sin(pair_angle_sl) * np.sin(elevation_angle))
    return elevation_angle, oriented_volume


def _robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if len(valid) == 0:
        return np.nan, np.nan
    center = float(np.median(valid))
    mad = float(np.median(np.abs(valid - center)))
    scale = 1.4826 * mad if mad > 1e-12 else float(np.std(valid, ddof=0))
    return center, scale


def _adaptive_membership_from_z(z_score: np.ndarray, alpha: float, z_thresh: float) -> np.ndarray:
    excess = np.maximum(np.asarray(z_score, dtype=float) - float(z_thresh), 0.0)
    return np.exp(-float(alpha) * excess)


def _spectral_drift_series(
    matrices: np.ndarray,
    *,
    dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = len(matrices)
    drift_energy = np.zeros(n_rows, dtype=float)
    ratio = np.ones(n_rows, dtype=float)
    proxy = np.zeros(n_rows, dtype=float)
    prev_matrix = np.zeros((3, 3), dtype=float)
    prev_energy: float | None = None
    for idx in range(n_rows):
        curr_matrix = np.asarray(matrices[idx], dtype=float)
        diff = curr_matrix - prev_matrix
        energy = float(np.sum(np.square(diff)))
        drift_energy[idx] = energy
        if prev_energy is not None:
            ratio[idx] = energy / (prev_energy + EPS)
            proxy[idx] = (ratio[idx] - 1.0) / (2.0 * float(dt_seconds))
        prev_matrix = curr_matrix.copy()
        prev_energy = energy
    return drift_energy, ratio, proxy


def _compute_lie_state_bundle(
    x: np.ndarray,
    ms_prior: np.ndarray,
    ml_prior: np.ndarray,
    sfreq: float = 1.0,
    chunk_size: int = 50000,
    collect_matrices: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
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

    k_all = np.zeros((n_rows, 3, 3), dtype=float) if collect_matrices else np.empty((0, 3, 3), dtype=float)
    a_all = np.zeros((n_rows, 3, 3), dtype=float) if collect_matrices else np.empty((0, 3, 3), dtype=float)
    s_all = np.zeros((n_rows, 3, 3), dtype=float) if collect_matrices else np.empty((0, 3, 3), dtype=float)

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

        if collect_matrices:
            k_all[start:stop] = k_t
            a_all[start:stop] = a_t
            s_all[start:stop] = s_t

        if len(k_t):
            prev_k = k_t[-1].copy()
            prev_d2 = float(d2_chunk[-1])

    features = {
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
    extras = {"k_t": k_all, "a_t": a_all, "s_t": s_all} if collect_matrices else {}
    return features, extras


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
    sfreq: float = 1.0,
    chunk_size: int = 50000,
) -> dict[str, np.ndarray]:
    features, _ = _compute_lie_state_bundle(
        x=x,
        ms_prior=ms_prior,
        ml_prior=ml_prior,
        sfreq=sfreq,
        chunk_size=chunk_size,
        collect_matrices=False,
    )
    return features


def compute_oriented_state_features(
    x: np.ndarray,
    ms_prior: np.ndarray,
    ml_prior: np.ndarray,
    min_state_energy: float = 1e-12,
    sfreq: float = 1.0,
    chunk_size: int = 50000,
) -> dict[str, np.ndarray]:
    n_rows = len(x)
    oriented = {name: np.full(n_rows, np.nan, dtype=float) for name in ORIENTED_STATE_FEATURE_COLUMNS}
    lie, extras = _compute_lie_state_bundle(
        x=x,
        ms_prior=ms_prior,
        ml_prior=ml_prior,
        sfreq=sfreq,
        chunk_size=chunk_size,
        collect_matrices=True,
    )
    if n_rows == 0:
        return oriented

    dt_seconds = 1.0 / float(sfreq) if float(sfreq) > 0 else 1.0
    spectral_orbit_drift_energy, spectral_orbit_divergence_ratio, spectral_orbit_divergence_proxy = (
        _spectral_drift_series(extras["a_t"], dt_seconds=dt_seconds)
    )
    spectral_strain_drift_energy, spectral_strain_divergence_ratio, spectral_strain_divergence_proxy = (
        _spectral_drift_series(extras["s_t"], dt_seconds=dt_seconds)
    )

    prev_basis: np.ndarray | None = None
    prev_unit_cols: np.ndarray | None = None
    pair_index_map = {"xs": (0, 1), "xl": (0, 2), "sl": (1, 2)}

    for idx in range(n_rows):
        q_row = np.column_stack((x[idx], ms_prior[idx], ml_prior[idx]))
        current_basis = _reference_basis_from_row(q_row, prev_basis)
        unit_cols = _row_unit_columns(q_row, min_state_energy=min_state_energy)

        if current_basis is None or unit_cols is None:
            prev_basis = current_basis
            prev_unit_cols = unit_cols
            continue

        reference_basis = current_basis if idx == 0 else prev_basis
        reference_units = unit_cols if idx == 0 else prev_unit_cols
        if reference_basis is None or reference_units is None:
            prev_basis = current_basis
            prev_unit_cols = unit_cols
            continue

        current_coords = reference_basis.T @ unit_cols
        prev_coords = reference_basis.T @ reference_units
        current_coord_units = []
        prev_coord_units = []
        valid_row = True
        for col_idx in range(3):
            curr_unit = _safe_unit_vector(current_coords[:, col_idx], floor=EPS)
            prev_unit = _safe_unit_vector(prev_coords[:, col_idx], floor=EPS)
            if curr_unit is None or prev_unit is None:
                valid_row = False
                break
            current_coord_units.append(curr_unit)
            prev_coord_units.append(prev_unit)
        if not valid_row:
            prev_basis = current_basis
            prev_unit_cols = unit_cols
            continue

        current_coord_units = np.column_stack(current_coord_units)
        prev_coord_units = np.column_stack(prev_coord_units)
        pair_align = {
            name: float(np.clip(np.dot(unit_cols[:, i_idx], unit_cols[:, j_idx]), -1.0, 1.0))
            for name, (i_idx, j_idx) in pair_index_map.items()
        }
        prev_normals = {
            name: _pair_normal_from_coords(prev_coord_units[:, i_idx], prev_coord_units[:, j_idx])
            for name, (i_idx, j_idx) in pair_index_map.items()
        }
        if any(normal is None for normal in prev_normals.values()):
            prev_basis = current_basis
            prev_unit_cols = unit_cols
            continue

        pair_metrics = {}
        current_normals = {}
        for name, (i_idx, j_idx) in pair_index_map.items():
            pair_angle, pair_quadrature, oriented_pair_angle, current_normal = _pair_orientation_from_prev_normal(
                pair_align=pair_align[name],
                curr_a_vec=current_coord_units[:, i_idx],
                curr_b_vec=current_coord_units[:, j_idx],
                prev_normal=prev_normals[name],
            )
            if current_normal is None or not np.isfinite(pair_quadrature) or not np.isfinite(oriented_pair_angle):
                valid_row = False
                break
            pair_metrics[name] = {
                "pair_angle": pair_angle,
                "pair_quadrature": pair_quadrature,
                "oriented_pair_angle": oriented_pair_angle,
            }
            current_normals[name] = current_normal
        if not valid_row:
            prev_basis = current_basis
            prev_unit_cols = unit_cols
            continue

        k_row = extras["k_t"][idx]
        a_row = extras["a_t"][idx]
        try:
            r_upper = np.linalg.cholesky(k_row).T
            omega = r_upper @ a_row @ np.linalg.inv(r_upper)
        except np.linalg.LinAlgError:
            prev_basis = current_basis
            prev_unit_cols = unit_cols
            continue
        if not np.all(np.isfinite(omega)):
            prev_basis = current_basis
            prev_unit_cols = unit_cols
            continue

        elevation_angle, oriented_volume = _oriented_volume_from_prev_normal(
            coord_x_vec=current_coord_units[:, 0],
            coord_s_vec=current_coord_units[:, 1],
            coord_l_vec=current_coord_units[:, 2],
            pair_angle_sl=pair_metrics["sl"]["pair_angle"],
            prev_normal=prev_normals["sl"],
        )
        if not np.isfinite(elevation_angle) or not np.isfinite(oriented_volume):
            prev_basis = current_basis
            prev_unit_cols = unit_cols
            continue

        orbit_norm = lie["lie_orbit_norm"][idx]
        strain_norm = lie["lie_strain_norm"][idx]
        commutator_norm = lie["lie_commutator_norm"][idx]
        if not np.isfinite(orbit_norm) or not np.isfinite(strain_norm) or not np.isfinite(commutator_norm):
            prev_basis = current_basis
            prev_unit_cols = unit_cols
            continue

        for name in pair_index_map:
            oriented[f"pair_align_{name}"][idx] = pair_align[name]
            oriented[f"pair_angle_{name}"][idx] = pair_metrics[name]["pair_angle"]
            oriented[f"pair_quadrature_{name}"][idx] = pair_metrics[name]["pair_quadrature"]
            oriented[f"pair_phase_ratio_{name}"][idx] = pair_metrics[name]["pair_quadrature"] / (pair_align[name] + EPS)
            oriented[f"oriented_pair_angle_{name}"][idx] = pair_metrics[name]["oriented_pair_angle"]

        oriented["oriented_angle_contrast_xs_xl"][idx] = (
            pair_metrics["xs"]["oriented_pair_angle"] - pair_metrics["xl"]["oriented_pair_angle"]
        )
        oriented["lie_orbit_xs"][idx] = float(omega[0, 1])
        oriented["lie_orbit_xl"][idx] = float(omega[0, 2])
        oriented["lie_orbit_sl"][idx] = float(omega[1, 2])
        oriented["lie_orbit_strain_ratio"][idx] = orbit_norm / (strain_norm + EPS)
        oriented["lie_orbit_dominance"][idx] = (orbit_norm - strain_norm) / (orbit_norm + strain_norm + EPS)
        oriented["lie_commutator_coupling_ratio"][idx] = commutator_norm / (orbit_norm + strain_norm + EPS)
        oriented["elevation_angle_x_over_sl"][idx] = elevation_angle
        oriented["oriented_volume_xsl"][idx] = oriented_volume
        oriented["structural_branch_sign_xsl"][idx] = 1.0 if oriented_volume >= 0.0 else -1.0
        oriented["negative_volume_excursion_xsl"][idx] = max(-oriented_volume, 0.0)
        oriented["phase_coherence_residual_xs_xl_sl"][idx] = (
            pair_metrics["xs"]["oriented_pair_angle"]
            - pair_metrics["xl"]["oriented_pair_angle"]
            - pair_metrics["sl"]["oriented_pair_angle"]
        )
        phi_abs = abs(oriented["phase_coherence_residual_xs_xl_sl"][idx])
        vol_abs = abs(oriented_volume)
        oriented["sheaf_defect_log_ratio"][idx] = float(np.log((phi_abs + EPS) / (vol_abs + EPS)))
        oriented["spectral_orbit_drift_energy"][idx] = spectral_orbit_drift_energy[idx]
        oriented["spectral_strain_drift_energy"][idx] = spectral_strain_drift_energy[idx]
        oriented["spectral_orbit_divergence_ratio"][idx] = spectral_orbit_divergence_ratio[idx]
        oriented["spectral_strain_divergence_ratio"][idx] = spectral_strain_divergence_ratio[idx]
        oriented["spectral_orbit_divergence_proxy"][idx] = spectral_orbit_divergence_proxy[idx]
        oriented["spectral_strain_divergence_proxy"][idx] = spectral_strain_divergence_proxy[idx]

        prev_basis = current_basis
        prev_unit_cols = unit_cols

    return oriented


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


def compute_adaptive_metric_features(
    features: dict[str, np.ndarray],
    cfg: AdaptiveMetricConfig,
) -> dict[str, np.ndarray]:
    if not cfg.feature_names:
        return {}

    n_rows = 0
    for name in cfg.feature_names:
        if name in features:
            n_rows = len(features[name])
            break
    if n_rows == 0:
        return {}

    out = {"baseline_locked": np.zeros(n_rows, dtype=float)}
    branch_feat = None
    branch_enabled = cfg.branch_feature_name is not None and cfg.branch_feature_name in features
    if branch_enabled:
        branch_feat = np.asarray(features[cfg.branch_feature_name], dtype=float)
        out["branch_baseline_locked"] = np.zeros(n_rows, dtype=float)
    for name in cfg.feature_names:
        out[f"adaptive_z_{name}"] = np.full(n_rows, np.nan, dtype=float)
        out[f"soft_membership_{name}"] = np.full(n_rows, np.nan, dtype=float)
        out[f"evidence_{name}"] = np.zeros(n_rows, dtype=float)
        if branch_enabled:
            out[f"branch_adaptive_z_{name}"] = np.full(n_rows, np.nan, dtype=float)
            out[f"branch_soft_membership_{name}"] = np.full(n_rows, np.nan, dtype=float)
            out[f"branch_evidence_{name}"] = np.zeros(n_rows, dtype=float)

    for name in cfg.feature_names:
        if name not in features:
            continue

        feat = np.asarray(features[name], dtype=float)

        burn_in_end = min(n_rows, cfg.burn_in_length)
        out["baseline_locked"][burn_in_end:] = 1.0
        if branch_enabled:
            out["branch_baseline_locked"][burn_in_end:] = 1.0

        baseline_median = np.nan
        baseline_scale = np.nan
        if burn_in_end > 0:
            baseline_median, baseline_scale = _robust_center_scale(feat[:burn_in_end])
        if np.isfinite(baseline_median) and np.isfinite(baseline_scale):
            z_score = (feat - baseline_median) / (baseline_scale + EPS)
            out[f"adaptive_z_{name}"] = z_score

            mu = _adaptive_membership_from_z(z_score, alpha=cfg.alpha, z_thresh=cfg.z_thresh)
            out[f"soft_membership_{name}"] = mu

            evidence = np.zeros(n_rows, dtype=float)
            evidence_val = 0.0

            for i in range(n_rows):
                if i < burn_in_end:
                    evidence[i] = 0.0
                else:
                    if not np.isfinite(mu[i]):
                        evidence[i] = evidence_val
                        continue
                    evidence_val = (1.0 - cfg.beta) * evidence_val + cfg.beta * (1.0 - mu[i])
                    evidence[i] = evidence_val
            out[f"evidence_{name}"] = evidence

        if branch_enabled and branch_feat is not None:
            branch_median_pos = np.nan
            branch_scale_pos = np.nan
            branch_median_neg = np.nan
            branch_scale_neg = np.nan
            if burn_in_end > 0:
                burn_in_feat = feat[:burn_in_end]
                burn_in_branch = branch_feat[:burn_in_end]
                finite_feat = np.isfinite(burn_in_feat)
                finite_branch = np.isfinite(burn_in_branch)
                pos_mask = finite_feat & finite_branch & (burn_in_branch >= float(cfg.branch_threshold))
                neg_mask = finite_feat & finite_branch & (burn_in_branch < float(cfg.branch_threshold))
                if int(np.count_nonzero(pos_mask)) >= int(cfg.min_branch_burn_in_count):
                    branch_median_pos, branch_scale_pos = _robust_center_scale(burn_in_feat[pos_mask])
                    if not np.isfinite(branch_scale_pos) or branch_scale_pos <= 0.0:
                        branch_median_pos = np.nan
                        branch_scale_pos = np.nan
                if int(np.count_nonzero(neg_mask)) >= int(cfg.min_branch_burn_in_count):
                    branch_median_neg, branch_scale_neg = _robust_center_scale(burn_in_feat[neg_mask])
                    if not np.isfinite(branch_scale_neg) or branch_scale_neg <= 0.0:
                        branch_median_neg = np.nan
                        branch_scale_neg = np.nan

            branch_z = np.full(n_rows, np.nan, dtype=float)
            branch_mu = np.full(n_rows, np.nan, dtype=float)
            for i in range(n_rows):
                feat_val = feat[i]
                if not np.isfinite(feat_val):
                    continue
                use_center = baseline_median
                use_scale = baseline_scale
                branch_val = branch_feat[i]
                if np.isfinite(branch_val):
                    if branch_val >= float(cfg.branch_threshold):
                        if np.isfinite(branch_median_pos) and np.isfinite(branch_scale_pos):
                            use_center = branch_median_pos
                            use_scale = branch_scale_pos
                    else:
                        if np.isfinite(branch_median_neg) and np.isfinite(branch_scale_neg):
                            use_center = branch_median_neg
                            use_scale = branch_scale_neg
                if not np.isfinite(use_center) or not np.isfinite(use_scale):
                    continue
                branch_z[i] = (feat_val - use_center) / (use_scale + EPS)
            finite_branch_z = np.isfinite(branch_z)
            branch_mu[finite_branch_z] = _adaptive_membership_from_z(
                branch_z[finite_branch_z],
                alpha=cfg.alpha,
                z_thresh=cfg.z_thresh,
            )
            out[f"branch_adaptive_z_{name}"] = branch_z
            out[f"branch_soft_membership_{name}"] = branch_mu

            branch_evidence = np.zeros(n_rows, dtype=float)
            branch_evidence_val = 0.0
            for i in range(n_rows):
                if i < burn_in_end:
                    branch_evidence[i] = 0.0
                else:
                    if not np.isfinite(branch_mu[i]):
                        branch_evidence[i] = branch_evidence_val
                        continue
                    branch_evidence_val = (1.0 - cfg.beta) * branch_evidence_val + cfg.beta * (1.0 - branch_mu[i])
                    branch_evidence[i] = branch_evidence_val
            out[f"branch_evidence_{name}"] = branch_evidence

    return out
