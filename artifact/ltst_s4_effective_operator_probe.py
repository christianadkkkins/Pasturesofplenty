from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    torch = None
    F = None
    nn = None
    DataLoader = None
    Dataset = object


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifact"
for path in (REPO_ROOT, ARTIFACT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


EPS = 1e-9
DEFAULT_GEOMETRY_RUN = ARTIFACT_DIR / "runs" / "ltst_full_86_20260405T035505Z"
DEFAULT_ORIENTED_RUN = ARTIFACT_DIR / "runs" / "ltst_full_86_oriented_backfill_20260410T230134Z"
DEFAULT_HISTORICAL_BASELINE_AUROC = 0.7384131971720346
DEFAULT_HISTORICAL_BASELINE_METRIC = "baseline_p95_temporal_volume_3"
REGIME_TO_PHENOTYPE = {
    "angle_only": "loose_orbit",
    "angle_first": "loose_orbit",
    "long_only": "constrained_orbit",
    "long_first": "constrained_orbit",
    "both_same_horizon": "constrained_orbit",
    "neither": "rigid_orbit",
}
CANONICAL_FEATURE_COLUMNS = (
    "curr_energy",
    "past_energy",
    "dot_centered",
    "twist_angle",
    "twist_sq",
    "ema_fast",
    "ema_slow",
    "drift_centered_sq",
    "drift_raw_sq",
    "mean_shift_sq",
    "phase_polarity",
    "energy_delta",
    "energy_asym",
    "drift_norm",
    "poincare_b",
    "gram_spread_sq",
    "cos_signed",
    "local_template_corr",
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
    "gram_trace",
    "gram_logdet",
    "lie_orbit_norm",
    "lie_strain_norm",
    "lie_commutator_norm",
    "lie_metric_drift",
    "gram_drift_energy",
    "gram_divergence_ratio",
    "gram_divergence_proxy",
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
DIRECT_COMPARISON_COLUMNS = (
    "probe_prob",
    "operator_rho",
    "operator_low_rank_leverage",
    "proj_volume_xsl",
    "lie_orbit_norm",
    "lie_orbit_dominance",
    "pair_angle_sl",
    "sheaf_defect_log_ratio",
    "poincare_b",
)
OPERATOR_CHANNEL_COLUMNS = (
    "operator_anti_norm",
    "operator_hermitian_norm",
    "operator_rho",
    "operator_low_rank_leverage",
    "operator_low_rank_alignment",
    "operator_mean_real_eig",
    "operator_mean_abs_imag_eig",
    "operator_spectral_radius",
)


@dataclass(frozen=True)
class ProbeConfig:
    geometry_run_dir: Path = DEFAULT_GEOMETRY_RUN
    oriented_run_dir: Path = DEFAULT_ORIENTED_RUN
    out_run_dir: Path | None = None
    records: tuple[str, ...] = ()
    state_size: int = 64
    horizon_beats: int = 64
    cooldown_beats: int = 32
    baseline_beats: int = 500
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    seed: int = 7
    window_len: int = 256
    window_stride: int = 128
    epochs: int = 8
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    skew_preservation_weight: float = 0.05
    hermitian_penalty_weight: float = 0.05
    low_rank_penalty_weight: float = 0.01
    specificity_target: float = 0.95
    min_window_valid: int = 32
    device: str = "auto"
    max_records: int | None = None
    num_workers: int = 0


@dataclass
class RecordSequence:
    record: str
    frame: pd.DataFrame
    features: np.ndarray
    labels: np.ndarray
    loss_mask: np.ndarray
    eval_mask: np.ndarray
    st_entry: np.ndarray
    branch_label: str
    regime: str
    phenotype_target: str


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_saved_frame(base_path: Path) -> pd.DataFrame:
    parquet_path = base_path.with_suffix(".parquet")
    csv_path = base_path.with_suffix(".csv")
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing saved frame for {base_path}")


def save_frame(df: pd.DataFrame, base_path: Path) -> None:
    df.to_csv(base_path.with_suffix(".csv"), index=False)
    try:
        df.to_parquet(base_path.with_suffix(".parquet"), index=False)
    except Exception:
        pass


def default_out_run_dir(project_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("ltst_s4_effective_operator_probe_%Y%m%dT%H%M%SZ")
    out_dir = project_root / "artifact" / "runs" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def parse_records_arg(records_arg: str, available_records: list[str]) -> list[str]:
    if records_arg.strip().lower() in {"", "all", "*"}:
        return sorted(available_records)
    wanted = [piece.strip() for piece in records_arg.split(",") if piece.strip()]
    missing = sorted(set(wanted) - set(available_records))
    if missing:
        raise ValueError(f"Requested records are not present: {', '.join(missing)}")
    return wanted


def regime_to_phenotype(regime: str) -> str:
    return REGIME_TO_PHENOTYPE.get(str(regime), "rigid_orbit")


def inverse_softplus(value: float) -> float:
    if value <= 0.0:
        return -20.0
    return float(math.log(math.expm1(value)))


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def rank_one_vector(state_size: int) -> np.ndarray:
    idx = np.arange(state_size, dtype=float)
    return np.sqrt(2.0 * idx + 1.0)


def legs_skew_matrix(state_size: int) -> np.ndarray:
    p = rank_one_vector(state_size)
    outer = 0.5 * np.outer(p, p)
    return np.triu(outer, k=1) - np.tril(outer, k=-1)


def legs_dense_operator(
    state_size: int,
    real_shift: float = -0.5,
    skew_scale: float = 1.0,
    low_rank_scale: float = 1.0,
) -> np.ndarray:
    identity = np.eye(state_size, dtype=np.complex128)
    p = rank_one_vector(state_size).astype(np.complex128)
    skew = legs_skew_matrix(state_size).astype(np.complex128)
    low_rank = 0.5 * np.outer(p, p)
    return (real_shift * identity) + (skew_scale * skew) - (low_rank_scale * low_rank)


def bilinear_discretize_dense(
    continuous_a: np.ndarray,
    continuous_b: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    identity = np.eye(continuous_a.shape[0], dtype=np.complex128)
    lhs = identity - 0.5 * delta * continuous_a
    rhs = identity + 0.5 * delta * continuous_a
    discrete_a = np.linalg.solve(lhs, rhs)
    discrete_b = np.linalg.solve(lhs, delta * continuous_b)
    return discrete_a, discrete_b


def anti_hermitian_split(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    anti = 0.5 * (matrix - matrix.conj().T)
    herm = 0.5 * (matrix + matrix.conj().T)
    return anti, herm


def compute_global_operator_diagnostics(
    operator_matrix: np.ndarray,
    low_rank_matrix: np.ndarray | None = None,
) -> dict[str, float]:
    anti, herm = anti_hermitian_split(operator_matrix.astype(np.complex128))
    anti_sq = float(np.sum(np.abs(anti) ** 2).real)
    herm_sq = float(np.sum(np.abs(herm) ** 2).real)
    total = anti_sq + herm_sq
    eigvals = np.linalg.eigvals(operator_matrix)
    out = {
        "anti_norm": math.sqrt(max(anti_sq, 0.0)),
        "hermitian_norm": math.sqrt(max(herm_sq, 0.0)),
        "rho": anti_sq / (total + EPS),
        "spectral_radius": float(np.max(np.abs(eigvals))),
        "mean_real_eig": float(np.mean(eigvals.real)),
        "mean_abs_imag_eig": float(np.mean(np.abs(eigvals.imag))),
        "pythagorean_residual": abs(total - float(np.sum(np.abs(operator_matrix) ** 2).real)),
    }
    if low_rank_matrix is not None:
        out["low_rank_norm"] = float(np.linalg.norm(low_rank_matrix))
    return out


def compute_effective_operator_metrics(
    discrete_a: np.ndarray,
    discrete_a_base: np.ndarray,
    state: np.ndarray,
) -> dict[str, float]:
    if not np.all(np.isfinite(state.real)) or not np.all(np.isfinite(state.imag)):
        return {"operator_valid": 0.0}
    ax = discrete_a @ state
    low_rank_response = (discrete_a - discrete_a_base) @ state
    z = np.column_stack([state, ax, low_rank_response]).astype(np.complex128)
    column_norms = np.linalg.norm(z, axis=0)
    if np.min(column_norms) <= EPS:
        return {"operator_valid": 0.0}
    try:
        q, r = np.linalg.qr(z, mode="reduced")
    except np.linalg.LinAlgError:
        return {"operator_valid": 0.0}
    diag = np.abs(np.diag(r))
    if np.min(diag) <= 1e-8:
        return {"operator_valid": 0.0}
    a_eff = q.conj().T @ discrete_a @ q
    anti, herm = anti_hermitian_split(a_eff)
    anti_sq = float(np.sum(np.abs(anti) ** 2).real)
    herm_sq = float(np.sum(np.abs(herm) ** 2).real)
    total = anti_sq + herm_sq
    eigvals = np.linalg.eigvals(a_eff)
    low_rank_norm = float(np.linalg.norm(low_rank_response))
    ax_norm = float(np.linalg.norm(ax))
    state_norm = float(np.linalg.norm(state))
    low_rank_alignment = float(
        np.abs(np.vdot(state, low_rank_response)) / (state_norm * low_rank_norm + EPS)
    )
    return {
        "operator_valid": 1.0,
        "operator_anti_norm": math.sqrt(max(anti_sq, 0.0)),
        "operator_hermitian_norm": math.sqrt(max(herm_sq, 0.0)),
        "operator_rho": anti_sq / (total + EPS),
        "operator_low_rank_leverage": low_rank_norm / (ax_norm + EPS),
        "operator_low_rank_alignment": low_rank_alignment,
        "operator_mean_real_eig": float(np.mean(eigvals.real)),
        "operator_mean_abs_imag_eig": float(np.mean(np.abs(eigvals.imag))),
        "operator_spectral_radius": float(np.max(np.abs(eigvals))),
        "operator_pythagorean_residual": abs(total - float(np.sum(np.abs(a_eff) ** 2).real)),
        "operator_active_cond": float(np.max(diag) / (np.min(diag) + EPS)),
    }


def derive_imminent_onset_labels(
    st_event: np.ndarray,
    st_entry: np.ndarray,
    horizon_beats: int,
) -> np.ndarray:
    next_entry_distance = np.full(len(st_event), np.inf, dtype=float)
    next_idx = np.inf
    for idx in range(len(st_event) - 1, -1, -1):
        if bool(st_entry[idx]):
            next_idx = float(idx)
        if np.isfinite(next_idx):
            next_entry_distance[idx] = next_idx - float(idx)
    return (~st_event) & (next_entry_distance >= 1.0) & (next_entry_distance <= float(horizon_beats))


def build_entry_mask(df: pd.DataFrame) -> np.ndarray:
    if "st_entry" in df.columns:
        return df["st_entry"].fillna(False).astype(bool).to_numpy(dtype=bool)
    st_event = df["st_event"].fillna(False).astype(bool).to_numpy(dtype=bool)
    prev = np.concatenate(([False], st_event[:-1]))
    return st_event & (~prev)


def finite_feature_mask(df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    values = df.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    status_series = df["status"] if "status" in df.columns else pd.Series(0, index=df.index)
    status = pd.to_numeric(status_series, errors="coerce").fillna(1).to_numpy(dtype=int)
    return np.all(np.isfinite(values), axis=1) & (status == 0)


def assign_branch_label(df: pd.DataFrame, baseline_beats: int) -> tuple[str, dict[str, float]]:
    eval_df = df.loc[df["eval_mask"]].copy()
    baseline = eval_df.head(baseline_beats)
    onset = eval_df.loc[eval_df["label_imminent_onset"]]
    if baseline.empty or onset.empty:
        return "other", {
            "baseline_median_proj_volume_xsl": np.nan,
            "baseline_median_lie_orbit_norm": np.nan,
            "onset_median_proj_volume_xsl": np.nan,
            "onset_median_lie_orbit_norm": np.nan,
            "delta_proj_volume_xsl": np.nan,
            "delta_lie_orbit_norm": np.nan,
        }
    baseline_vol = float(baseline["proj_volume_xsl"].median())
    baseline_orbit = float(baseline["lie_orbit_norm"].median())
    onset_vol = float(onset["proj_volume_xsl"].median())
    onset_orbit = float(onset["lie_orbit_norm"].median())
    delta_vol = onset_vol - baseline_vol
    delta_orbit = onset_orbit - baseline_orbit
    if delta_vol < 0.0 and delta_orbit > 0.0:
        label = "dominant"
    elif delta_vol > 0.0 and delta_orbit < 0.0:
        label = "reverse"
    else:
        label = "other"
    return label, {
        "baseline_median_proj_volume_xsl": baseline_vol,
        "baseline_median_lie_orbit_norm": baseline_orbit,
        "onset_median_proj_volume_xsl": onset_vol,
        "onset_median_lie_orbit_norm": onset_orbit,
        "delta_proj_volume_xsl": delta_vol,
        "delta_lie_orbit_norm": delta_orbit,
    }


def build_record_sequence(
    frame: pd.DataFrame,
    feature_columns: list[str],
    horizon_beats: int,
    baseline_beats: int,
) -> tuple[RecordSequence, dict[str, Any]]:
    record = str(frame["record"].iloc[0])
    regime = str(frame["regime"].iloc[0]) if "regime" in frame.columns else "unknown"
    if "phenotype_target" in frame.columns:
        phenotype = str(frame["phenotype_target"].iloc[0])
    else:
        phenotype = regime_to_phenotype(regime)
    work = frame.copy()
    st_event = work["st_event"].fillna(False).astype(bool).to_numpy(dtype=bool)
    st_entry = build_entry_mask(work)
    labels = derive_imminent_onset_labels(st_event=st_event, st_entry=st_entry, horizon_beats=horizon_beats)
    loss_mask = finite_feature_mask(work, feature_columns) & (~st_event)
    eval_mask = loss_mask.copy()
    work["st_entry"] = st_entry
    work["label_imminent_onset"] = labels
    work["loss_mask"] = loss_mask
    work["eval_mask"] = eval_mask
    features = work.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    branch_label, branch_summary = assign_branch_label(work, baseline_beats=baseline_beats)
    sequence = RecordSequence(
        record=record,
        frame=work,
        features=features,
        labels=labels.astype(bool),
        loss_mask=loss_mask.astype(bool),
        eval_mask=eval_mask.astype(bool),
        st_entry=st_entry.astype(bool),
        branch_label=branch_label,
        regime=regime,
        phenotype_target=phenotype,
    )
    summary = {
        "record": record,
        "regime": regime,
        "phenotype_target": phenotype,
        "branch_label": branch_label,
        "n_total": int(len(work)),
        "n_eval": int(np.sum(eval_mask)),
        "n_positive": int(np.sum(labels & eval_mask)),
        "n_events": int(np.sum(st_entry)),
        **branch_summary,
    }
    return sequence, summary


def make_record_splits(
    records_df: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for _, group in records_df.groupby("regime", sort=True):
        names = sorted(group["record"].astype(str).tolist())
        rng.shuffle(names)
        n_total = len(names)
        n_train = max(1, int(round(n_total * train_fraction)))
        n_val = int(round(n_total * val_fraction))
        n_test = n_total - n_train - n_val
        if n_total >= 3:
            if n_val == 0:
                n_val = 1
                n_train = max(1, n_train - 1)
            if n_test == 0:
                n_test = 1
                n_train = max(1, n_train - 1)
        else:
            n_val = 0
            n_test = max(1, n_total - n_train)
        while n_train + n_val + n_test > n_total:
            if n_train > 1:
                n_train -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                n_test -= 1
        train.extend(names[:n_train])
        val.extend(names[n_train : n_train + n_val])
        test.extend(names[n_train + n_val : n_train + n_val + n_test])
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def fit_standardizer(records: list[RecordSequence]) -> Standardizer:
    collected: list[np.ndarray] = []
    for record in records:
        if np.any(record.loss_mask):
            collected.append(record.features[record.loss_mask])
    if not collected:
        raise ValueError("No valid training rows available to fit the feature standardizer.")
    stacked = np.vstack(collected)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return Standardizer(mean=mean, std=std)


def apply_standardizer(records: list[RecordSequence], standardizer: Standardizer) -> list[RecordSequence]:
    transformed: list[RecordSequence] = []
    for record in records:
        scaled = standardizer.transform(record.features)
        transformed.append(
            RecordSequence(
                record=record.record,
                frame=record.frame.copy(),
                features=scaled.astype(np.float32),
                labels=record.labels.copy(),
                loss_mask=record.loss_mask.copy(),
                eval_mask=record.eval_mask.copy(),
                st_entry=record.st_entry.copy(),
                branch_label=record.branch_label,
                regime=record.regime,
                phenotype_target=record.phenotype_target,
            )
        )
    return transformed


class SequenceWindowDataset(Dataset):
    def __init__(
        self,
        records: list[RecordSequence],
        window_len: int,
        stride: int,
        min_window_valid: int,
    ) -> None:
        self.records = list(records)
        self.window_len = int(window_len)
        self.windows: list[tuple[int, int, int]] = []
        for rec_idx, record in enumerate(self.records):
            total = len(record.features)
            if total == 0:
                continue
            if total <= self.window_len:
                candidates = [(0, total)]
            else:
                starts = list(range(0, total - self.window_len + 1, stride))
                last_start = total - self.window_len
                if starts[-1] != last_start:
                    starts.append(last_start)
                candidates = [(start, start + self.window_len) for start in starts]
            for start, stop in candidates:
                if int(np.sum(record.loss_mask[start:stop])) >= int(min_window_valid):
                    self.windows.append((rec_idx, start, stop))
        if not self.windows:
            raise ValueError("No sequence windows met the minimum valid-row requirement.")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec_idx, start, stop = self.windows[idx]
        record = self.records[rec_idx]
        features = record.features[start:stop].astype(np.float32)
        labels = record.labels[start:stop].astype(np.float32)
        mask = record.loss_mask[start:stop].astype(np.float32)
        if len(features) < self.window_len:
            pad = self.window_len - len(features)
            features = np.pad(features, ((0, pad), (0, 0)), mode="constant", constant_values=0.0)
            labels = np.pad(labels, (0, pad), mode="constant", constant_values=0.0)
            mask = np.pad(mask, (0, pad), mode="constant", constant_values=0.0)
        return {
            "features": features,
            "labels": labels,
            "mask": mask,
        }


if nn is not None:
    class MinimalLegSProbe(nn.Module):
        def __init__(self, input_dim: int, state_size: int) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.state_size = int(state_size)
            self.input_encoder = nn.Linear(self.input_dim, 1)
            self.head = nn.Linear((2 * self.state_size) + 1, 1)
            p = torch.tensor(rank_one_vector(self.state_size), dtype=torch.float32)
            skew = torch.tensor(legs_skew_matrix(self.state_size), dtype=torch.float32)
            self.register_buffer("rank_one_p", p)
            self.register_buffer("skew_template", skew)
            self.register_buffer("identity", torch.eye(self.state_size, dtype=torch.complex64))
            self.real_shift_log = nn.Parameter(torch.tensor(inverse_softplus(0.5), dtype=torch.float32))
            self.skew_scale_log = nn.Parameter(torch.tensor(inverse_softplus(1.0), dtype=torch.float32))
            self.low_rank_scale_log = nn.Parameter(torch.tensor(inverse_softplus(1.0), dtype=torch.float32))
            self.delta_log = nn.Parameter(torch.tensor(inverse_softplus(0.1), dtype=torch.float32))
            b_init = p / torch.linalg.norm(p)
            self.b_real = nn.Parameter(b_init.clone())
            self.b_imag = nn.Parameter(torch.zeros_like(b_init))

        def _continuous_components(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            real_shift = -F.softplus(self.real_shift_log)
            skew_scale = F.softplus(self.skew_scale_log)
            low_rank_scale = F.softplus(self.low_rank_scale_log)
            delta = F.softplus(self.delta_log) + 1e-4
            return real_shift, skew_scale, low_rank_scale, delta

        def continuous_operator(self, include_low_rank: bool = True) -> torch.Tensor:
            real_shift, skew_scale, low_rank_scale, _ = self._continuous_components()
            low_rank = self.low_rank_matrix().to(torch.complex64)
            skew = self.skew_template.to(torch.complex64)
            a_cont = (real_shift.to(torch.complex64) * self.identity) + (skew_scale.to(torch.complex64) * skew)
            if include_low_rank:
                a_cont = a_cont - (low_rank_scale.to(torch.complex64) * low_rank)
            return a_cont

        def low_rank_matrix(self) -> torch.Tensor:
            return 0.5 * torch.outer(self.rank_one_p, self.rank_one_p)

        def global_operator_regularization_metrics(self) -> dict[str, torch.Tensor]:
            real_shift, skew_scale, low_rank_scale, delta = self._continuous_components()
            a_cont = self.continuous_operator(include_low_rank=True)
            anti = 0.5 * (a_cont - a_cont.conj().transpose(0, 1))
            herm = 0.5 * (a_cont + a_cont.conj().transpose(0, 1))
            anti_sq = torch.sum(torch.abs(anti) ** 2).real
            herm_sq = torch.sum(torch.abs(herm) ** 2).real
            total_sq = anti_sq + herm_sq
            low_rank = low_rank_scale * self.low_rank_matrix()
            low_rank_sq = torch.sum(torch.abs(low_rank.to(torch.complex64)) ** 2).real
            return {
                "anti_sq": anti_sq,
                "hermitian_sq": herm_sq,
                "total_sq": total_sq,
                "rho": anti_sq / total_sq.clamp_min(EPS),
                "hermitian_ratio": herm_sq / total_sq.clamp_min(EPS),
                "low_rank_ratio": low_rank_sq / total_sq.clamp_min(EPS),
                "low_rank_sq": low_rank_sq,
                "real_shift": real_shift,
                "skew_scale": skew_scale,
                "low_rank_scale": low_rank_scale,
                "delta": delta,
            }

        def global_operator_diagnostics(self) -> dict[str, float]:
            with torch.no_grad():
                metrics = self.global_operator_regularization_metrics()
                a_cont = self.continuous_operator(include_low_rank=True).detach().cpu().numpy().astype(np.complex128)
                low_rank = (
                    metrics["low_rank_scale"].detach().cpu().item()
                    * self.low_rank_matrix().detach().cpu().numpy().astype(np.float64)
                )
                out = compute_global_operator_diagnostics(operator_matrix=a_cont, low_rank_matrix=low_rank)
                out.update(
                    {
                        "real_shift": float(metrics["real_shift"].detach().cpu().item()),
                        "skew_scale": float(metrics["skew_scale"].detach().cpu().item()),
                        "low_rank_scale": float(metrics["low_rank_scale"].detach().cpu().item()),
                        "delta": float(metrics["delta"].detach().cpu().item()),
                    }
                )
                return out

        def input_vector(self) -> torch.Tensor:
            return torch.complex(self.b_real, self.b_imag)

        def discretized_operators(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            _, _, _, delta = self._continuous_components()
            delta_complex = delta.to(torch.complex64)
            b_vec = self.input_vector()
            a_full = self.continuous_operator(include_low_rank=True)
            a_base = self.continuous_operator(include_low_rank=False)
            lhs_full = self.identity - (0.5 * delta_complex * a_full)
            rhs_full = self.identity + (0.5 * delta_complex * a_full)
            a_bar = torch.linalg.solve(lhs_full, rhs_full)
            b_bar = torch.linalg.solve(lhs_full, delta_complex * b_vec)
            lhs_base = self.identity - (0.5 * delta_complex * a_base)
            rhs_base = self.identity + (0.5 * delta_complex * a_base)
            a_bar_base = torch.linalg.solve(lhs_base, rhs_base)
            return a_bar, b_bar, a_bar_base

        def forward(self, features: torch.Tensor, return_states: bool = False) -> dict[str, torch.Tensor]:
            batch, seq_len, _ = features.shape
            a_bar, b_bar, _ = self.discretized_operators()
            state = torch.zeros(batch, self.state_size, dtype=torch.complex64, device=features.device)
            logits: list[torch.Tensor] = []
            states: list[torch.Tensor] = []
            a_bar_t = a_bar.transpose(0, 1)
            for step in range(seq_len):
                u = self.input_encoder(features[:, step, :]).squeeze(-1)
                state = torch.matmul(state, a_bar_t) + (u.to(torch.complex64).unsqueeze(-1) * b_bar.unsqueeze(0))
                if return_states:
                    states.append(state)
                head_features = torch.cat([state.real, state.imag, u.unsqueeze(-1)], dim=-1)
                logits.append(self.head(head_features).squeeze(-1))
            out = {"logits": torch.stack(logits, dim=1)}
            if return_states:
                out["states"] = torch.stack(states, dim=1)
            return out


else:  # pragma: no cover
    class MinimalLegSProbe:  # type: ignore[override]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("PyTorch is required to instantiate the LTST S4 effective-operator probe.")


def require_torch() -> None:
    if torch is None or nn is None or F is None or DataLoader is None:
        raise RuntimeError(
            "PyTorch is required for the LTST S4 effective-operator probe. "
            "Install torch in the active Python environment before running this script."
        )


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def collate_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    features = torch.from_numpy(np.stack([item["features"] for item in batch], axis=0))
    labels = torch.from_numpy(np.stack([item["labels"] for item in batch], axis=0))
    mask = torch.from_numpy(np.stack([item["mask"] for item in batch], axis=0))
    return {"features": features, "labels": labels, "mask": mask}


def masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=pos_weight)
    weighted = loss * mask
    return weighted.sum() / mask.sum().clamp_min(1.0)


def auc_roc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(bool)
    s = scores.astype(float)
    finite = np.isfinite(s)
    y = y[finite]
    s = s[finite]
    n_pos = int(np.sum(y))
    n_neg = int(np.sum(~y))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy(dtype=float)
    pos_sum = float(ranks[y].sum())
    return (pos_sum - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(bool)
    s = scores.astype(float)
    finite = np.isfinite(s)
    y = y[finite]
    s = s[finite]
    n_pos = int(np.sum(y))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted.astype(float))
    fp = np.cumsum((~y_sorted).astype(float))
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / float(n_pos)
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def classification_summary(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(scores)
    y = y_true.astype(bool)[finite]
    s = scores.astype(float)[finite]
    if len(y) == 0:
        return {"n_rows": 0.0, "positive_fraction": np.nan, "auroc": np.nan, "pr_auc": np.nan}
    return {
        "n_rows": float(len(y)),
        "positive_fraction": float(np.mean(y)),
        "auroc": auc_roc(y, s),
        "pr_auc": pr_auc(y, s),
    }


def choose_score_direction(labels: np.ndarray, scores: np.ndarray) -> tuple[str, float]:
    auc_higher = auc_roc(labels, scores)
    auc_lower = auc_roc(labels, -scores)
    if np.isnan(auc_higher) and np.isnan(auc_lower):
        return "higher", float("nan")
    if np.isnan(auc_lower) or (not np.isnan(auc_higher) and auc_higher >= auc_lower):
        return "higher", auc_higher
    return "lower", auc_lower


def apply_episode_cooldown(raw_signal: np.ndarray, cooldown_beats: int) -> np.ndarray:
    signal = raw_signal.astype(bool)
    filtered = np.zeros_like(signal, dtype=bool)
    active = False
    cooldown_remaining = 0
    for idx, flag in enumerate(signal):
        if active:
            filtered[idx] = True
            if not flag:
                active = False
                cooldown_remaining = int(cooldown_beats)
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue
        if flag:
            active = True
            filtered[idx] = True
    return filtered


def evaluate_thresholded_detector(
    pred_df: pd.DataFrame,
    score_col: str,
    threshold: float,
    direction: str,
    cooldown_beats: int,
    horizon_beats: int,
) -> dict[str, float]:
    tp = fp = tn = fn = 0
    visible_events = 0
    total_events = 0
    all_alerts: list[np.ndarray] = []
    for _, group in pred_df.groupby("record", sort=False):
        finite_scores = pd.to_numeric(group[score_col], errors="coerce").to_numpy(dtype=float)
        eval_mask = group["eval_mask"].astype(bool).to_numpy(dtype=bool)
        labels = group["label_imminent_onset"].astype(bool).to_numpy(dtype=bool)
        raw = np.zeros(len(group), dtype=bool)
        finite_mask = eval_mask & np.isfinite(finite_scores)
        if direction == "lower":
            raw[finite_mask] = finite_scores[finite_mask] <= threshold
        else:
            raw[finite_mask] = finite_scores[finite_mask] >= threshold
        alert = apply_episode_cooldown(raw_signal=raw, cooldown_beats=cooldown_beats)
        active = alert & eval_mask
        tp += int(np.sum(active & labels))
        fp += int(np.sum(active & (~labels)))
        tn += int(np.sum((~active) & (~labels) & eval_mask))
        fn += int(np.sum((~active) & labels))
        st_entry = group["st_entry"].astype(bool).to_numpy(dtype=bool)
        event_indices = np.flatnonzero(st_entry)
        total_events += len(event_indices)
        for idx in event_indices:
            lo = max(0, idx - int(horizon_beats))
            if np.any(active[lo:idx]):
                visible_events += 1
        all_alerts.append(active[eval_mask])
    specificity = tn / max(tn + fp, 1)
    sensitivity = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    occupancy = float(np.mean(np.concatenate(all_alerts))) if all_alerts else np.nan
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "precision": float(precision),
        "alert_occupancy_fraction": float(occupancy),
        "visible_events": float(visible_events),
        "total_events": float(total_events),
        "event_visibility_fraction": float(visible_events / total_events) if total_events else np.nan,
    }


def candidate_thresholds(scores: np.ndarray, direction: str) -> np.ndarray:
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.asarray([], dtype=float)
    quantiles = np.linspace(0.05, 0.95, 37)
    values = np.quantile(finite, quantiles)
    values = np.unique(np.concatenate(([finite.min()], values, [finite.max()])))
    return np.sort(values) if direction == "lower" else np.sort(values)[::-1]


def select_threshold_from_validation(
    pred_df: pd.DataFrame,
    score_col: str,
    direction: str,
    cooldown_beats: int,
    horizon_beats: int,
    specificity_target: float,
) -> dict[str, float]:
    thresholds = candidate_thresholds(pd.to_numeric(pred_df[score_col], errors="coerce").to_numpy(dtype=float), direction)
    best: dict[str, float] | None = None
    relaxed_best: dict[str, float] | None = None
    for threshold in thresholds:
        metrics = evaluate_thresholded_detector(
            pred_df=pred_df,
            score_col=score_col,
            threshold=float(threshold),
            direction=direction,
            cooldown_beats=cooldown_beats,
            horizon_beats=horizon_beats,
        )
        candidate = {"threshold": float(threshold), **metrics}
        if metrics["specificity"] >= specificity_target:
            if best is None or (
                candidate["sensitivity"] > best["sensitivity"]
                or (
                    math.isclose(candidate["sensitivity"], best["sensitivity"], abs_tol=1e-9)
                    and candidate["alert_occupancy_fraction"] < best["alert_occupancy_fraction"]
                )
            ):
                best = candidate
        if metrics["specificity"] >= 0.90:
            if relaxed_best is None or (
                candidate["sensitivity"] > relaxed_best["sensitivity"]
                or (
                    math.isclose(candidate["sensitivity"], relaxed_best["sensitivity"], abs_tol=1e-9)
                    and candidate["alert_occupancy_fraction"] < relaxed_best["alert_occupancy_fraction"]
                )
            ):
                relaxed_best = candidate
    selected = best or relaxed_best
    if selected is None:
        selected = {
            "threshold": np.nan,
            **evaluate_thresholded_detector(
                pred_df=pred_df,
                score_col=score_col,
                threshold=np.nan,
                direction=direction,
                cooldown_beats=cooldown_beats,
                horizon_beats=horizon_beats,
            ),
        }
    selected["direction"] = direction
    return selected


def build_onset_enrichment(pred_df: pd.DataFrame, channels: tuple[str, ...]) -> pd.DataFrame:
    work = pred_df.loc[pred_df["eval_mask"]].copy()
    rows: list[dict[str, Any]] = []
    labels = work["label_imminent_onset"].astype(bool).to_numpy(dtype=bool)
    for channel in channels:
        scores = pd.to_numeric(work[channel], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(scores)
        if not np.any(finite):
            continue
        y = labels[finite]
        s = scores[finite]
        pos = s[y]
        neg = s[~y]
        neg_std = float(np.std(neg)) if len(neg) else np.nan
        onset_z = (float(np.mean(pos)) - float(np.mean(neg))) / (neg_std + EPS) if len(pos) and len(neg) else np.nan
        direction, directed_auroc = choose_score_direction(y, s)
        rows.append(
            {
                "channel": channel,
                "n_positive": float(np.sum(y)),
                "n_negative": float(np.sum(~y)),
                "positive_mean": float(np.mean(pos)) if len(pos) else np.nan,
                "negative_mean": float(np.mean(neg)) if len(neg) else np.nan,
                "negative_std": neg_std,
                "onset_enrichment_z": onset_z,
                "direction": direction,
                "auroc": directed_auroc,
                "pr_auc": pr_auc(y, s if direction == "higher" else -s),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["auroc", "onset_enrichment_z"], ascending=[False, False]).reset_index(drop=True)


def build_branch_enrichment(pred_df: pd.DataFrame, channels: tuple[str, ...]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for branch, group in pred_df.groupby("branch_label", sort=False):
        subset = build_onset_enrichment(group, channels)
        if subset.empty:
            continue
        subset.insert(0, "branch_label", branch)
        rows.append(subset)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_detector_comparison(
    val_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
    channels: tuple[str, ...],
    cooldown_beats: int,
    horizon_beats: int,
    specificity_target: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for channel in channels:
        val_scores = pd.to_numeric(val_pred[channel], errors="coerce").to_numpy(dtype=float)
        val_labels = val_pred["label_imminent_onset"].astype(bool).to_numpy(dtype=bool)
        direction, val_auroc = choose_score_direction(val_labels, val_scores)
        threshold_row = select_threshold_from_validation(
            pred_df=val_pred,
            score_col=channel,
            direction=direction,
            cooldown_beats=cooldown_beats,
            horizon_beats=horizon_beats,
            specificity_target=specificity_target,
        )
        test_metrics = evaluate_thresholded_detector(
            pred_df=test_pred,
            score_col=channel,
            threshold=float(threshold_row["threshold"]),
            direction=direction,
            cooldown_beats=cooldown_beats,
            horizon_beats=horizon_beats,
        )
        test_labels = test_pred["label_imminent_onset"].astype(bool).to_numpy(dtype=bool)
        test_scores = pd.to_numeric(test_pred[channel], errors="coerce").to_numpy(dtype=float)
        test_summary = classification_summary(test_labels, test_scores if direction == "higher" else -test_scores)
        rows.append(
            {
                "channel": channel,
                "direction": direction,
                "val_auroc": val_auroc,
                "val_threshold": threshold_row["threshold"],
                "test_auroc": test_summary["auroc"],
                "test_pr_auc": test_summary["pr_auc"],
                **test_metrics,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["specificity", "alert_occupancy_fraction", "test_auroc"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def load_historical_baseline(geometry_run_dir: Path) -> dict[str, Any]:
    benchmark_path = geometry_run_dir / "memory_taxonomy_v2" / "memory_taxonomy_benchmarks.csv"
    if not benchmark_path.exists():
        return {
            "historical_metric": DEFAULT_HISTORICAL_BASELINE_METRIC,
            "historical_auroc": DEFAULT_HISTORICAL_BASELINE_AUROC,
        }
    benchmarks = pd.read_csv(benchmark_path)
    row = benchmarks.loc[
        (benchmarks["task"].astype(str) == "loose_vs_rest")
        & (benchmarks["metric"].astype(str) == DEFAULT_HISTORICAL_BASELINE_METRIC)
    ]
    if row.empty:
        return {
            "historical_metric": DEFAULT_HISTORICAL_BASELINE_METRIC,
            "historical_auroc": DEFAULT_HISTORICAL_BASELINE_AUROC,
        }
    return {"historical_metric": str(row["metric"].iloc[0]), "historical_auroc": float(row["auroc"].iloc[0])}


def write_report(
    out_dir: Path,
    cfg: ProbeConfig,
    selected_features: list[str],
    split_summary: pd.DataFrame,
    model_summary: dict[str, Any],
    pooled_enrichment: pd.DataFrame,
    branch_enrichment: pd.DataFrame,
    detector_comparison: pd.DataFrame,
    historical_baseline: dict[str, Any],
    global_operator_df: pd.DataFrame,
) -> None:
    lines = [
        "# LTST S4 Effective-Operator Probe",
        "",
        "## Summary",
        "",
        f"- Geometry run: `{cfg.geometry_run_dir}`",
        f"- Oriented run: `{cfg.oriented_run_dir}`",
        f"- Selected geometry features: `{len(selected_features)}`",
        f"- State size: `{cfg.state_size}`",
        f"- Imminent-onset horizon: `{cfg.horizon_beats}` beats",
        f"- Cooldown for detector sweeps: `{cfg.cooldown_beats}` beats",
        (
            f"- Regularization weights: skew `{cfg.skew_preservation_weight}`, "
            f"Hermitian `{cfg.hermitian_penalty_weight}`, low-rank `{cfg.low_rank_penalty_weight}`"
        ),
        "",
        "## Record Splits",
        "",
        split_summary.to_markdown(index=False) if not split_summary.empty else "_No split summary available._",
        "",
        "## Model Competence",
        "",
        f"- Validation AUROC: `{model_summary['val_auroc']:.4f}`",
        f"- Validation PR-AUC: `{model_summary['val_pr_auc']:.4f}`",
        f"- Test AUROC: `{model_summary['test_auroc']:.4f}`",
        f"- Test PR-AUC: `{model_summary['test_pr_auc']:.4f}`",
        f"- Best epoch: `{int(model_summary['best_epoch'])}`",
        "",
        "## Global Operator Diagnostics",
        "",
        global_operator_df.to_markdown(index=False) if not global_operator_df.empty else "_No global operator diagnostics available._",
        "",
        "## Pooled Operator-Channel Onset Enrichment",
        "",
        pooled_enrichment.to_markdown(index=False) if not pooled_enrichment.empty else "_No pooled onset enrichment rows available._",
        "",
        "## Branch-Stratified Operator Behavior",
        "",
        branch_enrichment.to_markdown(index=False) if not branch_enrichment.empty else "_No branch-stratified rows available._",
        "",
        "## Detector Comparison",
        "",
        detector_comparison.to_markdown(index=False) if not detector_comparison.empty else "_No detector comparison rows available._",
        "",
        "## Historical Baseline Reference",
        "",
        (
            f"- Historical direct-geometry reference: `{historical_baseline['historical_metric']}` "
            f"with AUROC `{historical_baseline['historical_auroc']:.4f}` "
            "(published loose-vs-rest baseline, retained here as context rather than a matched onset task)."
        ),
    ]
    (out_dir / "ltst_s4_effective_operator_report.md").write_text("\n".join(lines), encoding="utf-8")


def infer_records(
    model: MinimalLegSProbe,
    records: list[RecordSequence],
    device: str,
    compute_operator: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        a_bar_t, _, a_bar_base_t = model.discretized_operators()
        a_bar = a_bar_t.detach().cpu().numpy().astype(np.complex128)
        a_bar_base = a_bar_base_t.detach().cpu().numpy().astype(np.complex128)
        for record in records:
            inputs = torch.from_numpy(record.features).unsqueeze(0).to(device=device, dtype=torch.float32)
            outputs = model(inputs, return_states=compute_operator)
            logits = outputs["logits"][0].detach().cpu().numpy().astype(np.float64)
            probs = sigmoid_np(logits)
            states = None
            if compute_operator:
                states = outputs["states"][0].detach().cpu().numpy().astype(np.complex128)
            for idx, (_, source_row) in enumerate(record.frame.iterrows()):
                item = {
                    "record": record.record,
                    "row_index": int(idx),
                    "beat_sample": int(source_row["beat_sample"]),
                    "label_imminent_onset": bool(record.labels[idx]),
                    "loss_mask": bool(record.loss_mask[idx]),
                    "eval_mask": bool(record.eval_mask[idx]),
                    "st_entry": bool(record.st_entry[idx]),
                    "regime": record.regime,
                    "phenotype_target": record.phenotype_target,
                    "branch_label": record.branch_label,
                    "probe_logit": float(logits[idx]),
                    "probe_prob": float(probs[idx]),
                }
                for channel in DIRECT_COMPARISON_COLUMNS:
                    if channel in item:
                        continue
                    if channel in source_row.index:
                        item[channel] = float(source_row[channel]) if pd.notna(source_row[channel]) else np.nan
                if compute_operator and states is not None and record.eval_mask[idx]:
                    item.update(compute_effective_operator_metrics(discrete_a=a_bar, discrete_a_base=a_bar_base, state=states[idx]))
                else:
                    item.update({name: np.nan for name in OPERATOR_CHANNEL_COLUMNS})
                    item["operator_valid"] = 0.0
                    item["operator_pythagorean_residual"] = np.nan
                    item["operator_active_cond"] = np.nan
                rows.append(item)
    return pd.DataFrame(rows)


def train_probe(
    cfg: ProbeConfig,
    train_records: list[RecordSequence],
    val_records: list[RecordSequence],
    feature_dim: int,
    device: str,
) -> tuple[MinimalLegSProbe, dict[str, Any], pd.DataFrame]:
    require_torch()
    train_ds = SequenceWindowDataset(
        records=train_records,
        window_len=cfg.window_len,
        stride=cfg.window_stride,
        min_window_valid=cfg.min_window_valid,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_windows,
    )
    model = MinimalLegSProbe(input_dim=feature_dim, state_size=cfg.state_size).to(device)
    init_global_diagnostics = model.global_operator_diagnostics()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    pos_count = float(sum(int(np.sum(record.labels & record.loss_mask)) for record in train_records))
    neg_count = float(sum(int(np.sum((~record.labels) & record.loss_mask)) for record in train_records))
    pos_weight_value = float(max(neg_count / max(pos_count, 1.0), 1.0))
    pos_weight = torch.tensor(pos_weight_value, device=device, dtype=torch.float32)
    history_rows: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_score = -np.inf
    best_epoch = -1

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_base_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_rho = 0.0
        batch_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            features = batch["features"].to(device=device, dtype=torch.float32)
            labels = batch["labels"].to(device=device, dtype=torch.float32)
            mask = batch["mask"].to(device=device, dtype=torch.float32)
            outputs = model(features)
            base_loss = masked_bce_loss(outputs["logits"], labels, mask, pos_weight=pos_weight)
            reg_metrics = model.global_operator_regularization_metrics()
            reg_loss = (
                (cfg.skew_preservation_weight * (1.0 - reg_metrics["rho"]))
                + (cfg.hermitian_penalty_weight * reg_metrics["hermitian_ratio"])
                + (cfg.low_rank_penalty_weight * reg_metrics["low_rank_ratio"])
            )
            loss = base_loss + reg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            epoch_base_loss += float(base_loss.detach().cpu().item())
            epoch_reg_loss += float(reg_loss.detach().cpu().item())
            epoch_rho += float(reg_metrics["rho"].detach().cpu().item())
            batch_count += 1
        val_pred = infer_records(model=model, records=val_records, device=device, compute_operator=True)
        val_eval = val_pred.loc[val_pred["eval_mask"]].copy()
        val_scores = pd.to_numeric(val_eval["probe_prob"], errors="coerce").to_numpy(dtype=float)
        val_labels = val_eval["label_imminent_onset"].astype(bool).to_numpy(dtype=bool)
        val_metrics = classification_summary(val_labels, val_scores)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(batch_count, 1),
                "train_base_loss": epoch_base_loss / max(batch_count, 1),
                "train_reg_loss": epoch_reg_loss / max(batch_count, 1),
                "train_global_rho": epoch_rho / max(batch_count, 1),
                "val_auroc": val_metrics["auroc"],
                "val_pr_auc": val_metrics["pr_auc"],
            }
        )
        score = float(val_metrics["auroc"]) if np.isfinite(val_metrics["auroc"]) else -np.inf
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")
    model.load_state_dict(best_state)
    final_val_pred = infer_records(model=model, records=val_records, device=device, compute_operator=True)
    final_val_eval = final_val_pred.loc[final_val_pred["eval_mask"]].copy()
    final_scores = pd.to_numeric(final_val_eval["probe_prob"], errors="coerce").to_numpy(dtype=float)
    final_labels = final_val_eval["label_imminent_onset"].astype(bool).to_numpy(dtype=bool)
    final_metrics = classification_summary(final_labels, final_scores)
    summary = {
        "best_epoch": best_epoch,
        "pos_weight": pos_weight_value,
        "val_auroc": final_metrics["auroc"],
        "val_pr_auc": final_metrics["pr_auc"],
        "global_operator_init": init_global_diagnostics,
        "global_operator_learned": model.global_operator_diagnostics(),
    }
    return model, summary, pd.DataFrame(history_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a minimal LTST S4-LegS onset probe on the canonical geometry stream and export active-subspace operator metrics."
    )
    parser.add_argument("--geometry-run-dir", type=Path, default=DEFAULT_GEOMETRY_RUN)
    parser.add_argument("--oriented-run-dir", type=Path, default=DEFAULT_ORIENTED_RUN)
    parser.add_argument("--out-run-dir", type=Path, default=None)
    parser.add_argument("--records", default="all")
    parser.add_argument("--state-size", type=int, default=64)
    parser.add_argument("--horizon-beats", type=int, default=64)
    parser.add_argument("--cooldown-beats", type=int, default=32)
    parser.add_argument("--baseline-beats", type=int, default=500)
    parser.add_argument("--window-len", type=int, default=256)
    parser.add_argument("--window-stride", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--skew-preservation-weight", type=float, default=0.05)
    parser.add_argument("--hermitian-penalty-weight", type=float, default=0.05)
    parser.add_argument("--low-rank-penalty-weight", type=float, default=0.01)
    parser.add_argument("--specificity-target", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    require_torch()
    cfg = ProbeConfig(
        geometry_run_dir=args.geometry_run_dir.resolve(),
        oriented_run_dir=args.oriented_run_dir.resolve(),
        out_run_dir=args.out_run_dir.resolve() if args.out_run_dir else None,
        state_size=args.state_size,
        horizon_beats=args.horizon_beats,
        cooldown_beats=args.cooldown_beats,
        baseline_beats=args.baseline_beats,
        window_len=args.window_len,
        window_stride=args.window_stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        skew_preservation_weight=args.skew_preservation_weight,
        hermitian_penalty_weight=args.hermitian_penalty_weight,
        low_rank_penalty_weight=args.low_rank_penalty_weight,
        specificity_target=args.specificity_target,
        seed=args.seed,
        device=args.device,
        max_records=args.max_records,
        num_workers=args.num_workers,
    )
    set_seeds(cfg.seed)
    out_dir = cfg.out_run_dir or default_out_run_dir(REPO_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    regime_df = pd.read_csv(cfg.oriented_run_dir / "regime.csv")
    available_records = sorted(regime_df["record"].astype(str).tolist())
    selected_records = parse_records_arg(args.records, available_records)
    if cfg.max_records is not None:
        selected_records = selected_records[: int(cfg.max_records)]

    sample_df = read_saved_frame(cfg.oriented_run_dir / "beat_level" / selected_records[0])
    selected_features = [name for name in CANONICAL_FEATURE_COLUMNS if name in sample_df.columns]
    if not selected_features:
        raise RuntimeError("No canonical geometry feature columns were found in the oriented beat-level run.")

    records: list[RecordSequence] = []
    record_rows: list[dict[str, Any]] = []
    for record in selected_records:
        frame = read_saved_frame(cfg.oriented_run_dir / "beat_level" / record)
        sequence, summary = build_record_sequence(
            frame=frame,
            feature_columns=selected_features,
            horizon_beats=cfg.horizon_beats,
            baseline_beats=cfg.baseline_beats,
        )
        records.append(sequence)
        record_rows.append(summary)
    record_summary = pd.DataFrame(record_rows).sort_values("record").reset_index(drop=True)
    split_map = make_record_splits(
        records_df=record_summary[["record", "regime"]].drop_duplicates(),
        train_fraction=cfg.train_fraction,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )
    split_lookup = {record: split for split, names in split_map.items() for record in names}
    record_summary["split"] = record_summary["record"].map(split_lookup).fillna("unused")
    record_summary.to_csv(out_dir / "ltst_s4_probe_record_summary.csv", index=False)

    split_summary = (
        record_summary.groupby(["split", "regime", "branch_label"], dropna=False)
        .agg(records=("record", "nunique"), positives=("n_positive", "sum"), eval_rows=("n_eval", "sum"))
        .reset_index()
    )
    split_summary.to_csv(out_dir / "ltst_s4_probe_split_summary.csv", index=False)

    record_lookup = {record.record: record for record in records}
    train_records = [record_lookup[name] for name in split_map["train"]]
    val_records = [record_lookup[name] for name in split_map["val"]]
    test_records = [record_lookup[name] for name in split_map["test"]]
    standardizer = fit_standardizer(train_records)
    scaled_lookup = {record.record: record for record in apply_standardizer(records, standardizer)}
    train_records_scaled = [scaled_lookup[name] for name in split_map["train"]]
    val_records_scaled = [scaled_lookup[name] for name in split_map["val"]]
    test_records_scaled = [scaled_lookup[name] for name in split_map["test"]]

    device = resolve_device(cfg.device)
    model, model_summary, history_df = train_probe(
        cfg=cfg,
        train_records=train_records_scaled,
        val_records=val_records_scaled,
        feature_dim=len(selected_features),
        device=device,
    )
    history_df.to_csv(out_dir / "ltst_s4_probe_training_history.csv", index=False)
    torch.save(model.state_dict(), out_dir / "ltst_s4_probe_state_dict.pt")

    val_pred = infer_records(model=model, records=val_records_scaled, device=device, compute_operator=True)
    test_pred = infer_records(model=model, records=test_records_scaled, device=device, compute_operator=True)
    save_frame(val_pred, out_dir / "ltst_s4_probe_val_predictions")
    save_frame(test_pred, out_dir / "ltst_s4_probe_test_predictions")

    val_eval = val_pred.loc[val_pred["eval_mask"]].copy()
    test_eval = test_pred.loc[test_pred["eval_mask"]].copy()
    test_summary = classification_summary(
        test_eval["label_imminent_onset"].astype(bool).to_numpy(dtype=bool),
        pd.to_numeric(test_eval["probe_prob"], errors="coerce").to_numpy(dtype=float),
    )
    model_summary.update({"test_auroc": test_summary["auroc"], "test_pr_auc": test_summary["pr_auc"]})
    global_operator_df = pd.DataFrame(
        [
            {"phase": "init", **model_summary["global_operator_init"]},
            {"phase": "learned", **model_summary["global_operator_learned"]},
        ]
    )
    global_operator_df.to_csv(out_dir / "ltst_s4_probe_global_operator_diagnostics.csv", index=False)

    pooled_enrichment = build_onset_enrichment(test_pred, OPERATOR_CHANNEL_COLUMNS + ("probe_prob",))
    pooled_enrichment.to_csv(out_dir / "ltst_s4_probe_operator_onset_enrichment.csv", index=False)
    branch_enrichment = build_branch_enrichment(
        test_pred.loc[test_pred["branch_label"].isin(["dominant", "reverse"])],
        OPERATOR_CHANNEL_COLUMNS + ("probe_prob",),
    )
    branch_enrichment.to_csv(out_dir / "ltst_s4_probe_branch_enrichment.csv", index=False)
    detector_comparison = build_detector_comparison(
        val_pred=val_pred,
        test_pred=test_pred,
        channels=DIRECT_COMPARISON_COLUMNS,
        cooldown_beats=cfg.cooldown_beats,
        horizon_beats=cfg.horizon_beats,
        specificity_target=cfg.specificity_target,
    )
    detector_comparison.to_csv(out_dir / "ltst_s4_probe_detector_comparison.csv", index=False)

    historical_baseline = load_historical_baseline(cfg.geometry_run_dir)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - t0,
        "config": json_safe(asdict(cfg)),
        "selected_features": list(selected_features),
        "model_summary": json_safe(model_summary),
        "historical_baseline": json_safe(historical_baseline),
    }
    (out_dir / "ltst_s4_probe_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(
        out_dir=out_dir,
        cfg=cfg,
        selected_features=selected_features,
        split_summary=split_summary,
        model_summary=model_summary,
        pooled_enrichment=pooled_enrichment,
        branch_enrichment=branch_enrichment,
        detector_comparison=detector_comparison,
        historical_baseline=historical_baseline,
        global_operator_df=global_operator_df,
    )
    print(f"[LTST-S4] Output directory: {out_dir}", flush=True)
    print(
        f"[LTST-S4] Test AUROC={model_summary['test_auroc']:.4f} "
        f"| rho-valid-rows={int(np.sum(pd.to_numeric(test_pred['operator_valid'], errors='coerce').fillna(0.0) > 0.0))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
