from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EPS = 1e-9
EXEMPLAR_RECORDS = ("s20021", "s20041", "s20151", "s30742")
BEAT_COLUMNS = [
    "record",
    "beat_sample",
    "status",
    "curr_energy",
    "past_energy",
    "poincare_b",
    "drift_norm",
    "energy_asym",
    "st_event",
]

FLAG_FEATURE_VALID = 1 << 0
FLAG_WARMUP = 1 << 1
FLAG_LOW_CURR_ENERGY = 1 << 2
FLAG_LOW_PAST_ENERGY = 1 << 3
FLAG_SATURATED = 1 << 4
FLAG_DIV_GUARD = 1 << 5
FLAG_REJECTED_BEAT = 1 << 6


@dataclass(frozen=True)
class ESP32ContractConfig:
    run_dir: Path
    out_dir_name: str = "esp32_contract_sim"
    records: tuple[str, ...] = EXEMPLAR_RECORDS
    baseline_window_beats: tuple[int, ...] = (64, 250, 500)
    low_energy_floor: float = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate the LTST V1 feature contract under ESP32-style fixed-point quantization.")
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "artifact" / "runs" / "ltst_full_86_20260405T035505Z"))
    parser.add_argument("--records", default=",".join(EXEMPLAR_RECORDS))
    return parser.parse_args()


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


def ensure_output_dir(run_dir: Path, out_dir_name: str) -> Path:
    out_dir = run_dir / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def read_beat_frame(beat_level_dir: Path, record: str) -> pd.DataFrame:
    parquet_path = beat_level_dir / f"{record}.parquet"
    csv_path = beat_level_dir / f"{record}.csv"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path, columns=BEAT_COLUMNS)
        except Exception:
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path, usecols=BEAT_COLUMNS)
    raise FileNotFoundError(f"Missing beat-level file for {record}.")


def quantize_unsigned(values: np.ndarray, int_bits: int, frac_bits: int) -> tuple[np.ndarray, np.ndarray]:
    scale = float(1 << frac_bits)
    max_raw = (1 << (int_bits + frac_bits)) - 1
    raw = np.rint(np.nan_to_num(values, nan=0.0) * scale)
    clipped = (raw < 0) | (raw > max_raw)
    raw = np.clip(raw, 0, max_raw).astype(np.uint32)
    return raw, clipped


def quantize_signed(values: np.ndarray, int_bits: int, frac_bits: int) -> tuple[np.ndarray, np.ndarray]:
    scale = float(1 << frac_bits)
    min_raw = -(1 << (int_bits + frac_bits))
    max_raw = (1 << (int_bits + frac_bits)) - 1
    raw = np.rint(np.nan_to_num(values, nan=0.0) * scale)
    clipped = (raw < min_raw) | (raw > max_raw)
    raw = np.clip(raw, min_raw, max_raw).astype(np.int32)
    return raw, clipped


def dequantize(raw: np.ndarray, frac_bits: int) -> np.ndarray:
    return raw.astype(np.float64) / float(1 << frac_bits)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return np.nan
    aa = pd.Series(a[mask]).rank(method="average").to_numpy(dtype=float)
    bb = pd.Series(b[mask]).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(aa, bb)[0, 1])


def build_packet_frame(df: pd.DataFrame, cfg: ESP32ContractConfig) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["beat_index"] = np.arange(len(out), dtype=np.uint32)

    pb_raw, pb_clip = quantize_unsigned(out["poincare_b"].to_numpy(dtype=float), int_bits=2, frac_bits=14)
    drift_raw, drift_clip = quantize_unsigned(out["drift_norm"].to_numpy(dtype=float), int_bits=4, frac_bits=12)
    asym_raw, asym_clip = quantize_signed(out["energy_asym"].to_numpy(dtype=float), int_bits=1, frac_bits=14)

    flags = np.zeros(len(out), dtype=np.uint16)
    finite_mask = np.isfinite(out["poincare_b"]) & np.isfinite(out["drift_norm"]) & np.isfinite(out["energy_asym"])
    flags |= np.where(finite_mask, FLAG_FEATURE_VALID, 0).astype(np.uint16)
    flags |= np.where(pd.to_numeric(out["status"], errors="coerce").fillna(0).to_numpy(dtype=int) != 0, FLAG_REJECTED_BEAT, 0).astype(np.uint16)
    flags |= np.where(pd.to_numeric(out["curr_energy"], errors="coerce").fillna(0.0).to_numpy(dtype=float) <= cfg.low_energy_floor, FLAG_LOW_CURR_ENERGY, 0).astype(np.uint16)
    flags |= np.where(pd.to_numeric(out["past_energy"], errors="coerce").fillna(0.0).to_numpy(dtype=float) <= cfg.low_energy_floor, FLAG_LOW_PAST_ENERGY, 0).astype(np.uint16)
    flags |= np.where(out["drift_norm"].fillna(0.0).to_numpy(dtype=float) <= EPS, FLAG_DIV_GUARD, 0).astype(np.uint16)
    flags |= np.where(pb_clip | drift_clip | asym_clip, FLAG_SATURATED, 0).astype(np.uint16)

    out["poincare_b_raw"] = pb_raw.astype(np.uint16)
    out["drift_norm_raw"] = drift_raw.astype(np.uint16)
    out["energy_asym_raw"] = asym_raw.astype(np.int16)
    out["quality_flags"] = flags

    out["poincare_b_emulated"] = dequantize(pb_raw, frac_bits=14)
    out["drift_norm_emulated"] = dequantize(drift_raw, frac_bits=12)
    out["energy_asym_emulated"] = dequantize(asym_raw, frac_bits=14)
    out["stiffness_proxy_ref"] = out["poincare_b"].to_numpy(dtype=float) / (out["drift_norm"].to_numpy(dtype=float) + EPS)
    out["stiffness_proxy_emulated"] = out["poincare_b_emulated"].to_numpy(dtype=float) / (out["drift_norm_emulated"].to_numpy(dtype=float) + EPS)
    return out


def window_median(series: pd.Series, n: int) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    return float(clean.head(int(n)).median())


def evaluate_record(packet_df: pd.DataFrame, cfg: ESP32ContractConfig) -> dict[str, Any]:
    valid_mask = (packet_df["quality_flags"].to_numpy(dtype=np.uint16) & FLAG_FEATURE_VALID) != 0
    valid = packet_df.loc[valid_mask].copy()
    result: dict[str, Any] = {"record": str(packet_df["record"].iloc[0]), "n_beats": int(len(packet_df)), "n_valid_beats": int(len(valid))}
    if valid.empty:
        return result

    result["poincare_b_mae"] = float(np.median(np.abs(valid["poincare_b"] - valid["poincare_b_emulated"])))
    result["drift_norm_mae"] = float(np.median(np.abs(valid["drift_norm"] - valid["drift_norm_emulated"])))
    result["energy_asym_mae"] = float(np.median(np.abs(valid["energy_asym"] - valid["energy_asym_emulated"])))
    result["poincare_b_spearman"] = spearman_corr(valid["poincare_b"].to_numpy(dtype=float), valid["poincare_b_emulated"].to_numpy(dtype=float))
    result["drift_norm_spearman"] = spearman_corr(valid["drift_norm"].to_numpy(dtype=float), valid["drift_norm_emulated"].to_numpy(dtype=float))

    asym_mask = np.abs(valid["energy_asym"].to_numpy(dtype=float)) >= 0.05
    if asym_mask.any():
        result["energy_asym_sign_agreement"] = float(
            np.mean(
                np.sign(valid.loc[asym_mask, "energy_asym"].to_numpy(dtype=float))
                == np.sign(valid.loc[asym_mask, "energy_asym_emulated"].to_numpy(dtype=float))
            )
        )
    else:
        result["energy_asym_sign_agreement"] = np.nan

    result["saturated_fraction"] = float(((valid["quality_flags"].to_numpy(dtype=np.uint16) & FLAG_SATURATED) != 0).mean())
    result["div_guard_fraction"] = float(((valid["quality_flags"].to_numpy(dtype=np.uint16) & FLAG_DIV_GUARD) != 0).mean())
    result["low_curr_energy_fraction"] = float(((valid["quality_flags"].to_numpy(dtype=np.uint16) & FLAG_LOW_CURR_ENERGY) != 0).mean())
    result["low_past_energy_fraction"] = float(((valid["quality_flags"].to_numpy(dtype=np.uint16) & FLAG_LOW_PAST_ENERGY) != 0).mean())

    for window in cfg.baseline_window_beats:
        result[f"poincare_b_median_abs_error_{window}"] = abs(window_median(valid["poincare_b"], window) - window_median(valid["poincare_b_emulated"], window))
        result[f"drift_norm_median_abs_error_{window}"] = abs(window_median(valid["drift_norm"], window) - window_median(valid["drift_norm_emulated"], window))
        result[f"stiffness_proxy_median_abs_error_{window}"] = abs(window_median(valid["stiffness_proxy_ref"], window) - window_median(valid["stiffness_proxy_emulated"], window))
        result[f"ref_poincare_b_median_{window}"] = window_median(valid["poincare_b"], window)
        result[f"emu_poincare_b_median_{window}"] = window_median(valid["poincare_b_emulated"], window)
        result[f"ref_drift_norm_median_{window}"] = window_median(valid["drift_norm"], window)
        result[f"emu_drift_norm_median_{window}"] = window_median(valid["drift_norm_emulated"], window)
        result[f"ref_stiffness_proxy_median_{window}"] = window_median(valid["stiffness_proxy_ref"], window)
        result[f"emu_stiffness_proxy_median_{window}"] = window_median(valid["stiffness_proxy_emulated"], window)
    return result


def write_report(out_dir: Path, summary_df: pd.DataFrame, ordering_df: pd.DataFrame) -> None:
    lines = [
        "# LTST ESP32 Contract Simulation",
        "",
        "Fixed-point emulation of the frozen V1 cardiac beat packet on the four benchmark LTST exemplar records.",
        "",
        "## Record Summary",
        summary_df.to_markdown(index=False),
        "",
        "## Ordering Check",
        ordering_df.to_markdown(index=False),
        "",
    ]
    (out_dir / "ltst_esp32_contract_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = tuple(piece.strip() for piece in str(args.records).split(",") if piece.strip())
    cfg = ESP32ContractConfig(run_dir=Path(args.run_dir).resolve(), records=records)
    out_dir = ensure_output_dir(cfg.run_dir, cfg.out_dir_name)
    beat_level_dir = cfg.run_dir / "beat_level"

    packet_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for record in cfg.records:
        beat_df = read_beat_frame(beat_level_dir, record)
        packet_df = build_packet_frame(beat_df, cfg)
        packet_parts.append(packet_df)
        summary_rows.append(evaluate_record(packet_df, cfg))

    packets_df = pd.concat(packet_parts, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    ordering_rows: list[dict[str, Any]] = []
    for metric in ["ref_poincare_b_median_500", "emu_poincare_b_median_500", "ref_drift_norm_median_500", "emu_drift_norm_median_500", "ref_stiffness_proxy_median_500", "emu_stiffness_proxy_median_500"]:
        ordered = summary_df[["record", metric]].sort_values(metric, ascending=("drift_norm" not in metric))
        ordering_rows.append({"metric": metric, "order": " < ".join(ordered["record"].astype(str).tolist())})
    ordering_df = pd.DataFrame(ordering_rows)

    packets_df.to_csv(out_dir / "ltst_esp32_contract_packets.csv", index=False)
    summary_df.to_csv(out_dir / "ltst_esp32_contract_summary.csv", index=False)
    ordering_df.to_csv(out_dir / "ltst_esp32_contract_ordering.csv", index=False)
    write_report(out_dir, summary_df, ordering_df)

    metadata = {"config": json_safe(asdict(cfg)), "n_records": int(len(summary_df)), "n_packets": int(len(packets_df))}
    with (out_dir / "ltst_esp32_contract_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"[LTST-ESP32] Output directory: {out_dir}", flush=True)
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
