from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ltst_v1_uart_decoder import (
    FRAME_LEN,
    FRAME_VERSION,
    MSG_TYPE_FEATURE_PACKET,
    PAYLOAD_LEN,
    SYNC,
    crc16_ccitt_false,
    decode_payload,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RUN_DIR = PROJECT_ROOT / "artifact" / "runs" / "ltst_full_86_20260405T035505Z"

PB_ORDER = ("s20021", "s20041", "s20151", "s30742")
DRIFT_ORDER = ("s20021", "s20041", "s20151", "s30742")
STIFFNESS_ORDER = ("s20021", "s20041", "s20151", "s30742")

FEATURE_VALID_FLAG = 0x0001
ENERGY_ASYM_ACTIVE_THRESHOLD = 0.05

THRESHOLDS: dict[str, float] = {
    "valid_overlap_fraction_min": 0.98,
    "poincare_b_mae_max": 0.03,
    "drift_norm_mae_max": 0.015,
    "poincare_b_spearman_min": 0.95,
    "drift_norm_spearman_min": 0.93,
    "energy_asym_sign_agreement_min": 0.90,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare one LTST ESP32 bench session against the frozen V1 prototype thresholds.")
    parser.add_argument("--session-dir", type=Path, required=True, help="Session folder created by ltst_v1_bench_session.py")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="LTST run directory, used only for reference links in the report")
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


def decode_capture(raw_bytes: bytes) -> tuple[list[dict[str, Any]], int]:
    buffer = bytearray(raw_bytes)
    packets: list[dict[str, Any]] = []
    bad_frames = 0

    while True:
        sync_pos = buffer.find(SYNC)
        if sync_pos < 0:
            break
        if sync_pos > 0:
            del buffer[:sync_pos]
        if len(buffer) < FRAME_LEN:
            break

        frame = bytes(buffer[:FRAME_LEN])
        version = frame[2]
        msg_type = frame[3]
        payload_len = int.from_bytes(frame[4:6], "little")
        payload = frame[6 : 6 + payload_len]
        crc_rx = int.from_bytes(frame[6 + payload_len : 8 + payload_len], "little")

        if version != FRAME_VERSION or msg_type != MSG_TYPE_FEATURE_PACKET or payload_len != PAYLOAD_LEN:
            bad_frames += 1
            del buffer[:2]
            continue

        crc_calc = crc16_ccitt_false(frame[2 : 6 + payload_len])
        if crc_calc != crc_rx:
            bad_frames += 1
            del buffer[:FRAME_LEN]
            continue

        beat_index, pb_raw, drift_raw, asym_raw, flags = decode_payload(payload)
        packets.append(
            {
                "capture_index": len(packets) + 1,
                "beat_index": int(beat_index),
                "poincare_b_raw": int(pb_raw),
                "drift_norm_raw": int(drift_raw),
                "energy_asym_raw": int(asym_raw),
                "quality_flags": int(flags),
                "poincare_b": float(pb_raw) / float(1 << 14),
                "drift_norm": float(drift_raw) / float(1 << 12),
                "energy_asym": float(asym_raw) / float(1 << 14),
                "crc16": int(crc_rx),
            }
        )
        del buffer[:FRAME_LEN]

    return packets, bad_frames


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return np.nan
    aa = pd.Series(a[mask]).rank(method="average").to_numpy(dtype=float)
    bb = pd.Series(b[mask]).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(aa, bb)[0, 1])


def order_string(records: list[str], descending: bool = False) -> str:
    return (" > " if descending else " < ").join(records)


def compare_ordering(df: pd.DataFrame, metric: str, expected_order: tuple[str, ...], ascending: bool) -> tuple[bool, str]:
    actual = df.sort_values(metric, ascending=ascending)["record"].astype(str).tolist()
    return tuple(actual) == tuple(expected_order), order_string(actual, descending=not ascending)


def load_session(session_dir: Path) -> tuple[dict[str, Any], Path]:
    metadata_path = session_dir / "session_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing session metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw_capture_path = session_dir / "uart1_capture.bin"
    if not raw_capture_path.exists():
        raise FileNotFoundError(f"Missing raw capture: {raw_capture_path}")
    return metadata, raw_capture_path


def build_comparison_frame(metadata: dict[str, Any], packets: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, int]]:
    expected_vectors = pd.DataFrame(metadata.get("vectors", []))
    if expected_vectors.empty:
        raise ValueError("Session metadata contains no vectors to compare.")
    expected_vectors["beat_index"] = expected_vectors["beat_index"].astype(int)

    packet_df = pd.DataFrame(packets)
    if packet_df.empty:
        packet_df = pd.DataFrame(columns=["beat_index", "capture_index", "poincare_b", "drift_norm", "energy_asym", "quality_flags"])
    else:
        packet_df = packet_df.sort_values(["beat_index", "capture_index"]).reset_index(drop=True)

    last_packets = packet_df.groupby("beat_index", as_index=False).tail(1).copy() if not packet_df.empty else packet_df.copy()
    duplicate_count = int(packet_df["beat_index"].duplicated(keep=False).sum()) if not packet_df.empty else 0
    extra_packets = int((~packet_df["beat_index"].isin(expected_vectors["beat_index"])).sum()) if not packet_df.empty else 0

    merged = expected_vectors.merge(last_packets, on="beat_index", how="left", suffixes=("_ref", "_obs"))
    merged["matched"] = merged["capture_index"].notna()
    merged["feature_valid"] = merged["quality_flags"].fillna(0).astype(int).map(lambda value: (value & FEATURE_VALID_FLAG) != 0)
    merged["use_for_metrics"] = merged["matched"] & merged["feature_valid"]

    merged["abs_error_poincare_b"] = np.abs(merged["poincare_b_ref"] - merged["poincare_b_obs"])
    merged["abs_error_drift_norm"] = np.abs(merged["drift_norm_ref"] - merged["drift_norm_obs"])
    merged["abs_error_energy_asym"] = np.abs(merged["energy_asym_ref"] - merged["energy_asym_obs"])
    merged["stiffness_proxy_ref"] = merged["poincare_b_ref"] / (merged["drift_norm_ref"] + 1e-9)
    merged["stiffness_proxy_obs"] = merged["poincare_b_obs"] / (merged["drift_norm_obs"] + 1e-9)

    return merged, {"duplicate_packets": duplicate_count, "extra_packets": extra_packets}


def summarise_metrics(comp_df: pd.DataFrame, bad_frames: int) -> dict[str, Any]:
    valid = comp_df.loc[comp_df["use_for_metrics"]].copy()
    metrics: dict[str, Any] = {
        "expected_packets": int(len(comp_df)),
        "matched_packets": int(comp_df["matched"].sum()),
        "valid_packets": int(valid.shape[0]),
        "valid_overlap_fraction": float(valid.shape[0] / max(1, len(comp_df))),
        "bad_frames": int(bad_frames),
    }

    if valid.empty:
        metrics.update(
            {
                "poincare_b_mae": np.nan,
                "drift_norm_mae": np.nan,
                "energy_asym_mae": np.nan,
                "poincare_b_spearman": np.nan,
                "drift_norm_spearman": np.nan,
                "energy_asym_sign_agreement": np.nan,
                "energy_asym_sign_basis": "no-valid-packets",
                "pb_ordering_preserved": False,
                "pb_actual_order": "",
                "drift_ordering_preserved": False,
                "drift_actual_order": "",
                "stiffness_ordering_preserved": False,
                "stiffness_actual_order": "",
                "pass_valid_overlap": metrics["valid_overlap_fraction"] >= THRESHOLDS["valid_overlap_fraction_min"],
                "pass_poincare_b_mae": False,
                "pass_drift_norm_mae": False,
                "pass_poincare_b_spearman": False,
                "pass_drift_norm_spearman": False,
                "pass_energy_asym_sign": False,
                "pass_ordering": False,
                "pass_all": False,
            }
        )
        return metrics

    metrics["poincare_b_mae"] = float(np.median(valid["abs_error_poincare_b"]))
    metrics["drift_norm_mae"] = float(np.median(valid["abs_error_drift_norm"]))
    metrics["energy_asym_mae"] = float(np.median(valid["abs_error_energy_asym"]))
    metrics["poincare_b_spearman"] = spearman_corr(valid["poincare_b_ref"].to_numpy(dtype=float), valid["poincare_b_obs"].to_numpy(dtype=float))
    metrics["drift_norm_spearman"] = spearman_corr(valid["drift_norm_ref"].to_numpy(dtype=float), valid["drift_norm_obs"].to_numpy(dtype=float))

    active_mask = np.abs(valid["energy_asym_ref"].to_numpy(dtype=float)) >= ENERGY_ASYM_ACTIVE_THRESHOLD
    if active_mask.any():
        sign_ref = np.sign(valid.loc[active_mask, "energy_asym_ref"].to_numpy(dtype=float))
        sign_obs = np.sign(valid.loc[active_mask, "energy_asym_obs"].to_numpy(dtype=float))
        metrics["energy_asym_sign_agreement"] = float(np.mean(sign_ref == sign_obs))
        metrics["energy_asym_sign_basis"] = f"active|ref|>={ENERGY_ASYM_ACTIVE_THRESHOLD:.2f}"
    else:
        sign_ref = np.sign(valid["energy_asym_ref"].to_numpy(dtype=float))
        sign_obs = np.sign(valid["energy_asym_obs"].to_numpy(dtype=float))
        metrics["energy_asym_sign_agreement"] = float(np.mean(sign_ref == sign_obs))
        metrics["energy_asym_sign_basis"] = "all-matched-exemplars"

    pb_ok, pb_order = compare_ordering(valid, "poincare_b_obs", PB_ORDER, ascending=True)
    drift_ok, drift_order = compare_ordering(valid, "drift_norm_obs", DRIFT_ORDER, ascending=False)
    stiffness_ok, stiffness_order = compare_ordering(valid, "stiffness_proxy_obs", STIFFNESS_ORDER, ascending=True)
    metrics["pb_ordering_preserved"] = bool(pb_ok)
    metrics["pb_actual_order"] = pb_order
    metrics["drift_ordering_preserved"] = bool(drift_ok)
    metrics["drift_actual_order"] = drift_order
    metrics["stiffness_ordering_preserved"] = bool(stiffness_ok)
    metrics["stiffness_actual_order"] = stiffness_order

    metrics["pass_valid_overlap"] = metrics["valid_overlap_fraction"] >= THRESHOLDS["valid_overlap_fraction_min"]
    metrics["pass_poincare_b_mae"] = metrics["poincare_b_mae"] <= THRESHOLDS["poincare_b_mae_max"]
    metrics["pass_drift_norm_mae"] = metrics["drift_norm_mae"] <= THRESHOLDS["drift_norm_mae_max"]
    metrics["pass_poincare_b_spearman"] = bool(np.isfinite(metrics["poincare_b_spearman"])) and metrics["poincare_b_spearman"] >= THRESHOLDS["poincare_b_spearman_min"]
    metrics["pass_drift_norm_spearman"] = bool(np.isfinite(metrics["drift_norm_spearman"])) and metrics["drift_norm_spearman"] >= THRESHOLDS["drift_norm_spearman_min"]
    metrics["pass_energy_asym_sign"] = bool(np.isfinite(metrics["energy_asym_sign_agreement"])) and metrics["energy_asym_sign_agreement"] >= THRESHOLDS["energy_asym_sign_agreement_min"]
    metrics["pass_ordering"] = bool(pb_ok and drift_ok and stiffness_ok)
    metrics["pass_all"] = bool(
        metrics["pass_valid_overlap"]
        and metrics["pass_poincare_b_mae"]
        and metrics["pass_drift_norm_mae"]
        and metrics["pass_poincare_b_spearman"]
        and metrics["pass_drift_norm_spearman"]
        and metrics["pass_energy_asym_sign"]
        and metrics["pass_ordering"]
    )
    return metrics


def write_report(
    session_dir: Path,
    run_dir: Path,
    comp_df: pd.DataFrame,
    metrics: dict[str, Any],
    notes: dict[str, int],
) -> None:
    summary_rows = [
        {"check": "valid_overlap_fraction", "value": metrics["valid_overlap_fraction"], "threshold": f">= {THRESHOLDS['valid_overlap_fraction_min']:.2f}", "pass": metrics["pass_valid_overlap"]},
        {"check": "poincare_b_mae", "value": metrics["poincare_b_mae"], "threshold": f"<= {THRESHOLDS['poincare_b_mae_max']:.3f}", "pass": metrics["pass_poincare_b_mae"]},
        {"check": "drift_norm_mae", "value": metrics["drift_norm_mae"], "threshold": f"<= {THRESHOLDS['drift_norm_mae_max']:.3f}", "pass": metrics["pass_drift_norm_mae"]},
        {"check": "poincare_b_spearman", "value": metrics["poincare_b_spearman"], "threshold": f">= {THRESHOLDS['poincare_b_spearman_min']:.2f}", "pass": metrics["pass_poincare_b_spearman"]},
        {"check": "drift_norm_spearman", "value": metrics["drift_norm_spearman"], "threshold": f">= {THRESHOLDS['drift_norm_spearman_min']:.2f}", "pass": metrics["pass_drift_norm_spearman"]},
        {"check": f"energy_asym_sign_agreement ({metrics['energy_asym_sign_basis']})", "value": metrics["energy_asym_sign_agreement"], "threshold": f">= {THRESHOLDS['energy_asym_sign_agreement_min']:.2f}", "pass": metrics["pass_energy_asym_sign"]},
        {"check": "ordering_preserved", "value": f"pb={metrics['pb_actual_order']} | drift={metrics['drift_actual_order']} | stiffness={metrics['stiffness_actual_order']}", "threshold": "preserve exemplar ladder", "pass": metrics["pass_ordering"]},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(session_dir / "bench_compare_metrics.csv", index=False)

    comp_df.to_csv(session_dir / "bench_compare_table.csv", index=False)

    report_lines = [
        "# LTST V1 Bench Comparator",
        "",
        f"Session: [{session_dir.name}]({session_dir})",
        f"Reference LTST run: [{run_dir.name}]({run_dir})",
        "",
        f"Overall result: `{'PASS' if metrics['pass_all'] else 'FAIL'}`",
        "",
        "## Threshold Summary",
        summary_df.to_markdown(index=False),
        "",
        "## Session Notes",
        f"- Expected packets: `{metrics['expected_packets']}`",
        f"- Matched packets: `{metrics['matched_packets']}`",
        f"- Valid packets: `{metrics['valid_packets']}`",
        f"- Bad frames discarded: `{metrics['bad_frames']}`",
        f"- Duplicate beat indices seen in capture: `{notes['duplicate_packets']}`",
        f"- Extra packets with unexpected beat indices: `{notes['extra_packets']}`",
        "",
        "## Per-Record Comparison",
        comp_df[
            [
                "record",
                "beat_index",
                "matched",
                "feature_valid",
                "poincare_b_ref",
                "poincare_b_obs",
                "abs_error_poincare_b",
                "drift_norm_ref",
                "drift_norm_obs",
                "abs_error_drift_norm",
                "energy_asym_ref",
                "energy_asym_obs",
                "abs_error_energy_asym",
            ]
        ].to_markdown(index=False),
        "",
        "## Expected Ordering",
        f"- `poincare_b` ascending: `{' < '.join(PB_ORDER)}`",
        f"- `drift_norm` descending: `{' > '.join(DRIFT_ORDER)}`",
        f"- host-side stiffness proxy ascending: `{' < '.join(STIFFNESS_ORDER)}`",
        "",
    ]
    (session_dir / "bench_compare_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    metadata = {
        "thresholds": json_safe(THRESHOLDS),
        "metrics": json_safe(metrics),
        "notes": json_safe(notes),
        "session_dir": str(session_dir),
        "run_dir": str(run_dir),
    }
    (session_dir / "bench_compare_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    session_dir = args.session_dir.resolve()
    run_dir = args.run_dir.resolve()

    metadata, raw_capture_path = load_session(session_dir)
    packets, bad_frames = decode_capture(raw_capture_path.read_bytes())
    comp_df, notes = build_comparison_frame(metadata, packets)
    metrics = summarise_metrics(comp_df, bad_frames=bad_frames)
    write_report(session_dir, run_dir, comp_df, metrics, notes)

    print(session_dir)
    print(f"pass_all={int(bool(metrics['pass_all']))}")
    print(f"valid_packets={metrics['valid_packets']}")


if __name__ == "__main__":
    main()
