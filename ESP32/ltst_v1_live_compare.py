from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ltst_v1_bench_compare import THRESHOLDS, json_safe
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
DEFAULT_MANIFEST = SCRIPT_DIR / "generated" / "ltst_v1_replay_samples_manifest.json"
BEAT_RE = re.compile(
    r"^\[beat\]\s+record=(?P<record>\S+)\s+beat=(?P<beat>\d+)\s+sample=(?P<sample>\d+)\s+rr=(?P<rr>\d+)\s+feature_valid=(?P<feature_valid>\d+)\s+pb=(?P<pb>-?\d+\.\d+)\s+drift=(?P<drift>-?\d+\.\d+)\s+asym=(?P<asym>-?\d+\.\d+)\s+flags=0x(?P<flags>[0-9A-Fa-f]+)\s+conf=(?P<conf>-?\d+\.\d+)\s+thr=(?P<thr>-?\d+\.\d+)"
)

DETECTOR_THRESHOLDS = {
    "recall_min": 0.98,
    "precision_min": 0.98,
    "median_abs_offset_max": 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare one replay_samples ESP32 session against the LTST manifest reference.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--record", help="Override the record name stored in session metadata.")
    parser.add_argument("--match-tolerance", type=int, default=10)
    return parser.parse_args()


def decode_capture(raw_bytes: bytes) -> list[dict[str, Any]]:
    buffer = bytearray(raw_bytes)
    packets: list[dict[str, Any]] = []
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
            del buffer[:2]
            continue
        if crc16_ccitt_false(frame[2 : 6 + payload_len]) != crc_rx:
            del buffer[:FRAME_LEN]
            continue
        beat_index, pb_raw, drift_raw, asym_raw, flags = decode_payload(payload)
        packets.append(
            {
                "beat_index": int(beat_index),
                "poincare_b": float(pb_raw) / float(1 << 14),
                "drift_norm": float(drift_raw) / float(1 << 12),
                "energy_asym": float(asym_raw) / float(1 << 14),
                "feature_valid": bool(flags & 0x0001),
                "quality_flags": int(flags),
            }
        )
        del buffer[:FRAME_LEN]
    return packets


def parse_console_beats(console_text: str, record_filter: str | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for line in console_text.splitlines():
        match = BEAT_RE.match(line.strip())
        if not match:
            continue
        row = {
            "record": match.group("record"),
            "beat_index": int(match.group("beat")),
            "sample_index": int(match.group("sample")),
            "rr_samples": int(match.group("rr")),
            "feature_valid_console": bool(int(match.group("feature_valid"))),
            "poincare_b_console": float(match.group("pb")),
            "drift_norm_console": float(match.group("drift")),
            "energy_asym_console": float(match.group("asym")),
            "quality_flags_console": int(match.group("flags"), 16),
            "detector_confidence": float(match.group("conf")),
            "detector_threshold": float(match.group("thr")),
        }
        if record_filter is None or row["record"] == record_filter:
            rows.append(row)
    return pd.DataFrame(rows)


def greedy_match(expected_df: pd.DataFrame, detected_df: pd.DataFrame, tolerance: int) -> pd.DataFrame:
    matches: list[dict[str, Any]] = []
    used_expected: set[int] = set()
    for det in detected_df.sort_values("sample_index").itertuples(index=False):
        expected_candidates = expected_df.loc[~expected_df["match_id"].isin(used_expected)].copy()
        if expected_candidates.empty:
            continue
        expected_candidates["abs_delta"] = np.abs(expected_candidates["sample_index"] - det.sample_index)
        eligible = expected_candidates.loc[expected_candidates["abs_delta"] <= tolerance].sort_values(["abs_delta", "beat_index"])
        if eligible.empty:
            continue
        best = eligible.iloc[0]
        match_id = int(best["match_id"])
        used_expected.add(match_id)
        matches.append(
            {
                "expected_match_id": match_id,
                "expected_beat_index": int(best["beat_index"]),
                "expected_sample_index": int(best["sample_index"]),
                "detected_beat_index": int(det.beat_index),
                "detected_sample_index": int(det.sample_index),
                "sample_offset": int(det.sample_index - int(best["sample_index"])),
            }
        )
    return pd.DataFrame(matches)


def main() -> None:
    args = parse_args()
    session_dir = args.session_dir.resolve()
    metadata = json.loads((session_dir / "session_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest_path.resolve().read_text(encoding="utf-8"))
    record = args.record or metadata.get("source_record")
    if not record:
        raise ValueError("No record provided and no source_record found in session metadata")

    record_entry = manifest.get("records", {}).get(record)
    if record_entry is None:
        raise KeyError(f"Record '{record}' not found in manifest {args.manifest_path}")

    expected_df = pd.DataFrame(record_entry.get("expected_beats", []))
    if expected_df.empty:
        raise ValueError(f"No expected beats found for {record}")
    expected_df["match_id"] = np.arange(len(expected_df), dtype=int)

    packets = decode_capture((session_dir / "uart1_capture.bin").read_bytes())
    packet_df = pd.DataFrame(packets)
    console_df = parse_console_beats((session_dir / "console_output.txt").read_text(encoding="utf-8", errors="replace"), record_filter=record)
    if console_df.empty:
        raise ValueError("No [beat] sideband lines found in console output")

    merged_detected = console_df.merge(packet_df, on="beat_index", how="left", suffixes=("", "_packet"))
    matches_df = greedy_match(expected_df, merged_detected, tolerance=int(args.match_tolerance))

    matched_expected = int(matches_df["expected_match_id"].nunique()) if not matches_df.empty else 0
    matched_detected = int(matches_df.shape[0])
    recall = float(matched_expected / max(1, len(expected_df)))
    precision = float(matched_detected / max(1, len(merged_detected)))
    median_abs_offset = float(np.median(np.abs(matches_df["sample_offset"]))) if not matches_df.empty else np.nan

    joined = matches_df.merge(expected_df, left_on="expected_match_id", right_on="match_id", how="left")
    joined = joined.merge(merged_detected, left_on="detected_beat_index", right_on="beat_index", how="left", suffixes=("_expected", "_detected"))

    valid = joined.loc[joined["feature_valid_expected"] & joined["feature_valid_console"]].copy() if not joined.empty else joined.copy()
    if not valid.empty:
        valid["abs_error_poincare_b"] = np.abs(valid["poincare_b_expected"] - valid["poincare_b_detected"])
        valid["abs_error_drift_norm"] = np.abs(valid["drift_norm_expected"] - valid["drift_norm_detected"])
        valid["abs_error_energy_asym"] = np.abs(valid["energy_asym_expected"] - valid["energy_asym_detected"])
        pb_mae = float(np.median(valid["abs_error_poincare_b"]))
        drift_mae = float(np.median(valid["abs_error_drift_norm"]))
        sign_ref = np.sign(valid["energy_asym_expected"].to_numpy(dtype=float))
        sign_obs = np.sign(valid["energy_asym_detected"].to_numpy(dtype=float))
        sign_agreement = float(np.mean(sign_ref == sign_obs))
    else:
        pb_mae = np.nan
        drift_mae = np.nan
        sign_agreement = np.nan

    summary = {
        "record": record,
        "expected_beats": int(len(expected_df)),
        "detected_beats": int(len(merged_detected)),
        "matched_beats": int(matched_detected),
        "recall": recall,
        "precision": precision,
        "median_abs_offset": median_abs_offset,
        "valid_feature_pairs": int(len(valid)),
        "poincare_b_mae": pb_mae,
        "drift_norm_mae": drift_mae,
        "energy_asym_sign_agreement": sign_agreement,
        "pass_recall": recall >= DETECTOR_THRESHOLDS["recall_min"],
        "pass_precision": precision >= DETECTOR_THRESHOLDS["precision_min"],
        "pass_offset": bool(np.isfinite(median_abs_offset)) and median_abs_offset <= DETECTOR_THRESHOLDS["median_abs_offset_max"],
        "pass_poincare_b_mae": bool(np.isfinite(pb_mae)) and pb_mae <= THRESHOLDS["poincare_b_mae_max"],
        "pass_drift_norm_mae": bool(np.isfinite(drift_mae)) and drift_mae <= THRESHOLDS["drift_norm_mae_max"],
        "pass_energy_sign": bool(np.isfinite(sign_agreement)) and sign_agreement >= THRESHOLDS["energy_asym_sign_agreement_min"],
    }
    summary["pass_all"] = bool(
        summary["pass_recall"]
        and summary["pass_precision"]
        and summary["pass_offset"]
        and summary["pass_poincare_b_mae"]
        and summary["pass_drift_norm_mae"]
        and summary["pass_energy_sign"]
    )

    (session_dir / "live_compare_matches.csv").write_text(matches_df.to_csv(index=False), encoding="utf-8")
    (session_dir / "live_compare_table.csv").write_text(joined.to_csv(index=False), encoding="utf-8")
    (session_dir / "live_compare_summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")

    report = "\n".join(
        [
            "# LTST V1 Live Replay Comparator",
            "",
            f"Record: `{record}`",
            f"Overall result: `{'PASS' if summary['pass_all'] else 'FAIL'}`",
            "",
            "## Detector",
            f"- Recall: `{summary['recall']:.4f}`",
            f"- Precision: `{summary['precision']:.4f}`",
            f"- Median absolute beat offset: `{summary['median_abs_offset']}` samples",
            "",
            "## Primitive Agreement",
            f"- Valid feature pairs: `{summary['valid_feature_pairs']}`",
            f"- `poincare_b` median absolute error: `{summary['poincare_b_mae']}`",
            f"- `drift_norm` median absolute error: `{summary['drift_norm_mae']}`",
            f"- `energy_asym` sign agreement: `{summary['energy_asym_sign_agreement']}`",
        ]
    )
    (session_dir / "live_compare_report.md").write_text(report, encoding="utf-8")

    print(session_dir)
    print(f"pass_all={int(summary['pass_all'])}")


if __name__ == "__main__":
    main()
