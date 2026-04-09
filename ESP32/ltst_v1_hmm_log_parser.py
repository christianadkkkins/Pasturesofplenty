from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HMM_RE = re.compile(
    r"^\[hmm\]\s+record=(?P<record>\S+)\s+beat=(?P<beat>\d+)\s+state=(?P<state>\S+)\s+score=(?P<score>-?\d+\.\d+)\s+episode=(?P<episode>\d+)\s+start=(?P<start>\d+)\s+active_duration=(?P<duration>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse ESP32 sideband HMM logs and optionally compare them to a host reference.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, help="Optional host-side HMM reference CSV with beat_index/state columns.")
    parser.add_argument("--record", help="Optional record filter.")
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


def parse_hmm_lines(console_text: str, record_filter: str | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for line in console_text.splitlines():
        match = HMM_RE.match(line.strip())
        if not match:
            continue
        row = {
            "record": match.group("record"),
            "beat_index": int(match.group("beat")),
            "state": match.group("state"),
            "score": float(match.group("score")),
            "episode_id": int(match.group("episode")),
            "episode_start_beat": int(match.group("start")),
            "active_duration_beats": int(match.group("duration")),
        }
        if record_filter is None or row["record"] == record_filter:
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    session_dir = args.session_dir.resolve()
    console_text = (session_dir / "console_output.txt").read_text(encoding="utf-8", errors="replace")
    hmm_df = parse_hmm_lines(console_text, record_filter=args.record)
    if hmm_df.empty:
        raise ValueError("No [hmm] lines found in console output")

    summary: dict[str, Any] = {
        "row_count": int(len(hmm_df)),
        "state_counts": hmm_df["state"].value_counts().to_dict(),
        "episode_count": int(hmm_df["episode_id"].replace(0, np.nan).dropna().nunique()),
    }

    if args.reference_csv is not None:
        ref_df = pd.read_csv(args.reference_csv.resolve())
        required = {"beat_index", "state"}
        missing = required - set(ref_df.columns)
        if missing:
            raise ValueError(f"Reference CSV missing required columns: {sorted(missing)}")
        merged = hmm_df.merge(ref_df, on="beat_index", how="inner", suffixes=("_device", "_ref"))
        if not merged.empty:
            summary["state_agreement"] = float(np.mean(merged["state_device"] == merged["state_ref"]))
            if "episode_start_beat" in ref_df.columns:
                summary["median_episode_start_abs_diff"] = float(
                    np.median(np.abs(merged["episode_start_beat_device"] - merged["episode_start_beat_ref"]))
                )
            if "active_duration_beats" in ref_df.columns:
                denom = np.maximum(np.abs(merged["active_duration_beats_ref"]), 1)
                summary["median_duration_relative_error"] = float(
                    np.median(np.abs(merged["active_duration_beats_device"] - merged["active_duration_beats_ref"]) / denom)
                )
        else:
            summary["state_agreement"] = np.nan

    (session_dir / "hmm_sideband.csv").write_text(hmm_df.to_csv(index=False), encoding="utf-8")
    (session_dir / "hmm_sideband_summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")

    report_lines = [
        "# LTST V1 HMM Sideband Parser",
        "",
        f"- Rows parsed: `{summary['row_count']}`",
        f"- Episode count: `{summary['episode_count']}`",
        f"- State counts: `{summary['state_counts']}`",
    ]
    if "state_agreement" in summary:
        report_lines.append(f"- State agreement: `{summary['state_agreement']}`")
    if "median_episode_start_abs_diff" in summary:
        report_lines.append(f"- Median episode start abs diff: `{summary['median_episode_start_abs_diff']}`")
    if "median_duration_relative_error" in summary:
        report_lines.append(f"- Median duration relative error: `{summary['median_duration_relative_error']}`")
    (session_dir / "hmm_sideband_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(session_dir)
    print(f"rows={summary['row_count']}")


if __name__ == "__main__":
    main()
