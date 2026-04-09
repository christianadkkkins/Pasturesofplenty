from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RUN_DIR = PROJECT_ROOT / "artifact" / "runs" / "ltst_full_86_20260405T035505Z"
DEFAULT_WINDOWS_CSV = DEFAULT_RUN_DIR / "transition_hmm" / "ltst_transition_hmm_windows.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a replay-sample HMM reference CSV from the host LTST transition-HMM windows.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--windows-csv", type=Path, default=DEFAULT_WINDOWS_CSV)
    parser.add_argument("--manifest-path", type=Path, help="Override manifest path from session metadata.")
    parser.add_argument("--record", help="Override source record from session metadata.")
    parser.add_argument("--chunksize", type=int, default=200000)
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


def build_episode_annotations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df = df.copy()
        df["episode_start_beat"] = []
        df["active_duration_beats"] = []
        return df

    episode_start_beats: list[int] = []
    active_duration_beats: list[int] = []
    in_episode = False
    episode_start = 0

    for row in df.itertuples(index=False):
        state = str(row.state)
        beat_index = int(row.beat_index)
        is_excited = state in {"transition", "active"}
        if is_excited and not in_episode:
            in_episode = True
            episode_start = beat_index
        elif not is_excited:
            in_episode = False
            episode_start = 0

        episode_start_beats.append(episode_start if in_episode else 0)
        if state == "active" and in_episode:
            active_duration_beats.append(int(beat_index - episode_start + 1))
        else:
            active_duration_beats.append(0)

    out = df.copy()
    out["episode_start_beat"] = episode_start_beats
    out["active_duration_beats"] = active_duration_beats
    return out


def load_record_windows(windows_csv: Path, record: str, chunksize: int) -> pd.DataFrame:
    usecols = ["record", "beat_sample", "state"]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(windows_csv, usecols=usecols, chunksize=int(chunksize)):
        matched = chunk.loc[chunk["record"].astype(str) == record, usecols[1:]].copy()
        if not matched.empty:
            parts.append(matched)
    if not parts:
        return pd.DataFrame(columns=["beat_sample", "state"])
    frame = pd.concat(parts, ignore_index=True)
    frame["beat_sample"] = frame["beat_sample"].astype(int)
    frame["state"] = frame["state"].astype(str)
    return frame.sort_values("beat_sample").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    session_dir = args.session_dir.resolve()
    metadata = json.loads((session_dir / "session_metadata.json").read_text(encoding="utf-8"))
    manifest_path = args.manifest_path.resolve() if args.manifest_path else Path(metadata["replay_manifest_path"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = args.record or metadata.get("source_record")
    if not record:
        raise ValueError("No source record found in session metadata and no --record override was provided")

    record_entry = manifest.get("records", {}).get(record)
    if record_entry is None:
        raise KeyError(f"Record '{record}' not found in manifest {manifest_path}")

    expected_df = pd.DataFrame(record_entry.get("expected_beats", []))
    if expected_df.empty:
        raise ValueError(f"No expected beats found for {record}")

    windows_df = load_record_windows(args.windows_csv.resolve(), record=record, chunksize=int(args.chunksize))
    if windows_df.empty:
        raise ValueError(f"No host transition-HMM rows found for record {record} in {args.windows_csv}")

    sample_to_beat = expected_df[["beat_index", "original_beat_sample", "sample_index"]].copy()
    sample_to_beat["beat_index"] = sample_to_beat["beat_index"].astype(int)
    sample_to_beat["original_beat_sample"] = sample_to_beat["original_beat_sample"].astype(int)
    sample_to_beat["sample_index"] = sample_to_beat["sample_index"].astype(int)

    merged = sample_to_beat.merge(
        windows_df,
        left_on="original_beat_sample",
        right_on="beat_sample",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlap between replay manifest beat samples and host transition-HMM windows")

    reference = merged[["beat_index", "sample_index", "original_beat_sample", "state"]].copy()
    reference = reference.sort_values("beat_index").reset_index(drop=True)
    reference = build_episode_annotations(reference)

    summary = {
        "record": record,
        "reference_rows": int(len(reference)),
        "state_counts": reference["state"].value_counts().to_dict(),
        "manifest_path": str(manifest_path),
        "windows_csv": str(args.windows_csv.resolve()),
    }

    out_csv = session_dir / "hmm_reference.csv"
    out_json = session_dir / "hmm_reference_summary.json"
    out_csv.write_text(reference.to_csv(index=False), encoding="utf-8")
    out_json.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")

    print(out_csv)
    print(f"rows={summary['reference_rows']}")


if __name__ == "__main__":
    main()
