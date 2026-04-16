from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from typing import Any
from pathlib import Path


def add_interval(store: dict[str, list[tuple[int, int]]], record: str, start: int, end: int) -> None:
    if end < start:
        return
    store[record].append((start, end))


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: list[list[int]] = [[intervals[0][0], intervals[0][1]]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + 1:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def interval_length(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start + 1 for start, end in intervals)


def intersection_length(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    i = 0
    j = 0
    total = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if start <= end:
            total += end - start + 1
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def compute_metrics_from_tables(
    *,
    record_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    pre_event_beats: int,
    record_filter: set[str] | None = None,
) -> list[tuple[str, float | int]]:
    record_lengths: dict[str, int] = {}
    for row in record_rows:
        record = str(row["record"])
        if record_filter is not None and record not in record_filter:
            continue
        record_lengths[record] = int(row["n_beats"])

    lead_pos: dict[str, list[tuple[int, int]]] = defaultdict(list)
    strict_pos: dict[str, list[tuple[int, int]]] = defaultdict(list)
    n_events = 0
    for row in event_rows:
        record = str(row["record"])
        if record not in record_lengths:
            continue
        start = int(row["event_start_idx"])
        end = int(row["event_end_idx"])
        add_interval(strict_pos, record, start, end)
        add_interval(lead_pos, record, max(0, start - pre_event_beats), end)
        n_events += 1

    pred: dict[str, list[tuple[int, int]]] = defaultdict(list)
    raw_pred_rows: list[tuple[str, int, int]] = []
    for row in episode_rows:
        if row.get("method", "hmm") != "hmm":
            continue
        record = str(row["record"])
        if record not in record_lengths:
            continue
        start = int(row["start_idx"])
        end = int(row["end_idx"])
        add_interval(pred, record, start, end)
        raw_pred_rows.append((record, start, end))

    lead_pos = {key: merge_intervals(value) for key, value in lead_pos.items()}
    strict_pos = {key: merge_intervals(value) for key, value in strict_pos.items()}
    pred = {key: merge_intervals(value) for key, value in pred.items()}

    false_positive_episode_count = 0
    for record, start, end in raw_pred_rows:
        overlaps = False
        for pos_start, pos_end in lead_pos.get(record, []):
            if pos_end < start:
                continue
            if pos_start > end:
                break
            overlaps = True
            break
        if not overlaps:
            false_positive_episode_count += 1

    total_beats = sum(record_lengths.values())
    pred_beats = sum(interval_length(intervals) for intervals in pred.values())
    lead_pos_beats = sum(interval_length(intervals) for intervals in lead_pos.values())
    strict_pos_beats = sum(interval_length(intervals) for intervals in strict_pos.values())
    lead_tp = sum(intersection_length(pred.get(record, []), lead_pos.get(record, [])) for record in record_lengths)
    strict_tp = sum(intersection_length(pred.get(record, []), strict_pos.get(record, [])) for record in record_lengths)
    lead_fp = pred_beats - lead_tp
    strict_fp = pred_beats - strict_tp
    lead_tn = (total_beats - lead_pos_beats) - lead_fp
    strict_tn = (total_beats - strict_pos_beats) - strict_fp
    lead_fn = lead_pos_beats - lead_tp
    strict_fn = strict_pos_beats - strict_tp

    def ratio(num: int, den: int) -> float:
        return float(num) / float(den) if den else 0.0

    return [
        ("n_events", n_events),
        ("pre_event_beats", pre_event_beats),
        ("total_beats", total_beats),
        ("predicted_positive_beats", pred_beats),
        ("lead_positive_beats", lead_pos_beats),
        ("strict_positive_beats", strict_pos_beats),
        ("lead_true_positive_beats", lead_tp),
        ("lead_false_positive_beats", lead_fp),
        ("lead_true_negative_beats", lead_tn),
        ("lead_false_negative_beats", lead_fn),
        ("strict_true_positive_beats", strict_tp),
        ("strict_false_positive_beats", strict_fp),
        ("strict_true_negative_beats", strict_tn),
        ("strict_false_negative_beats", strict_fn),
        ("lead_specificity", ratio(lead_tn, lead_tn + lead_fp)),
        ("lead_sensitivity", ratio(lead_tp, lead_tp + lead_fn)),
        ("lead_precision", ratio(lead_tp, lead_tp + lead_fp)),
        ("strict_specificity", ratio(strict_tn, strict_tn + strict_fp)),
        ("strict_sensitivity", ratio(strict_tp, strict_tp + strict_fn)),
        ("strict_precision", ratio(strict_tp, strict_tp + strict_fp)),
        ("alert_occupancy_fraction", ratio(pred_beats, total_beats)),
        ("episode_count", len(raw_pred_rows)),
        ("false_positive_episode_count", false_positive_episode_count),
        ("false_positive_episode_fraction", ratio(false_positive_episode_count, len(raw_pred_rows))),
    ]


def compute_metrics(run_dir: Path, pre_event_beats: int, record_filter: set[str] | None = None) -> list[tuple[str, float | int]]:
    with (run_dir / "ltst_transition_hmm_record_summary.csv").open(newline="", encoding="utf-8") as handle:
        record_rows = list(csv.DictReader(handle))
    with (run_dir / "ltst_transition_hmm_event_table.csv").open(newline="", encoding="utf-8") as handle:
        event_rows = list(csv.DictReader(handle))
    with (run_dir / "ltst_transition_hmm_episodes.csv").open(newline="", encoding="utf-8") as handle:
        episode_rows = list(csv.DictReader(handle))
    return compute_metrics_from_tables(
        record_rows=record_rows,
        event_rows=event_rows,
        episode_rows=episode_rows,
        pre_event_beats=pre_event_beats,
        record_filter=record_filter,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive confusion-style metrics for a completed LTST transition-HMM run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to a completed transition_hmm* output directory.")
    parser.add_argument("--pre-event-beats", type=int, default=500, help="Positive lookback window used for lead-aware metrics.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <run-dir>/ltst_transition_hmm_derived_metrics.csv.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_path = args.output.resolve() if args.output else run_dir / "ltst_transition_hmm_derived_metrics.csv"
    metrics = compute_metrics(run_dir=run_dir, pre_event_beats=args.pre_event_beats)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for metric, value in metrics:
            writer.writerow([metric, value])

    print(output_path)


if __name__ == "__main__":
    main()
