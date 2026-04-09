from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RUN_DIR = PROJECT_ROOT / "artifact" / "runs" / "ltst_full_86_20260405T035505Z"
DEFAULT_OUTPUT_HEADER = SCRIPT_DIR / "generated" / "ltst_v1_replay_samples_generated.h"
DEFAULT_OUTPUT_MANIFEST = SCRIPT_DIR / "generated" / "ltst_v1_replay_samples_manifest.json"
DEFAULT_RECORDS = ("s20021", "s20041", "s20151", "s30742")


@dataclass(frozen=True)
class ExportConfig:
    run_dir: Path
    output_header: Path
    output_manifest: Path
    records: tuple[str, ...]
    pn_dir: str
    channel_index: int
    max_valid_beats: int
    pre_roll_samples: int
    post_roll_samples: int
    q15_target_peak: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LTST replay-sample assets for the ESP32 live cardiac path.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-header", type=Path, default=DEFAULT_OUTPUT_HEADER)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--records", default=",".join(DEFAULT_RECORDS))
    parser.add_argument("--pn-dir", default="ltstdb/1.0.0")
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument("--max-valid-beats", type=int, default=128)
    parser.add_argument("--pre-roll-samples", type=int, default=400)
    parser.add_argument("--post-roll-samples", type=int, default=200)
    parser.add_argument("--q15-target-peak", type=int, default=16384)
    return parser.parse_args()


def parse_records(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


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


def load_beat_frame(run_dir: Path, record: str) -> pd.DataFrame:
    csv_path = run_dir / "beat_level" / f"{record}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing beat-level file for {record}: {csv_path}")
    return pd.read_csv(csv_path)


def load_signal_excerpt(cfg: ExportConfig, record: str, start_sample: int, end_sample: int) -> np.ndarray:
    try:
        import wfdb  # type: ignore
    except Exception as exc:
        raise RuntimeError("wfdb is required to export LTST replay samples") from exc

    rec = wfdb.rdrecord(
        record,
        pn_dir=cfg.pn_dir,
        sampfrom=int(start_sample),
        sampto=int(end_sample),
        channels=[int(cfg.channel_index)],
        physical=True,
    )
    if rec.p_signal is None or rec.p_signal.shape[1] == 0:
        raise RuntimeError(f"No physical signal returned for {record}")
    signal = np.asarray(rec.p_signal[:, 0], dtype=np.float64)
    return signal


def build_record_asset(cfg: ExportConfig, record: str) -> tuple[dict[str, Any], np.ndarray]:
    beat_df = load_beat_frame(cfg.run_dir, record)
    beat_df = beat_df.loc[beat_df["beat_symbol"].astype(str) == "N"].copy()
    beat_df["status"] = beat_df["status"].fillna(999).astype(int)
    beat_df["beat_sample"] = beat_df["beat_sample"].astype(int)

    valid = beat_df.loc[beat_df["status"] == 0].copy()
    if valid.empty:
        raise RuntimeError(f"{record}: no valid beats available in beat-level export")

    chosen_valid = valid.head(int(cfg.max_valid_beats)).copy()
    first_sample = int(chosen_valid["beat_sample"].min())
    last_sample = int(chosen_valid["beat_sample"].max())
    start_sample = max(0, first_sample - int(cfg.pre_roll_samples))
    end_sample = int(last_sample + cfg.post_roll_samples + 1)

    excerpt_df = beat_df.loc[(beat_df["beat_sample"] >= start_sample) & (beat_df["beat_sample"] < end_sample)].copy()
    signal = load_signal_excerpt(cfg, record, start_sample=start_sample, end_sample=end_sample)
    centered = signal - float(np.nanmedian(signal))
    peak_abs = float(np.max(np.abs(centered))) if centered.size else 0.0
    sample_scale = float(cfg.q15_target_peak) / max(peak_abs, 1e-6)
    q15 = np.clip(np.rint(centered * sample_scale), -32768, 32767).astype(np.int16)

    expected_beats: list[dict[str, Any]] = []
    for beat_index, row in enumerate(excerpt_df.itertuples(index=False), start=1):
        feature_valid = int(getattr(row, "status")) == 0
        expected_beats.append(
            {
                "beat_index": int(beat_index),
                "sample_index": int(getattr(row, "beat_sample") - start_sample),
                "original_beat_sample": int(getattr(row, "beat_sample")),
                "rr_samples": int(round(float(getattr(row, "rr_samples")))) if np.isfinite(getattr(row, "rr_samples")) else 0,
                "feature_valid": bool(feature_valid),
                "poincare_b": float(getattr(row, "poincare_b")) if feature_valid and np.isfinite(getattr(row, "poincare_b")) else None,
                "drift_norm": float(getattr(row, "drift_norm")) if feature_valid and np.isfinite(getattr(row, "drift_norm")) else None,
                "energy_asym": float(getattr(row, "energy_asym")) if feature_valid and np.isfinite(getattr(row, "energy_asym")) else None,
            }
        )

    manifest_entry = {
        "record": record,
        "channel_index": int(cfg.channel_index),
        "pn_dir": cfg.pn_dir,
        "start_sample": int(start_sample),
        "end_sample": int(end_sample),
        "sample_count": int(len(q15)),
        "sample_scale": float(sample_scale),
        "expected_beats": expected_beats,
    }
    return manifest_entry, q15


def format_c_array(name: str, values: np.ndarray) -> str:
    chunks: list[str] = []
    line: list[str] = []
    for idx, value in enumerate(values.tolist(), start=1):
        line.append(str(int(value)))
        if idx % 12 == 0:
            chunks.append("    " + ", ".join(line))
            line = []
    if line:
        chunks.append("    " + ", ".join(line))
    body = ",\n".join(chunks)
    return f"static const int16_t {name}[] = {{\n{body}\n}};\n"


def write_header(cfg: ExportConfig, assets: list[tuple[dict[str, Any], np.ndarray]]) -> None:
    cfg.output_header.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "#ifndef LTST_V1_REPLAY_SAMPLES_GENERATED_H",
        "#define LTST_V1_REPLAY_SAMPLES_GENERATED_H",
        "",
        "#include <stdint.h>",
        "",
        "#ifdef __cplusplus",
        "extern \"C\" {",
        "#endif",
        "",
        "typedef struct ltst_v1_generated_trace_t {",
        "    const char *record_name;",
        "    const int16_t *samples_q15;",
        "    uint32_t sample_count;",
        "    float sample_scale;",
        "} ltst_v1_generated_trace_t;",
        "",
    ]

    for entry, samples in assets:
        array_name = f"LTST_V1_TRACE_{entry['record'].upper()}"
        lines.append(format_c_array(array_name, samples))

    lines.append(f"#define LTST_V1_GENERATED_TRACE_COUNT {len(assets)}u")
    lines.append("static const ltst_v1_generated_trace_t LTST_V1_GENERATED_TRACES[] = {")
    for entry, _samples in assets:
        array_name = f"LTST_V1_TRACE_{entry['record'].upper()}"
        lines.append(
            f"    {{\"{entry['record']}\", {array_name}, {entry['sample_count']}u, {entry['sample_scale']:.8f}f}},"
        )
    lines.extend(
        [
            "};",
            "",
            "#ifdef __cplusplus",
            "}",
            "#endif",
            "",
            "#endif /* LTST_V1_REPLAY_SAMPLES_GENERATED_H */",
            "",
        ]
    )
    cfg.output_header.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = ExportConfig(
        run_dir=args.run_dir.resolve(),
        output_header=args.output_header.resolve(),
        output_manifest=args.output_manifest.resolve(),
        records=parse_records(args.records),
        pn_dir=str(args.pn_dir),
        channel_index=int(args.channel_index),
        max_valid_beats=int(args.max_valid_beats),
        pre_roll_samples=int(args.pre_roll_samples),
        post_roll_samples=int(args.post_roll_samples),
        q15_target_peak=int(args.q15_target_peak),
    )

    assets: list[tuple[dict[str, Any], np.ndarray]] = []
    manifest = {
        "config": json_safe(cfg.__dict__),
        "records": {},
    }
    for record in cfg.records:
        entry, samples = build_record_asset(cfg, record)
        assets.append((entry, samples))
        manifest["records"][record] = json_safe(entry)

    cfg.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_header(cfg, assets)
    cfg.output_manifest.write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")

    print(cfg.output_header)
    print(cfg.output_manifest)


if __name__ == "__main__":
    main()
