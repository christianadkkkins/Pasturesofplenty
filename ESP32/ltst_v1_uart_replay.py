from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RECORDS = ("s20021", "s20041", "s20151", "s30742")
DEFAULT_RUN_DIR = PROJECT_ROOT / "artifact" / "runs" / "ltst_full_86_20260405T035505Z"
FEATURE_VALID_FLAG = 0x0001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send LTST exemplar replay commands over UART0 to the ESP32 bring-up app.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="LTST run directory containing invariant_summary.csv")
    parser.add_argument("--records", default=",".join(DEFAULT_RECORDS), help="Comma-separated exemplar records to replay")
    parser.add_argument("--serial", help="Console UART port, for example COM5. Not required with --dry-run.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--line-delay-ms", type=int, default=150, help="Delay between commands in milliseconds")
    parser.add_argument("--boot-delay-ms", type=int, default=300, help="Delay after opening the port before the first command")
    parser.add_argument("--flags-hex", default="0001", help="Flags to send with each exemplar packet, default FEATURE_VALID")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without opening a serial port")
    parser.add_argument("--status-after", action="store_true", help="Send a final status command after replay")
    return parser.parse_args()


def parse_records_arg(records_arg: str) -> list[str]:
    return [piece.strip() for piece in str(records_arg).split(",") if piece.strip()]


def load_exemplar_vectors(run_dir: Path, records: list[str]) -> list[dict[str, object]]:
    invariant_path = run_dir / "invariant_summary.csv"
    df = pd.read_csv(invariant_path)
    subset = df.loc[df["record"].astype(str).isin(records)].copy()
    found = subset["record"].astype(str).tolist()
    missing = [record for record in records if record not in found]
    if missing:
        raise FileNotFoundError(f"Missing exemplar rows in invariant_summary.csv: {', '.join(missing)}")

    subset["record"] = pd.Categorical(subset["record"], categories=records, ordered=True)
    subset = subset.sort_values("record").reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for beat_index, (_, row) in enumerate(subset.iterrows(), start=1):
        rows.append(
            {
                "record": str(row["record"]),
                "beat_index": int(beat_index),
                "poincare_b": float(row["poincare_b_p50"]),
                "drift_norm": float(row["median_drift_norm"]),
                "energy_asym": float(row["median_energy_asym"]),
            }
        )
    return rows


def build_commands(vectors: list[dict[str, object]], flags_hex: str, status_after: bool) -> list[str]:
    commands = ["mode replay"]
    for vector in vectors:
        commands.append(
            "packet {beat_index} {poincare_b:.10f} {drift_norm:.10f} {energy_asym:.10f} {flags}".format(
                beat_index=int(vector["beat_index"]),
                poincare_b=float(vector["poincare_b"]),
                drift_norm=float(vector["drift_norm"]),
                energy_asym=float(vector["energy_asym"]),
                flags=str(flags_hex),
            )
        )
    if status_after:
        commands.append("status")
    return commands


def print_plan(vectors: list[dict[str, object]], commands: list[str]) -> None:
    print("LTST exemplar replay plan:", file=sys.stderr)
    for vector in vectors:
        print(
            "  {record}: beat={beat_index} pb={poincare_b:.6f} drift={drift_norm:.6f} asym={energy_asym:.6f}".format(
                record=str(vector["record"]),
                beat_index=int(vector["beat_index"]),
                poincare_b=float(vector["poincare_b"]),
                drift_norm=float(vector["drift_norm"]),
                energy_asym=float(vector["energy_asym"]),
            ),
            file=sys.stderr,
        )
    print("commands:", file=sys.stderr)
    for command in commands:
        print(f"  {command}", file=sys.stderr)


def run_serial(port: str, baud: int, timeout: float, boot_delay_ms: int, line_delay_ms: int, commands: list[str]) -> None:
    try:
        import serial  # type: ignore
    except Exception as exc:
        raise RuntimeError("pyserial is required unless --dry-run is used") from exc

    with serial.Serial(port=port, baudrate=baud, timeout=timeout, write_timeout=timeout) as ser:
        time.sleep(max(0.0, boot_delay_ms / 1000.0))
        for command in commands:
            line = command + "\n"
            ser.write(line.encode("ascii"))
            ser.flush()
            time.sleep(max(0.0, line_delay_ms / 1000.0))


def main() -> None:
    args = parse_args()
    records = parse_records_arg(args.records)
    vectors = load_exemplar_vectors(args.run_dir.resolve(), records)
    commands = build_commands(vectors, flags_hex=str(args.flags_hex), status_after=bool(args.status_after))
    print_plan(vectors, commands)

    if args.dry_run:
        for command in commands:
            print(command)
        return

    if not args.serial:
        raise SystemExit("--serial is required unless --dry-run is used")

    run_serial(
        port=str(args.serial),
        baud=int(args.baud),
        timeout=float(args.timeout),
        boot_delay_ms=int(args.boot_delay_ms),
        line_delay_ms=int(args.line_delay_ms),
        commands=commands,
    )


if __name__ == "__main__":
    main()
