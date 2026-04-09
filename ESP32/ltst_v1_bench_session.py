from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ltst_v1_uart_capture import count_frames_from_buffer
from ltst_v1_uart_decoder import (
    FRAME_LEN,
    FRAME_VERSION,
    MSG_TYPE_FEATURE_PACKET,
    PAYLOAD_LEN,
    SYNC,
    crc16_ccitt_false,
    decode_payload,
)
from ltst_v1_uart_replay import build_commands, load_exemplar_vectors, parse_records_arg


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RUN_DIR = PROJECT_ROOT / "artifact" / "runs" / "ltst_full_86_20260405T035505Z"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifact" / "runs"


@dataclass(frozen=True)
class BenchSessionConfig:
    run_dir: Path
    output_root: Path
    source_mode: str
    records: tuple[str, ...]
    record: str | None
    flags_hex: str
    manifest_path: Path | None
    console_serial: str | None
    data_serial: str | None
    console_baud: int = 115200
    data_baud: int = 115200
    console_timeout: float = 1.0
    data_timeout: float = 0.5
    chunk_size: int = 256
    line_delay_ms: int = 150
    boot_delay_ms: int = 300
    final_console_drain_ms: int = 500
    progress_seconds: float = 1.0
    status_after: bool = True
    dry_run: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one LTST UART replay session end-to-end: send commands, capture UART1, and decode the capture.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-mode", choices=["replay_packets", "replay_samples"], default="replay_packets")
    parser.add_argument("--records", default="s20021,s20041,s20151,s30742")
    parser.add_argument("--record", default="s20021", help="Record name for replay_samples mode.")
    parser.add_argument("--flags-hex", default="0001")
    parser.add_argument("--manifest-path", type=Path, default=PROJECT_ROOT / "ESP32" / "generated" / "ltst_v1_replay_samples_manifest.json")
    parser.add_argument("--console-serial", help="UART0 console port, for example COM5. Required unless --dry-run.")
    parser.add_argument("--data-serial", help="UART1 binary port, for example COM6. Required unless --dry-run.")
    parser.add_argument("--console-baud", type=int, default=115200)
    parser.add_argument("--data-baud", type=int, default=115200)
    parser.add_argument("--console-timeout", type=float, default=1.0)
    parser.add_argument("--data-timeout", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--line-delay-ms", type=int, default=150)
    parser.add_argument("--boot-delay-ms", type=int, default=300)
    parser.add_argument("--final-console-drain-ms", type=int, default=500)
    parser.add_argument("--progress-seconds", type=float, default=1.0)
    parser.add_argument("--no-status-after", action="store_true", help="Do not append a final status command.")
    parser.add_argument("--dry-run", action="store_true", help="Write the session folder and command log without opening serial ports.")
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create_session_dir(output_root: Path) -> Path:
    session_dir = output_root / f"ltst_v1_bench_session_{utc_stamp()}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Replay manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def decode_capture_bytes(raw_bytes: bytes) -> tuple[list[str], int, int]:
    buffer = bytearray(raw_bytes)
    decoded_lines: list[str] = []
    frame_count = 0
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

        frame_count += 1
        beat_index, pb_raw, drift_raw, asym_raw, flags = decode_payload(payload)
        poincare_b = pb_raw / float(1 << 14)
        drift_norm = drift_raw / float(1 << 12)
        energy_asym = asym_raw / float(1 << 14)
        decoded_lines.append(
            f"[{frame_count:04d}] beat={beat_index} pb_raw={pb_raw} drift_raw={drift_raw} asym_raw={asym_raw} "
            f"pb={poincare_b:.6f} drift={drift_norm:.6f} asym={energy_asym:.6f} flags=0x{flags:04X} crc=0x{crc_rx:04X}"
        )
        del buffer[:FRAME_LEN]

    return decoded_lines, frame_count, bad_frames


def capture_worker(
    port: str,
    baud: int,
    timeout: float,
    chunk_size: int,
    expected_frames: int,
    raw_path: Path,
    stop_event: threading.Event,
    progress_seconds: float,
    result: dict[str, Any],
) -> None:
    try:
        import serial  # type: ignore
    except Exception as exc:
        result["error"] = f"pyserial is required for capture: {exc}"
        return

    total_bytes = 0
    frame_count = 0
    parser_buffer = bytearray()
    start = time.monotonic()
    last_progress = start

    try:
        with serial.Serial(port=port, baudrate=baud, timeout=timeout) as ser:
            with raw_path.open("wb") as fp:
                while not stop_event.is_set():
                    chunk = ser.read(max(1, int(chunk_size)))
                    now = time.monotonic()
                    if chunk:
                        fp.write(chunk)
                        fp.flush()
                        total_bytes += len(chunk)
                        parser_buffer.extend(chunk)
                        frame_count = count_frames_from_buffer(parser_buffer, frame_count)
                    if progress_seconds > 0 and (now - last_progress) >= progress_seconds:
                        print(f"[bench-capture] elapsed={now - start:.1f}s bytes={total_bytes} frames={frame_count}", file=sys.stderr)
                        last_progress = now
                    if expected_frames > 0 and frame_count >= expected_frames:
                        break
    except Exception as exc:
        result["error"] = str(exc)
        return

    result["bytes"] = total_bytes
    result["frames"] = frame_count


def drain_console(ser: Any, duration_ms: int) -> str:
    deadline = time.monotonic() + max(0.0, duration_ms / 1000.0)
    parts: list[bytes] = []
    while time.monotonic() < deadline:
        chunk = ser.read(256)
        if chunk:
            parts.append(chunk)
        else:
            time.sleep(0.02)
    if not parts:
        return ""
    return b"".join(parts).decode("utf-8", errors="replace")


def run_console_session(cfg: BenchSessionConfig, commands: list[str], console_log_path: Path) -> None:
    try:
        import serial  # type: ignore
    except Exception as exc:
        raise RuntimeError("pyserial is required for console replay") from exc

    log_parts: list[str] = []
    with serial.Serial(port=str(cfg.console_serial), baudrate=cfg.console_baud, timeout=cfg.console_timeout, write_timeout=cfg.console_timeout) as ser:
        time.sleep(max(0.0, cfg.boot_delay_ms / 1000.0))
        boot_text = drain_console(ser, 200)
        if boot_text:
            log_parts.append("=== boot ===\n" + boot_text)
        for command in commands:
            ser.write((command + "\n").encode("ascii"))
            ser.flush()
            time.sleep(max(0.0, cfg.line_delay_ms / 1000.0))
            reply = drain_console(ser, cfg.line_delay_ms)
            log_parts.append(f"> {command}\n{reply}")
        tail = drain_console(ser, cfg.final_console_drain_ms)
        if tail:
            log_parts.append("=== tail ===\n" + tail)

    write_text(console_log_path, "\n".join(log_parts))


def main() -> None:
    args = parse_args()
    cfg = BenchSessionConfig(
        run_dir=args.run_dir.resolve(),
        output_root=args.output_root.resolve(),
        source_mode=str(args.source_mode),
        records=tuple(parse_records_arg(args.records)),
        record=str(args.record) if args.record else None,
        flags_hex=str(args.flags_hex),
        manifest_path=args.manifest_path.resolve() if args.manifest_path else None,
        console_serial=args.console_serial,
        data_serial=args.data_serial,
        console_baud=int(args.console_baud),
        data_baud=int(args.data_baud),
        console_timeout=float(args.console_timeout),
        data_timeout=float(args.data_timeout),
        chunk_size=int(args.chunk_size),
        line_delay_ms=int(args.line_delay_ms),
        boot_delay_ms=int(args.boot_delay_ms),
        final_console_drain_ms=int(args.final_console_drain_ms),
        progress_seconds=float(args.progress_seconds),
        status_after=not bool(args.no_status_after),
        dry_run=bool(args.dry_run),
    )

    if not cfg.dry_run and (not cfg.console_serial or not cfg.data_serial):
        raise SystemExit("--console-serial and --data-serial are required unless --dry-run is used")

    session_dir = create_session_dir(cfg.output_root)
    raw_capture_path = session_dir / "uart1_capture.bin"
    decoded_text_path = session_dir / "decoded_packets.txt"
    commands_path = session_dir / "commands_sent.txt"
    console_log_path = session_dir / "console_output.txt"
    metadata_path = session_dir / "session_metadata.json"

    metadata: dict[str, Any] = {
        "config": json_safe(asdict(cfg)),
        "session_dir": str(session_dir),
        "vectors": [],
        "commands": [],
    }

    expected_frames = 0
    if cfg.source_mode == "replay_packets":
        vectors = load_exemplar_vectors(cfg.run_dir, list(cfg.records))
        commands = build_commands(vectors, flags_hex=cfg.flags_hex, status_after=cfg.status_after)
        metadata["vectors"] = vectors
    else:
        manifest_missing = False
        try:
            manifest = load_manifest(cfg.manifest_path) if cfg.manifest_path is not None else {"records": {}}
        except FileNotFoundError:
            manifest_missing = True
            manifest = {"records": {}}
        record_name = cfg.record or ""
        record_entry = manifest.get("records", {}).get(record_name)
        commands = [
            "mode replay_samples",
            f"record {record_name}",
            "hmm on",
        ]
        if cfg.status_after:
            commands.append("status")
        metadata["source_record"] = record_name
        metadata["replay_manifest_path"] = str(cfg.manifest_path)
        if record_entry is None:
            message = (
                f"Record '{record_name}' not found in replay manifest {cfg.manifest_path}. "
                "Generate replay assets with ltst_v1_export_replay_samples.py first."
            )
            if manifest_missing:
                message = (
                    f"Replay manifest not found at {cfg.manifest_path}. "
                    "Generate replay assets with ltst_v1_export_replay_samples.py first."
                )
            if not cfg.dry_run:
                raise KeyError(message)
            metadata["replay_manifest_warning"] = message
            metadata["replay_manifest_entry"] = None
        else:
            expected_frames = int(len(record_entry.get("expected_beats", [])))
            metadata["replay_manifest_entry"] = record_entry

    metadata["commands"] = commands
    write_text(commands_path, "\n".join(commands) + "\n")

    if cfg.dry_run:
        write_text(console_log_path, "dry-run: console session not executed\n")
        raw_capture_path.write_bytes(b"")
        write_text(decoded_text_path, "")
        metadata["capture_bytes"] = 0
        metadata["capture_frames"] = 0
        metadata["capture_bad_frames"] = 0
        write_text(metadata_path, json.dumps(metadata, indent=2))
        print(session_dir)
        return

    stop_event = threading.Event()
    capture_result: dict[str, Any] = {}
    capture_thread = threading.Thread(
        target=capture_worker,
        args=(
            str(cfg.data_serial),
            cfg.data_baud,
            cfg.data_timeout,
            cfg.chunk_size,
            expected_frames if cfg.source_mode == "replay_samples" else len(metadata["vectors"]),
            raw_capture_path,
            stop_event,
            cfg.progress_seconds,
            capture_result,
        ),
        daemon=True,
    )
    capture_thread.start()
    time.sleep(0.1)

    try:
        run_console_session(cfg, commands, console_log_path)
    finally:
        capture_thread.join(timeout=5.0)
        stop_event.set()
        capture_thread.join(timeout=1.0)

    if "error" in capture_result:
        raise RuntimeError(f"capture failed: {capture_result['error']}")

    raw_bytes = raw_capture_path.read_bytes() if raw_capture_path.exists() else b""
    decoded_lines, frame_count, bad_frames = decode_capture_bytes(raw_bytes)
    write_text(decoded_text_path, ("\n".join(decoded_lines) + ("\n" if decoded_lines else "")))

    metadata["capture_bytes"] = int(capture_result.get("bytes", len(raw_bytes)))
    metadata["capture_frames"] = int(frame_count)
    metadata["capture_bad_frames"] = int(bad_frames)
    write_text(metadata_path, json.dumps(metadata, indent=2))

    print(session_dir)
    print(f"frames={frame_count}")
    print(f"bytes={metadata['capture_bytes']}")


if __name__ == "__main__":
    main()
