from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ltst_v1_uart_decoder import (
    FRAME_LEN,
    FRAME_VERSION,
    MSG_TYPE_FEATURE_PACKET,
    PAYLOAD_LEN,
    SYNC,
    crc16_ccitt_false,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RUN_ROOT = PROJECT_ROOT / "artifact" / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture raw LTST V1 UART1 frame bytes to a .bin file.")
    parser.add_argument("--serial", required=True, help="UART1 serial port, for example COM6.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0.5, help="Serial read timeout in seconds.")
    parser.add_argument("--output", type=Path, help="Output .bin path. Defaults to artifact/runs under the repo root with a UTC timestamp.")
    parser.add_argument("--chunk-size", type=int, default=256, help="Serial read chunk size in bytes.")
    parser.add_argument("--duration", type=float, default=0.0, help="Capture duration in seconds. 0 means run until Ctrl+C or frame count is reached.")
    parser.add_argument("--count", type=int, default=0, help="Stop after this many valid feature frames. 0 means no explicit frame limit.")
    parser.add_argument("--progress-seconds", type=float, default=1.0, help="How often to print capture progress.")
    return parser.parse_args()


def default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("ltst_v1_uart_capture_%Y%m%dT%H%M%SZ.bin")
    return (DEFAULT_RUN_ROOT / stamp).resolve()


def count_frames_from_buffer(buffer: bytearray, frame_count: int) -> int:
    while True:
        sync_pos = buffer.find(SYNC)
        if sync_pos < 0:
            if len(buffer) > 1:
                del buffer[:-1]
            return frame_count
        if sync_pos > 0:
            del buffer[:sync_pos]
        if len(buffer) < FRAME_LEN:
            return frame_count

        frame = bytes(buffer[:FRAME_LEN])
        version = frame[2]
        msg_type = frame[3]
        payload_len = int.from_bytes(frame[4:6], "little")
        crc_rx = int.from_bytes(frame[6 + payload_len : 8 + payload_len], "little")

        if version != FRAME_VERSION or msg_type != MSG_TYPE_FEATURE_PACKET or payload_len != PAYLOAD_LEN:
            del buffer[:2]
            continue

        crc_calc = crc16_ccitt_false(frame[2 : 6 + payload_len])
        if crc_calc != crc_rx:
            del buffer[:FRAME_LEN]
            continue

        frame_count += 1
        del buffer[:FRAME_LEN]


def run_capture(args: argparse.Namespace) -> tuple[Path, int, int]:
    try:
        import serial  # type: ignore
    except Exception as exc:
        raise RuntimeError("pyserial is required for capture") from exc

    output_path = (args.output.resolve() if args.output else default_output_path())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    total_bytes = 0
    parser_buffer = bytearray()
    start = time.monotonic()
    last_progress = start

    with serial.Serial(port=args.serial, baudrate=args.baud, timeout=args.timeout) as ser:
        with output_path.open("wb") as fp:
            while True:
                chunk = ser.read(max(1, int(args.chunk_size)))
                now = time.monotonic()

                if chunk:
                    fp.write(chunk)
                    fp.flush()
                    total_bytes += len(chunk)
                    parser_buffer.extend(chunk)
                    frame_count = count_frames_from_buffer(parser_buffer, frame_count)

                if args.progress_seconds > 0 and (now - last_progress) >= args.progress_seconds:
                    elapsed = now - start
                    print(
                        f"[capture] elapsed={elapsed:.1f}s bytes={total_bytes} frames={frame_count} output={output_path}",
                        file=sys.stderr,
                    )
                    last_progress = now

                if args.count > 0 and frame_count >= args.count:
                    break
                if args.duration > 0 and (now - start) >= args.duration:
                    break

    return output_path, total_bytes, frame_count


def main() -> None:
    args = parse_args()
    try:
        output_path, total_bytes, frame_count = run_capture(args)
    except KeyboardInterrupt:
        print("[capture] interrupted by user", file=sys.stderr)
        raise SystemExit(130)

    print(f"output={output_path}")
    print(f"bytes={total_bytes}")
    print(f"frames={frame_count}")


if __name__ == "__main__":
    main()
