from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


SYNC = b"\xA5\x5A"
FRAME_VERSION = 0x01
MSG_TYPE_FEATURE_PACKET = 0x01
PAYLOAD_LEN = 12
FRAME_LEN = 2 + 1 + 1 + 2 + PAYLOAD_LEN + 2
PACKET_STRUCT = struct.Struct("<IHHhH")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode LTST V1 UART frames from a file or serial port.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Binary frame capture file.")
    source.add_argument("--serial", help="Serial port to read from, for example COM5.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--count", type=int, default=0, help="Stop after decoding this many packets. 0 means no explicit limit.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Serial timeout in seconds.")
    return parser.parse_args()


def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def decode_payload(payload: bytes) -> tuple[int, int, int, int, int]:
    return PACKET_STRUCT.unpack(payload)


def print_packet(index: int, payload: bytes, crc_expected: int) -> None:
    beat_index, pb_raw, drift_raw, asym_raw, flags = decode_payload(payload)
    poincare_b = pb_raw / float(1 << 14)
    drift_norm = drift_raw / float(1 << 12)
    energy_asym = asym_raw / float(1 << 14)
    print(
        f"[{index:04d}] beat={beat_index} pb_raw={pb_raw} drift_raw={drift_raw} asym_raw={asym_raw} "
        f"pb={poincare_b:.6f} drift={drift_norm:.6f} asym={energy_asym:.6f} flags=0x{flags:04X} crc=0x{crc_expected:04X}"
    )


def consume_buffer(buffer: bytearray, packet_limit: int, packet_count: int) -> int:
    while True:
        sync_pos = buffer.find(SYNC)
        if sync_pos < 0:
            if len(buffer) > 1:
                del buffer[:-1]
            return packet_count
        if sync_pos > 0:
            del buffer[:sync_pos]
        if len(buffer) < FRAME_LEN:
            return packet_count

        frame = bytes(buffer[:FRAME_LEN])
        version = frame[2]
        msg_type = frame[3]
        payload_len = int.from_bytes(frame[4:6], "little")
        payload = frame[6 : 6 + payload_len]
        crc_rx = int.from_bytes(frame[6 + payload_len : 8 + payload_len], "little")

        if version != FRAME_VERSION or msg_type != MSG_TYPE_FEATURE_PACKET or payload_len != PAYLOAD_LEN:
            del buffer[:2]
            continue

        crc_calc = crc16_ccitt_false(frame[2 : 6 + payload_len])
        if crc_calc != crc_rx:
            print(f"[warn] crc mismatch: got 0x{crc_rx:04X} expected 0x{crc_calc:04X}", file=sys.stderr)
            del buffer[:FRAME_LEN]
            continue

        packet_count += 1
        print_packet(packet_count, payload, crc_rx)
        del buffer[:FRAME_LEN]
        if packet_limit > 0 and packet_count >= packet_limit:
            return packet_count


def run_file(path: Path, packet_limit: int) -> int:
    buffer = bytearray(path.read_bytes())
    return consume_buffer(buffer, packet_limit, 0)


def run_serial(port: str, baud: int, timeout: float, packet_limit: int) -> int:
    try:
        import serial  # type: ignore
    except Exception as exc:
        raise RuntimeError("pyserial is required for --serial mode") from exc

    packet_count = 0
    buffer = bytearray()
    with serial.Serial(port=port, baudrate=baud, timeout=timeout) as ser:
        while packet_limit <= 0 or packet_count < packet_limit:
            chunk = ser.read(256)
            if not chunk:
                break
            buffer.extend(chunk)
            packet_count = consume_buffer(buffer, packet_limit, packet_count)
    return packet_count


def main() -> None:
    args = parse_args()
    if args.input is not None:
        count = run_file(args.input, args.count)
    else:
        count = run_serial(args.serial, args.baud, args.timeout, args.count)
    print(f"decoded_packets={count}", file=sys.stderr)


if __name__ == "__main__":
    main()
