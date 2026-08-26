#!/usr/bin/env python3
"""Generates a minimal synthetic GGUF to test the M0 parser without
having to download a real model. Two small F32 tensors and three
metadata entries.

Usage: python tools/make_test_gguf.py [tests/test.gguf]
"""
import array
import struct
import sys


def s(x: str) -> bytes:
    """GGUF string: u64 length + UTF-8 bytes, no terminator."""
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "tests/test.gguf"

    buf = b"GGUF"
    buf += struct.pack("<I", 3)  # version
    buf += struct.pack("<Q", 2)  # n_tensors
    buf += struct.pack("<Q", 3)  # n_kv

    # metadata: string=8, u32=4, array=9
    buf += s("general.architecture") + struct.pack("<I", 8) + s("nanotest")
    buf += s("general.alignment") + struct.pack("<I", 4) + struct.pack("<I", 32)
    buf += (s("nanotest.dims") + struct.pack("<I", 9)
            + struct.pack("<I", 4) + struct.pack("<Q", 2)   # array of u32, 2 elements
            + struct.pack("<II", 4, 8))

    # tensor directory: 2 F32 tensors (type 0) of 8x4 = 32 elements = 128 bytes
    buf += (s("tok_embd.weight") + struct.pack("<I", 2)
            + struct.pack("<QQ", 8, 4) + struct.pack("<I", 0)
            + struct.pack("<Q", 0))
    buf += (s("output.weight") + struct.pack("<I", 2)
            + struct.pack("<QQ", 8, 4) + struct.pack("<I", 0)
            + struct.pack("<Q", 128))

    buf += b"\x00" * ((-len(buf)) % 32)  # padding up to alignment

    buf += array.array("f", [i * 0.5 for i in range(32)]).tobytes()
    buf += array.array("f", [float(i) for i in range(32)]).tobytes()

    with open(out, "wb") as fp:
        fp.write(buf)
    print(f"written {out} ({len(buf)} bytes)")


if __name__ == "__main__":
    main()
