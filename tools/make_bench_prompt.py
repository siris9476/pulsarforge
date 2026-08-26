#!/usr/bin/env python3
"""Generates a long prompt for the M5 benchmark (nf bench), by
repeating a base paragraph. Measuring throughput doesn't need "real"
text, just volume — unlike the correctness golden values, the oracle
doesn't come into play here.

Usage: python tools/make_bench_prompt.py [tests/bench_prompt.txt] [n_repeats]
"""
import sys

PARAGRAPH = (
    "The capital of France is Paris, a city rich in history and culture. "
    "Italy is famous for its cuisine, its art, and its landscapes. "
    "Computer science studies how to represent and process information. "
    "A language model learns to predict the next token from large amounts of text. "
)


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "tests/bench_prompt.txt"
    n_repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    with open(out, "wb") as f:
        f.write((PARAGRAPH * n_repeat).encode("utf-8"))
    print(f"written {out} ({n_repeat} repeats, ~{len(PARAGRAPH) * n_repeat} bytes)")


if __name__ == "__main__":
    main()
