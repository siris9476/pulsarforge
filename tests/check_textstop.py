#!/usr/bin/env python3
"""Regression test for text-based stopping (candidate "2" post-M6).

Why this test exists: nf_stopset/is_stop_token trust that the model
SAMPLES the atomic id of a special token to stop. But the model could
instead WRITE the text of that token ("<|endoftext|>") using ordinary
BPE sub-words ("<", "|", "endoftext", "|", ">"), escaping the id-based
check. `nf debugstop A|B` exercises the buffering algorithm
(stream_piece_through_stopfilter in nf.c) with two hand-built cases,
with no need for a real model (a model can't be deterministically
forced to write a special token as text):

  A: the special token arrives split across several BPE pieces -> it
     must be suppressed entirely from the output.
  B: a prefix that looks like the start of a stop sequence but then
     diverges must not block output forever -> all the text must still
     be printed.

Usage: python tests/check_textstop.py
"""
import subprocess
import sys

NF = "./nf.exe" if sys.platform == "win32" else "./nf"

CASES = {
    "A": b"Hello world",
    "B": b"Hello <|im_middle|> world",
}


def main() -> int:
    ok = True
    for case, expected in CASES.items():
        r = subprocess.run([NF, "debugstop", case], capture_output=True)
        if r.returncode != 0:
            print(f"[FAIL] case {case}: nf debugstop failed: {r.stderr.decode(errors='replace')}")
            ok = False
            continue
        got = r.stdout
        status = "OK" if got == expected else "FAIL"
        if got != expected:
            ok = False
        print(f"[{status}] case {case}: expected={expected!r} got={got!r}")

    print()
    print("TEXT STOP VALIDATED" if ok else "TEXT STOP FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
