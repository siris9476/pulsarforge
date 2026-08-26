#!/usr/bin/env python3
"""Regression test: sampling (temp>0) with the same seed must produce
the SAME output every time it runs.

Why this test exists: a real OpenMP race condition (the matvec/attention
loop variable declared outside the pragma, with no explicit `private()`
— MSVC/OpenMP 2.0 wasn't privatizing it correctly) made generation
non-deterministic even with a fixed seed, discovered while investigating
a discrepancy between 'nf generate' and 'nf chat' that turned out to be
just race-condition noise, not a real difference between the two
commands. Fixed by adding `private(r)`/`private(h)` to all omp pragmas.
This test guards that fix over time: if someone touches the OpenMP loops
and reintroduces the race, this test catches it immediately, without
having to re-investigate a seemingly random "wrong answer" from scratch.

Usage: python tests/check_determinism.py [gguf_path]
"""
import subprocess
import sys

GGUF = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B-Q8_0.gguf"
NF = "./nf.exe" if sys.platform == "win32" else "./nf"
PROMPT_FILE = "tests/m3_prompt.txt"  # already present from M3, reused


def run_once():
    r = subprocess.run([NF, "generate", GGUF, "--file", PROMPT_FILE,
                        "--n-predict", "200", "--temp", "0.7", "--top-p", "0.9",
                        "--seed", "42", "--repeat-penalty", "1.1"],
                       capture_output=True)
    if r.returncode != 0:
        sys.exit(f"nf generate failed:\n{r.stderr.decode(errors='replace')}")
    return r.stdout


def main() -> int:
    runs = [run_once() for _ in range(3)]
    ok = all(r == runs[0] for r in runs[1:])
    print(f"[{'OK' if ok else 'FAIL'}] 3 identical runs seed=42/temp=0.7: "
         f"{'same output byte for byte' if ok else 'DIVERGENCE — possible OpenMP race condition'}")
    if not ok:
        for i, r in enumerate(runs):
            print(f"--- run {i} ({len(r)} bytes) ---")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
