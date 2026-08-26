#!/usr/bin/env python3
"""Full regression battery in a single command.

Why it exists: the battery had grown to ~9 manual commands — and an
inconvenient battery gets run less often, which is how regressions slip
through unnoticed. This script reduces it to:

  venv-tools/Scripts/python.exe tests/run_all.py            (base suite, ~4-5 min)
  venv-tools/Scripts/python.exe tests/run_all.py --full     (+ 4B and 8B selfcheck)

Runs each check as a subprocess (same interpreter), collects PASS/FAIL
and prints a summary. Exit code 0 only if everything passes.

Notes on criteria already documented elsewhere:
- check_m2 on Q4_K_M fails BY DESIGN (cosine 0.9758 < 0.99: a real
  format loss, measured in M4) — the suite only runs it on Q8_0.
- check_m3 prints the greedy-vs-oracle prefix as informational data;
  its exit code only reflects execution errors.
"""
import os
import subprocess
import sys
import time

# The global torch (XPU build) crashes on import and `gguf` is missing:
# the tests that call the HF oracle (M3) or gguf.quants (M4) need the
# venv-tools Python. If it exists, it's used for ALL sub-tests;
# otherwise it falls back to the current interpreter (CI/other
# machines).
_VENV_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "venv-tools", "Scripts", "python.exe")
PY = _VENV_PY if os.path.exists(_VENV_PY) else sys.executable
NF = ".\\nf.exe" if sys.platform == "win32" else "./nf"

BASE = [
    ("M1 tokenizer 0.6B vs HF oracle",
     [PY, "tests/check_m1.py"]),
    ("M1 tokenizer 4B vs ITS OWN oracle",
     [PY, "tests/check_m1.py", "models/Qwen3-4B-Q4_K_M.gguf",
      "models/Qwen3-4B-tokenizer.json"]),
    ("M2 logits Q8_0 vs golden fp32",
     [PY, "tests/check_m2.py", "models/Qwen3-0.6B-Q8_0.gguf"]),
    ("M3 greedy generation vs oracle",
     [PY, "tests/check_m3.py"]),
    ("M4 dequant Q4_K/Q6_K vs gguf.quants",
     [PY, "tests/check_m4_dequant.py"]),
    ("embed vs NF_DEBUG_DUMP_ALLLAYERS (dense)",
     [PY, "tests/check_embed.py"]),
    ("M6 chat + cross-process session",
     [PY, "tests/check_m6.py"]),
    ("Determinism (fixed seed, temp>0)",
     [PY, "tests/check_determinism.py", "models/Qwen3-0.6B-Q8_0.gguf"]),
    ("Text stop (nf debugstop)",
     [PY, "tests/check_textstop.py"]),
    ("M8 concurrent server (thread-per-connection, no corruption)",
     [PY, "tests/check_serve_concurrency.py"]),
    ("Selfcheck 0.6B (cache == forward)",
     [NF, "selfcheck", "models/Qwen3-0.6B-Q8_0.gguf",
      "--file", "tests/m3_prompt.txt"]),
]

FULL = [
    ("Selfcheck 4B",
     [NF, "selfcheck", "models/Qwen3-4B-Q4_K_M.gguf",
      "--file", "tests/m3_prompt.txt"]),
    ("Selfcheck 8B",
     [NF, "selfcheck", "models/Qwen3-8B-Q4_K_M.gguf",
      "--file", "tests/m3_prompt.txt"]),
    ("M4 dequant on the 4B",
     [PY, "tests/check_m4_dequant.py", "models/Qwen3-4B-Q4_K_M.gguf"]),
    ("M4 dequant on the 8B",
     [PY, "tests/check_m4_dequant.py", "models/Qwen3-8B-Q4_K_M.gguf"]),
    ("embed vs NF_DEBUG_DUMP_ALLLAYERS (MoE)",
     [PY, "tests/check_embed.py", "models/Qwen3-30B-A3B-Q4_K_M.gguf", "10"]),
]


def main() -> int:
    suite = BASE + (FULL if "--full" in sys.argv else [])
    results = []
    t_all = time.time()
    for name, cmd in suite:
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True)
        dt = time.time() - t0
        ok = r.returncode == 0
        results.append((name, ok, dt))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  ({dt:.0f}s)")
        if not ok:
            tail = (r.stdout + r.stderr).decode(errors="replace").splitlines()
            for line in tail[-8:]:
                print(f"       | {line}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} passed "
          f"in {time.time() - t_all:.0f}s"
          + ("" if n_fail == 0 else f"  —  {n_fail} FAILED"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
