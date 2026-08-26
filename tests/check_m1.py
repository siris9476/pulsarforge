#!/usr/bin/env python3
"""M1 regression test: compares the C tokenizer (nf tokenize) against
the reference HuggingFace tokenizer on a battery of prompts.

Usage: python tests/check_m1.py [gguf_path] [tokenizer.json_path]
  The second argument is ALWAYS an explicit local path to a
  tokenizer.json (default: models/Qwen3-0.6B-tokenizer.json) — same
  pattern for every member of the family (0.6B/4B/8B), downloaded with
  curl (see README.md, Setup section). No dependency on the local
  HuggingFace cache: it used to silently rely on a side effect of
  tools/golden.py (AutoTokenizer.from_pretrained), an implicit coupling
  between two scripts that was never declared anywhere — removed to be
  as explicit as the 4B/8B cases, which already used a direct path.
Requires: pip install tokenizers
"""
import os
import subprocess
import sys

MODEL = sys.argv[2] if len(sys.argv) > 2 else "models/Qwen3-0.6B-tokenizer.json"
GGUF = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B-Q8_0.gguf"
NF = "./nf.exe" if sys.platform == "win32" else "./nf"

PROMPTS = [
    "ciao mondo",
    "La capitale d'Italia e' Roma, e il 2026 e' un anno bellissimo!",
    "Hello, world! It's a test: don't we love C?",
    "   spazi    multipli   e\ttab",
    "riga uno\nriga due\n\n\nriga tre",
    "perché però così è già più",           # Italian accents (UTF-8 multibyte)
    "prezzo: 1234,56 EUR (sconto -15%)",
    "x = (a+b)*c/d; // commento",
    "...punteggiatura?!?...",
    " inizia con spazio e finisce con spazio ",
    "CamelCase snake_case kebab-case UPPER",
    # special tokens (M6): must come out atomic (151644/151645), not
    # split by regular BPE — see nf_tokenizer.c, match_special()
    "<|im_start|>user\nciao<|im_end|>\n<|im_start|>assistant\n",
]


def find_tokenizer_json() -> str:
    """Direct path to a tokenizer.json — no cache, no side effect from
    other scripts: if missing, download it with curl (see README.md,
    Setup section)."""
    if not os.path.exists(MODEL):
        sys.exit(f"tokenizer.json not found: {MODEL}\n"
                 f"Download it with curl — see README.md, Setup section.")
    return MODEL


def main() -> int:
    import tempfile

    # use the `tokenizers` library (Rust) directly: no torch, which
    # crashes on import on this machine because of the XPU driver
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(find_tokenizer_json())

    failures = 0
    for prompt in PROMPTS:
        golden = tok.encode(prompt).ids

        # the prompt is passed via a UTF-8 file: argv on Windows uses
        # the system codepage and would corrupt non-ASCII characters.
        # BINARY write: text mode on Windows would translate \n into
        # \r\n and skew the comparison.
        with tempfile.NamedTemporaryFile("wb", suffix=".txt",
                                         delete=False) as tmp:
            tmp.write(prompt.encode("utf-8"))
            tmp_path = tmp.name
        r = subprocess.run([NF, "tokenize", GGUF, "--file", tmp_path],
                           capture_output=True, text=False)
        os.unlink(tmp_path)
        ours = [int(line.split(b"\t")[0])
                for line in r.stdout.splitlines() if b"\t" in line]

        ok = golden == ours
        print(f"[{'OK' if ok else 'FAIL'}] {prompt[:50]!r}")
        if not ok:
            failures += 1
            print(f"       golden: {golden}")
            print(f"       ours:   {ours}")

    print(f"\n{len(PROMPTS) - failures}/{len(PROMPTS)} prompts identical to the HF oracle")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
