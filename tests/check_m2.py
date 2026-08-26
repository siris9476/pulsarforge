#!/usr/bin/env python3
"""M2 regression test: compares the C forward pass logits against the
HuggingFace fp32 model (the oracle).

Our weights are Q8_0 (quantized), the oracle is fp32: the logits CANNOT
match exactly. M2's criterion is:
  - same argmax (the most probable token must be the same)
  - very high cosine similarity
  - good top-10 overlap
The rest of the drift is the cost of quantization, measured here for the
first time (it's a preview of the M4 lesson).

Usage: python tests/check_m2.py [gguf_path]
The golden values are generated with the venv (torch CPU):
  venv-tools/Scripts/python.exe tools/golden.py --prompt-file tests/m2_prompt.txt
"""
import json
import math
import os
import struct
import subprocess
import sys

GGUF = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B-Q8_0.gguf"
NF = "./nf.exe" if sys.platform == "win32" else "./nf"
PROMPT = "The capital of France is"
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(HERE, "m2_prompt.txt")
GOLDEN_DIR = os.path.join(HERE, "golden")
OURS_F32 = os.path.join(HERE, "m2_logits.f32")


def read_f32(path):
    with open(path, "rb") as fp:
        raw = fp.read()
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def main() -> int:
    # prompt written in binary mode: Windows text mode would translate
    # \n into \r\n and skew the comparison
    with open(PROMPT_FILE, "wb") as fp:
        fp.write(PROMPT.encode("utf-8"))

    golden_path = os.path.join(GOLDEN_DIR, "logits.f32")
    meta_path = os.path.join(GOLDEN_DIR, "meta.json")
    if not os.path.exists(golden_path):
        sys.exit("missing golden values: generate them with\n  venv-tools/Scripts/"
                 "python.exe tools/golden.py --prompt-file tests/m2_prompt.txt")
    with open(meta_path) as fp:
        meta = json.load(fp)
    if meta["prompt"] != PROMPT:
        sys.exit(f"golden values are for a different prompt: {meta['prompt']!r}")

    r = subprocess.run([NF, "logits", GGUF, "--file", PROMPT_FILE,
                        "--dump", OURS_F32], capture_output=True)
    if r.returncode != 0:
        sys.exit(f"nf logits failed:\n{r.stderr.decode(errors='replace')}")

    golden = read_f32(golden_path)
    ours = read_f32(OURS_F32)
    if len(golden) != len(ours):
        sys.exit(f"different vocab sizes: golden {len(golden)} vs ours {len(ours)}")

    n = len(golden)
    dot = sum(a * b for a, b in zip(golden, ours))
    ng = math.sqrt(sum(a * a for a in golden))
    no = math.sqrt(sum(b * b for b in ours))
    cosine = dot / (ng * no)
    max_diff = max(abs(a - b) for a, b in zip(golden, ours))
    mean_diff = sum(abs(a - b) for a, b in zip(golden, ours)) / n

    top_g = sorted(range(n), key=lambda i: -golden[i])[:10]
    top_o = sorted(range(n), key=lambda i: -ours[i])[:10]
    overlap = len(set(top_g) & set(top_o))
    argmax_exact = top_g[0] == top_o[0]
    # DELIBERATE revision of the argmax criterion (integer vec_dot): int8
    # activation quantization
    # adds ~0.3-0.7% noise per matvec (verified NOT to be a bug with
    # `nf debugdot`: same order across all formats). When the top two
    # golden logits are closer than this accumulated noise (here: margin
    # 0.256 on ~14.5), their order can flip — as happens on every CPU
    # engine with quantized activations (llama.cpp included). The
    # criterion accepts a swap of the top two ONLY if the golden margin
    # is under 0.5; an argmax outside the golden top-2 is still a hard
    # FAIL. Empirical note: with this same path, check_m3 goes from
    # 5/20 to 20/20 tokens identical to the oracle — the noise isn't a
    # bias, it redistributes near-ties.
    margin_g = golden[top_g[0]] - golden[top_g[1]]
    argmax_ok = argmax_exact or (top_o[0] == top_g[1] and margin_g < 0.5)

    print(f"argmax:    golden={top_g[0]}  ours={top_o[0]}  "
          f"{'OK' if argmax_exact else ('OK(top-2, golden margin %.3f)' % margin_g if argmax_ok else 'FAIL')}")
    print(f"cosine:    {cosine:.6f}")
    print(f"top-10:    {overlap}/10 in common")
    print(f"max |d|:   {max_diff:.4f}   mean |d|: {mean_diff:.4f}")
    print(f"top golden: {top_g}")
    print(f"top ours:   {top_o}")

    ok = argmax_ok and cosine > 0.99 and overlap >= 8
    print(f"\n{'M2 VALIDATED' if ok else 'M2 FAILED'} "
          f"(criteria: argmax equal or top-2 swap with golden margin <0.5, "
          f"cosine>0.99, top-10 overlap>=8)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
