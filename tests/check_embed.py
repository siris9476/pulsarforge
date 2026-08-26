#!/usr/bin/env python3
"""Validates `nf embed --layer N` by comparing it
bit-exact with NF_DEBUG_DUMP_ALLLAYERS on the same prompt — two
different C paths (nf_model_embed via a truncated nf_forward_layer, vs
nf_model_forward with the debug hook) that must produce the SAME
hidden state by construction: both run the identical computation up
to the chosen layer, then either stop (embed) or continue
(forward+dump, which however dumps X at that layer too before going
on). This is not a retrieval-quality test (that's a downstream
consumer's job) — just that the engine returns exactly the hidden
state it claims to return.

Usage: venv-tools/Scripts/python.exe tests/check_embed.py [gguf] [layer]
"""
import re
import subprocess
import sys
import tempfile
import os

import numpy as np

NF = "./nf.exe" if sys.platform == "win32" else "./nf"
GGUF = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B-Q8_0.gguf"
LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 15
PROMPT_FILE = "tests/m3_prompt.txt"


def n_embd(gguf_path):
    r = subprocess.run([NF, "inspect", gguf_path], capture_output=True)
    m = re.search(r"embedding_length\s+(\d+)", r.stdout.decode())
    if not m:
        sys.exit("nf: could not read embedding_length from 'nf inspect'")
    return int(m.group(1))


def main() -> int:
    E = n_embd(GGUF)
    with tempfile.TemporaryDirectory() as td:
        dump_prefix = os.path.join(td, "fwd")
        # NF_KV_F16: nf_model_embed deliberately uses the f16 cache
        # (contract: embeddings don't change with cache modes); since
        # the kv-q8 default, the comparison dump must be forced to
        # the same representation, otherwise two different quantizations
        # of the same stream would be compared.
        r = subprocess.run([NF, "logits", GGUF, "--file", PROMPT_FILE],
                           capture_output=True,
                           env={**os.environ, "NF_DEBUG_DUMP_ALLLAYERS": dump_prefix,
                                "NF_KV_F16": "1"})
        if r.returncode != 0:
            sys.exit(f"nf logits (with dump) failed:\n{r.stderr.decode(errors='replace')}")

        dump_path = f"{dump_prefix}_l{LAYER:02d}.bin"
        X = np.fromfile(dump_path, dtype=np.float32).reshape(-1, E)
        golden = X[-1]  # last token, same pooling as nf_model_embed

        embed_out = os.path.join(td, "embed.bin")
        r = subprocess.run([NF, "embed", GGUF, "--file", PROMPT_FILE,
                            "--layer", str(LAYER), "--dump", embed_out],
                           capture_output=True)
        if r.returncode != 0:
            sys.exit(f"nf embed failed:\n{r.stderr.decode(errors='replace')}")
        got = np.fromfile(embed_out, dtype=np.float32)

    d = np.abs(golden.astype(np.float64) - got.astype(np.float64))
    ok = d.max() < 1e-6
    print(f"[{'OK' if ok else 'FAIL'}] {GGUF} layer={LAYER} dim={E} "
          f"max|diff|={d.max():.3e}")
    print(f"EMBED {'VALIDATED' if ok else 'FAILED'} "
          f"(nf embed vs NF_DEBUG_DUMP_ALLLAYERS, same prompt)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
