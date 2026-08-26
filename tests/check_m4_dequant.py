#!/usr/bin/env python3
"""M4 test: checks our C dequantizer (Q4_K, Q6_K) against the official
`gguf.quants` reference from llama.cpp/GGUF, element by element on real
rows from an actual model.

Why this test exists: during M4 development, a comparison with the
fp32 oracle (check_m2.py) showed a cosine of 0.976 for Q4_K_M — much
lower than 0.9998 for Q8_0. Before accepting that number as "the cost
of 4-bit quantization", this test isolates the question: is OUR
dequantizer correct? (it was an off-by-one in the Q6_K type id: 14, not
15 — found and fixed thanks to exactly this test).

Requires: pip install gguf (in venv-tools)
Usage: venv-tools/Scripts/python.exe tests/check_m4_dequant.py [gguf_path]
(the comparison is type-independent: it uses the tensor's actual type
in the file — with Qwen3-0.6B-Q5_K_M.gguf it validates the Q5_K kernel,
etc.)
"""
import subprocess
import sys

import numpy as np
from gguf import quants
from gguf.gguf_reader import GGUFReader

NF = "./nf.exe" if sys.platform == "win32" else "./nf"
GGUF = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B-Q4_K_M.gguf"

import sys as _sys
LAYER = int(_sys.argv[2]) if len(_sys.argv) > 2 else 0

CASES = [
    (f"blk.{LAYER}.attn_q.weight", False, None),  # attn_q: 4/5 bit depending on the file
    (f"blk.{LAYER}.attn_v.weight", True,  None),  # attn_v: the most sensitive tensor
]


def golden_rows(tensor_name, n_rows=2):
    r = GGUFReader(GGUF)
    t = next(x for x in r.tensors if x.name == tensor_name)
    qtype = quants.GGMLQuantizationType(t.tensor_type)
    dequant = quants.dequantize(t.data, qtype)
    return qtype.name, [dequant[i].astype(np.float64) for i in range(n_rows)]


def our_row(row, is_v):
    r = subprocess.run([NF, "debugwq", GGUF, str(LAYER), str(row)] +
                       (["--v"] if is_v else []), capture_output=True)
    if r.returncode != 0:
        sys.exit(f"nf debugwq failed:\n{r.stderr.decode(errors='replace')}")
    return np.array([float(x) for x in r.stdout.decode().split()],
                    dtype=np.float64)


def main() -> int:
    all_ok = True
    for tensor_name, is_v, expect_type in CASES:
        actual_type, golden = golden_rows(tensor_name)
        if expect_type and actual_type != expect_type:
            print(f"note: {tensor_name} is {actual_type}, not the expected {expect_type}")
        for row_idx, g in enumerate(golden):
            o = our_row(row_idx, is_v)
            d = np.abs(g - o)
            ok = d.max() < 1e-6
            all_ok &= ok
            print(f"[{'OK' if ok else 'FAIL'}] {tensor_name} ({actual_type}) "
                 f"row {row_idx}: max|diff|={d.max():.3e}  "
                 f"identical={np.sum(d < 1e-6)}/{len(d)}")

    print(f"\n{'M4 DEQUANT VALIDATED' if all_ok else 'M4 DEQUANT FAILED'} "
          f"(element-by-element comparison with gguf.quants)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
