"""Validates pulsarforge's 5 new dequantizers (Q2_K, Q3_K, IQ2_XXS,
IQ3_XXS, IQ1_S) against the Python oracle gguf.quants.dequantize, on
REAL RAW BYTES taken from the actual GGUF (glm52-merged.gguf, GLM-5.2
Unsloth UD-IQ1_S, 202GB) via `nf debugdequant5`.

For each type: <prefix>_q.bin are the quantized bytes read DIRECTLY
from the real file (same bytes, no transformation), <prefix>_x.f32 is
our C dequant. Here we compare them against the independent oracle.
"""
import numpy as np
from gguf import quants
from gguf.constants import GGMLQuantizationType

CASES = [
    ("q2k",    84,  GGMLQuantizationType.Q2_K),
    ("q3k",    110, GGMLQuantizationType.Q3_K),
    ("iq2xxs", 66,  GGMLQuantizationType.IQ2_XXS),
    ("iq3xxs", 98,  GGMLQuantizationType.IQ3_XXS),
    ("iq1s",   50,  GGMLQuantizationType.IQ1_S),
]

print(f"{'type':10s} {'n_val':>10s} {'max|d|':>12s} {'RMSE':>12s} {'cosine':>10s}")
all_ok = True
for name, blkbytes, qtype in CASES:
    x = np.fromfile(f"{name}_x.f32", dtype=np.float32)
    qbytes = np.fromfile(f"{name}_q.bin", dtype=np.uint8)
    n_blocks = qbytes.size // blkbytes
    assert n_blocks * blkbytes == qbytes.size, f"{name}: unexpected byte size"
    assert n_blocks * 256 == x.size, f"{name}: n_blocks={n_blocks} doesn't match x.size={x.size}"

    y = quants.dequantize(qbytes.reshape(n_blocks, blkbytes), qtype)
    y = np.asarray(y).reshape(-1).astype(np.float32)

    d = x.astype(np.float64) - y.astype(np.float64)
    rmse = np.sqrt(np.mean(d * d))
    maxd = np.max(np.abs(d))
    nx = np.linalg.norm(x.astype(np.float64))
    ny = np.linalg.norm(y.astype(np.float64))
    cos = float(np.dot(x.astype(np.float64), y.astype(np.float64)) / (nx * ny + 1e-30))

    status = "OK" if (maxd < 1e-3 and cos > 0.99999) else "FAIL"
    if status == "FAIL":
        all_ok = False
    print(f"{name:10s} {x.size:10d} {maxd:12.4e} {rmse:12.4e} {cos:10.6f}  {status}")

print()
print("PASS" if all_ok else "FAIL", "- pulsarforge C vs gguf.quants.dequantize oracle comparison (real GGUF bytes)")
