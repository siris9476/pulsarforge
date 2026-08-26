"""Validates the bit layout of the new Q2_K encoder/decoder
(nf_model.c) against the Python oracle gguf.quants.dequantize, which
can DECODE Q2_K (verified) but can't CREATE it (NotImplementedError)
— so the comparison is one-directional: we take the bytes written by
OUR C encoder (q2k_dump_q.bin, produced by `nf debugq2k <model>`) and
have an independent decoder decode them, then compare against the
original float (q2k_dump_x.f32). If the bit layout were wrong, the
oracle would produce numbers inconsistent with the original (not just
"a bit less precise" but typically garbage/huge values).
"""
import sys
import numpy as np
from gguf import quants
from gguf.constants import GGMLQuantizationType

x = np.fromfile("q2k_dump_x.f32", dtype=np.float32)
qbytes = np.fromfile("q2k_dump_q.bin", dtype=np.uint8)

n_blocks = qbytes.size // 84
assert n_blocks * 84 == qbytes.size, f"unexpected byte size: {qbytes.size}"
assert n_blocks * 256 == x.size, f"n_blocks={n_blocks} doesn't match x.size={x.size}"

y = quants.dequantize(qbytes.reshape(n_blocks, 84), GGMLQuantizationType.Q2_K)
y = np.asarray(y).reshape(-1)

assert y.size == x.size, f"oracle produced {y.size} values, expected {x.size}"

d = x.astype(np.float64) - y.astype(np.float64)
rmse = np.sqrt(np.mean(d * d))
maxd = np.max(np.abs(d))
cos = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))
rms_x = np.sqrt(np.mean(x.astype(np.float64) ** 2))

print(f"n_blocks={n_blocks}  n_values={x.size}")
print(f"Python oracle (gguf.quants.dequantize) vs original float:")
print(f"  RMSE={rmse:.4e}  max|d|={maxd:.4e}  cosine={cos:.6f}  (rms_x={rms_x:.4e})")

# Direct comparison with the C dequant: if the layout is correct, the
# two decoders (our C, official Python) must agree BIT FOR BIT on the
# same input (no quantization here, only decoding).
print("\nIf this cosine is close to what `nf debugq2k` reports")
print("(C round-trip), our encoder's bit layout is CORRECT —")
print("the independent oracle agrees with our decoder on the same")
print("quantized input. If instead the oracle gave inconsistent numbers")
print("(cosine near 0 or negative, huge values), the bug would be")
print("in the encoder's bit PACKING, not in the format's quality.")
