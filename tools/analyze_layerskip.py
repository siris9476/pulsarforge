# Phase A of the layer-skip drafter: offline analysis of the
# NF_GLM_LAYERSKIP_TRACE trace (x_before/x_after per MoE layer, per
# decode position) -> mean/min cosine of the WHOLE block's
# (attention+MoE, not just the FFN) contribution to the residual,
# filtering out the DSA indexer's "full" layers (never skip
# candidates, see glm_layerskip_should_skip in nf_model.c) and
# ranking the rest by skippability.
#
# Usage: python tools/analyze_layerskip.py [trace_path]
# Default trace: tools/real_layerskip_trace.bin (14 tokens, prompt_ciao,
# NF_GLM_STREAM_GB=8/IO_THREADS=6).
import sys
import struct
import collections

import numpy as np

TRACE = sys.argv[1] if len(sys.argv) > 1 else "tools/real_layerskip_trace.bin"

with open(TRACE, "rb") as f:
    magic, E = struct.unpack("<ii", f.read(8))
    assert magic == 0x5453424C, hex(magic)
    recs = []
    while True:
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        layer, pos = struct.unpack("<ii", hdr)
        xb = np.frombuffer(f.read(4 * E), dtype=np.float32)
        xa = np.frombuffer(f.read(4 * E), dtype=np.float32)
        recs.append((layer, pos, xb, xa))

print(f"E={E}  n_records={len(recs)}")

per_layer = collections.defaultdict(list)
for layer, pos, xb, xa in recs:
    nb = np.linalg.norm(xb)
    na = np.linalg.norm(xa)
    cos = float(np.dot(xb, xa) / (nb * na + 1e-12))
    rel = float(np.linalg.norm(xa - xb) / (nb + 1e-12))
    per_layer[layer].append((cos, rel))

# indexer fallback formula (freq=4, offset=3 — confirmed by the real
# model's load log, which reports the two GGUF keys explicitly
# missing): v = l - offset + 1, clamped to 0, full = (v % freq)==0.
# Must match nf_model.c EXACTLY (the L->has_indexer assignment in
# nf_model_load).
OFFSET, FREQ = 3, 4
full = set()
for l in range(78):
    v = max(0, l - OFFSET + 1)
    if v % FREQ == 0:
        full.add(l)

moe_layers = sorted(per_layer.keys())
eligible = [l for l in moe_layers if l not in full]
excluded_full = [l for l in moe_layers if l in full]
print(f"full layers (0..77): {sorted(full)}")
print(f"MoE layers traced: {len(moe_layers)}, excluded (full): {len(excluded_full)} -> {excluded_full}")
print(f"eligible pool: {len(eligible)}")

rows = []
for l in eligible:
    coss = [c for c, r in per_layer[l]]
    rels = [r for c, r in per_layer[l]]
    rows.append((l, float(np.mean(coss)), float(np.min(coss)), float(np.mean(rels))))
rows.sort(key=lambda r: -r[1])

print()
print(f"{'rank':>4} {'layer':>5} {'cos_mean':>9} {'cos_min':>8} {'rel_mean':>9}")
for i, (l, cm, cmin, rm) in enumerate(rows):
    print(f"{i+1:4d} {l:5d} {cm:9.5f} {cmin:8.5f} {rm:9.5f}")

print()
print("chosen set (top-20, see glm_layerskip_should_skip/SET78 in nf_model.c):")
top20 = sorted(l for l, *_ in rows[:20])
print(top20)
