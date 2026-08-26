# Cross-layer routing prediction: while computing layer L, how well
# does layer L+1's router (resident, 6MB F32) applied to input xb_L
# predict the experts layer L+1 will ACTUALLY choose? Measured offline
# from the NF_GLM_XB_TRACE trace — the "measured BEFORE building" version
# of the PILOT idea (which was built blind and gave -53%).
import struct, sys
import numpy as np
from gguf import GGUFReader

TRACE = "tools/real_xb_trace.bin"
GGUF = "models/glm52-merged.gguf"

buf = open(TRACE, "rb").read()
magic, E, K = struct.unpack_from("<3i", buf, 0)
assert magic == 0x52544258
recsz = 8 + 4*E + 8*K
recs = {}   # (layer,pos) -> (xb, midx)
off = 12
while off + recsz <= len(buf):
    l, p = struct.unpack_from("<2i", buf, off)
    xb = np.frombuffer(buf, np.float32, E, off+8)
    midx = np.frombuffer(buf, np.int32, K, off+8+4*E)
    recs[(l, p)] = (xb, midx)
    off += recsz
layers = sorted({l for l, _ in recs})
poss = sorted({p for _, p in recs})
print(f"trace: {len(recs)} records, layers {layers[0]}..{layers[-1]}, pos {poss[0]}..{poss[-1]}")

print("loading router+bias from the GGUF (header + small F32 tensors only)...")
r = GGUFReader(GGUF)
tens = {t.name: t for t in r.tensors}
def router(l):
    W = np.array(tens[f"blk.{l}.ffn_gate_inp.weight"].data, dtype=np.float32).reshape(256, E)
    b = np.array(tens[f"blk.{l}.exp_probs_b.bias"].data, dtype=np.float32)
    return W, b

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

sample_layers = [3, 10, 20, 30, 40, 50, 60, 70, 76]   # predicts L+1
cache = {}
for base in sample_layers:
    tgt = base + 1
    if tgt not in layers: continue
    if tgt not in cache: cache[tgt] = router(tgt)
    W, b = cache[tgt]
    ov1 = []; ovreal = []
    for p in poss:
        if (base, p) not in recs or (tgt, p) not in recs: continue
        xb_base, _ = recs[(base, p)]
        _, midx_tgt = recs[(tgt, p)]
        choice = sigmoid(W @ xb_base) + b
        pred = set(np.argsort(choice)[-8:].tolist())
        ov1.append(len(pred & set(midx_tgt.tolist()))/8.0)
        # sanity check: with the TRUE input xb_{L+1}, does the recomputed selection match?
        xb_tgt, _ = recs[(tgt, p)]
        ch2 = sigmoid(W @ xb_tgt) + b
        pred2 = set(np.argsort(ch2)[-8:].tolist())
        ovreal.append(len(pred2 & set(midx_tgt.tolist()))/8.0)
    print(f"  L{base}->L{tgt}: pred with xb_L = {np.mean(ov1)*100:5.1f}%   (sanity with true xb_L+1: {np.mean(ovreal)*100:5.1f}%)  n={len(ov1)}")

# aggregate over ALL adjacent layers
tot = []
for base in layers:
    tgt = base + 1
    if tgt not in layers: continue
    if tgt not in cache: cache[tgt] = router(tgt)
    W, b = cache[tgt]
    for p in poss:
        if (base, p) not in recs or (tgt, p) not in recs: continue
        xb_base, _ = recs[(base, p)]
        _, midx_tgt = recs[(tgt, p)]
        choice = sigmoid(W @ xb_base) + b
        pred = set(np.argsort(choice)[-8:].tolist())
        tot.append(len(pred & set(midx_tgt.tolist()))/8.0)
print(f"GLOBAL (all {len(layers)-1} L->L+1 hops, {len(tot)} observations): mean predicted overlap = {np.mean(tot)*100:.1f}%")
