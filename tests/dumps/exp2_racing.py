"""Experiment 2 (falsification): racing argmax over the logits with
exact per-block bounds (Cauchy-Schwarz).

Theory to falsify: the true logit argmax can be certified by reading
only a fraction of the output head's blocks. For every row r: after k
blocks, logit_r ∈ [partial_r − U_r, partial_r + U_r] with
U_r = Σ_{b remaining} ||W_rb||·||x_b|| (Cauchy-Schwarz, exact). A row
dies once its best possible value drops below the leader's guaranteed
minimum. Cost = average fraction of (row, block) pairs read. Verdict
positive if the cost drops below ~40%; negative otherwise.
"""
import numpy as np
from gguf import quants
from gguf.gguf_reader import GGUFReader

GGUF = "models/Qwen3-4B-Q4_K_M.gguf"
N_TOK, E, NB = 72, 2560, 10          # 10 blocks of 256
RMS_EPS = 1e-6

r = GGUFReader(GGUF)
tensors = {t.name: t for t in r.tensors}

def deq(name):
    t = tensors[name]
    return quants.dequantize(t.data, quants.GGMLQuantizationType(t.tensor_type))

print("dequantizing the output head (tied)...")
W = deq("token_embd.weight").astype(np.float32)        # [152k, 2560]
w_out = deq("output_norm.weight").astype(np.float32)   # [2560]
V = W.shape[0]

X = np.fromfile("tests/dumps/x4b_l35.bin", dtype=np.float32).reshape(N_TOK, E)

# per (row, block) norms — the theory's "offline repacking"
Wb = W.reshape(V, NB, 256)
Wn = np.linalg.norm(Wb, axis=2)                        # [V, NB]

frac_reads, survivors_log, exact = [], [], 0
samples = range(24, N_TOK)
for p in samples:
    x = X[p].astype(np.float32)
    ss = np.mean(x.astype(np.float64) ** 2)
    xb = (x / np.sqrt(ss + RMS_EPS) * w_out).astype(np.float32)
    xn = np.linalg.norm(xb.reshape(NB, 256), axis=1)   # [NB]

    order = np.argsort(-xn)          # most energetic blocks first
    true_arg = int(np.argmax(W @ xb))

    partial = np.zeros(V, dtype=np.float32)
    U = (Wn * xn).sum(axis=1)        # initial total uncertainty
    alive = np.ones(V, dtype=bool)
    reads, surv = 0, []
    for step, b in enumerate(order):
        sl = slice(b * 256, (b + 1) * 256)
        reads += int(alive.sum())
        partial[alive] += Wb[alive, b, :] @ xb[sl]
        U[alive] = np.maximum(U[alive] - Wn[alive, b] * xn[b], 0.0)
        best_lower = np.max(partial[alive] - U[alive])
        alive &= (partial + U) >= best_lower
        surv.append(int(alive.sum()))
        if alive.sum() == 1:
            break
    winner = int(np.argmax(np.where(alive, partial, -np.inf)))
    exact += (winner == true_arg)
    frac_reads.append(reads / (V * NB))
    survivors_log.append(surv)

print(f"samples: {len(frac_reads)}  exact argmax: {exact}/{len(frac_reads)}")
print(f"fraction of (row,block) read: mean {np.mean(frac_reads):.1%}  "
      f"median {np.median(frac_reads):.1%}  worst {np.max(frac_reads):.1%}")
s = survivors_log[len(survivors_log) // 2]
print(f"survivors per step (median sample): {s}")
