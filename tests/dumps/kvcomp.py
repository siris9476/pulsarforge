"""Experiment A (falsification): composable KV for RAG.

Theory: by saving the K's PRE-RoPE, a chunk's KV cache computed in
ISOLATION can be relocated to any absolute position (RoPE is a
rotation whose angle depends only on position) and composed with
other caches — in the style of ds4_kvstore + DwarfStar's
topology-neutral snapshot. The approximation being measured: the
isolated chunk never "saw" the context preceding it (no
cross-attention during its own encoding).

Scenarios (Qwen3-0.6B Q8_0 dequantized to fp32, NumPy forward pass):
  FULL      : the whole prompt in one normal forward pass (exact baseline)
  COMPOSED  : KV(prefix+DocA) normal; KV(DocB) isolated and RELOCATED
              (rope at the right positions); question evaluated on top
  MISPLACED : same as COMPOSED but without relocation (rope at
              0-based positions) — control: how much does position matter?
The question asks about DocB's content (the transplanted chunk).
"""
import glob
import os
import numpy as np
from gguf import quants
from gguf.gguf_reader import GGUFReader
from tokenizers import Tokenizer

GGUF = "models/Qwen3-0.6B-Q8_0.gguf"
E, NH, NKV, HD, NL, NFF = 1024, 16, 8, 128, 28, 3072
ROPE_BASE = 1e6
EPS = 1e-6

tok_path = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json"))[0]
tok = Tokenizer.from_file(tok_path)

print("dequantizing the weights (0.6B fp32)...")
r = GGUFReader(GGUF)
T = {t.name: t for t in r.tensors}
def deq(n):
    t = T[n]
    return quants.dequantize(t.data, quants.GGMLQuantizationType(t.tensor_type)).astype(np.float32)

embd = deq("token_embd.weight")
out_norm = deq("output_norm.weight")
Ls = []
for l in range(NL):
    p = f"blk.{l}."
    Ls.append({k: deq(p + n) for k, n in [
        ("an", "attn_norm.weight"), ("wq", "attn_q.weight"),
        ("wk", "attn_k.weight"), ("wv", "attn_v.weight"),
        ("wo", "attn_output.weight"), ("qn", "attn_q_norm.weight"),
        ("kn", "attn_k_norm.weight"), ("fn", "ffn_norm.weight"),
        ("wg", "ffn_gate.weight"), ("wu", "ffn_up.weight"),
        ("wd", "ffn_down.weight")]})

def rms(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * w

def rope(v, pos):
    """v: [n, heads, HD], pos: [n] — NEOX variant (pairs i, i+HD/2)."""
    half = HD // 2
    freq = ROPE_BASE ** (-2.0 * np.arange(half) / HD)
    a = pos[:, None] * freq[None, :]
    c, s = np.cos(a)[:, None, :], np.sin(a)[:, None, :]
    x0, x1 = v[..., :half], v[..., half:]
    return np.concatenate([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1)

def encode(ids, pos, KP, V):
    """Forward pass of `ids` at positions `pos`, attending to the caches
    KP/V (pre-RoPE K per layer, with the position array in KP[l][1]).
    Updates the caches in place; returns the last position's logits."""
    n = len(ids)
    X = embd[ids].astype(np.float32)
    pos = np.asarray(pos, dtype=np.float64)
    for l, L in enumerate(Ls):
        xb = rms(X, L["an"])
        q = (xb @ L["wq"].T).reshape(n, NH, HD)
        k = (xb @ L["wk"].T).reshape(n, NKV, HD)
        v = (xb @ L["wv"].T).reshape(n, NKV, HD)
        q = rms(q, L["qn"])
        k = rms(k, L["kn"])          # PRE-RoPE: this is what gets saved
        q = rope(q, pos)
        KP[l][0].append(k)
        KP[l][1].append(pos)
        V[l].append(v)
        k_all = np.concatenate(KP[l][0])
        p_all = np.concatenate(KP[l][1])
        v_all = np.concatenate(V[l])
        kr = rope(k_all, p_all)      # relocation: rope on the fly
        att = np.zeros((n, NH, HD), dtype=np.float32)
        for h in range(NH):
            kvh = h // (NH // NKV)
            sc = (q[:, h, :] @ kr[:, kvh, :].T) / np.sqrt(HD)
            mask = pos[:, None] < p_all[None, :]          # causality
            sc = np.where(mask, -1e30, sc)
            w = np.exp(sc - sc.max(axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True)
            att[:, h, :] = w @ v_all[:, kvh, :]
        X = X + att.reshape(n, NH * HD) @ L["wo"].T
        xb = rms(X, L["fn"])
        g = xb @ L["wg"].T
        X = X + ((g / (1 + np.exp(-g))) * (xb @ L["wu"].T)) @ L["wd"].T
    return rms(X[-1], out_norm) @ embd.T

def new_cache():
    return [[[], []] for _ in range(NL)], [[] for _ in range(NL)]

def generate(KP, V, ids, pos_start, n_out=8):
    pos = list(range(pos_start, pos_start + len(ids)))
    logits = encode(ids, pos, KP, V)
    first = logits.copy()
    out = []
    p = pos[-1]
    for _ in range(n_out):
        t = int(np.argmax(logits))
        if t in (151645, 151643):
            break
        out.append(t)
        p += 1
        logits = encode([t], [p], KP, V)
    return tok.decode(out), first

def vs_full(name, first, first_full):
    cos = float(np.dot(first, first_full) /
                (np.linalg.norm(first) * np.linalg.norm(first_full)))
    same = int(np.argmax(first)) == int(np.argmax(first_full))
    print(f"  {name}: logit cosine={cos:.6f}  max|d|={np.abs(first - first_full).max():.3f}  "
          f"argmax {'same' if same else 'DIFFERENT'}")

PRE  = ("<|im_start|>system\nAnswer in a single word.<|im_end|>\n"
        "<|im_start|>user\nDocument A: Marco's cat is named Felix.\n")
DOCB = "Document B: the capital of Australia is Canberra.\n"
QST  = ("Question: what is the capital of Australia?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")

ids_pre = tok.encode(PRE).ids
ids_b   = tok.encode(DOCB).ids
ids_q   = tok.encode(QST).ids
n1, n2 = len(ids_pre), len(ids_b)

# FULL: exact baseline
KP, V = new_cache()
encode(ids_pre + ids_b, list(range(n1 + n2)), KP, V)
ans_full, first_full = generate(KP, V, ids_q, n1 + n2)

# COMPOSED: DocB encoded in isolation (positions 0..n2, no context),
# then RELOCATED to positions n1..n1+n2 via pre-RoPE
KPb, Vb = new_cache()
encode(ids_b, list(range(n2)), KPb, Vb)
KP, V = new_cache()
encode(ids_pre, list(range(n1)), KP, V)
for l in range(NL):
    KP[l][0] += KPb[l][0]
    KP[l][1] += [np.asarray(p) + n1 for p in KPb[l][1]]   # shift positions
    V[l] += Vb[l]
ans_comp, first_comp = generate(KP, V, ids_q, n1 + n2)

# MISPLACED: same transplant but WITHOUT the position shift
KP, V = new_cache()
encode(ids_pre, list(range(n1)), KP, V)
for l in range(NL):
    KP[l][0] += KPb[l][0]
    KP[l][1] += KPb[l][1]                                  # 0-based positions (wrong)
    V[l] += Vb[l]
ans_mis, first_mis = generate(KP, V, ids_q, n1 + n2)

print(f"\nFULL      (full recompute):          {ans_full!r}")
print(f"COMPOSED  (isolated+shifted chunk):   {ans_comp!r}")
print(f"MISPLACED (no shift, control):        {ans_mis!r}")
print("\nlogit fidelity of the first answer token (vs FULL):")
vs_full("COMPOSED ", first_comp, first_full)
vs_full("MISPLACED", first_mis, first_full)
