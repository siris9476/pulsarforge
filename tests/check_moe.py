#!/usr/bin/env python3
"""MoE validation (Qwen3-30B-A3B): layer-0 router + top-K experts
against an INDEPENDENT Python reimplementation, on the actual weights
from the file (dequantized with gguf.quants, not with our own C
dequantizer — otherwise a shared bug in the assumption would stay
invisible, the same lesson as the tied-embedding bug on the 8B).

Why not an fp32 HuggingFace oracle (as M2/M3 do for the dense Qwen3):
Qwen3-30B-A3B in bf16/fp32 doesn't fit in 32GB of RAM (the weights
alone would be ~60GB in bf16), so transformers can't load it on this
machine. The validation here instead isolates the NEW piece of code
(router + expert dispatch, never tested before) from the rest
(attention, RMSNorm, RoPE — unchanged and already validated on the
dense path): nf_debug_moe computes ONLY the layer-0 router+experts
starting from an already-computed embedding, and dumps both the input
(xb, post-ffn_norm) and the weighted output. Here the SAME step is
recomputed in Python, from the actual weights in the file, and the two
are compared: same xb in -> same expected output, if the router/expert
math is implemented correctly in C.

Formula (verified against the official HF transformers source code,
modeling_qwen3_moe.py: Qwen3MoeTopKRouter.forward — not assumed):
  1. logits = W_router @ xb                (F32)
  2. probs  = softmax(logits) over ALL experts
  3. top-K by probs value
  4. renormalize the K weights to sum to 1  (norm_topk_prob=true,
     confirmed in the model's ACTUAL config.json on HuggingFace, not
     a library default)
  5. out = Sum_k weight_k * down_k(silu(gate_k(xb)) * up_k(xb))

Requires: pip install gguf (in venv-tools)
Usage: venv-tools/Scripts/python.exe tests/check_moe.py [token_id]
"""
import subprocess
import sys
import tempfile
import os

import numpy as np
from gguf import quants
from gguf.gguf_reader import GGUFReader

NF = "./nf.exe" if sys.platform == "win32" else "./nf"
GGUF = "models/Qwen3-30B-A3B-Q4_K_M.gguf"
TOKEN = int(sys.argv[1]) if len(sys.argv) > 1 else 1187


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def main() -> int:
    reader = GGUFReader(GGUF)
    tensors = {t.name: t for t in reader.tensors}

    n_expert = 128
    n_expert_used = 8

    xb_f = tempfile.NamedTemporaryFile(suffix=".f32", delete=False)
    out_f = tempfile.NamedTemporaryFile(suffix=".f32", delete=False)
    xb_path, out_path = xb_f.name, out_f.name
    xb_f.close()
    out_f.close()

    r = subprocess.run([NF, "debugmoe", GGUF, str(TOKEN), xb_path, out_path],
                       capture_output=True)
    if r.returncode != 0:
        sys.exit(f"nf debugmoe failed:\n{r.stderr.decode(errors='replace')}")
    sys.stderr.write(r.stderr.decode(errors="replace"))

    xb = np.fromfile(xb_path, dtype=np.float32).astype(np.float64)
    c_out = np.fromfile(out_path, dtype=np.float32).astype(np.float64)
    os.unlink(xb_path)
    os.unlink(out_path)

    # router: F32, no dequant needed. gguf reader returns
    # (n_expert, n_embd) — already the right orientation for W @ xb.
    router_w = tensors["blk.0.ffn_gate_inp.weight"].data.astype(np.float64)
    logits = router_w @ xb
    probs = softmax(logits)
    top_idx = np.argsort(-probs)[:n_expert_used]
    top_w = probs[top_idx]
    top_w = top_w / top_w.sum()

    print(f"token {TOKEN}: experts chosen (Python) = "
         f"{list(zip(top_idx.tolist(), np.round(top_w, 4).tolist()))}")

    gate_t = tensors["blk.0.ffn_gate_exps.weight"]
    up_t   = tensors["blk.0.ffn_up_exps.weight"]
    down_t = tensors["blk.0.ffn_down_exps.weight"]
    gate_qtype = quants.GGMLQuantizationType(gate_t.tensor_type)
    up_qtype   = quants.GGMLQuantizationType(up_t.tensor_type)
    down_qtype = quants.GGMLQuantizationType(down_t.tensor_type)

    py_out = np.zeros_like(xb)
    for e, w in zip(top_idx, top_w):
        ge = quants.dequantize(gate_t.data[e:e + 1], gate_qtype)[0].astype(np.float64)
        ue = quants.dequantize(up_t.data[e:e + 1], up_qtype)[0].astype(np.float64)
        de = quants.dequantize(down_t.data[e:e + 1], down_qtype)[0].astype(np.float64)
        g = ge @ xb
        u = ue @ xb
        silu = g / (1.0 + np.exp(-g))
        h = silu * u
        d = de @ h
        py_out += w * d

    cos = float(np.dot(py_out, c_out) / (np.linalg.norm(py_out) * np.linalg.norm(c_out)))
    max_ad = float(np.abs(py_out - c_out).max())
    rel_rms = float(np.sqrt(np.mean((py_out - c_out) ** 2)) / np.sqrt(np.mean(py_out ** 2)))

    print(f"cosine C vs Python: {cos:.6f}  max|diff|={max_ad:.4e}  rel_rms={rel_rms:.4e}")

    # criterion: same spirit as check_m2 (Q4_K_M is never bit-exact,
    # but the cosine must be very high — here we compare C AGAINST AN
    # INDEPENDENT REIMPLEMENTATION on the SAME quantized weights, so we
    # expect agreement even tighter than a real fp32 oracle: the only
    # possible difference is floating-point summation order inside the
    # dot product, not different quantization noise.
    ok = cos > 0.999
    print("MOE VALIDATED" if ok else "MOE FAILED — check the router/expert formula")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
