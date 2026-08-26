# forge_tiny.py — generates a TINY .forge container from
# glm_tiny_sparse.gguf (F32), for developing/testing the C loader in
# seconds instead of against the 403GB container. Same layout as the
# real converter: [NFRG][len][json][1024-aligned tensors], HF names,
# per-tensor experts i4gs64, dense i8gs64, norms/router f32.
# kv_b_proj RECONSTRUCTED from attn_k_b (de-transposed per head) +
# attn_v_b, so the loader exercises the same split as the real one.
# Internal dimensions not a multiple of 64 stay at f32 (a kind already
# supported everywhere).
# Usage: python tools/forge_tiny.py [dest]
import json, struct, sys
import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from forge_quant import quant_gs64, pack_nibbles, GS
from gguf import GGUFReader

ALIGN = 1024

def fld(r, k):
    f = r.fields.get(k)
    if f is None: return None
    v = f.parts[f.data[0]].tolist()
    return v[0] if isinstance(v, list) and len(v) == 1 else v

def q_i8(w):
    O, I = w.shape
    g = w.reshape(O, I // GS, GS)
    s = (np.abs(g).max(axis=2) / 127.0).astype(np.float16)
    sf = s.astype(np.float32); sf[sf == 0] = 1.0
    q = np.clip(np.rint(g / sf[:, :, None]), -128, 127).astype(np.int8)
    return np.concatenate([q.reshape(O, I).view(np.uint8),
                           s.view(np.uint16).view(np.uint8).reshape(O, -1)], axis=1)

def q_i4(w):
    q, s = quant_gs64(w)
    O = w.shape[0]
    return np.concatenate([pack_nibbles(q).reshape(O, -1),
                           s.view(np.uint16).view(np.uint8).reshape(O, -1)], axis=1)

def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else "tools/glm_tiny_sparse.forge"
    r = GGUFReader("tools/glm_tiny_sparse.gguf")
    T = {}
    for t in r.tensors:
        # gguf-py: data already in reversed ne[] order -> numpy shape ok
        T[t.name] = np.array(t.data, dtype=np.float32).reshape(
            [int(d) for d in reversed(t.shape)]) if t.data.dtype != np.float32 \
            else t.data.reshape([int(d) for d in reversed(t.shape)]).astype(np.float32)

    NL   = int(fld(r, "glm-dsa.block_count"))
    E    = int(fld(r, "glm-dsa.embedding_length"))
    NH   = int(fld(r, "glm-dsa.attention.head_count"))
    KLR  = int(fld(r, "glm-dsa.attention.kv_lora_rank"))
    NOPE = int(fld(r, "glm-dsa.attention.key_length")) - int(fld(r, "glm-dsa.rope.dimension_count") or fld(r, "glm-dsa.attention.rope_dimension") or 0)
    # when in doubt: NOPE from the k_b tensor itself (ne=[NOPE, KLR, NH] -> numpy shape [NH, KLR, NOPE])
    kb = T[f"blk.0.attn_k_b.weight"]
    NOPE = kb.shape[-1]
    vb = T[f"blk.0.attn_v_b.weight"]
    VH = vb.shape[-2] if vb.ndim == 3 else vb.shape[0] // NH
    cfg = {
        "num_hidden_layers": NL,
        "first_k_dense_replace": int(fld(r, "glm-dsa.leading_dense_block_count")),
        "n_routed_experts": int(fld(r, "glm-dsa.expert_count")),
        "num_experts_per_tok": int(fld(r, "glm-dsa.expert_used_count")),
        "hidden_size": E,
        "intermediate_size": int(fld(r, "glm-dsa.feed_forward_length")),
        "moe_intermediate_size": int(fld(r, "glm-dsa.expert_feed_forward_length")),
        "num_attention_heads": NH,
        "q_lora_rank": int(fld(r, "glm-dsa.attention.q_lora_rank")),
        "kv_lora_rank": KLR,
        "qk_rope_head_dim": int(fld(r, "glm-dsa.rope.dimension_count")),
        "qk_nope_head_dim": NOPE,
        "v_head_dim": VH,
        "vocab_size": T["token_embd.weight"].shape[0],
        "rope_theta": float(fld(r, "glm-dsa.rope.freq_base")),
        "rms_norm_eps": float(fld(r, "glm-dsa.attention.layer_norm_rms_epsilon")),
        "index_n_heads": int(fld(r, "glm-dsa.attention.indexer.head_count")),
        "index_head_dim": int(fld(r, "glm-dsa.attention.indexer.key_length")),
        "index_topk": int(fld(r, "glm-dsa.attention.indexer.top_k")),
        "n_shared_experts": 1,
        "norm_topk_prob": 1.0,
        "routed_scaling_factor": float(fld(r, "glm-dsa.expert_weights_scale") or 1.0),
    }

    out = {}   # HF name -> (kind, np array f32 [O,I] or vector)
    def put(name, w, prefer):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 1 or prefer == "f32" or w.shape[-1] % GS != 0:
            out[name] = ("f32", w)
        else:
            out[name] = (prefer, w)

    NM = {  # blk.N.<gguf> -> model.layers.N.<hf>
        "attn_norm.weight": ("input_layernorm.weight", "f32"),
        "ffn_norm.weight": ("post_attention_layernorm.weight", "f32"),
        "attn_q_a_norm.weight": ("self_attn.q_a_layernorm.weight", "f32"),
        "attn_kv_a_norm.weight": ("self_attn.kv_a_layernorm.weight", "f32"),
        "attn_q_a.weight": ("self_attn.q_a_proj.weight", "i8gs64"),
        "attn_q_b.weight": ("self_attn.q_b_proj.weight", "i8gs64"),
        "attn_kv_a_mqa.weight": ("self_attn.kv_a_proj_with_mqa.weight", "i8gs64"),
        "attn_output.weight": ("self_attn.o_proj.weight", "i8gs64"),
        "ffn_gate.weight": ("mlp.gate_proj.weight", "i8gs64"),
        "ffn_up.weight": ("mlp.up_proj.weight", "i8gs64"),
        "ffn_down.weight": ("mlp.down_proj.weight", "i8gs64"),
        "ffn_gate_inp.weight": ("mlp.gate.weight", "f32"),
        "exp_probs_b.bias": ("mlp.gate.e_score_correction_bias", "f32"),
        "exp_probs_b": ("mlp.gate.e_score_correction_bias", "f32"),
        "ffn_gate_shexp.weight": ("mlp.shared_experts.gate_proj.weight", "i8gs64"),
        "ffn_up_shexp.weight": ("mlp.shared_experts.up_proj.weight", "i8gs64"),
        "ffn_down_shexp.weight": ("mlp.shared_experts.down_proj.weight", "i8gs64"),
        "indexer.attn_q_b.weight": ("self_attn.indexer.wq_b.weight", "i8gs64"),
        "indexer.attn_k.weight": ("self_attn.indexer.wk.weight", "i8gs64"),
        "indexer.proj.weight": ("self_attn.indexer.weights_proj.weight", "i8gs64"),
        "indexer.k_norm.weight": ("self_attn.indexer.k_norm.weight", "f32"),
        "indexer.k_norm.bias": ("self_attn.indexer.k_norm.bias", "f32"),
    }
    for name, w in T.items():
        if name == "token_embd.weight": put("model.embed_tokens.weight", w, "i4gs64"); continue
        if name == "output.weight":     put("lm_head.weight", w, "i4gs64"); continue
        if name == "output_norm.weight": put("model.norm.weight", w, "f32"); continue
        if not name.startswith("blk."): continue
        L = int(name.split(".")[1])
        rest = name.split(".", 2)[2]
        pre = f"model.layers.{L}."
        if rest in NM:
            hf, kind = NM[rest]
            put(pre + hf, w, kind)
        elif rest in ("attn_k_b.weight", "attn_v_b.weight"):
            pass   # handled below (kv_b merge)
        elif rest.startswith("ffn_") and rest.endswith("_exps.weight"):
            mat = {"ffn_gate_exps.weight": "gate_proj",
                   "ffn_up_exps.weight": "up_proj",
                   "ffn_down_exps.weight": "down_proj"}[rest]
            for e in range(w.shape[0]):
                put(pre + f"mlp.experts.{e}.{mat}.weight", w[e], "i4gs64")
        else:
            raise SystemExit(f"unmapped tensor: {name}")

    # kv_b_proj: per-head rows [NOPE (K) | VH (V)] x KLR.
    # attn_k_b GGUF: pre-transposed, numpy [NH, KLR, NOPE] -> K rows = transpose
    # attn_v_b GGUF: natural, numpy [NH, VH, KLR] -> V rows direct
    for L in range(NL):
        kb = T[f"blk.{L}.attn_k_b.weight"].reshape(NH, KLR, NOPE)
        vb = T[f"blk.{L}.attn_v_b.weight"].reshape(NH, VH, KLR)
        rows = []
        for h in range(NH):
            rows.append(kb[h].T)          # [NOPE, KLR]
            rows.append(vb[h])            # [VH, KLR]
        put(f"model.layers.{L}.self_attn.kv_b_proj.weight",
            np.concatenate(rows, axis=0), "i8gs64")

    # write-out: same scheme as the real converter (json + aligned tensors)
    def payload(kind, w):
        if kind == "f32":   return w.reshape(-1).astype(np.float32).tobytes()
        if kind == "i4gs64": return q_i4(w).tobytes()
        return q_i8(w).tobytes()
    bufs = {n: payload(k, w) for n, (k, w) in out.items()}
    def directory(off0):
        off, d = off0, {}
        for n in sorted(out):
            off = (off + ALIGN - 1) // ALIGN * ALIGN
            k, w = out[n]
            shape = list(w.shape) if w.ndim == 2 else [int(w.size)]
            d[n] = {"offset": off, "shape": shape, "kind": k, "shard": "tiny"}
            off += len(bufs[n])
        return d, off
    probe, _ = directory(0)
    hdr = ((8 + 8 + len(json.dumps({"version": 1, "config": cfg, "tensors": probe}).encode()) + 4096) // ALIGN + 1) * ALIGN
    d, total = directory(hdr)
    meta = {"version": 1, "config": cfg, "tensors": d, "header_bytes": hdr}
    with open(dest, "wb") as f:
        f.write(b"NFRG")
        jb = json.dumps(meta).encode()
        f.write(struct.pack("<Q", len(jb)))
        f.write(jb)
        for n in sorted(out):
            f.seek(d[n]["offset"]); f.write(bufs[n])
        f.truncate(total)
    print(f"tiny .forge written: {dest} ({total/1e6:.1f}MB, {len(out)} tensors)")

if __name__ == "__main__":
    main()
