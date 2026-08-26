# forge_oracle.py — INDEPENDENT ORACLE for the tiny .forge.
#
# Independence of ASSUMPTIONS: this forward pass is derived LINE BY
# LINE from the official HuggingFace modeling code
# (modular_glm_moe_dsa.py + modular_deepseek_v32.py +
# modeling_deepseek_v3.py), NOT from our engine's design. Points kept
# faithful to the HF source:
#   - attention: q = [q_pass(NOPE) | q_rot(RPE)] (nope FIRST), rope
#     INTERLEAVED in pairs (x0,x1) with a per-pair angle;
#   - indexer: q = [q_rot | q_pass] (rope FIRST), k = LayerNorm(wk(x)),
#     scores = relu(q.k * D^-0.5), head weights = weights_proj(x)*H^-0.5,
#     causal topk;
#   - router: sigmoid; selection on sigmoid+bias; WEIGHTS = sigmoid
#     WITHOUT bias of the selected ones, normalized if norm_topk_prob,
#     x scaling;
#   - MoE: weighted sum of experts + shared; FFN = down(silu(gate)*up).
#
# Comparison: logits from the engine's EVAL PATH (nf forgelogits) for
# each teacher-forced position. Usage:
#   python tools/forge_oracle.py <tiny.forge> <ids-csv> <eng_logits.bin>
import json, struct, sys
import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from forge_quant import GS

def load_forge(path):
    f = open(path, "rb")
    assert f.read(4) == b"NFRG"
    n = struct.unpack("<Q", f.read(8))[0]
    meta = json.loads(f.read(n))
    T = {}
    for name, e in meta["tensors"].items():
        O, I = (e["shape"] + [1])[:2]
        f.seek(e["offset"])
        if e["kind"] == "f32":
            w = np.frombuffer(f.read(O * I * 4), dtype=np.float32).reshape(
                e["shape"]).astype(np.float32)
        elif e["kind"] == "i4gs64":
            rb = I // 2 + I // GS * 2
            b = np.frombuffer(f.read(O * rb), dtype=np.uint8).reshape(O, rb)
            lo = (b[:, :I // 2] & 0xF).astype(np.int8)
            hi = (b[:, :I // 2] >> 4).astype(np.int8)
            lo[lo > 7] -= 16; hi[hi > 7] -= 16
            q = np.empty((O, I), dtype=np.int8)
            q[:, 0::2] = lo; q[:, 1::2] = hi
            sc = b[:, I // 2:].copy().view(np.uint16).view(np.float16)
            w = (q.reshape(O, I // GS, GS).astype(np.float32)
                 * sc.astype(np.float32)[:, :, None]).reshape(O, I)
        else:  # i8gs64
            rb = I + I // GS * 2
            b = np.frombuffer(f.read(O * rb), dtype=np.uint8).reshape(O, rb)
            q = b[:, :I].view(np.int8).astype(np.float32)
            sc = b[:, I:].copy().view(np.uint16).view(np.float16)
            w = (q.reshape(O, I // GS, GS)
                 * sc.astype(np.float32)[:, :, None]).reshape(O, I)
        T[name] = w
    return meta["config"], T

def rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w

def layernorm(x, w, b, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b

def rope_interleave(x, pos, theta, rpe):
    # x: [S, H, rpe] (or [S, 1, rpe]); pairs (x0,x1) per frequency
    inv = 1.0 / theta ** (np.arange(0, rpe, 2, dtype=np.float64) / rpe)
    ang = np.asarray(pos, dtype=np.float64)[:, None] * inv[None, :]  # [S, rpe/2]
    c = np.cos(ang)[:, None, :]
    s = np.sin(ang)[:, None, :]
    xe, xo = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = xe * c - xo * s
    out[..., 1::2] = xe * s + xo * c
    return out

def silu(x):
    return x / (1.0 + np.exp(-x))

ENGSEL = {}
MARG = {"max": 0.0}

def main():
    fpath, csv, engbin = sys.argv[1], sys.argv[2], sys.argv[3]
    import os
    tr = os.environ.get("ORACLE_ENGINE_SEL")
    if tr:
        b = open(tr, "rb").read()
        i = 0
        while i < len(b):
            layer, p, keep = struct.unpack_from("<3i", b, i); i += 12
            sel = struct.unpack_from(f"<{keep}i", b, i); i += 4 * keep
            ENGSEL.setdefault((layer, p), list(sel))
    ids = [int(t) for t in csv.replace(" ", "").split(",") if t]
    cfg, T = load_forge(fpath)
    S = len(ids)
    E = int(cfg["hidden_size"]); NH = int(cfg["num_attention_heads"])
    NOPE = int(cfg["qk_nope_head_dim"]); RPE = int(cfg["qk_rope_head_dim"])
    VH = int(cfg["v_head_dim"]); KLR = int(cfg["kv_lora_rank"])
    NL = int(cfg["num_hidden_layers"]); ND = int(cfg["first_k_dense_replace"])
    NEXP = int(cfg["n_routed_experts"]); K = int(cfg["num_experts_per_tok"])
    IXH = int(cfg["index_n_heads"]); IXD = int(cfg["index_head_dim"])
    IXTOP = int(cfg["index_topk"])
    theta = float(cfg["rope_theta"]); eps = float(cfg["rms_norm_eps"])
    rsf = float(cfg.get("routed_scaling_factor", 1.0))
    norm_topk = bool(cfg.get("norm_topk_prob", 1.0))
    QD = NOPE + RPE
    pos = np.arange(S)

    def t(name):
        return T[name]

    x = t("model.embed_tokens.weight")[ids].astype(np.float32)   # [S, E]
    prev_topk = None
    for l in range(NL):
        P = f"model.layers.{l}."
        h = rmsnorm(x, t(P + "input_layernorm.weight"), eps)
        # ---- MLA attention (HF: GlmMoeDsaAttention.forward) ----
        q_resid = rmsnorm(h @ t(P + "self_attn.q_a_proj.weight").T,
                          t(P + "self_attn.q_a_layernorm.weight"), eps)
        q = (q_resid @ t(P + "self_attn.q_b_proj.weight").T).reshape(S, NH, QD)
        q_pass, q_rot = q[..., :NOPE], q[..., NOPE:]
        ckv = h @ t(P + "self_attn.kv_a_proj_with_mqa.weight").T   # [S, KLR+RPE]
        kv_pass, k_rot = ckv[:, :KLR], ckv[:, KLR:]
        k_lat = rmsnorm(kv_pass, t(P + "self_attn.kv_a_layernorm.weight"), eps)
        q_rot = rope_interleave(q_rot, pos, theta, RPE)
        k_rot = rope_interleave(k_rot[:, None, :], pos, theta, RPE)[:, 0, :]
        k_rot = k_rot.astype(np.float16).astype(np.float32)   # f16 cache
        kvb = (k_lat @ t(P + "self_attn.kv_b_proj.weight").T
               ).reshape(S, NH, NOPE + VH)
        # engine's f16 cache (Gkv): the declared K/V representation is
        # f16 — the oracle emulates it (representation, not math: on
        # the random tiny model, near-tied selections would otherwise
        # flip from pure rounding noise)
        kvb = kvb.astype(np.float16).astype(np.float32)
        k_nope, v = kvb[..., :NOPE], kvb[..., NOPE:]
        # ---- DSA indexer (HF: GlmMoeDsaIndexer.forward) ----
        ixk_name = P + "self_attn.indexer.wk.weight"
        if ixk_name in T:
            iq = (q_resid @ t(P + "self_attn.indexer.wq_b.weight").T
                  ).reshape(S, IXH, IXD)
            iq_rot, iq_pass = iq[..., :RPE], iq[..., RPE:]
            ik = layernorm(h @ t(ixk_name).T,
                           t(P + "self_attn.indexer.k_norm.weight"),
                           t(P + "self_attn.indexer.k_norm.bias"))
            ik_rot, ik_pass = ik[:, :RPE], ik[:, RPE:]
            iq_rot = rope_interleave(iq_rot, pos, theta, RPE)
            ik_rot = rope_interleave(ik_rot[:, None, :], pos, theta, RPE)[:, 0, :]
            iq = np.concatenate([iq_rot, iq_pass], -1)
            ik = np.concatenate([ik_rot, ik_pass], -1)
            ik = ik.astype(np.float16).astype(np.float32)     # Ixk f16 cache
            sc = np.maximum(np.einsum("shd,td->sht", iq, ik) * IXD ** -0.5, 0.0)
            wts = (h @ t(P + "self_attn.indexer.weights_proj.weight").T
                   ) * IXH ** -0.5
            iscore = np.einsum("sh,sht->st", wts, sc)
            causal = pos[None, :] > pos[:, None]
            iscore[causal] = -np.inf
            topk = min(IXTOP, S)
            prev_topk = np.argsort(-iscore, axis=-1)[:, :topk]
            # margin mode (ORACLE_ENGINE_SEL=trace.bin): FORCES the
            # engine's selections and measures how much the oracle's
            # scores disagree with them — a disagreement under fp
            # noise is a near-tie (a tolerance class), a large one is
            # a semantic bug.
            import os
            tr = os.environ.get("ORACLE_ENGINE_SEL")
            if tr:
                for s_ in range(S):
                    key = (l, s_)
                    if key in ENGSEL and s_ + 1 > IXTOP:
                        es = ENGSEL[key]
                        osel = set(int(v) for v in prev_topk[s_][:min(IXTOP, s_+1)])
                        rng = float(iscore[s_][np.isfinite(iscore[s_])].max()
                                    - iscore[s_][np.isfinite(iscore[s_])].min() + 1e-30)
                        for e_ in es:
                            if e_ not in osel:
                                worst_kept = min(float(iscore[s_][v]) for v in osel)
                                marg = (worst_kept - float(iscore[s_][e_])) / rng
                                MARG["max"] = max(MARG["max"], marg)
                        prev_topk[s_] = np.array(sorted(es) +
                            [v for v in prev_topk[s_] if int(v) not in es])[:len(prev_topk[s_])]
            import os
            if os.environ.get("ORACLE_DUMP_SEL"):
                for s_ in range(S):
                    if s_ + 1 > IXTOP:
                        print(f"oracle sel L{l} pos{s_}:",
                              sorted(int(v) for v in prev_topk[s_][:min(IXTOP, s_+1)]))
        keep = np.zeros((S, S), dtype=bool)
        for s_ in range(S):
            keep[s_, prev_topk[s_]] = True
        keep &= pos[None, :] <= pos[:, None]
        # ---- sparse attention ----
        qf = np.concatenate([q_pass, q_rot], -1)                  # [S,H,QD]
        kf = np.concatenate(
            [k_nope, np.broadcast_to(k_rot[:, None, :], (S, NH, RPE))], -1)
        att = np.einsum("shd,thd->sht", qf, kf) * QD ** -0.5      # [S,H,T]
        att[~keep[:, None, :].repeat(NH, 1).transpose(0, 1, 2)] = -np.inf
        att = att - att.max(-1, keepdims=True)
        p = np.exp(att); p /= p.sum(-1, keepdims=True)
        o = np.einsum("sht,thd->shd", p, v).reshape(S, NH * VH)
        x = x + o @ t(P + "self_attn.o_proj.weight").T
        # ---- FFN / MoE ----
        h2 = rmsnorm(x, t(P + "post_attention_layernorm.weight"), eps)
        if l < ND:
            g = h2 @ t(P + "mlp.gate_proj.weight").T
            u = h2 @ t(P + "mlp.up_proj.weight").T
            x = x + (silu(g) * u) @ t(P + "mlp.down_proj.weight").T
        else:
            logits_r = h2 @ t(P + "mlp.gate.weight").T
            scores = 1.0 / (1.0 + np.exp(-logits_r))
            choice = scores + t(P + "mlp.gate.e_score_correction_bias")
            topi = np.argsort(-choice, axis=-1)[:, :K]
            w_k = np.take_along_axis(scores, topi, -1)
            if norm_topk:
                w_k = w_k / (w_k.sum(-1, keepdims=True) + 1e-20)
            w_k = w_k * rsf
            out = np.zeros_like(x)
            for s_ in range(S):
                for j in range(K):
                    e = int(topi[s_, j])
                    EP = P + f"mlp.experts.{e}."
                    g = h2[s_] @ t(EP + "gate_proj.weight").T
                    u = h2[s_] @ t(EP + "up_proj.weight").T
                    out[s_] += w_k[s_, j] * (
                        (silu(g) * u) @ t(EP + "down_proj.weight").T)
            gs = h2 @ t(P + "mlp.shared_experts.gate_proj.weight").T
            us = h2 @ t(P + "mlp.shared_experts.up_proj.weight").T
            out += (silu(gs) * us) @ t(P + "mlp.shared_experts.down_proj.weight").T
            x = x + out
    xb = rmsnorm(x, t("model.norm.weight"), eps)
    logits = xb @ t("lm_head.weight").T                            # [S, V]

    # ---- comparison with the engine ----
    f = open(engbin, "rb")
    n_e = struct.unpack("<I", f.read(4))[0]
    v_e = struct.unpack("<Q", f.read(8))[0]
    eng = np.frombuffer(f.read(), dtype=np.float32).reshape(n_e, v_e)
    assert n_e == S and v_e == logits.shape[1], (n_e, v_e, logits.shape)
    ok = True
    for s_ in range(S):
        a, b = logits[s_], eng[s_]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        am = int(np.argmax(a)) == int(np.argmax(b))
        print(f"pos {s_}: cosine {cos:.6f}  argmax "
              f"{'OK' if am else 'DIFFERENT'} ({int(np.argmax(a))} vs {int(np.argmax(b))})")
        ok &= cos > 0.999 and am
    if ENGSEL:
        print(f"max selection disagreement margin (rel. to range): {MARG['max']:.2e}")
    print("ORACLE GATE:", "GREEN" if ok else "RED")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
