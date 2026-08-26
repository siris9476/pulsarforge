#!/usr/bin/env python3
"""Converts the synthetic GLM-5.2 (glm-dsa) toy model produced by
colibri's own c/tools/make_glm_oracle.py (safetensors + config.json,
written to c/glm_tiny/) into a GGUF v3 with the "glm-dsa"
architecture, to validate pulsarforge's naive implementation
(nf_model_forward_glm) against the real HF oracle (transformers
GlmMoeDsaForCausalLM).

Throwaway (for THIS test ONLY): no quantization, all tensors F32, no
generalization to real models — a one-off that maps EXACTLY the toy
model's tensors from colibri (5 layers: 3 dense + 2 MoE, all "full"
for the DSA indexer) to the GGUF names nf_model.c reads for glm-dsa
(see nf_model_load in nf_model.c).

Usage: python tools/make_glm_tiny_gguf.py <glm_tiny_dir> <out.gguf>
  <glm_tiny_dir> is colibri's own c/glm_tiny/ output directory
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

GGUF_U32 = 4
GGUF_F32 = 6
GGUF_STR = 8
GGUF_ARR = 9


def s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


class GGUFWriter:
    def __init__(self):
        self.kv = []          # (key_bytes_incl_type, ) already serialized
        self.tensors = []      # (name, dims_ggml, data_bytes)

    def add_str(self, key, val):
        self.kv.append(s(key) + struct.pack("<I", GGUF_STR) + s(val))

    def add_u32(self, key, val):
        self.kv.append(s(key) + struct.pack("<I", GGUF_U32) + struct.pack("<I", int(val)))

    def add_f32(self, key, val):
        self.kv.append(s(key) + struct.pack("<I", GGUF_F32) + struct.pack("<f", float(val)))

    def add_str_array(self, key, vals):
        buf = s(key) + struct.pack("<I", GGUF_ARR)
        buf += struct.pack("<I", GGUF_STR) + struct.pack("<Q", len(vals))
        for v in vals:
            buf += s(v)
        self.kv.append(buf)

    def add_tensor(self, name, arr: np.ndarray):
        """arr: numpy array in PyTorch order (row-major, last dim
        contiguous). GGUF/ggml dims go "in reverse" (ne[0] = fastest
        dim = numpy's last) — same bytes, only the dimension order in
        the directory changes. Always F32 (type=0)."""
        arr = np.ascontiguousarray(arr.astype(np.float32))
        dims_ggml = list(reversed(arr.shape))
        self.tensors.append((name, dims_ggml, arr.tobytes()))

    def write(self, path):
        n_tensors = len(self.tensors)
        n_kv = len(self.kv)
        header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", n_tensors, n_kv)
        kv_blob = b"".join(self.kv)

        # directory: name, n_dims, dims (u64 each), ggml type (u32), offset (u64)
        alignment = 32
        dir_blob = b""
        data_blob = b""
        offset = 0
        for name, dims, data in self.tensors:
            dir_blob += s(name)
            dir_blob += struct.pack("<I", len(dims))
            for d in dims:
                dir_blob += struct.pack("<Q", d)
            dir_blob += struct.pack("<I", 0)   # ggml type F32
            dir_blob += struct.pack("<Q", offset)
            pad = (-len(data)) % alignment
            data_blob += data + b"\x00" * pad
            offset += len(data) + pad

        pre_data = header + kv_blob + dir_blob
        pad = (-len(pre_data)) % alignment
        pre_data += b"\x00" * pad

        with open(path, "wb") as f:
            f.write(pre_data)
            f.write(data_blob)
        print(f"written {path}: {n_tensors} tensors, {n_kv} metadata entries, "
              f"{len(pre_data) + len(data_blob)} bytes")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    src = Path(sys.argv[1])
    out = sys.argv[2]

    cfg = json.load(open(src / "config.json"))
    st = safe_open(str(src / "model.safetensors"), framework="np")

    def t(name):
        return st.get_tensor(name)

    E = cfg["hidden_size"]
    NL = cfg["num_hidden_layers"]
    NH = cfg["num_attention_heads"]
    NKV = cfg.get("num_key_value_heads", NH)
    VOCAB = cfg["vocab_size"]
    Q_LORA = cfg["q_lora_rank"]
    KV_LORA = cfg["kv_lora_rank"]
    QK_ROPE = cfg["qk_rope_head_dim"]
    QK_NOPE = cfg["qk_nope_head_dim"]
    V_HEAD = cfg["v_head_dim"]
    HEAD_DIM = QK_NOPE + QK_ROPE
    NE = cfg["n_routed_experts"]
    K_USED = cfg["num_experts_per_tok"]
    N_SHARED = cfg["n_shared_experts"]
    FF_MOE = cfg["moe_intermediate_size"]
    FF_DENSE = cfg["intermediate_size"]
    FIRST_DENSE = cfg["first_k_dense_replace"]
    IDX_NH = cfg["index_n_heads"]
    IDX_HD = cfg["index_head_dim"]
    IDX_TOPK = cfg["index_topk"]
    ROUTED_SCALE = cfg["routed_scaling_factor"]
    RMS_EPS = cfg["rms_norm_eps"]
    ROPE_THETA = cfg["rope_parameters"]["rope_theta"]
    IDX_TYPES = cfg.get("indexer_types", ["full"] * NL)

    print(f"config: E={E} NL={NL} NH={NH} Q_LORA={Q_LORA} KV_LORA={KV_LORA} "
          f"HEAD_DIM={HEAD_DIM} (nope={QK_NOPE}+rope={QK_ROPE}) V_HEAD={V_HEAD} "
          f"NE={NE} K={K_USED} FF_MOE={FF_MOE} FF_DENSE={FF_DENSE} "
          f"FIRST_DENSE={FIRST_DENSE} IDX_NH={IDX_NH} IDX_HD={IDX_HD} "
          f"IDX_TOPK={IDX_TOPK} indexer_types={IDX_TYPES}")

    assert HEAD_DIM * NH == t(f"model.layers.0.self_attn.q_b_proj.weight").shape[0]

    # MTP ("nextn"): the synthetic toy model has the extra layer ONLY
    # if make_glm_oracle.py was extended to produce it (the hybrid
    # oracle — see the "Hybrid MTP (nextn) oracle" block in there) —
    # auto-detected from the presence of nextn.eh_proj.weight in the
    # safetensors, the same "no flag, auto-detect" scheme already used
    # for the indexer/layer types. If absent: behavior IDENTICAL to
    # before this feature (no extra tensor/key written).
    all_keys = set(st.keys())
    HAS_MTP = "nextn.eh_proj.weight" in all_keys
    print(f"MTP (nextn) present in the toy model: {HAS_MTP}")

    w = GGUFWriter()
    w.add_str("general.architecture", "glm-dsa")
    w.add_u32("general.alignment", 32)
    pfx = "glm-dsa"
    # block_count includes the MTP layer when present, EXACTLY like
    # the real (Unsloth) GGUF — see nf_model_load, bug 2 "block_count
    # includes the MTP layer": num_hidden_layers (NL) stays the true
    # count of normal layers, block_count is NL+1 and
    # nextn_predict_layers=1 tells the loader to subtract it. Before
    # this feature block_count was always NL (nextn=0 by default, no
    # adjustment) — this is the first synthetic toy model exercising
    # that subtraction.
    w.add_u32(f"{pfx}.block_count", NL + (1 if HAS_MTP else 0))
    w.add_u32(f"{pfx}.embedding_length", E)
    w.add_u32(f"{pfx}.attention.head_count", NH)
    w.add_u32(f"{pfx}.attention.head_count_kv", NKV)
    w.add_f32(f"{pfx}.rope.freq_base", ROPE_THETA)
    w.add_f32(f"{pfx}.attention.layer_norm_rms_epsilon", RMS_EPS)
    # NOTE: on the real (Unsloth) GGUF, "attention.key_length"/"value_length"
    # (WITHOUT suffix) are the width of the COMPRESSED CACHE
    # (kv_lora_rank+qk_rope_dim for key, kv_lora_rank for value),
    # different from the per-head width ("_mla"). Both pairs are
    # written here, with DELIBERATELY DIFFERENT values, so the test
    # catches whether the loader reads the wrong key (a real bug found
    # by inspecting the actual model — see the comment in nf_model.c).
    w.add_u32(f"{pfx}.attention.key_length", KV_LORA + QK_ROPE)
    w.add_u32(f"{pfx}.attention.key_length_mla", HEAD_DIM)
    w.add_u32(f"{pfx}.expert_count", NE)
    w.add_u32(f"{pfx}.expert_used_count", K_USED)
    w.add_u32(f"{pfx}.expert_feed_forward_length", FF_MOE)
    w.add_u32(f"{pfx}.attention.kv_lora_rank", KV_LORA)
    w.add_u32(f"{pfx}.attention.q_lora_rank", Q_LORA)
    w.add_u32(f"{pfx}.rope.dimension_count", QK_ROPE)
    w.add_u32(f"{pfx}.attention.value_length", KV_LORA)
    w.add_u32(f"{pfx}.attention.value_length_mla", V_HEAD)
    w.add_u32(f"{pfx}.leading_dense_block_count", FIRST_DENSE)
    w.add_u32(f"{pfx}.expert_shared_count", N_SHARED)
    w.add_u32(f"{pfx}.feed_forward_length", FF_DENSE)
    w.add_u32(f"{pfx}.attention.indexer.head_count", IDX_NH)
    w.add_u32(f"{pfx}.attention.indexer.key_length", IDX_HD)
    w.add_u32(f"{pfx}.attention.indexer.top_k", IDX_TOPK)
    w.add_f32(f"{pfx}.expert_weights_scale", ROUTED_SCALE)
    w.add_str_array(f"{pfx}.attention.indexer.types", IDX_TYPES)
    if HAS_MTP:
        w.add_u32(f"{pfx}.nextn_predict_layers", 1)

    w.add_tensor("token_embd.weight", t("model.embed_tokens.weight"))
    w.add_tensor("output.weight", t("lm_head.weight"))
    w.add_tensor("output_norm.weight", t("model.norm.weight"))

    # MTP: blk.NL is the extra layer (attn+moe, same names/shapes as
    # any normal layer — see the "Hybrid MTP oracle" comment in
    # make_glm_oracle.py) when HAS_MTP. Reuses the SAME loop body
    # VERBATIM: the only difference is the indexer, always "full" for
    # the MTP layer (there's no IDX_TYPES[NL] entry in config.json,
    # the extra layer doesn't exist in the original config — the
    # oracle built it with a temporary "full" indexer just for
    # initialization).
    for l in range(NL + (1 if HAS_MTP else 0)):
        p = f"model.layers.{l}."
        w.add_tensor(f"blk.{l}.attn_norm.weight", t(p + "input_layernorm.weight"))
        w.add_tensor(f"blk.{l}.attn_q_a_norm.weight", t(p + "self_attn.q_a_layernorm.weight"))
        w.add_tensor(f"blk.{l}.attn_kv_a_norm.weight", t(p + "self_attn.kv_a_layernorm.weight"))
        w.add_tensor(f"blk.{l}.ffn_norm.weight", t(p + "post_attention_layernorm.weight"))

        w.add_tensor(f"blk.{l}.attn_q_a.weight", t(p + "self_attn.q_a_proj.weight"))
        w.add_tensor(f"blk.{l}.attn_q_b.weight", t(p + "self_attn.q_b_proj.weight"))
        w.add_tensor(f"blk.{l}.attn_kv_a_mqa.weight", t(p + "self_attn.kv_a_proj_with_mqa.weight"))

        # Real bug found while loading the actual GGUF (Unsloth/llama.cpp):
        # there's no combined attn_kv_b for glm-dsa — it's split into
        # two per-head 3D tensors. kv_b_proj.weight (HF, combined) has
        # shape [NH*(QK_NOPE+V_HEAD), KV_LORA] (nn.Linear:
        # out_features x in_features); sliced per head and per part
        # (nope/v).
        #   attn_v_b: already in the "natural" direction (contracts
        #   over the incoming KV_LORA) — no transpose, numpy shape
        #   [NH, V_HEAD, KV_LORA] -> ggml ne=[KV_LORA, V_HEAD, NH].
        #   attn_k_b: upstream llama.cpp stores it PRE-TRANSPOSED for
        #   its own absorption trick (contracts over QK_NOPE, produces
        #   the latent space — empirically verified against `nf
        #   inspect --tensors` on the real GGUF: ne=[qk_nope_head_dim,
        #   kv_lora_rank, n_head], NOT [kv_lora_rank, qk_nope_head_dim,
        #   n_head] as symmetry with attn_v_b would suggest). The K
        #   slice is transposed with swapaxes before writing, so the
        #   synthetic tensor reproduces EXACTLY the same axis
        #   convention as the real one (otherwise the test wouldn't
        #   exercise the transpose bug in nf_model.c at all).
        kv_b = t(p + "self_attn.kv_b_proj.weight").reshape(NH, QK_NOPE + V_HEAD, KV_LORA)
        k_b = np.swapaxes(kv_b[:, :QK_NOPE, :], 1, 2)   # [NH, KV_LORA, QK_NOPE] (transposed)
        v_b = kv_b[:, QK_NOPE:, :]                       # [NH, V_HEAD, KV_LORA]  (natural)
        w.add_tensor(f"blk.{l}.attn_k_b.weight", k_b)
        w.add_tensor(f"blk.{l}.attn_v_b.weight", v_b)
        w.add_tensor(f"blk.{l}.attn_output.weight", t(p + "self_attn.o_proj.weight"))

        # indexer: present on ALL layers in this toy model
        # (indexer_types all "full", see the glm_tiny/config.json
        # produced by make_glm_oracle.py) — the MTP layer (l==NL) has
        # no entry of its own in IDX_TYPES (it never existed in the
        # original config), always "full" by construction.
        is_full_idx = (l >= NL) or (IDX_TYPES[l] == "full")
        if is_full_idx:
            w.add_tensor(f"blk.{l}.indexer.attn_q_b.weight", t(p + "self_attn.indexer.wq_b.weight"))
            w.add_tensor(f"blk.{l}.indexer.attn_k.weight", t(p + "self_attn.indexer.wk.weight"))
            w.add_tensor(f"blk.{l}.indexer.proj.weight", t(p + "self_attn.indexer.weights_proj.weight"))
            w.add_tensor(f"blk.{l}.indexer.k_norm.weight", t(p + "self_attn.indexer.k_norm.weight"))
            w.add_tensor(f"blk.{l}.indexer.k_norm.bias", t(p + "self_attn.indexer.k_norm.bias"))

        if l < FIRST_DENSE:
            w.add_tensor(f"blk.{l}.ffn_gate.weight", t(p + "mlp.gate_proj.weight"))
            w.add_tensor(f"blk.{l}.ffn_up.weight", t(p + "mlp.up_proj.weight"))
            w.add_tensor(f"blk.{l}.ffn_down.weight", t(p + "mlp.down_proj.weight"))
        else:
            w.add_tensor(f"blk.{l}.ffn_gate_inp.weight", t(p + "mlp.gate.weight"))
            w.add_tensor(f"blk.{l}.exp_probs_b", t(p + "mlp.gate.e_score_correction_bias"))

            gate = np.stack([t(p + f"mlp.experts.{e}.gate_proj.weight") for e in range(NE)], axis=0)
            up = np.stack([t(p + f"mlp.experts.{e}.up_proj.weight") for e in range(NE)], axis=0)
            down = np.stack([t(p + f"mlp.experts.{e}.down_proj.weight") for e in range(NE)], axis=0)
            w.add_tensor(f"blk.{l}.ffn_gate_exps.weight", gate)
            w.add_tensor(f"blk.{l}.ffn_up_exps.weight", up)
            w.add_tensor(f"blk.{l}.ffn_down_exps.weight", down)

            w.add_tensor(f"blk.{l}.ffn_gate_shexp.weight", t(p + "mlp.shared_experts.gate_proj.weight"))
            w.add_tensor(f"blk.{l}.ffn_up_shexp.weight", t(p + "mlp.shared_experts.up_proj.weight"))
            w.add_tensor(f"blk.{l}.ffn_down_shexp.weight", t(p + "mlp.shared_experts.down_proj.weight"))

    # MTP: the 4 tensors belonging to the "nextn" head, top-level in
    # the safetensors (nextn.eh_proj/enorm/hnorm/shared_head_norm.weight
    # — see the "Hybrid MTP oracle" comment in make_glm_oracle.py),
    # written under blk.<NL>.nextn.* — the SAME convention as the real
    # GGUF (blk.78.nextn.{eh_proj,enorm,hnorm,shared_head_norm},
    # verified with `nf inspect --tensors` on the actual model).
    if HAS_MTP:
        w.add_tensor(f"blk.{NL}.nextn.eh_proj.weight", t("nextn.eh_proj.weight"))
        w.add_tensor(f"blk.{NL}.nextn.enorm.weight", t("nextn.enorm.weight"))
        w.add_tensor(f"blk.{NL}.nextn.hnorm.weight", t("nextn.hnorm.weight"))
        w.add_tensor(f"blk.{NL}.nextn.shared_head_norm.weight", t("nextn.shared_head_norm.weight"))

    w.write(out)


if __name__ == "__main__":
    main()
