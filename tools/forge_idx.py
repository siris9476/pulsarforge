# forge_idx.py — distills the .forge container's JSON directory into a
# BINARY fixed-record index (.forge.idx), easy to read from C:
#
#   [magic "NFIX" u32][version u32]
#   [n_cfg u32] then n_cfg pairs (32B zero-padded ascii key, f64 value)
#   [n_dense u32] then n_dense records: 96B ascii name, kind u32
#       (0=f32 1=i4gs64 2=i8gs64), shape u32 x2 (O, I; I=1 for vectors),
#       offset u64
#   [n_layers_moe u32][n_experts u32] then, for each moe layer in order
#       and for each expert 0..255: THREE u64 (gate, up, down offset) —
#       the fetch will do ONE pread from min(offset) for the whole
#       contiguous record, but explicit offsets keep the index
#       independent of layout.
#
# Usage: python tools/forge_idx.py [path.forge]
import json, struct, sys
import numpy as np

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "glm52.forge"
    f = open(path, "rb")
    assert f.read(4) == b"NFRG"
    n = struct.unpack("<Q", f.read(8))[0]
    meta = json.loads(f.read(n))
    cfg = meta["config"]
    tensors = meta["tensors"]

    # numeric config useful to the loader (from HF's config.json)
    keys = ["num_hidden_layers", "first_k_dense_replace", "n_routed_experts",
            "num_experts_per_tok", "hidden_size", "intermediate_size",
            "moe_intermediate_size", "num_attention_heads", "q_lora_rank",
            "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim",
            "v_head_dim", "vocab_size", "rope_theta", "rms_norm_eps",
            "routed_scaling_factor", "norm_topk_prob", "n_shared_experts",
            "index_n_heads", "index_head_dim", "index_topk"]
    cfg_pairs = []
    for k in keys:
        if k in cfg:
            v = cfg[k]
            cfg_pairs.append((k, float(v) if not isinstance(v, bool) else float(int(v))))

    dense, expert = {}, {}
    for name, e in tensors.items():
        if ".mlp.experts." in name:
            # model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight
            p = name.split(".")
            L, E = int(p[2]), int(p[5])
            m = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}[p[6]]
            expert[(L, E, m)] = e["offset"]
        else:
            kind = {"f32": 0, "i4gs64": 1, "i8gs64": 2}[e["kind"]]
            shape = e["shape"] + [1]
            dense[name] = (kind, shape[0], shape[1], e["offset"])

    layers = sorted({k[0] for k in expert})
    n_exp = max(k[1] for k in expert) + 1
    out = open(path + ".idx", "wb")
    out.write(b"NFIX"); out.write(struct.pack("<I", 2))
    out.write(struct.pack("<I", len(cfg_pairs)))
    for k, v in cfg_pairs:
        out.write(struct.pack("<32sd", k.encode(), v))
    out.write(struct.pack("<I", len(dense)))
    for name in sorted(dense):
        kind, O, I, off = dense[name]
        out.write(struct.pack("<96sIIIQ", name.encode(), kind, O, I, off))
    out.write(struct.pack("<II", len(layers), n_exp))
    # v2: for each moe layer, the KINDs of the three expert matrices
    # (uniform within a layer by construction: same shapes) — 0=f32
    # 1=i4 2=i8. On the real model these are always (1,1,1); on the
    # tiny model, internal dimensions not a multiple of 64 can leave a
    # matrix at f32.
    kmap = {"f32": 0, "i4gs64": 1, "i8gs64": 2}
    for L in layers:
        for mm_ in ("gate_proj", "up_proj", "down_proj"):
            e0 = tensors[f"model.layers.{L}.mlp.experts.0.{mm_}.weight"]
            out.write(struct.pack("<I", kmap[e0["kind"]]))
    for L in layers:
        for E in range(n_exp):
            for m in range(3):
                out.write(struct.pack("<Q", expert[(L, E, m)]))
    out.close()
    print(f"idx written: {len(cfg_pairs)} cfg, {len(dense)} dense, "
          f"{len(layers)} moe layers x {n_exp} experts")
    # sanity: the expert record is ADJACENT in name-sort order
    # (alphabetical: down|gate|up) — no gaps beyond 1024-byte alignment.
    # If so, the fetch does ONE pread from "down" for the whole record.
    # (Kinds can differ within a record — on the tiny model, internal
    # dimensions not a multiple of 64 stay at f32 — so we compare the
    # TRUE sizes, not a uniform stride.)
    def tsize(name):
        e = tensors[name]
        O, I = (e["shape"] + [1])[:2]
        if e["kind"] == "f32": return O * I * 4
        if e["kind"] == "i4gs64": return O * (I // 2 + I // 64 * 2)
        return O * (I + I // 64 * 2)
    def al(x): return (x + 1023) // 1024 * 1024
    bad = 0
    span = None
    for L in layers:
        for E in range(n_exp):
            names3 = [f"model.layers.{L}.mlp.experts.{E}.{mm}.weight"
                      for mm in ("down_proj", "gate_proj", "up_proj")]
            offs = [tensors[nm]["offset"] for nm in names3]
            d, g, u = offs
            ok = (d < g < u and al(d + tsize(names3[0])) == g
                  and al(g + tsize(names3[1])) == u)
            if not ok: bad += 1
            else: span = u + tsize(names3[2]) - d
    print(f"  record adjacency (down|gate|up): "
          f"{'ALL OK, one pread from down' if bad == 0 else f'{bad} BROKEN'}"
          f" (record span {span} bytes)")

if __name__ == "__main__":
    main()
