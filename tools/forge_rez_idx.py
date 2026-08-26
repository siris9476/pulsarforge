# forge_rez_idx.py — the two Python-side halves of the HUF reconversion:
#   --prep <src.forgez>       generates <src>.frames for forge_rez.exe:
#       [u64 dense_end][u64 n] then per frame (sorted by ASC zoff):
#       u64 j, zoff, zlen, span
#   --idx <src.forgez> <dst>  generates <dst>.idx (v4: layout IDENTICAL
#       to v3 but version=4 = Huff0 frames) reading the dst's .rezman
#       and the forgez's manifest json (for dense and rel_g/rel_u/span).
import io, json, os, struct, sys

def load_forgez(src):
    f = io.open(src, "rb")
    assert f.read(4) == b"NFRZ"
    (jn,) = struct.unpack("<Q", f.read(8))
    meta = json.loads(f.read(jn))
    f.close()
    man = json.load(io.open(src + ".manifest.json", encoding="utf-8"))
    return meta, man

def tsize(e):
    O, I = (e["shape"] + [1])[:2]
    if e["kind"] == "f32":    return O * I * 4
    if e["kind"] == "i4gs64": return O * (I // 2 + I // 64 * 2)
    return O * (I + I // 64 * 2)

def recs(meta):
    tensors = meta["tensors"]
    rec = {}
    for name, e in tensors.items():
        if ".mlp.experts." in name:
            p = name.split(".")
            r = rec.setdefault((int(p[2]), int(p[5])), {})
            r[p[6]] = (e["offset"], tsize(e))
    for k, r in rec.items():
        d, g, u = r["down_proj"], r["gate_proj"], r["up_proj"]
        r["off"] = d[0]; r["span"] = u[0] + u[1] - d[0]
        r["rel_g"] = g[0] - d[0]; r["rel_u"] = u[0] - d[0]
    order = sorted(rec, key=lambda k: rec[k]["off"])
    return rec, order

def prep(src):
    meta, man = load_forgez(src)
    rec, order = recs(meta)
    # dense_end = end of the forgez's head region (first frame zoff)
    frames = [(int(j), z[0], z[1]) for j, z in man["frames"].items()]
    frames.sort(key=lambda x: x[1])
    dense_end = frames[0][1]
    out = io.open(src + ".frames", "wb")
    out.write(struct.pack("<QQ", dense_end, len(frames)))
    for j, zoff, zlen in frames:
        span = rec[order[j]]["span"]
        out.write(struct.pack("<4Q", j, zoff, zlen, span))
    out.close()
    print(f"frames: {len(frames)}, head {dense_end/1e9:.2f}GB")

def build_idx(src, dst):
    meta, man = load_forgez(src)
    rec, order = recs(meta)
    tensors = meta["tensors"]
    # rezman: [done_from][h_end][n] + n x (j, hoff, hlen)
    rm = io.open(dst + ".rezman", "rb")
    done_from, h_end, n = struct.unpack("<3Q", rm.read(24))
    assert done_from == 0, f"incomplete reconversion (done_from={done_from})"
    hmap = {}
    for _ in range(n):
        j, hoff, hlen = struct.unpack("<3Q", rm.read(24))
        hmap[j] = (hoff, hlen)
    rm.close()
    assert len(hmap) == len(order), (len(hmap), len(order))

    dense = {nm: e for nm, e in tensors.items() if ".mlp.experts." not in nm}
    layers = sorted({k[0] for k in rec})
    n_exp = max(k[1] for k in rec) + 1
    cfg = meta["config"]
    keys = ["num_hidden_layers", "first_k_dense_replace", "n_routed_experts",
            "num_experts_per_tok", "hidden_size", "intermediate_size",
            "moe_intermediate_size", "num_attention_heads", "q_lora_rank",
            "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim",
            "v_head_dim", "vocab_size", "rope_theta", "rms_norm_eps",
            "routed_scaling_factor", "norm_topk_prob", "n_shared_experts",
            "index_n_heads", "index_head_dim", "index_topk"]
    pairs = [(k, float(cfg[k]) if not isinstance(cfg[k], bool)
              else float(int(cfg[k]))) for k in keys if k in cfg]
    kmap = {"f32": 0, "i4gs64": 1, "i8gs64": 2}
    idx = io.open(dst + ".idx", "wb")
    idx.write(b"NFIX"); idx.write(struct.pack("<I", 4))
    idx.write(struct.pack("<I", len(pairs)))
    for k, v in pairs:
        idx.write(struct.pack("<32sd", k.encode(), v))
    idx.write(struct.pack("<I", len(dense)))
    for name in sorted(dense):
        e = tensors[name]
        O, I = (e["shape"] + [1])[:2]
        idx.write(struct.pack("<96sIIIQ", name.encode(), kmap[e["kind"]],
                              O, I, man["dense"][name]))
    idx.write(struct.pack("<II", len(layers), n_exp))
    for L in layers:
        for mm in ("gate_proj", "up_proj", "down_proj"):
            e0 = tensors[f"model.layers.{L}.mlp.experts.0.{mm}.weight"]
            idx.write(struct.pack("<I", kmap[e0["kind"]]))
    pos_of = {k: j for j, k in enumerate(order)}
    for L in layers:
        for E in range(n_exp):
            j = pos_of[(L, E)]
            r = rec[(L, E)]
            hoff, hlen = hmap[j]
            idx.write(struct.pack("<5Q", hoff, hlen, r["span"],
                                  r["rel_g"], r["rel_u"]))
    idx.close()
    print(f"idx v4 written: {dst}.idx")

if __name__ == "__main__":
    if sys.argv[1] == "--prep":
        prep(sys.argv[2])
    elif sys.argv[1] == "--idx":
        build_idx(sys.argv[2], sys.argv[3])
