# forge_zip.py — converts glm52.forge into glm52.forgez (experts
# zstd-9 compressed per record, dense/JSON verbatim) TRUNCATING the
# source in stages: the free space (229GB after the GGUF) isn't enough
# for a full copy (~351GB), so each stage compresses a tail of the
# source, VERIFIES it bit-exact against the original still present,
# writes the manifest, and ONLY THEN truncates. Resuming: relaunch and
# it picks up from the manifest.
#
#   .forgez : [b"NFRZ"][u64 json_len][JSON verbatim from .forge]
#             [dense verbatim, 1024-aligned] [zstd frames, 4096-aligned]
#   .forgez.idx (v3): like v2 but new dense offsets and per expert
#             5 x u64: zoff, zlen, span, rel_gate, rel_up (down rel=0)
#
# Usage: python tools/forge_zip.py [--dry]
import io, json, os, struct, sys, time, shutil
from concurrent.futures import ThreadPoolExecutor
import zstandard

args = [a for a in sys.argv[1:] if not a.startswith("--")]
SRC = args[0] if args else "glm52.forge"
DST = args[1] if len(args) > 1 else SRC + "z"
MAN = DST + ".manifest.json"
LVL = 9
SAFETY = 15 * 1024**3          # free-space margin that must never be invaded
RATIO_EST = 1.10               # CONSERVATIVE estimate for planning stages
DRY = "--dry" in sys.argv
KEEP = "--keep" in sys.argv    # NEVER truncate the source (tiny/test)

def free_bytes():
    return shutil.disk_usage(os.path.dirname(SRC)).free

def al(x, a):
    return (x + a - 1) // a * a

def tsize(e):
    O, I = (e["shape"] + [1])[:2]
    if e["kind"] == "f32":    return O * I * 4
    if e["kind"] == "i4gs64": return O * (I // 2 + I // 64 * 2)
    return O * (I + I // 64 * 2)

def load_meta():
    f = io.open(SRC, "rb")
    assert f.read(4) == b"NFRG"
    n = struct.unpack("<Q", f.read(8))[0]
    raw = f.read(n)
    f.close()
    return raw, json.loads(raw)

def manifest_read():
    if os.path.exists(MAN):
        return json.load(io.open(MAN, encoding="utf-8"))
    return None

def manifest_write(m):
    tmp = MAN + ".tmp"
    io.open(tmp, "w", encoding="utf-8").write(json.dumps(m))
    os.replace(tmp, MAN)

def main():
    raw_json, meta = load_meta()
    tensors = meta["tensors"]

    dense = {}   # name -> (kind,O,I,off_orig,size)
    rec = {}     # (L,E) -> dict(down,gate,up,span,rel_g,rel_u)
    for name, e in tensors.items():
        if ".mlp.experts." in name:
            p = name.split(".")
            L, E, mm = int(p[2]), int(p[5]), p[6]
            r = rec.setdefault((L, E), {})
            r[mm] = (e["offset"], tsize(e))
        else:
            dense[name] = (e["kind"], e["offset"], tsize(e))
    for k, r in rec.items():
        d, g, u = r["down_proj"], r["gate_proj"], r["up_proj"]
        assert d[0] < g[0] < u[0], k
        r["off"] = d[0]
        r["span"] = u[0] + u[1] - d[0]
        r["rel_g"] = g[0] - d[0]
        r["rel_u"] = u[0] - d[0]
    order = sorted(rec, key=lambda k: rec[k]["off"])   # ascending offset
    n_rec = len(order)
    tot_span = sum(rec[k]["span"] for k in order)
    print(f"expert records: {n_rec}, total span {tot_span/1e9:.1f}GB, "
          f"dense {sum(d[2] for d in dense.values())/1e9:.1f}GB, "
          f"free {free_bytes()/1e9:.1f}GB")
    if DRY:
        return

    man = manifest_read()
    src_size = os.path.getsize(SRC)
    if man is None:
        # --- .forgez head: header + JSON + dense verbatim ---------------
        out = io.open(DST, "wb")
        out.write(b"NFRZ")
        out.write(struct.pack("<Q", len(raw_json)))
        out.write(raw_json)
        src = io.open(SRC, "rb")
        new_dense = {}
        for name in sorted(dense):
            kind, off, size = dense[name]
            pos = al(out.tell(), 1024)
            out.write(b"\0" * (pos - out.tell()))
            src.seek(off)
            remaining = size
            while remaining:
                b = src.read(min(remaining, 64 << 20))
                out.write(b)
                remaining -= len(b)
            new_dense[name] = pos
        src.close()
        pos = al(out.tell(), 4096)
        out.write(b"\0" * (pos - out.tell()))
        out.flush(); os.fsync(out.fileno()); out.close()
        man = {"dense": new_dense, "frames": {}, "src_trunc": src_size,
               "z_end": os.path.getsize(DST),
               "done_upto": n_rec}   # done_upto: index in `order` NOT
                                     # yet converted (from the tail)
        manifest_write(man)
        print(f"head written: dense copied, .forgez={os.path.getsize(DST)/1e9:.1f}GB")

    # NB: ZstdCompressor is NOT thread-safe on the same instance
    # (silent native crash, seen firsthand): a NEW compressor per call
    # — the overhead is nothing compared to the ~300ms of compression.
    def compress_one(raw):
        return zstandard.ZstdCompressor(level=LVL).compress(raw)
    dctx = zstandard.ZstdDecompressor()

    while man["done_upto"] > 0:
        # --- plan the stage against the REAL free space ------------------
        budget = free_bytes() - SAFETY
        if budget <= 0:
            raise SystemExit("not enough space for a stage: clean abort")
        take, spend = [], 0
        i = man["done_upto"] - 1
        while i >= 0:
            need = al(int(rec[order[i]]["span"] / RATIO_EST), 4096)
            if spend + need > budget:
                break
            take.append(i); spend += need; i -= 1
        if not take:
            raise SystemExit("no record fits in the budget: clean abort")
        lo = take[-1]
        cut_off = rec[order[lo]]["off"]   # new truncation boundary
        print(f"stage: records {lo}..{man['done_upto']-1} "
              f"({sum(rec[order[j]]['span'] for j in take)/1e9:.1f}GB source, "
              f"budget {budget/1e9:.1f}GB)")

        # --- compress (thread pool: zstd releases the GIL) ---------------
        # resuming after a mid-stage crash: frames past z_end are
        # orphans not registered in the manifest — discard, start clean.
        src = io.open(SRC, "rb")
        out = io.open(DST, "r+b")
        out.truncate(man["z_end"])
        out.seek(0, 2)
        t0 = time.time()
        def read_rec(j):
            r = rec[order[j]]
            src.seek(r["off"])
            return src.read(r["span"])
        with ThreadPoolExecutor(max_workers=4) as ex:
            batch = 32   # 32 x ~20MB raw + blob ~= 1.2GB peak RAM
            done_ct = 0
            for b0 in range(man["done_upto"] - 1, lo - 1, -batch):
                js = list(range(b0, max(b0 - batch, lo - 1), -1))
                raws = [read_rec(j) for j in js]
                blobs = list(ex.map(compress_one, raws))
                for j, blob, rw in zip(js, blobs, raws):
                    # IMMEDIATE check: the decompressed frame must be
                    # bit-identical to the source just read
                    if dctx.decompress(blob, max_output_size=len(rw)) != rw:
                        raise SystemExit(f"CHECK FAILED on record {j}: "
                                         "no truncation, source intact")
                    pos = al(out.tell(), 4096)
                    out.write(b"\0" * (pos - out.tell()))
                    out.write(blob)
                    man["frames"][str(j)] = [pos, len(blob)]
                done_ct += len(js)
                if done_ct % 1024 < batch:
                    el = time.time() - t0
                    print(f"  {done_ct}/{len(take)} records, "
                          f"{out.tell()/1e9:.1f}GB out, {el/60:.1f}min", flush=True)
        out.flush(); os.fsync(out.fileno()); out.close()
        src.close()

        # --- second check FROM THE FILE (what's actually on disk) -------
        print("  checking from disk...", flush=True)
        src = io.open(SRC, "rb")
        z = io.open(DST, "rb")
        for j in take:
            r = rec[order[j]]
            zoff, zlen = man["frames"][str(j)]
            z.seek(zoff)
            blob = z.read(zlen)
            src.seek(r["off"])
            if dctx.decompress(blob, max_output_size=r["span"]) != src.read(r["span"]):
                raise SystemExit(f"DISK CHECK FAILED on record {j}: "
                                 "no truncation, source intact")
        src.close(); z.close()

        # --- manifest BEFORE truncation, then truncate --------------------
        man["done_upto"] = lo
        man["src_trunc"] = cut_off
        man["z_end"] = os.path.getsize(DST)
        manifest_write(man)
        if KEEP:
            print("  --keep: source NOT truncated", flush=True)
        else:
            f = io.open(SRC, "r+b")
            f.truncate(cut_off)
            f.close()
            print(f"  truncated the source to {cut_off/1e9:.1f}GB, "
                  f"free {free_bytes()/1e9:.1f}GB", flush=True)

    # --- idx v3 ---------------------------------------------------------
    layers = sorted({k[0] for k in rec})
    n_exp = max(k[1] for k in rec) + 1
    idx = io.open(DST + ".idx", "wb")
    idx.write(b"NFIX"); idx.write(struct.pack("<I", 3))
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
    idx.write(struct.pack("<I", len(pairs)))
    for k, v in pairs:
        idx.write(struct.pack("<32sd", k.encode(), v))
    kmap = {"f32": 0, "i4gs64": 1, "i8gs64": 2}
    dn = {n: (kmap[dense[n][0]],) for n in dense}
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
            zoff, zlen = man["frames"][str(j)]
            idx.write(struct.pack("<5Q", zoff, zlen, r["span"],
                                  r["rel_g"], r["rel_u"]))
    idx.close()
    print(f"idx v3 written. .forgez={os.path.getsize(DST)/1e9:.1f}GB, "
          f"remaining source={os.path.getsize(SRC)/1e9:.1f}GB")
    print("NOTE: the leftover .forge must be deleted BY HAND after the "
          "engine's own gates pass (this script won't touch it).")

if __name__ == "__main__":
    main()
