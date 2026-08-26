# forge_convert.py — streaming FP8 -> .forge v1 container converter.
#
# Usage:
#   python tools/forge_convert.py --plan-only          # exact size, zero download
#   python tools/forge_convert.py --dest glm52.forge [--tmp C:\...\tmp]
#
# Design (product build-out, point 0 + space decision):
# - DETERMINISTIC plan: every tensor's offsets are computed ahead of
#   time from the index + config; shards can arrive/write in any order,
#   resuming is just a list of completed shards (manifest json).
# - SPARSE file on NTFS: physical allocation follows the bytes written
#   (disk margin is tight: 404GB free, container estimated at ~405GB).
# - shard temp file on C: (default), deleted as soon as it's converted.
# - quant: experts int4-gs64, dense/heads int8-gs64 (same group
#   structure, shareable kernel), norms/bias/router raw f32.
# - per-tensor validation in the log: relative error + cosine.
import argparse, json, os, struct, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forge_quant import dequant_fp8, quant_gs64, pack_nibbles, GS

REPO = "zai-org/GLM-5.2-FP8"
MAGIC = b"NFRG"
VERSION = 1
ALIGN = 1024

# tensors that stay raw f32 (small, or where fidelity is everything:
# router and correction bias = the measured near-ties)
def keep_f32(name):
    return (name.endswith("norm.weight") or name.endswith("norm.bias")
            or ".layernorm" in name or name.endswith("_layernorm.weight")
            or name.endswith(".bias")
            or ".mlp.gate.weight" in name          # router
            or "e_score_correction_bias" in name)

def is_expert(name):
    return ".mlp.experts." in name

# v1: the MTP layer (78) stays OUT of the container — on this class of
# machine MTP is a regression we measured (1.75x) and it's also
# auto-disabled by colibrì (doctor): ~5.4GB saved. A v2 container when
# a RAM budget justifies it.
def skip_tensor(name):
    return name.startswith("model.layers.78.")

# embeddings and head at int4-gs64 (the everyday GGUF already keeps
# them at ~4 bit and QA is clean on that front): -0.95GB
def force_i4(name):
    return name in ("model.embed_tokens.weight", "lm_head.weight")

def gs64_bytes(shape, bits):
    O, I = shape
    assert I % GS == 0, (shape,)
    per_row = I * bits // 8 + (I // GS) * 2       # nibble/int8 + f16 scale
    return O * per_row

def f32_bytes(shape):
    n = 1
    for d in shape: n *= d
    return n * 4

def aligned(x):
    return (x + ALIGN - 1) // ALIGN * ALIGN

def fetch_all_headers(tmp, use_cache=True):
    """Downloads (or reuses) ONLY the safetensors headers of the 141
    shards via HTTP range (~8MB each max, headers much smaller): the
    EXACT shape of every tensor comes from here. Disk cache in
    tmp/headers/."""
    import urllib.request
    hdrdir = os.path.join(tmp, "forge_headers")
    os.makedirs(hdrdir, exist_ok=True)
    api = json.load(open(os.path.join(hdrdir, "index.json")) ) if (
        use_cache and os.path.exists(os.path.join(hdrdir, "index.json"))) else None
    if api is None:
        with urllib.request.urlopen(
                f"https://huggingface.co/{REPO}/resolve/main/model.safetensors.index.json",
                timeout=120) as r:
            api = json.load(r)
        json.dump(api, open(os.path.join(hdrdir, "index.json"), "w"))
    shards = sorted(set(api["weight_map"].values()))
    headers = {}
    for sh in shards:
        cache = os.path.join(hdrdir, sh + ".hdr.json")
        if use_cache and os.path.exists(cache):
            headers[sh] = json.load(open(cache))
            continue
        url = f"https://huggingface.co/{REPO}/resolve/main/{sh}"
        req = urllib.request.Request(url, headers={"Range": "bytes=0-131071"})
        with urllib.request.urlopen(req, timeout=120) as r:
            b = r.read()
        n = struct.unpack("<Q", b[:8])[0]
        if 8 + n > len(b):
            req = urllib.request.Request(url, headers={"Range": f"bytes=0-{8+n-1}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                b = r.read()
        h = json.loads(b[8:8 + n])
        h.pop("__metadata__", None)
        json.dump(h, open(cache, "w"))
        headers[sh] = h
        print(f"  header {sh}: {len(h)} tensors")
    return api, headers

def build_plan_from_headers(api, headers, cfg):
    wm = api["weight_map"]
    entries = []   # (name, shape, dtype_kind, shard, size)
    for sh, h in headers.items():
        for name, meta in h.items():
            if name.endswith("_scale_inv") or skip_tensor(name):
                continue
            shape = meta["shape"]
            if keep_f32(name) or len(shape) == 1:
                kind, size = "f32", f32_bytes(shape)
            elif is_expert(name) or force_i4(name):
                kind, size = "i4gs64", gs64_bytes(shape, 4)
            else:
                kind, size = "i8gs64", gs64_bytes(shape, 8)
            entries.append((name, shape, kind, sh, size))
    entries.sort(key=lambda e: (is_expert(e[0]), e[0]))   # dense first, then experts
    # pass 1: provisional directory json to estimate its size
    def directory(off0):
        off = off0
        d = {}
        for name, shape, kind, sh, size in entries:
            off = aligned(off)
            d[name] = {"offset": off, "shape": shape, "kind": kind, "shard": sh}
            off += size
        return d, off
    probe, _ = directory(0)
    meta = {"version": VERSION, "config": cfg, "tensors": probe}
    jlen = len(json.dumps(meta).encode())
    hdr = aligned(8 + 8 + jlen + 65536)     # magic+len+json with a safety margin
    d, total = directory(hdr)
    meta = {"version": VERSION, "config": cfg, "tensors": d, "header_bytes": hdr}
    return meta, total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="glm52.forge")
    ap.add_argument("--tmp", default=os.path.join(os.environ.get("LOCALAPPDATA", "/tmp"), "forge_tmp"))
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--limit-shards", type=int, default=0, help="for testing: convert only N shards")
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)

    import urllib.request
    cfg_path = os.path.join(a.tmp, "config.json")
    if not os.path.exists(cfg_path):
        urllib.request.urlretrieve(
            f"https://huggingface.co/{REPO}/resolve/main/config.json", cfg_path)
    cfg = json.load(open(cfg_path))

    print("plan: downloading/reusing headers for the 141 shards (range-request only)...")
    api, headers = fetch_all_headers(a.tmp)
    meta, total = build_plan_from_headers(api, headers, cfg)
    n_exp = sum(1 for n in meta["tensors"] if is_expert(n))
    n_dense = len(meta["tensors"]) - n_exp
    print(f"PLAN: {len(meta['tensors'])} tensors ({n_exp} expert, {n_dense} dense/other)")
    print(f"CONTAINER SIZE: {total/1e9:.2f} GB  (ceiling: <= 404 GB)")
    if a.plan_only:
        return

    if total / 1e9 > 404:
        raise SystemExit("the plan exceeds the 404GB free: stop and reconsider")

    manifest_path = a.dest + ".manifest.json"
    done = set()
    if os.path.exists(manifest_path):
        done = set(json.load(open(manifest_path))["shards_done"])
        print(f"resuming: {len(done)} shards already converted")
    # create the sparse file + write header/json once
    if not os.path.exists(a.dest):
        with open(a.dest, "wb") as f:
            f.write(MAGIC)
            jb = json.dumps(meta).encode()
            f.write(struct.pack("<Q", len(jb)))
            f.write(jb)
        os.system(f'fsutil sparse setflag "{a.dest}" >nul 2>&1')
    from huggingface_hub import hf_hub_download
    shards = sorted(set(api["weight_map"].values()))
    if a.limit_shards: shards = shards[:a.limit_shards]
    t0 = time.time()
    for i, sh in enumerate(shards):
        if sh in done: continue
        t1 = time.time()
        p = hf_hub_download(REPO, sh, local_dir=a.tmp)
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
            h.pop("__metadata__", None)
            base = 8 + n
            out = open(a.dest, "r+b")
            wrote = 0
            for name, m2 in h.items():
                if name.endswith("_scale_inv") or name not in meta["tensors"]:
                    continue
                e = meta["tensors"][name]
                o0, o1 = m2["data_offsets"]
                f.seek(base + o0); raw = f.read(o1 - o0)
                # UNIFORM load to f32 based on source dtype (bug found by
                # the 1-shard test run: lm_head/embed are BF16, no FP8
                # scale — the quantizer assumed FP8 everywhere)
                if m2["dtype"] == "F8_E4M3":
                    sk = name + "_scale_inv"
                    so = h[sk]["data_offsets"]; f.seek(base + so[0])
                    sc = np.frombuffer(f.read(so[1]-so[0]), dtype=np.float32).reshape(h[sk]["shape"])
                    w = dequant_fp8(raw, sc, m2["shape"])
                elif m2["dtype"] == "BF16":
                    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                    w = u.view(np.float32).reshape(m2["shape"])
                elif m2["dtype"] == "F32":
                    w = np.frombuffer(raw, dtype=np.float32).reshape(m2["shape"]).copy()
                elif m2["dtype"] == "F16":
                    w = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(m2["shape"])
                else:
                    raise SystemExit(f"unexpected dtype {m2['dtype']}: {name}")
                if e["kind"] == "f32":
                    buf = w.reshape(-1).astype(np.float32).tobytes()
                else:
                    if e["kind"] == "i4gs64":
                        q, scale = quant_gs64(w)
                        qb = pack_nibbles(q)
                    else:  # i8gs64
                        O, I = w.shape
                        g = w.reshape(O, I // GS, GS)
                        s8 = (np.abs(g).max(axis=2) / 127.0).astype(np.float16)
                        sf = s8.astype(np.float32); sf[sf == 0] = 1.0
                        q = np.clip(np.rint(g / sf[:, :, None]), -128, 127).astype(np.int8)
                        qb, scale = q.tobytes(), s8
                    # per-row layout: [row quant | row scale] — contiguous
                    O, I = m2["shape"]
                    qrow = np.frombuffer(qb, dtype=np.uint8).reshape(O, -1)
                    srow = scale.view(np.uint16).reshape(O, -1)
                    buf = np.concatenate(
                        [qrow, srow.view(np.uint8).reshape(O, -1)], axis=1).tobytes()
                out.seek(e["offset"]); out.write(buf); wrote += len(buf)
            # lesson from the power-off incident: the manifest must NEVER
            # be ahead of the actual data on disk — flush+fsync the shard
            # BEFORE declaring it done.
            out.flush(); os.fsync(out.fileno())
            out.close()
        os.remove(p)
        done.add(sh)
        json.dump({"shards_done": sorted(done)}, open(manifest_path, "w"))
        dt, el = time.time() - t1, time.time() - t0
        print(f"[{len(done)}/{len(shards)}] {sh}: {wrote/1e6:.0f}MB in {dt:.0f}s "
              f"(total {el/60:.1f}min)")
    print("CONVERSION COMPLETE")

if __name__ == "__main__":
    main()
