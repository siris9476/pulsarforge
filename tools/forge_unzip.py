# forge_unzip.py — ROLLBACK: reconstructs glm52.forge (byte-identical
# to the original) from glm52.forgez + the truncated remainder, with no
# download.
# - remainder: first 18.6GB of the original (header+JSON+low dense) intact
# - dense above the cut: from the .forgez dense block (head, never truncated)
# - experts: frame decompression, written at the ORIGINAL OFFSETS
#   (from the .forgez's verbatim JSON), in ASCENDING order so that the
#   progressive truncation of the .forgez's TAIL frees space while the
#   .forge grows back. The old glm52.forge.idx (never touched) becomes
#   valid again at the end of the run.
import io, json, os, struct, sys, shutil
import zstandard

FORGE = "glm52.forge"
FZ = "glm52.forgez"
MAN = FZ + ".manifest.json"
ORIG_SIZE = 403631504384
SAFETY = 12 * 1024**3
DRY = "--dry" in sys.argv

def tsize(e):
    O, I = (e["shape"] + [1])[:2]
    if e["kind"] == "f32":    return O * I * 4
    if e["kind"] == "i4gs64": return O * (I // 2 + I // 64 * 2)
    return O * (I + I // 64 * 2)

man = json.load(io.open(MAN, encoding="utf-8"))
fz = io.open(FZ, "rb")
assert fz.read(4) == b"NFRZ"
(jn,) = struct.unpack("<Q", fz.read(8))
meta = json.loads(fz.read(jn))
tensors = meta["tensors"]

rec = {}
for name, e in tensors.items():
    if ".mlp.experts." in name:
        p = name.split(".")
        r = rec.setdefault((int(p[2]), int(p[5])), {})
        r[p[6]] = (e["offset"], tsize(e))
for k, r in rec.items():
    d, g, u = r["down_proj"], r["gate_proj"], r["up_proj"]
    r["off"] = d[0]
    r["span"] = u[0] + u[1] - d[0]
order = sorted(rec, key=lambda k: rec[k]["off"])
pos_of = {k: j for j, k in enumerate(order)}

res_size = os.path.getsize(FORGE)
free = shutil.disk_usage(os.path.dirname(FORGE)).free
todo = [k for k in order if rec[k]["off"] >= res_size]
print(f"remainder {res_size/1e9:.1f}GB, forgez {os.path.getsize(FZ)/1e9:.1f}GB, "
      f"free {free/1e9:.1f}GB, records to restore {len(todo)}")
if DRY:
    sys.exit(0)

dctx = zstandard.ZstdDecompressor()
out = io.open(FORGE, "r+b")

# 1) experts, ascending original offset; truncate .forgez in chunks
written = 0
z_trunc = os.path.getsize(FZ)
for n, k in enumerate(todo):
    j = pos_of[k]
    zoff, zlen = man["frames"][str(j)]
    fz.seek(zoff)
    raw = dctx.decompress(fz.read(zlen), max_output_size=rec[k]["span"])
    assert len(raw) == rec[k]["span"], (k, len(raw))
    out.seek(rec[k]["off"])
    out.write(raw)
    written += len(raw)
    # frames in ascending original offset sit at the TAIL of the
    # forgez: everything past zoff has already been consumed ->
    # truncatable once the free-space margin drops below SAFETY.
    if shutil.disk_usage(os.path.dirname(FORGE)).free < SAFETY:
        out.flush(); os.fsync(out.fileno())
        fz.close()
        t = io.open(FZ, "r+b"); t.truncate(zoff); t.close()
        z_trunc = zoff
        fz = io.open(FZ, "rb")
        print(f"  [{n+1}/{len(todo)}] forgez truncated to {zoff/1e9:.1f}GB, "
              f"free {shutil.disk_usage(os.path.dirname(FORGE)).free/1e9:.1f}GB",
              flush=True)
    if (n + 1) % 2000 == 0:
        print(f"  {n+1}/{len(todo)} records, {written/1e9:.1f}GB", flush=True)

# 2) dense above the remainder: from the forgez dense block (intact head)
nd = 0
for name, e in tensors.items():
    if ".mlp.experts." in name:
        continue
    off, size = e["offset"], tsize(e)
    if off + size <= res_size:
        continue   # already in the original remainder
    fz.seek(man["dense"][name])
    out.seek(off)
    remaining = size
    while remaining:
        b = fz.read(min(remaining, 64 << 20))
        out.write(b)
        remaining -= len(b)
    nd += 1
print(f"dense tensors restored above the cut: {nd}")

if os.path.getsize(FORGE) < ORIG_SIZE:
    out.seek(ORIG_SIZE - 1); out.write(b"\0")   # original padding tail
out.flush(); os.fsync(out.fileno()); out.close()
fz.close()
print(f"reconstructed: {os.path.getsize(FORGE)} bytes (expected {ORIG_SIZE})")
print("recommended final check: run a 24-token chat (reference text)"
      " and THEN manually delete the leftover glm52.forgez + manifest.")
