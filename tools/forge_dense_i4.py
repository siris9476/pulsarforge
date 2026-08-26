# forge_dense_i4.py — container v2 IN-PLACE: dense i8gs64 -> i4gs64.
#
# Why (A/B/A test): dense tensors at int8 cost ~17GB of RAM residency
# and squeeze the expert cache down to 398 slots, BELOW a token's
# working set (the "cycling cache" pathology from earlier in the
# project). At int4, residency drops to ~8.5GB and the cache can rise
# to ~750 slots with NF_GLM_STREAM_GB=15. Double quantization
# (fp8->i8->i4): the i8 step is nearly lossless, the extra noise is
# negligible compared to going straight to i4 — the quality bench
# remains the judge.
#
# In-place: i4 is SHORTER than i8 -> it's rewritten at the SAME
# offset (the tail becomes dead space on disk, irrelevant); the "kind"
# in the header's JSON is patched byte-for-byte ("i8gs64" and "i4gs64"
# have the same length). Then regenerate the idx with forge_idx.py.
# RECOVERY in case quality is rejected: re-download just the dense
# tensors via range-request from the FP8 source (~19GB) and reconvert
# them to i8.
import json, struct, sys, time
import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from forge_quant import quant_gs64, pack_nibbles, GS

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "glm52.forge"
    f = open(path, "r+b")
    assert f.read(4) == b"NFRG"
    jlen = struct.unpack("<Q", f.read(8))[0]
    jpos = 12
    raw = bytearray(f.read(jlen))
    meta = json.loads(raw.decode())
    targets = [(n, e) for n, e in meta["tensors"].items()
               if e["kind"] == "i8gs64"]
    print(f"i8gs64 dense tensors to convert: {len(targets)}")
    t0 = time.time()
    done = 0
    for n, e in sorted(targets, key=lambda t: t[1]["offset"]):
        O, I = e["shape"]
        rb8 = I + I // GS * 2
        f.seek(e["offset"])
        buf = np.frombuffer(f.read(O * rb8), dtype=np.uint8).reshape(O, rb8)
        q8 = buf[:, :I].view(np.int8).astype(np.float32)
        sc = buf[:, I:].copy().view(np.uint16).view(np.float16).astype(np.float32)
        w = (q8.reshape(O, I // GS, GS) * sc[:, :, None]).reshape(O, I)
        q4, s4 = quant_gs64(w)
        out = np.concatenate(
            [pack_nibbles(q4).reshape(O, -1),
             s4.view(np.uint16).view(np.uint8).reshape(O, -1)], axis=1)
        f.seek(e["offset"]); f.write(out.tobytes())
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(targets)} ({time.time()-t0:.0f}s)")
    # patch the "kind" in the JSON byte-for-byte (same length)
    txt = raw.decode()
    for n, e in targets:
        key = json.dumps(n) + ': {"offset": ' + str(e["offset"])
        i = txt.find(key)
        assert i >= 0, n
        j = txt.find('"i8gs64"', i)
        assert 0 <= j < i + 400, n
        txt = txt[:j] + '"i4gs64"' + txt[j + 8:]
    assert len(txt.encode()) == jlen
    f.seek(jpos); f.write(txt.encode())
    f.flush()
    import os
    os.fsync(f.fileno())
    f.close()
    print(f"DONE: {done} tensors in {time.time()-t0:.0f}s — regenerate the idx")

if __name__ == "__main__":
    main()
