# forge_verify.py — integrity check for the .forge container.
# Samples the tensors (all of them if --full): a healthy quantized
# tensor has f16 scales at the end of each row that are NOT all zero;
# a healthy f32 tensor isn't all zero. A "hole" from lost pages (abrupt
# power-off on a sparse file) shows up as an entirely zeroed region.
# Usage: python tools/forge_verify.py [--full] [--sample N]
import argparse, json, struct, sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="glm52.forge")
    ap.add_argument("--sample", type=int, default=800)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    f = open(a.path, "rb")
    assert f.read(4) == b"NFRG", "wrong magic"
    n = struct.unpack("<Q", f.read(8))[0]
    meta = json.loads(f.read(n))
    tensors = meta["tensors"]
    names = sorted(tensors)
    if not a.full:
        step = max(1, len(names) // a.sample)
        names = names[::step]
    bad = []
    by_shard_bad = {}
    for i, name in enumerate(names):
        e = tensors[name]
        O, I = (e["shape"] + [1])[:2]
        # read a chunk from the start, middle, and end of the tensor
        if e["kind"] == "f32":
            size = O * I * 4
        elif e["kind"] == "i4gs64":
            size = O * (I // 2 + I // 64 * 2)
        else:
            size = O * (I + I // 64 * 2)
        ok = True
        for off in (0, size // 2, max(0, size - 65536)):
            f.seek(e["offset"] + off)
            chunk = f.read(min(65536, size - off))
            if chunk and not any(chunk):
                ok = False
                break
        if not ok:
            bad.append(name)
            by_shard_bad[e["shard"]] = by_shard_bad.get(e["shard"], 0) + 1
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(names)} checked, {len(bad)} zeroed")
    print(f"CHECKED {len(names)} tensors: {len(bad)} with ALL-ZERO regions")
    if by_shard_bad:
        print("affected shards (reconvert by removing them from the manifest):")
        for sh in sorted(by_shard_bad):
            print(f"  {sh}: {by_shard_bad[sh]} zeroed tensors")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
