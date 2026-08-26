# forge_kern_gate.py — generates the binary files for `nf debugforgekern`:
# quantized rows (i4gs64 and i8gs64) + x + reference y (python dequant
# -> f32 dot) + row-0 dequant for the bit-exact check.
import struct, sys
import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from forge_quant import quant_gs64, pack_nibbles, dequant_gs64, GS

def emit(path, kind4, O=64, I=1024, seed=7):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((O, I)).astype(np.float32) * 0.05
    if kind4:
        q, scale = quant_gs64(w)
        qrow = pack_nibbles(q).reshape(O, -1)
    else:
        g = w.reshape(O, I // GS, GS)
        scale = (np.abs(g).max(axis=2) / 127.0).astype(np.float16)
        sf = scale.astype(np.float32); sf[sf == 0] = 1.0
        q = np.clip(np.rint(g / sf[:, :, None]), -128, 127).astype(np.int8)
        qrow = q.reshape(O, I).view(np.uint8)
    rows = np.concatenate([qrow, scale.view(np.uint16).view(np.uint8).reshape(O, -1)], axis=1)
    # reference: EXACT dequant (f16->f32 then int*scale, same arithmetic
    # as the C side) and dot in f64 for the matvec comparison
    wq = (q.reshape(O, I // GS, GS).astype(np.float32)
          * scale.astype(np.float32)[:, :, None]).reshape(O, I)
    x = rng.standard_normal(I).astype(np.float32)
    y = (wq.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("<3i", O, I, 1 if kind4 else 0))
        f.write(rows.tobytes())
        f.write(x.tobytes())
        f.write(y.tobytes())
        f.write(wq[0].tobytes())
    print(path, "written:", rows.shape, "y[0..2] =", y[:3])

if __name__ == "__main__":
    emit("tests/forge_kern_i4.bin", True)
    emit("tests/forge_kern_i8.bin", False)
