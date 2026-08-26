"""Builds the per-layer steering vector: mean direction
(convergent − orbiting) of the Q4_K_M hidden states, normalized
per layer. Output: steer.bin (n_layers × E float32)."""
import numpy as np

L, E = 28, 1024
N_BAD, N_GOOD = 348, 380
SKIP = 40   # skip the prompt (~30 tok) + transient

out = np.zeros((L, E), dtype=np.float32)
for l in range(L):
    bad = np.fromfile(f"tests/dumps/steer/bad_l{l:02d}.bin",
                      dtype=np.float32).reshape(N_BAD, E)[SKIP:]
    good = np.fromfile(f"tests/dumps/steer/good_l{l:02d}.bin",
                       dtype=np.float32).reshape(N_GOOD, E)[SKIP:]
    d = good.mean(axis=0) - bad.mean(axis=0)
    out[l] = d          # RAW: the norm grows with layer like ||x||,
                        # so a single alpha scales sensibly everywhere
    if l in (0, 7, 14, 21, 27):
        xn = np.linalg.norm(bad, axis=1).mean()
        print(f"layer {l:2d}: ||dir||={np.linalg.norm(d):.3f}  ||x||_mean={xn:.1f}")

out.tofile("tests/dumps/steer/steer.bin")
print(f"wrote steer.bin ({L}x{E} f32, raw per-layer differences)")
