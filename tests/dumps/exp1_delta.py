"""Experiment 1 (falsification): how 'compressible' is the delta of
the residual stream between adjacent positions?

Theory to falsify (inference as a video codec): if
Δx = x_t − x_{t-1} has its energy concentrated in a few coordinates,
W·x_t = y_{t-1} + W·Δx_truncated can be computed paying only a
fraction of the matvec. Verdict positive if the top-10% of coordinates
carries >~90% of Δx's energy; negative if the energy is spread out.
"""
import glob
import numpy as np

N_TOK, E = 72, 2560

print(f"{'layer':>5} {'cos(x_t,x_t-1)':>15} {'top10% energy':>15} {'top25% energy':>15}")
for path in sorted(glob.glob("tests/dumps/x4b_l*.bin")):
    l = int(path.split("_l")[1][:2])
    X = np.fromfile(path, dtype=np.float32).reshape(N_TOK, E).astype(np.float64)
    # delta between adjacent positions (skip the first 8: prompt transient)
    D = X[9:] - X[8:-1]
    cos = np.mean([np.dot(X[t], X[t-1]) / (np.linalg.norm(X[t]) * np.linalg.norm(X[t-1]) + 1e-12)
                   for t in range(9, N_TOK)])
    e = D * D
    e_sorted = np.sort(e, axis=1)[:, ::-1]
    tot = e_sorted.sum(axis=1) + 1e-12
    top10 = e_sorted[:, :E // 10].sum(axis=1) / tot
    top25 = e_sorted[:, :E // 4].sum(axis=1) / tot
    if l in (0, 5, 11, 17, 23, 29, 35):
        print(f"{l:>5} {cos:>15.4f} {np.mean(top10):>14.1%} {np.mean(top25):>14.1%}")
