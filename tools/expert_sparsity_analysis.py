# -*- coding: utf-8 -*-
"""Phase 0 — intra-expert sparsity measurement on GLM-5.2 quantized to
~2.3 bits.

The first-ever (as far as we know) measurement of how much of the
neuron sparsity INSIDE routed experts (arXiv 2605.08575: up to 90% of
neurons disableable at 95% accuracy on a 400B bf16 MoE) survives the
IQ1_S/IQ2_XXS/IQ3_XXS/IQ4_XS quantization of a model streamed from
disk. Offline measurement only: NO runtime mechanism.

Input:
  - tools/real_xb_trace.bin  (NF_GLM_XB_TRACE: for each MoE layer and
    decode position, the expert input xb post-ffn_norm + final midx[8]
    + mw[8])
  - the real GGUF (expert matrices read/dequantized via
    `nf debugdequant5`, C kernels bit-exact vs gguf-py)

For each (sampled layer, position, selected expert):
  g = silu(G @ xb), u = U @ xb, h = g*u, y_full = D @ h
  (G,U: [FF=2048 rows x E=6144 cols]; D: [E rows x FF cols] — matvec
   semantics: y[r] = sum_c W[r][c]*x[c], verified against
   expert_slice/matvec)
  criterion |g| (implementable gate-first) and |g*u| (oracle, upper bound):
  keep the top-p rows of h, zero the rest, y_sparse = D @ h_masked.
Metrics: per-expert cosine and relative L2, and on the mw-weighted
MIXTURE (the one that matters for quality). Byte economics with the
REAL quant types.
"""
import json
import os
import struct
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUF = os.path.join(ROOT, "models", "glm52-merged.gguf")
TRACE = os.path.join(ROOT, "tools", "real_xb_trace.bin")
NF = os.path.join(ROOT, "nf.exe")
TMPPREFIX = os.path.join(ROOT, "tools", "_spars_tmp")
OUT_JSON = os.path.join(ROOT, "tools", "expert_sparsity_results.json")

FF = 2048
MAGIC = 0x52544258
LAYERS = [3, 10, 20, 30, 40, 50, 60, 70, 77]
PS = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
CRITS = ["g", "gu"]          # |g| = implementable gate-first; |g*u| = oracle
ADAPTIVE = (0.5, 0.15, 2)    # top-2 experts by mw -> p=0.5, the rest p=0.15
SB_BYTES = {10: 84, 11: 110, 16: 66, 18: 98, 19: 50, 23: 136}
TYPE_NAME = {10: "Q2_K", 11: "Q3_K", 16: "IQ2_XXS", 18: "IQ3_XXS",
             19: "IQ1_S", 23: "IQ4_XS"}


def read_trace(path):
    with open(path, "rb") as f:
        buf = f.read()
    magic, E, K = struct.unpack_from("<3i", buf, 0)
    assert magic == MAGIC, "wrong xb trace magic"
    off = 12
    recsz = 8 + 4 * E + 8 * K
    recs = {}
    while off + recsz <= len(buf):
        layer, pos = struct.unpack_from("<2i", buf, off)
        off += 8
        xb = np.frombuffer(buf, np.float32, E, off).copy()
        off += 4 * E
        midx = np.frombuffer(buf, np.int32, K, off).copy()
        off += 4 * K
        mw = np.frombuffer(buf, np.float32, K, off).copy()
        off += 4 * K
        recs[(layer, pos)] = (xb, midx, mw)
    return E, K, recs


def dequant(tensor, row0, nrows, cols):
    r = subprocess.run([NF, "debugdequant5", GGUF, tensor,
                        str(row0), str(nrows), TMPPREFIX],
                       capture_output=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError("debugdequant5 failed on %s [%d,%d): %s"
                           % (tensor, row0, row0 + nrows,
                              r.stderr.decode(errors="replace")[-500:]))
    x = np.fromfile(TMPPREFIX + "_x.f32", np.float32)
    for suf in ("_x.f32", "_q.bin"):
        try:
            os.remove(TMPPREFIX + suf)
        except OSError:
            pass
    assert x.size == nrows * cols
    return x.reshape(nrows, cols)


def silu(x):
    return x / (1.0 + np.exp(-x))


def cos_rel(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    c = float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0
    rl2 = float(np.linalg.norm(a - b) / na) if na > 0 else 0.0
    return c, rl2


def layer_types_and_bytes(reader, E):
    """Real quant types and byte sizes for (gate,up,down) of every MoE
    layer, from the GGUF metadata (no data reads). Returns {layer: dict}."""
    info = {}
    for t in reader.tensors:
        name = t.name
        if not name.startswith("blk."):
            continue
        parts = name.split(".")
        l = int(parts[1])
        kind = parts[2]
        if kind not in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            continue
        tt = int(t.tensor_type)
        d = info.setdefault(l, {})
        d[kind] = {"type": tt, "bytes": int(t.n_bytes)}
    return info


def main():
    print("== reading xb trace ==")
    E, K, recs = read_trace(TRACE)
    layers_in_trace = sorted({l for (l, _) in recs})
    positions = sorted({p for (_, p) in recs})
    xs = np.stack([v[0] for v in recs.values()])
    print("E=%d K=%d records=%d MoE_layers=%d..%d positions=%s" %
          (E, K, len(recs), layers_in_trace[0], layers_in_trace[-1],
           positions))
    assert np.all(np.isfinite(xs)), "xb with NaN/inf in the trace"
    print("xb sanity: mean |xb|=%.4f min per-record std=%.4f" %
          (float(np.mean(np.linalg.norm(xs, axis=1))),
           float(np.min(np.std(xs, axis=1)))))

    # sampled positions: 4 spread out along the trace
    npos = len(positions)
    pos_sample = sorted({positions[0], positions[npos // 3],
                         positions[2 * npos // 3], positions[-1]})
    print("sampled positions:", pos_sample)

    # GGUF metadata: types/bytes per layer + router validation
    print("== GGUF metadata (gguf-py, header only) ==")
    import gguf
    reader = gguf.GGUFReader(GGUF)
    tinfo = layer_types_and_bytes(reader, E)

    # --- trace validation against the router (a check in its own right,
    # but valuable: confirms xb REALLY is the MoE input and that the
    # matrix/vector orientation assumed here matches the engine's) ---
    print("== router validation (2 samples) ==")
    val_layers = [LAYERS[0], LAYERS[-1]]
    for vl in val_layers:
        xb, midx, mw = recs[(vl, pos_sample[0])]
        gi = None
        pb = None
        for t in reader.tensors:
            if t.name == "blk.%d.ffn_gate_inp.weight" % vl:
                gi = np.array(t.data, np.float32).reshape(-1, E)
            if t.name == "blk.%d.exp_probs_b.bias" % vl:
                pb = np.array(t.data, np.float32).ravel()
        assert gi is not None and pb is not None
        logits = gi @ xb
        sig = 1.0 / (1.0 + np.exp(-logits))
        choice = sig + pb
        top8 = set(np.argsort(-choice)[:K].tolist())
        assert top8 == set(midx.tolist()), \
            "router mismatch layer %d: %s vs %s" % (vl, top8, set(midx))
        # mw must be proportional to sig[midx] (renorm x scale)
        ratio = mw / sig[midx]
        assert np.max(np.abs(ratio / ratio[0] - 1.0)) < 1e-3
        print("layer %d: recomputed top-8 router == traced midx, "
              "mw proportional to sigmoid (scale %.4f) OK" %
              (vl, float(ratio[0])))

    # --- main analysis ---
    results = []       # per-expert
    mixtures = []      # per (layer,pos)
    for L in LAYERS:
        types = {k: tinfo[L][k]["type"] for k in
                 ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")}
        print("== layer %d (gate=%s up=%s down=%s) ==" %
              (L, TYPE_NAME[types["ffn_gate_exps"]],
               TYPE_NAME[types["ffn_up_exps"]],
               TYPE_NAME[types["ffn_down_exps"]]))
        samples = {p: recs[(L, p)] for p in pos_sample}
        occs = {}
        for p, (xb, midx, mw) in samples.items():
            for k in range(K):
                occs.setdefault(int(midx[k]), []).append((p, float(mw[k])))
        print("  unique experts: %d/%d (overlap between positions %.0f%%)" %
              (len(occs), 4 * K, 100.0 * (1 - len(occs) / (4.0 * K))))
        ystore = {}    # (pos, e) -> {"full": y, ("g",p): y, ("gu",p): y}
        for e, occ in sorted(occs.items()):
            G = dequant("blk.%d.ffn_gate_exps.weight" % L, e * FF, FF, E)
            U = dequant("blk.%d.ffn_up_exps.weight" % L, e * FF, FF, E)
            D = dequant("blk.%d.ffn_down_exps.weight" % L, e * E, E, FF)
            for p, w in occ:
                xb = samples[p][0]
                g = silu(G @ xb)
                u = U @ xb
                h = g * u
                y_full = D @ h
                entry = {"full": y_full}
                for crit in CRITS:
                    order = np.argsort(-np.abs(g if crit == "g" else h))
                    for pf in PS:
                        keep = order[:max(1, int(round(pf * FF)))]
                        y_sp = D[:, keep] @ h[keep]
                        entry[(crit, pf)] = y_sp
                        c, rl2 = cos_rel(y_full, y_sp)
                        results.append({"layer": L, "pos": p, "expert": e,
                                        "mw": w, "crit": crit, "p": pf,
                                        "cos": c, "rel_l2": rl2})
                ystore[(p, e)] = entry
            del G, U, D
        # mw-weighted mixture for (layer, pos)
        for p in pos_sample:
            xb, midx, mw = samples[p]
            Yf = np.zeros(E, np.float32)
            for k in range(K):
                Yf += mw[k] * ystore[(p, int(midx[k]))]["full"]
            mrec = {"layer": L, "pos": p, "Yfull_norm":
                    float(np.linalg.norm(Yf))}
            for crit in CRITS:
                for pf in PS:
                    Ys = np.zeros(E, np.float32)
                    for k in range(K):
                        Ys += mw[k] * ystore[(p, int(midx[k]))][(crit, pf)]
                    c, rl2 = cos_rel(Yf, Ys)
                    mrec["%s_p%g_cos" % (crit, pf)] = c
                    mrec["%s_p%g_rl2" % (crit, pf)] = rl2
            # adaptive: top-2 experts by mw -> high p, rest low p (|g|)
            p_hi, p_lo, n_hi = ADAPTIVE
            rank = np.argsort(-mw)
            Ys = np.zeros(E, np.float32)
            for i, k in enumerate(rank):
                pf = p_hi if i < n_hi else p_lo
                Ys += mw[k] * ystore[(p, int(midx[k]))][("g", pf)]
            c, rl2 = cos_rel(Yf, Ys)
            mrec["adaptive_cos"] = c
            mrec["adaptive_rl2"] = rl2
            mixtures.append(mrec)

    # --- byte economics: EXACT over the whole model, from GGUF metadata ---
    tot_gate = sum(v["ffn_gate_exps"]["bytes"] for v in tinfo.values())
    tot_up = sum(v["ffn_up_exps"]["bytes"] for v in tinfo.values())
    tot_down = sum(v["ffn_down_exps"]["bytes"] for v in tinfo.values())
    tot = tot_gate + tot_up + tot_down
    econ = {"tot_gate_gb": tot_gate / 2**30, "tot_up_gb": tot_up / 2**30,
            "tot_down_gb": tot_down / 2**30, "tot_gb": tot / 2**30,
            "frac": {}}
    for pf in PS:
        econ["frac"][pf] = (tot_gate + pf * (tot_up + tot_down)) / tot
    p_hi, p_lo, n_hi = ADAPTIVE
    p_eff = (n_hi * p_hi + (K - n_hi) * p_lo) / K
    econ["frac"]["adaptive(p_eff=%.3f)" % p_eff] = \
        (tot_gate + p_eff * (tot_up + tot_down)) / tot
    tinfo_out = {str(l): {k: {"type": TYPE_NAME.get(v["type"], v["type"]),
                              "bytes": v["bytes"]}
                          for k, v in d.items()}
                 for l, d in tinfo.items()}

    with open(OUT_JSON, "w") as f:
        json.dump({"E": E, "K": K, "pos_sample": pos_sample,
                   "layers": LAYERS, "ps": PS,
                   "per_expert": results, "mixtures": mixtures,
                   "econ": econ, "tinfo": tinfo_out}, f)
    print("results saved to", OUT_JSON)

    # --- summary tables ---
    def agg(rows, key):
        return (float(np.mean(rows)), float(np.min(rows))) if len(rows) \
            else (float("nan"),) * 2

    print("\n===== PER-EXPERT: mean/min cosine per (criterion, p) =====")
    for crit in CRITS:
        print("criterion |%s|:" % ("g" if crit == "g" else "g*u"))
        hdr = "  p:      " + "".join("%12g" % pf for pf in PS)
        print(hdr)
        for L in LAYERS:
            cells = []
            for pf in PS:
                v = [r["cos"] for r in results
                     if r["layer"] == L and r["crit"] == crit
                     and r["p"] == pf]
                cells.append("%7.4f/%7.4f" % agg(v, None) if v else "   -")
            print("  L%-3d  " % L + " ".join(cells))
        for pf in PS:
            v = [r["cos"] for r in results
                 if r["crit"] == crit and r["p"] == pf]
            m, mn = agg(v, None)
            print("  GLOBAL p=%-5g mean=%.4f min=%.4f" % (pf, m, mn))

    print("\n===== mw-weighted MIXTURE: mean/min cosine =====")
    for crit in CRITS:
        print("criterion |%s|:" % ("g" if crit == "g" else "g*u"))
        for L in LAYERS:
            cells = []
            for pf in PS:
                v = [m["%s_p%g_cos" % (crit, pf)] for m in mixtures
                     if m["layer"] == L]
                cells.append("%.4f/%.4f" % agg(v, None))
            print("  L%-3d  " % L + " ".join(cells))
        for pf in PS:
            v = [m["%s_p%g_cos" % (crit, pf)] for m in mixtures]
            m_, mn = agg(v, None)
            print("  GLOBAL p=%-5g mean=%.5f min=%.5f  mean_rl2=%.4f" %
                  (pf, m_, mn,
                   float(np.mean([m["%s_p%g_rl2" % (crit, pf)]
                                  for m in mixtures]))))
    va = [m["adaptive_cos"] for m in mixtures]
    print("ADAPTIVE (top-%d mw p=%g, rest p=%g, |g|): mean=%.5f min=%.5f" %
          (ADAPTIVE[2], ADAPTIVE[0], ADAPTIVE[1],
           float(np.mean(va)), float(np.min(va))))

    print("\n===== BYTE ECONOMICS (exact, whole model) =====")
    print("gate=%.1fGB up=%.1fGB down=%.1fGB tot=%.1fGB" %
          (econ["tot_gate_gb"], econ["tot_up_gb"], econ["tot_down_gb"],
           econ["tot_gb"]))
    for kf, vf in econ["frac"].items():
        print("  p=%s -> expert byte fraction = %.3f" % (kf, vf))


if __name__ == "__main__":
    main()
