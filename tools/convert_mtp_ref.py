#!/usr/bin/env python3
"""Converts glm_tiny/mtp_ref.json (hybrid MTP oracle, HF-computed — see
the "Hybrid MTP (nextn) oracle" block in colibri's own
c/tools/make_glm_oracle.py) into a compact binary that
nf_debug_mtp_oracle (nf_model.c) reads to compare nf_mtp_predict_glm
against the REAL logits computed by HF (actual GlmMoeDsaDecoderLayer +
GlmMoeDsaRMSNorm, only the enorm/hnorm/eh_proj/concat-order wiring is
ours).

Format: header (2x int32: n_positions, vocab), then for each position:
t (int32), tok_t+1 (int32), true_tok_t+2 (int32), logits[vocab] (float32).

Usage: python tools/convert_mtp_ref.py <mtp_ref.json> <out.bin>
"""
import json
import struct
import sys


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    d = json.load(open(sys.argv[1]))
    preds = d["predictions"]
    vocab = len(preds[0]["logits"])
    with open(sys.argv[2], "wb") as f:
        f.write(struct.pack("<ii", len(preds), vocab))
        for p in preds:
            assert len(p["logits"]) == vocab
            f.write(struct.pack("<iii", p["t"], p["tok_t+1_used_as_input"],
                                p["true_tok_t+2_teacher_forced"]))
            f.write(struct.pack(f"<{vocab}f", *p["logits"]))
    print(f"written {sys.argv[2]}: {len(preds)} positions, vocab={vocab}")


if __name__ == "__main__":
    main()
