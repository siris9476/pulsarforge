#!/usr/bin/env python3
"""Generates the reference "golden vectors" with HuggingFace transformers.

This is the oracle every milestone of the C engine is validated
against (DwarfStar's central lesson: without an official reference, an
inference engine is indistinguishable from one that's silently wrong).

Usage:
    python tools/golden.py --model Qwen/Qwen3-0.6B \
        --prompt "The capital of France is" --out tests/golden

Requires: pip install transformers (and torch, unless --tokens-only)

Output in <out>/:
    meta.json   model, prompt, token count
    tokens.txt  the prompt's token ids, one per line   (oracle for M1)
    logits.f32  last-position logits, raw float32 (oracle for M2,
                skipped with --tokens-only)
"""
import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--prompt-file", default=None,
                    help="read the prompt from a UTF-8 file (exact bytes, "
                         "avoids argv codepage issues on Windows)")
    ap.add_argument("--out", default="tests/golden")
    ap.add_argument("--tokens-only", action="store_true",
                    help="tokenizer only (M1), doesn't need torch")
    args = ap.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, "rb") as fp:
            args.prompt = fp.read().decode("utf-8")

    from transformers import AutoTokenizer

    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt)["input_ids"]

    with open(os.path.join(args.out, "tokens.txt"), "w") as fp:
        for t in ids:
            fp.write(f"{t}\n")
    print("tokens:", ids)

    vocab_size = None
    if not args.tokens_only:
        import torch
        from transformers import AutoModelForCausalLM

        # float32 to get the highest-precision reference: the
        # quantization drift (M4) is measured against this.
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor([ids])).logits[0, -1, :].float()
        logits.numpy().tofile(os.path.join(args.out, "logits.f32"))
        vocab_size = int(logits.shape[0])

    with open(os.path.join(args.out, "meta.json"), "w") as fp:
        json.dump({
            "model": args.model,
            "prompt": args.prompt,
            "n_tokens": len(ids),
            "vocab_size": vocab_size,
        }, fp, indent=2)

    print(f"golden values written to {args.out}/ ({len(ids)} tokens)")


if __name__ == "__main__":
    main()
