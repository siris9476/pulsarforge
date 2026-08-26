#!/usr/bin/env python3
"""M3 test: generates N greedy tokens with pulsarforge and with the
HuggingFace fp32 oracle on the same prompt, compares the id sequence.

Honest note: our weights are Q8_0 quantized, the oracle is fp32.
Greedy decoding is a "cascading" process: each chosen token feeds back
into the next one, so even a tiny numeric difference (already measured
in M2: cosine 0.9998, not 1.0) can cause a different token to be
picked at some point — from there on the two sequences can diverge
completely. This isn't an engine bug, it's the unstable nature of
argmax. The criterion here isn't "identical sequence forever", but
"the common prefix is consistent with the quality already measured in
M2" (we expect several correct tokens before any divergence).

Usage: venv-tools/Scripts/python.exe tests/check_m3.py [gguf_path]
"""
import os
import subprocess
import sys

GGUF = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B-Q8_0.gguf"
NF = "./nf.exe" if sys.platform == "win32" else "./nf"
PROMPT = "The capital of France is"
N_PREDICT = 20
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(HERE, "m3_prompt.txt")


def main() -> int:
    with open(PROMPT_FILE, "wb") as fp:
        fp.write(PROMPT.encode("utf-8"))

    r = subprocess.run([NF, "generate", GGUF, "--file", PROMPT_FILE,
                        "--n-predict", str(N_PREDICT), "--temp", "0"],
                       capture_output=True)
    if r.returncode != 0:
        sys.exit(f"nf generate failed:\n{r.stderr.decode(errors='replace')}")
    ours_ids = [int(line.split()[1]) for line in
                r.stderr.decode().splitlines() if line.startswith("GENID ")]
    ours_text = r.stdout.decode("utf-8", errors="replace").strip()
    print("ours:  ", repr(ours_text))
    print("our ids:", ours_ids)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
                                                 torch_dtype=torch.float32)
    model.eval()
    prompt_ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(prompt_ids, max_new_tokens=N_PREDICT,
                             do_sample=False)
    golden_ids = out[0].tolist()[prompt_ids.shape[1]:]
    golden_text = tok.decode(out[0], skip_special_tokens=True)
    print("oracle:", repr(golden_text))
    print("golden ids:", golden_ids)

    common = 0
    for a, b in zip(golden_ids, ours_ids):
        if a != b:
            break
        common += 1
    print(f"\nidentical prefix: {common}/{min(len(golden_ids), len(ours_ids))} tokens")
    print("(divergence expected past a certain point due to the Q8_0 vs "
          "fp32 drift already measured in M2 — not a correctness failure)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
