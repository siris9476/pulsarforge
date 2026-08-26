# PulsarForge — operational guide

> This is the detailed command tour: setup, the original Qwen3-family
> milestone roadmap, model recommendations, tests, and a full usage
> walkthrough. For the project pitch, headline numbers, and
> comparisons against colibrì/DwarfStar/llama.cpp, see
> [README.md](README.md). For the fastest path to chatting with the
> flagship GLM-5.2 744B model via the `.forgezh` container, see
> [GETTING_STARTED.md](GETTING_STARTED.md). The Italian original of
> this document — the primary language the engineering actually
> happened in — is [README.it.md](README.it.md).

A **didactic** CPU-only LLM inference engine, written in C. It started
as a narrow bet on a single model family, **Qwen3** —
inspired by the lessons of [DwarfStar](https://github.com/antirez) and
llama.cpp, with a different goal: **understand by building**, not
compete on speed. It has since grown to validate five architectures
bit-exact or argmax-exact against an independent oracle (`qwen3`,
`qwen3moe`, `deepseek2`/MLA, `olmoe`, and `glm-dsa` — GLM-5.2, a 744B
MoE) and to run cross-platform bit-exact on Windows and Linux. This
guide covers the original Qwen3-family engine mechanics in full detail;
the GLM-5.2 flagship path has its own guide (linked above) because its
container format, streaming, and auto-config are substantial enough to
deserve one.

## What it is (and isn't)

- **Is** a small, readable engine that loads a Qwen3 GGUF, tokenizes,
  runs the forward pass, and generates text — validated at every step
  against the official HuggingFace implementation.
- **Isn't** a generic GGUF runner: one model family at a time, deeply
  validated (DwarfStar's "narrow bet," in miniature).
- **Doesn't aim** to beat llama.cpp on speed: on this hardware class
  its kernels are years of SIMD tuning. It aims to be explainable line
  by line.

## Reference hardware

x86-64 laptop with no dedicated GPU: HP EliteBook 840 G5 (i7-8550U, 4
cores / 8 threads, AVX2, 32GB RAM). Development and validation on
**Qwen3-0.6B** (fast to iterate on); the same code has to run
**quantized Qwen3-8B** within the 32GB budget.

## Setup (models and Python environment)

`models/` and `venv-tools/` are deliberately out of git (8.9GB of
GGUF, a whole venv) — recreate them like this, before build/test:

```
# Python venv for the tests (torch CPU — the global torch on this
# machine crashes on import):
python -m venv venv-tools
venv-tools/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
venv-tools/Scripts/pip install transformers tokenizers gguf

# models (mkdir models if it doesn't exist yet):
curl -sL -o models/Qwen3-0.6B-Q8_0.gguf   "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"
curl -sL -o models/Qwen3-0.6B-Q4_K_M.gguf "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf"
curl -sL -o models/Qwen3-0.6B-Q5_K_M.gguf "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q5_K_M.gguf"
curl -sL -o models/Qwen3-0.6B-Q6_K.gguf   "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q6_K.gguf"
curl -sL -o models/Qwen3-4B-Q4_K_M.gguf   "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf"
curl -sL -o models/Qwen3-8B-Q4_K_M.gguf   "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"

# MoE (Qwen3-30B-A3B, 18.6GB — 128 experts, 8 active per token):
curl -sL -o models/Qwen3-30B-A3B-Q4_K_M.gguf "https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf"

# tokenizer oracle, one per family member (check_m1.py validates each
# against ITS OWN tokenizer.json, never assumes it's identical across
# 0.6B/4B/8B; no HuggingFace cache involved):
curl -sL -o models/Qwen3-0.6B-tokenizer.json "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json"
curl -sL -o models/Qwen3-4B-tokenizer.json   "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/tokenizer.json"
curl -sL -o models/Qwen3-8B-tokenizer.json   "https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer.json"
```

## Roadmap (the original Qwen3-family milestones — all complete)

| Milestone | Content | Success criterion |
|---|---|---|
| **M0** ✅ | GGUF reader: header, metadata, tensors | inspects a real GGUF with no errors |
| **M1** ✅ | Byte-level BPE tokenizer | 11/11 prompts identical to the HF oracle |
| **M2** ✅ | fp32 forward pass, one token | matching argmax, cosine 0.9998, top-10 10/10 |
| **M3** ✅ | KV cache + generation | bit-exact selfcheck; greedy 5/20 = oracle then diverges (Q8_0 drift) |
| **M4** ✅ | Quantization: weights never expanded, inline dequant | Q8_0 603.9MB/cosine 0.9998; Q4_K_M 372.4MB/cosine 0.9758 |
| **M5** ✅ | mmap + OpenMP + AVX2 | 4.25x (ctx 128) → 2.91x (ctx 512); load 0.19s→0.04s |
| **M6** ✅ | Chat template, REPL, KV on disk | multi-turn conversation, session resumed across separate processes |

Beyond the original roadmap (see [RETROSPECTIVE.md](RETROSPECTIVE.md)
for detail): repetition penalty, vectorized attention, precomputed RoPE, batched prefill,
**integer kernels** (Q8 activations + maddubs), **f16 KV cache**, a
separate output head (Qwen3 8B+), textual stop, `--no-think`,
speculative decoding (`--draft`), **MoE** (Qwen3-30B-A3B: router +
top-8/128 experts).
Then the project grew far past this family entirely — five
architectures, a 744B flagship, a custom streaming container, a
cross-platform port — that story is in [README.md](README.md).

The discipline (from DwarfStar): **no milestone is done until it's
validated against the oracle**. Golden vectors are generated with
`tools/golden.py` (requires `pip install torch transformers`).

## Models tried and recommendations (on the reference machine)

| Model | RAM | Decode | Recommended use |
|---|---|---|---|
| Qwen3-0.6B Q4_K_M | 372MB | ~15 tok/s | raw speed, **`--no-think` only** (thinking collapses at Q4 on 0.6B) |
| Qwen3-0.6B Q6_K | 472MB | ~13 tok/s | the best 0.6B all-rounder |
| **Qwen3-4B Q4_K_M** | 2.5GB | ~2.6 tok/s | **the sweet spot**: `--no-think` for daily use, thinking for reasoning tasks |
| Qwen3-8B Q4_K_M | 4.8GB | ~1.2 tok/s | maximum quality, with patience |
| Qwen3-30B-A3B Q4_K_M (MoE) | 17.7GB | ~3.1 tok/s | large-model quality, faster than the 4B (only 8/128 experts active per token) |

(This table predates several MoE-affecting optimizations described in
[RETROSPECTIVE.md](RETROSPECTIVE.md); the 30B-A3B decode figure is
higher on the current engine — see README.md's headline numbers for
the up-to-date measurement.)

## Tests

The whole regression battery in one command:

```
venv-tools/Scripts/python.exe tests/run_all.py          # base suite (~2-3 min)
venv-tools/Scripts/python.exe tests/run_all.py --full   # + selfcheck/dequant on 4B and 8B
```

## Build

**Windows (MSVC / Visual Studio 2022):**
```
build.bat
```

**Linux (gcc) — mirrors build.bat, produces `./nf`:**
```
sh build_posix.sh nf
```

Cross-platform bit-exactness is validated: same source, same
`-ffp-contract=off`-guarded transcendentals (`nf_math.h`), byte-identical
output on Windows and Linux for the same model. See
[README.md](README.md) for how that was verified.

## Usage

```
# M0 — inspection (quick test with no model: a synthetic GGUF):
python tools/make_test_gguf.py tests/test.gguf
nf.exe inspect tests/test.gguf --tensors

# M0 — with the real model (in models/, downloaded from Qwen/Qwen3-0.6B-GGUF):
nf.exe inspect models/Qwen3-0.6B-Q8_0.gguf

# M1 — tokenization (inline ASCII text, or --file for UTF-8):
nf.exe tokenize models/Qwen3-0.6B-Q8_0.gguf "hello world"
nf.exe tokenize models/Qwen3-0.6B-Q8_0.gguf --file prompt.txt

# M1 — regression against the HuggingFace oracle:
python tests/check_m1.py

# M2 — last-position logits + comparison against the fp32 oracle:
nf.exe logits models/Qwen3-0.6B-Q8_0.gguf --file prompt.txt
venv-tools/Scripts/python.exe tests/check_m2.py

# M3 — generation (greedy by default; --temp/--top-p for sampling):
nf.exe generate models/Qwen3-0.6B-Q8_0.gguf --file prompt.txt --n-predict 30
nf.exe generate models/Qwen3-0.6B-Q8_0.gguf --file prompt.txt --n-predict 30 --temp 0.7 --top-p 0.9 --seed 7

# Speculative decoding (greedy only): a small model proposes K tokens,
# the big one verifies them in one pass — output is GUARANTEED
# identical to non-speculative. Honest note: on this machine it isn't
# faster (decode is compute-bound after the integer kernels); it
# stays useful on memory-bound machines/kernels.
nf.exe generate models/Qwen3-8B-Q4_K_M.gguf --file prompt.txt --n-predict 64 \
    --draft models/Qwen3-0.6B-Q4_K_M.gguf --draft-k 6

# M3 — self-consistency (incremental cache vs full recompute, no oracle):
nf.exe selfcheck models/Qwen3-0.6B-Q8_0.gguf --file prompt.txt

# M3 — greedy comparison against the HuggingFace oracle:
venv-tools/Scripts/python.exe tests/check_m3.py

# M4 — the exact same command as M2, but on a Q4_K_M GGUF instead of
# Q8_0 (weights stay quantized in RAM, dequant is inline in the dot product):
venv-tools/Scripts/python.exe tests/check_m2.py models/Qwen3-0.6B-Q4_K_M.gguf

# M4 — check the Q4_K/Q6_K dequantizer against the official reference:
venv-tools/Scripts/python.exe tests/check_m4_dequant.py

# M5 — prefill/decode benchmark across the context frontier (CSV):
python tools/make_bench_prompt.py tests/bench_prompt.txt
nf.exe bench models/Qwen3-0.6B-Q4_K_M.gguf --file tests/bench_prompt.txt \
    --ctx-start 128 --ctx-max 1024 --step-mul 2 --gen-tokens 16

# M6 — interactive chat (ChatML, KV reused across turns). Default is
# greedy (--temp 0): on a 0.6B model, sampling (temp>0) can derail
# reasoning on code/logic tasks — verified by comparing greedy against
# the HF fp32 oracle.
nf.exe chat models/Qwen3-0.6B-Q4_K_M.gguf --system "You are a concise assistant."
# on aggressive quants (Q4_K_M) extended thinking can spiral and never
# converge: --no-think skips it and the model answers directly (same
# question: from 900 tokens burned with no answer to a correct answer
# in 32). Reliable thinking needs Q5_K_M or above.
nf.exe chat models/Qwen3-0.6B-Q4_K_M.gguf --system "You are a concise assistant." --no-think
# for variety instead of reliability, enable sampling explicitly:
nf.exe chat models/Qwen3-0.6B-Q4_K_M.gguf --temp 0.7 --top-p 0.9
# /quit or /exit to leave. --session file.bin to save/resume:
nf.exe chat models/Qwen3-0.6B-Q4_K_M.gguf --session conversation.bin
# (the second invocation, even in a brand new process, resumes where
# the first one left off — see tests/check_m6.py)

# M6 — multi-turn chat + session regression, across two separate processes:
python tests/check_m6.py

# MoE (Qwen3-30B-A3B) — the exact same generate/chat/bench commands;
# the architecture (dense qwen3 or qwen3moe) is detected from the GGUF:
nf.exe generate models/Qwen3-30B-A3B-Q4_K_M.gguf --file prompt.txt --n-predict 30
nf.exe chat models/Qwen3-30B-A3B-Q4_K_M.gguf --system "You are a concise assistant."

# MoE — router+experts validation against an independent Python
# reimplementation on the real weights (no HF oracle: the model doesn't
# fit in RAM in bf16 on this machine):
venv-tools/Scripts/python.exe tests/check_moe.py
```

## Environment variables (GLM-5.2 streaming tuning)

GETTING_STARTED.md and README.md mention that fifteen measured
defaults engage automatically on glm-dsa models (`.forge`/`.forgezh`
containers or a `general.architecture=glm-dsa` GGUF), in `chat`,
`generate`, and `serve` alike (`serve` inherits the same auto-config —
the benches stay fully explicit), and that any variable you set
yourself always wins. Here they are by
name, since the auto-config banner only prints a count, not which
ones fired. Values differ between the GGUF path and the `.forge`
path where noted; `NF_GLM_AUTO=0` disables all of them at once.

| Variable | Default (GGUF / `.forge`) | What it does |
|---|---|---|
| `NF_PRIORITY` | `high` | Process priority. |
| `NF_GLM_STREAM_GB` | `8` / `15` | Expert-cache streaming budget in GB. |
| `NF_GLM_IO_THREADS` | `6` | I/O threads for expert streaming. |
| `NF_GLM_PREFETCH` | `1` | Enables expert prefetch. |
| `NF_GLM_QPRIO` | `1` | Priority queue for prefetch requests. |
| `NF_GLM_OVERLAP` | `1` | Overlaps I/O with compute. |
| `NF_GLM_MISS_SKIP` | `0.3` | Miss-skip threshold — **changes the output** (opt-in by design, never silently the wrong default): output depends on cache history, so two runs with different cache state can diverge. |
| `NF_MOE_DYNK` | `0.4` | Dynamic-k threshold for expert routing. |
| `NF_GLM_PF_TOPK` | `0` / `3` | Prefetch diet: only rank 0-2 predictions get anticipated (`.forge` only, never measured on GGUF). |
| `NF_GLM_CACHE_BIAS` | `0.05` | Cache-aware router: residency bonus on near-tie ranks ≥4. |
| `NF_GLM_PREFILL_BATCH` | `0` / `1` | Batched prefill (disabled automatically under `--draft`, never validated together). |
| `NF_GLM_PREFILL_CHUNK` | `32` | Batch size for batched prefill. |
| `NF_FORGE_INT8` | `0` / `1` | Int8 activation kernels. |
| `NF_MOE_TOPK_EFF` | off / `6` | Expert-truncation k (`.forge` only). |
| `NF_GLM_CACHE_BIAS_PREFILL` | off / `0.05` | Cache-aware router bias applied during prefill too (`.forge` only). |

A few more knobs exist outside the auto-config, off by default:

| Variable | What it does |
|---|---|
| `NF_KV_F16=1` (or `--kv-f16`) | Restores the f16 KV cache; the default is per-head int8 (half the memory). |
| `NF_MODEL_PIN_GB=N` | Pins N GB of the model's own mmap pages resident (Windows) — useful together with `.expw` when the combined footprint would otherwise exceed ~85% of physical RAM and get zero residency. |
| `NF_EXPERT_PIN_GB=N` | Pins N GB of expert-cache mmap pages resident — mmap branch only, has no effect once streaming is active. |
| `NF_GLM_THINK=1` | Reopens the `<think>` reasoning block for glm-dsa models (an empty think block is the default there, `--no-think`'s effect is redundant on glm). |
| `NF_LONGMEM=1` | Enables long-memory sparse attention over landmark blocks (`NF_LONGMEM_BLOCK`/`_SINK`/`_WINDOW`/`_TOPK` tune it — defaults 128/64/512/16). |
| `NF_LONGMEM_DISK=1` | Backs long-memory beyond the RAM budget onto disk (requires `NF_LONGMEM`; `NF_LONGMEM_DISK_PATH` sets the location, `NF_LONGMEM_DISK_POOL` the pool size). |
| `NF_GLM_MLA_ABSORB=1` | glm-dsa only: MLA attention directly on the cached latent instead of reconstructing full K/V — cuts the KV cache to ~2% of the default at the cost of disabling batched prefill, `--draft`/MTP/layer-skip drafting, and session save/load (all loudly rejected, not silently ignored). The fix suggested by the engine's own OOM message. |
| `NF_GLM_MTP=1` | glm-dsa only: loads the model's MTP (multi-token-prediction) head, required before `--mtp-draft` (below) can be used. |

`nf generate` also has two glm-dsa-specific speculative-decoding modes,
alternatives to `--draft`/`--draft-k` (a separate small model) that
use the SAME model instead — greedy-only, output is byte-identical to
plain greedy generation:

| Flag | What it does |
|---|---|
| `--mtp-draft` | Self-speculates using the model's own MTP head (requires `NF_GLM_MTP=1`). |
| `--layerskip-draft` | Self-speculates using a layer-skipped forward pass of the same model as the drafter. |

## Conceptual background

The concepts behind each milestone (tokens, tensors, KV cache, logits,
softmax, quantization, GGUF/GGML) came out of studying
[DwarfStar](https://github.com/antirez), the project that preceded
this one — kept as private study notes, not part of this repository.
