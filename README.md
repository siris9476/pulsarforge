# PulsarForge

A from-scratch, **CPU-only LLM inference engine in C11** that pushes a
**744-billion-parameter model** through a 2018 laptop with 32GB of RAM
and a USB SSD — five architectures validated bit-exact against an
independent oracle, cross-platform bit-exactness on Windows and Linux,
and a public engineering retrospective covering every measurement,
including the ones that failed.

> **Status: research engine.** Windows is the primary, most-measured
> path; Linux (native or WSL) is a validated second target — same
> source, byte-identical output on the same model. Want to run it?
> See [GETTING_STARTED.md](GETTING_STARTED.md) — build, tokenizer bootstrap,
> one-time conversion, chat and API in five steps.

*Questo README è in inglese per il pubblico open source; il README
operativo ([README.it.md](README.it.md)) è in italiano — la sua
traduzione aggiornata in inglese è [OPERATIONS.md](OPERATIONS.md).*

## What this is

PulsarForge started as a didactic engine for Qwen3-0.6B:
understand inference by building it, validating every milestone against
the HuggingFace reference. It grew into something rarer:

- **Five architectures**, each validated bit-exact or argmax-exact
  against an independent oracle: `qwen3`, `qwen3moe`, `deepseek2`
  (MLA), `olmoe`, and `glm-dsa` — GLM-5.2, a 744B MoE with compressed-Q
  MLA, a sparse DSA indexer, and 256 experts per layer.
- **Expert streaming on pure CPU**: the 202GB GGUF lives on a USB SSD;
  experts are fetched per-token through an explicit `NO_BUFFERING`
  reader with an LRU cache, cross-layer predictive prefetch (80.8%
  recall, measured offline *before* the code was written), arrival-order
  compute/IO overlap, and a priority queue where demand fetches overtake
  speculation.
- **A measurement discipline** behind every number below: baseline
  before building, interleaved A/B/A runs against thermal drift,
  counter-proofs, and honest negative verdicts — see
  [RETROSPECTIVE.md](RETROSPECTIVE.md) for the story, including the
  experiments that didn't work.

## Headline numbers (HP EliteBook 840 G5: i7-8550U, 4C/8T AVX2, 32GB RAM, USB SSD)

| What | Result |
|---|---|
| GLM-5.2 744B (202GB GGUF), first naive run | 196 s/token |
| Same model, same laptop, after the campaign | **9.3 s/token interactive (21×)** |
| **Same model in PulsarForge's own `.forge` container** | **~9 s/token, clean termination (see below)** |
| Batch mode (8 correlated streams, layer-major sweep) | **~2.9 s/token aggregate (68×)** |
| Qwen3-30B-A3B MoE chat | ~6.8 tok/s from the first token |
| Qwen3-4B | ~2.6 tok/s — the machine's sweet spot |

The remaining walls are named and measured, not guessed: the USB chain
tops out at 580–590MB/s real (Gen2 confirmed by probing cold regions),
and the 15W PL1 caps the CPU. The diary closes the fetch axis with a
measured verdict on every possible move — don't fetch (cache is at the
routing-locality ceiling, flat from 8GB to 20GB, above simulated
Bélády), fetch less (lossless compression: 0.1%; sub-expert grain loses
to 26MB/s 4K-random reads), fetch earlier (prefetch is capacity-bound),
fetch elsewhere (striping and cross-token prefetch: built, falsified).

## Measured findings we believe are new

- **The router is already a near-oracle on GLM-5.2**: correlation
  ρ=0.859 between router weights and true expert importance —
  contradicting published expectations for the DeepSeek family (weak
  correlation reported in arXiv 2406.18219 and follow-ups). This is
  what makes dynamic-k truncation safe here.
- **GLM-5.2 is "dense in use"**: no near-twin experts (max 1.2%
  similarity), no lazy layers, no dead positions, experts incompressible
  (ratio 1.000). The whole "skip/deduplicate parts of the model" family
  of optimizations is closed by how the model is, with citable numbers.
- **Nobody publishes numbers in this regime** (700B-class model,
  ≤32GB RAM, USB storage). The community results we could find need
  96GB RAM + a 24GB GPU, or 512GB EPYC machines. At this RAM class,
  PulsarForge is the measured state of the art — partly for lack of
  competitors, which is exactly why the numbers are published here.

## The `.forge` container: our own format, and why it exists

The head-to-head below exposed a trade nobody should have to make: our
2.17-bit GGUF was 7× faster than colibrì but often failed to emit a
stop token (an artifact-level fragility their deprecated per-row int4
also had — it lives in the quantization, not the engine); colibrì's
int4 terminated cleanly but crawled. So PulsarForge grew what colibrì
has: **its own container**, converted directly from the official FP8
weights by a streaming converter (141 shards, one at a time, ~6GB of
temp space, resumable, numerically validated), with the bit budget
spent where this project's *measurements* say it matters — experts at
int4 group-64 (the quality point colibrì's fix proved), router and
correction biases at f32 (that's where the measured near-ties live),
one contiguous 20MB record per expert (one `pread` instead of three).

Result, measured A/B/A interleaved on the same laptop, byte-identical
across repeat runs:

| Stack | Terminates cleanly? | s/token |
|---|---|---|
| 2.17-bit GGUF + PulsarForge (retired) | no | ~6–9.3 |
| **`.forge` int4 + PulsarForge** | **yes** | **~9** |
| int4 + colibrì | yes | 63–66 |

**Later update:** a two-day, measurement-led marathon took the
same scenario from 682s to **~300s** (several levers, every one bit-gated:
prefetch diet, a Huffman-compressed container `.forgezh` at 351GB with
parallel block decode, batched prefill with fetch/compute overlap and
batched attention, fused multi-activation kernels, and a novel
**cache-aware router** — residency bonus on near-tie ranks ≥4, factual
quality 3/3 at both tested strengths). Decode now runs at **~4.5 s/token**
(~14× colibrì per token), with k=6 expert truncation and a cache-aware
router both bracketed by measured quality cliffs on BOTH sides; a full 24-token run *including model load*
finishes before colibrì's 8-token decode alone.

Even at that earlier `.forge` milestone (before the marathon above),
PulsarForge already matched colibrì-class quality at ~7× colibrì's
speed — and the same price the fragile GGUF used to cost. The last
1.6× of *that* stretch came from a measurement-led
refinement worth retelling: dense weights at int8 cost ~17GB of RAM
residency, squeezing the expert cache below one token's working set
(the cache *cycled*); requantizing them to int4 **in place** (687
tensors, 264 seconds, no re-download — the JSON kind strings patched
byte-for-byte) halved residency, freed the cache to ~750 slots, and
dropped read traffic to exactly the artifact-size ratio.

```
# convert once (resumable; ~a day of streaming on this line):
python tools/forge_convert.py --dest glm52.forge
python tools/forge_idx.py     glm52.forge

# daily driving (auto-config included; the tokenizer is found
# automatically at models/glm52-tokenizer.gguf):
nf chat glm52.forge --session my.nfs
```

(That's the uncompressed `.forge` container from before the later
update above. For a new setup, use the compressed `.forgezh` container
instead — see [GETTING_STARTED.md](GETTING_STARTED.md) for the
conversion pipeline.)

## The ecosystem

What was two artisans' territory a year ago (colibrì, DwarfStar) is now
a category. The closest relatives, verified at the source:

| Project | Stack | Model / speed | Hardware floor |
|---|---|---|---|
| [flash-moe](https://github.com/danveloper/flash-moe) | C + Metal | Qwen3.5-397B @ 4.4 tok/s | M3 Max 48GB, **17.5 GB/s** SSD |
| [pulsar](https://github.com/giannisanni/pulsar) | Rust + CUDA | **GLM-5.2 744B @ 2.7 tok/s** | 2× 16GB CUDA GPUs, NVMe, io_uring |
| [sepia](https://github.com/rinaldofesta/sepia) | (Mac) | Inkling 975B | 128GB Mac |
| [moe-stream](https://github.com/GOBA-AI-Labs/moe-stream) | Rust | 80B class | 24GB Mac |
| [DwarfStar/ds4](https://github.com/antirez/ds4) | C, Metal/CUDA/ROCm | DeepSeek V4 class, distributed | 96-128GB Macs+ |
| llama.cpp ([PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294)) | C++ | expert streaming entering mainline | — |
| **PulsarForge** | **C, CPU-only** | **GLM-5.2 744B @ ~0.22 tok/s** | **32GB RAM, 580-590MB/s USB SSD (real), zero GPU** |

Same technique family everywhere — experts streamed from disk, hot-set
caching, routing-aware I/O. The niche nobody else occupies is the
hardware floor: pulsar runs the *same model* ~12× faster on ~30× the
I/O+compute budget; flash-moe's SSD alone moves ~30× our bytes/second.
PulsarForge is the existence proof for the bottom of the curve — a
744B frontier model, bit-gated quality, on a laptop that predates the
model by seven years.

**vs [colibrì](https://github.com/JustVugg/colibri)** — the closest
project in spirit (CPU inference of GLM-class MoE with SSD streaming).
Head-to-head on OLMoE-1B-7B (same model, same prompt, greedy, this
laptop): both engines — built with zero shared code — produce the
*identical token-for-token continuation*, a strong mutual correctness
validation. On speed PulsarForge decodes ~3 tok/s vs 0.14 for colibrì's
`olmoe.c`; after we patched an AVX2 kernel *into their prototype* it
reached 0.25, still ~12× slower — but that file is their validation
prototype, not their optimized main engine.

So we ran the real one. **Head-to-head on GLM-5.2 itself** (August
2026): colibrì v1.5.0 main engine, their recommended int4-gs64
container (~429GB), this same laptop, same USB SSD, same greedy
prompt, interleaved runs. Result: colibrì **~64.5 s/token** (8 tokens
in 502–530s, 3% expert residency, its own printout: 0.02 tok/s) vs
PulsarForge **~9.3 s/token** — **~7× faster per token**; our entire
run *including model load* finished before their decode alone. Honest
decomposition: ~1.8× of that comes from bytes (their int4 artifact
moves ~11GB/token, our 2.17-bit moves ~6 — a format choice, not an
engine one); the remaining ~4× is stack (8GB expert cache at 40–66%
hit + 80.8% cross-layer prefetch + I/O overlap, vs their 3% residency
at the cap=1 floor this machine's RAM forces on them). **Quality went
the other way**: their int4 answered cleanly and self-terminated at 8
tokens, twice out of twice; our 2.17-bit opens with the right answer
and then fails to stop — the price of extreme quantization, which
their own deprecated per-row int4 container also paid (documented
non-termination loops, their issue #455): cross-engine evidence that
termination fragility lives in the artifact, not the engine. Fairness
notes on the record: colibrì ran RAM-starved (overcommit, cache floor;
with 128GB it declares ~1.8 tok/s), while PulsarForge ran a config
tuned for weeks on this exact machine. Delightful convergence: their
auto-tuner disabled MTP on this box with the same reasoning our
measurements had already established ("widens the expert union, adds
disk reads").

**vs DwarfStar (antirez)** — the project this one grew from, and the
inventor of the SSD-streaming idea. A head-to-head is impossible *by
design*, and that difference is the point: DwarfStar's streaming is
GPU-only (`ds4.c` refuses `--ssd-streaming` on CPU; its overlap works
because the GPU computes while a CPU thread reads). PulsarForge streams
experts on **pure CPU**, where the computing threads are the same ones
that would contend for I/O — the regime DwarfStar explicitly does not
enter. The projects explore disjoint regimes of the same problem.
Inherited from DwarfStar: the streaming idea, the RoPE `theta_scale`
trick, the "narrow bet" discipline. Contributed back: a quantified
Known Issue (per-head cos/sin redundancy in its CPU reference path),
documented in its AGENT.md.

**vs llama.cpp** — on models that fit in RAM, measured on this machine
with official binaries, interleaved A/B/A: long-context decode (d4096)
**won 2.1–2.4×**, 30B MoE decode **won ~1.3×**, MoE prefill **won
1.25×**, long prefill **won 1.02×**, short decode **tied at 100.0%**
(the memory-bandwidth wall both engines hit at the same point), short
dense prefill **tied at 101.7%** (a different, compute-side tie, not
the same bandwidth ceiling). llama.cpp remains the right tool for
almost everyone; these numbers exist to show what a readable
single-purpose engine can reach on its reference hardware.

## Quick start

```
build.bat              # Windows, MSVC (VS2022) — primary target
sh build_posix.sh nf   # Linux, gcc — validated bit-exact against Windows
```

Model downloads, Python test venv, and the full milestone-by-milestone
command tour are in [OPERATIONS.md](OPERATIONS.md) (English) /
[README.it.md](README.it.md) (Italian original). The regression suite
is one command: `venv-tools/Scripts/python.exe tests/run_all.py`.

Daily driving a model that fits in RAM:

```
nf.exe chat models/Qwen3-4B-Q4_K_M.gguf --system "You are a concise assistant."
```

Daily driving the 744B (yes, really — one command, ~4.5s/token) with the
compressed `.forgezh` container (see
[GETTING_STARTED.md](GETTING_STARTED.md) for the full conversion
pipeline, from the original FP8 weights):

```
nf.exe chat glm52.forgezh --session my.nfs
```

The measured-best configuration (fifteen tuning knobs) applies itself
automatically on glm GGUFs (`NF_GLM_AUTO=0` to disable; any variable
you set yourself wins). `--session` makes conversation history free on
resume (~4.5 s/token of history *not* re-prefilled). A cost-aware
per-turn cap (96 tokens ≈ 7 min worst case on this hardware) protects
against the failure mode of extreme quants: at 2.17 bits/weight the
model often answers correctly in the first sentence and then fails to
emit a stop token — at every quality level, approximations exonerated
by byte-identical A/B (it's the model, not the engine). Ask for
brevity *inside your question* ("Answer in a few words") — it works;
a system prompt at this bitrate destabilizes instead.

## Honest limits

- Tuned on one machine. The techniques are general; the constants
  (cache budget, thread counts, dynamic-k threshold) are this laptop's.
- Windows/MSVC is the primary, most-measured path. Linux (`build_posix.sh`,
  gcc) is validated for correctness — cross-platform bit-exact output,
  functional `nf serve` on native Ubuntu and WSL2 — but hasn't had the
  months of performance tuning Windows has.
- This is a learning engine with real results, not a product. If you
  want to *run* models, use llama.cpp. If you want to *understand* what
  running them costs and why, read [RETROSPECTIVE.md](RETROSPECTIVE.md).

## Credits

[DwarfStar](https://github.com/antirez) for the lessons and the
streaming idea; [colibrì](https://github.com/JustVugg/colibri) for the
independent cross-validation target and an honest rival worth
measuring against; llama.cpp for being the reference that kept every
claim here disciplined.

Built with assistance from Claude Fable 5 (Anthropic).

## License

MIT (see `LICENSE`). The vendored zstd decoder (`zstd/`) is BSD-3-Clause,
Copyright Meta Platforms — see `zstd/LICENSE`.
