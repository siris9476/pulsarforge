# Research: MoE expert streaming from SSD — genealogy, state of the art, alternatives

Four web research passes run in parallel (primary,
open, verified sources: arXiv, GitHub PRs/issues, official blogs),
following an in-source analysis of colibrì and DwarfStar's own
streaming implementations. Legend: [V] =
verified against a primary source; [S] = from a snippet, source
identified but not read in full; [P] = prior knowledge, unverified.

---

## Part 1 — Genealogy of SSD streaming (2021-2026)

### The roots (2021-2022)
- **ZeRO-Infinity** (Microsoft, Apr 2021) [V]: first use of NVMe as a
  memory tier for giant networks — but for TRAINING.
  [arxiv.org/abs/2104.07857](https://arxiv.org/abs/2104.07857)
- **SE-MoE** (Baidu, May 2022) [V]: first hierarchical streaming of
  MoE experts in inference (CPU-GPU, round-robin, no prediction).
  [arxiv.org/abs/2205.10034](https://arxiv.org/abs/2205.10034)
- **DeepSpeed-Inference** (Jun 2022) [V]: first formal proposal of
  weights on NVMe for INFERENCE. [arxiv.org/abs/2207.00032](https://arxiv.org/abs/2207.00032)

### The two-school fork (March 2023)
- **FlexGen** [V]: explicit SCHEDULED offload (linear programming
  over GPU/CPU/disk). [arxiv.org/abs/2303.06865](https://arxiv.org/abs/2303.06865)
- **llama.cpp mmap** (PR #613, jart+slaren) [V]: disk as a PASSIVE
  tier via the OS page cache. All subsequent history is the
  comparison between these two schools — including our own
  ReadFile-vs-mmap measurement (2x per byte on cold reads).

### Speculative prefetch is born — three times (2023)
- **Pre-gated MoE** (Aug 23, 2023) [V]: block N's gate selects
  block N+1's experts — deterministic but requires training the
  model. [arxiv.org/abs/2308.12066](https://arxiv.org/abs/2308.12066)
- **EdgeMoE** (Aug 28, 2023) [V]: training-free statistical
  preloading. [arxiv.org/abs/2308.14352](https://arxiv.org/abs/2308.14352)
- **Eliseev & Mazur** (Dec 28, 2023) [V]: THE mechanism colibrì uses —
  applying layer L+1's gate to layer L's hidden states, zero
  training. + per-layer LRU cache + mixed quantization. Direct
  ancestor of router-lookahead. [arxiv.org/abs/2312.17238](https://arxiv.org/abs/2312.17238)

### Fame (Dec 2023) and maturation (2024)
- **LLM in a flash** (Apple, Dec 12, 2023) [V]: windowing, row-column
  bundling, a sparsity predictor. 4-5x on CPU vs naive. The flash
  tier becomes mainstream. [arxiv.org/abs/2312.11514](https://arxiv.org/abs/2312.11514)
- **PowerInfer** (SJTU, Dec 16, 2023, independent concurrent work)
  [V]: hot/cold neuron power-law — but NO disk (GPU-CPU).
  [arxiv.org/abs/2312.12456](https://arxiv.org/abs/2312.12456)
- **PowerInfer-2** (Jun 2024) [V]: flash tier on smartphones; beats
  Apple by 3.84x by restructuring COMPUTE to overlap I/O and compute
  (I/O idle: from 76.7% to 13.7%). [arxiv.org/abs/2406.06282](https://arxiv.org/abs/2406.06282)
- **MoE-Infinity** (Jan 2024) [V]: per-sequence tracing, explicit SSD
  tier on personal machines. [arxiv.org/abs/2401.14361](https://arxiv.org/abs/2401.14361)
- **Fiddler** (Feb 2024) [V]: flips the problem — run the expert on
  the CPU instead of moving the weights (ancestor of KTransformers).
  [arxiv.org/abs/2402.07033](https://arxiv.org/abs/2402.07033)
- **ProMoE** (Oct 2024) [V]: learned predictor with lead time k>1.
  [arxiv.org/abs/2410.22134](https://arxiv.org/abs/2410.22134)
- **HOBBIT** (Nov 2024) [V]: on cache miss, loads the REDUCED-precision
  version of the less critical expert. [arxiv.org/abs/2411.01433](https://arxiv.org/abs/2411.01433)

### The present (2025-2026)
- llama.cpp `--cpu-moe`/`-ot exps=CPU` [V]: experts in RAM, disk
  passive. No explicit expert streaming in mainline (confirmed:
  discussion #19163).
- ACTIVE streaming from NVMe reaches consumer engines only in 2026:
  **ds4/DwarfStar** (May 2026, antirez) and **colibrì** (Jul 2026,
  JustVugg, GLM 744B on ~25GB RAM, 0.05-0.1 tok/s cold on CPU).
- Contrarian view [V]: "SSD Offloading Considered Harmful in Energy
  Efficiency" (Aug 2025) — prefetch hides latency but not
  energy (flash ~102.4 pJ/b vs HBM 4.2). [arxiv.org/abs/2508.06978](https://arxiv.org/abs/2508.06978)

### Timeline summary
```
2021 Apr  ZeRO-Infinity          NVMe as a tier — training
2022 Jun  DeepSpeed-Inference    weights on NVMe for inference (first paper)
2023 Mar  FlexGen / llama.cpp    scheduled offload || passive mmap (the two schools)
2023 Aug  Pre-gated / EdgeMoE    "knowing L+1's experts in advance" is born
2023 Dec  Apple flash / PowerInfer / Eliseev&Mazur   fame + LRU/speculative prefetch
2024      MoE-Infinity, Fiddler, PowerInfer-2, ProMoE, HOBBIT   maturation
2025      KTransformers / llama.cpp -ot   experts in RAM, disk passive
2026      ds4 (May) / colibrì (Jul)   active NVMe streaming in consumer engines in C
```

---

## Part 2 — The cross-cutting finding: prefetch is the WEAK lever

Three independent ablations [V] converge against the intuition:
- **HOBBIT**: mixed-precision load = 1.19-1.57x on its own; prefetch =
  ~10% and only in prefill ("minimal decoding gains").
- **ProMoE** (the paper ABOUT prefetch): most of the gain comes from
  reordering; "prefetching alone" has minimal impact.
- **DALI 2026** (tested on Qwen3-30B-A3B!): placement 4.1x >>
  cache replacement +38% >> prefetch +9%. [arxiv.org/abs/2602.03495](https://arxiv.org/abs/2602.03495)

Structural reason: predicting L+1 buys at most one layer of lead
time; every miss is wasted bandwidth. **"Reduce the bytes, don't
read them faster."**

---

## Part 3 — The levers that pay off (for our CPU/Windows/USB-SSD 580-590MB/s context)

1. **Better quant, no streaming** (~zero effort): Q4_K_M (17.4
   GiB, PPL 9.209) is NOT on the Pareto front. **UD-Q4_K_XL: 16.5
   GiB, PPL 9.169** (smaller AND better, standard K-quants already
   supported). **IQ4_KS (ik_llama): 15.3 GiB, PPL 8.99 — below
   BF16** (suspected QAT), but needs a new kernel. IQ2_M: 9.7 GiB,
   PPL 9.98 (guaranteed all-in-RAM, quality to be validated).
   Sources: ubergarm gist, ik_llama discussion #359. [V]
2. **Cache-aware routing bias** (arXiv 2412.00099) [V]: bias on the
   router logits toward experts already in cache. Miss 28%→7%
   (DeepSeek-V2-Lite) / 35%→16% (Qwen1.5-MoE) with a half-model
   cache; task loss <0.1%; training-free; LOW complexity
   (a bias before top-k + a residency bitmap). Loses
   bit-exactness BY CONSTRUCTION → PPL gate (precedent: mixed
   ubatch). Average expert lifetime in cache: from ~19-26 tokens to 55-76.
3. **Layout for co-activation** (precedent: mbolt, Jul 2026,
   llama.cpp discussion #18758) [V]: co-activated experts written
   adjacently (usage profile; permuted router rows). On
   Qwen3-Next-80B-A3B: reads/token 1418→~370 (2.23x), end-to-end
   5.80→9.00-9.80 tok/s (1.55-1.63x); pread beats mmap by +13-14%
   (independent confirmation of our own measurement). KLD/PPL gate.
   Theoretical basis [V] (arXiv 2510.05497): the top 10% of expert
   pairs = 60-80% of co-activations, up to 20-40x above the
   random baseline; Qwen clusters more strongly than DeepSeek.
4. **HOBBIT mixed-precision on miss** [V]: double copy on disk
   (Q4+Q2); on a miss of a low-gate expert, load the small one.
   67% high load / 30% low / 3% skip; <1% quality loss; 1.19-1.57x.
5. **Low-gate tail skipping** [V]: on Qwen3-30B-A3B greedy holds
   stable down to k=5 out of 8 (arXiv 2602.02443) — sampling
   degrades it. ~37% less load, ~20 lines. MoDES (2511.15690):
   per-layer thresholds beat a naive single threshold.
6. **Speculative decoding as an I/O amortizer** (SpecMoEOff,
   arXiv 2508.21706) [V]: verifying K tokens per pass = one load
   per K tokens. 2.1x average on Mixtral offload. Note: our own
   multi-session mixed ubatch already does the same amortization
   in a different dimension.

## Part 4 — Windows I/O: dry verdicts

- **A pool of synchronous ReadFile calls is already the right
  choice** at 3MB blocks; IOCP buys nothing (diskspd #118) [S].
  QD 4-8 from 4-6 threads saturates a Gen3.
- **FILE_FLAG_NO_BUFFERING**: only makes sense WITH an application-
  level LRU (avoids double-caching + memcpy); modest gain; every
  reread comes back cold. [P]
- **DirectStorage without a GPU**: works but useless at 3MB/request
  (its strength is batching many small requests; GDeflate would
  steal the 4 cores). [S]

## Part 5 — Key data on Qwen3-30B-A3B

- **It's in the MOST cache-friendly group there is** (arXiv 2505.16056,
  measured directly) [V]: routing consistency SRP@16 = 55.9 >
  Mixtral 49.3 >> DeepSeek-V2-Lite 37.8. It has both of the two
  properties that predict cache-friendliness (MoE at every layer, no
  shared experts). Cache knee ≈ 2x active parameters
  (~2.3GB for the hot set), segment hit rate ≥60%.
- **Global skew evens out** (MoE-Infinity) [V]: "after 1000
  sequences the reuse counts even out" — the exploitable skew
  is per-request/temporal (co-occurrence, short-range locality), NOT
  static-global. This explains the null result of our own Phase B
  (pinning by global heat): it measured the wrong thing.
- Adjacent-token locality: ~30% chance of reselecting the previous
  token's expert vs 12.5% random baseline (arXiv 2412.00099) [V].

## Part 6 — Our own numbers that the research confirms/contextualizes

- ReadFile ~2x mmap per byte on cold reads (nf debugexpwio) —
  independently confirmed by mbolt (+13-14% end-to-end).
- Phase B pinning null — predicted by MoE-Infinity ("evens out").
- Phase 0 synchronous streaming: default resident ~6.6
  tok/s (N=8); synchronous streaming at 12GB = 4.57/4.88 (~71%); 14GB
  identical to 12 (hit-rate saturated). An earlier disaster at 4/8GB
  was the cache size, not synchronous itself. NOTE: at a 12GB budget
  no RAM is actually saved (18.6 pinned base + 12 cache ≈ 30.6GB).
- Double-digit claims from other systems (9.93x, 16.7x...) are almost
  always against weak baselines (bare mmap, DeepSpeed); against a
  well-built LRU the honest margin is 1.3-2x.

## Main sources
HOBBIT [arxiv.org/abs/2411.01433](https://arxiv.org/abs/2411.01433) · REAP [arxiv.org/abs/2510.13999](https://arxiv.org/abs/2510.13999) ·
Local Routing Consistency [arxiv.org/abs/2505.16056](https://arxiv.org/abs/2505.16056) · Cache-aware
routing [arxiv.org/abs/2412.00099](https://arxiv.org/abs/2412.00099) · ProMoE [arxiv.org/abs/2410.22134](https://arxiv.org/abs/2410.22134) ·
DALI [arxiv.org/abs/2602.03495](https://arxiv.org/abs/2602.03495) · Fate [arxiv.org/abs/2502.12224](https://arxiv.org/abs/2502.12224) ·
MoE-Infinity [arxiv.org/abs/2401.14361](https://arxiv.org/abs/2401.14361) · Fiddler [arxiv.org/abs/2402.07033](https://arxiv.org/abs/2402.07033) ·
PowerInfer-2 [arxiv.org/abs/2406.06282](https://arxiv.org/abs/2406.06282) · LLM-in-a-flash
[arxiv.org/abs/2312.11514](https://arxiv.org/abs/2312.11514) · Eliseev&Mazur [arxiv.org/abs/2312.17238](https://arxiv.org/abs/2312.17238) ·
SpecMoEOff [arxiv.org/abs/2508.21706](https://arxiv.org/abs/2508.21706) · co-activation
[arxiv.org/abs/2510.05497](https://arxiv.org/abs/2510.05497) · mbolt github.com/doramirdor/mbolt +
llama.cpp discussion #18758 · quant sizes: ubergarm gist + ik_llama
discussion #359 · energy [arxiv.org/abs/2508.06978](https://arxiv.org/abs/2508.06978)
