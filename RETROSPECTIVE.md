# pulsarforge: a retrospective

This is the technical story behind pulsarforge, distilled from the
project's own day-by-day build log — about 19,500 lines, written in
real time from the first GGUF parser to a 744B-parameter model running
on a laptop with no GPU: what was actually built, in what order, and
what each step measured.

## What this project actually is

pulsarforge is a CPU-only LLM inference engine, written from scratch
in C11, by one person, as an educational bet: understand how inference
really works by building it, not by reading someone else's kernel. It
does not try to out-optimize llama.cpp's hand-tuned SIMD kernels —
that fight isn't winnable by a single developer, and the diary says so
on its first page. What it does try to do, and mostly succeeds at, is
run models the "normal" hardware tier can't touch, by treating the
disk as a real memory tier instead of an afterthought — culminating in
a 744-billion-parameter mixture-of-experts model, GLM-5.2, streamed
from a USB SSD on 32GB of RAM with zero GPU.

The approach — validate against an independent reference before
trusting anything, keep the scope narrow, measure before optimizing —
comes directly from studying [DwarfStar](https://github.com/antirez),
antirez's own from-scratch inference engine and this project's
direct methodological predecessor.

Every number below was checked against an independent reference before
it was trusted, and re-measured whenever the surrounding engine
changed underneath it. That's not a stylistic choice — it's the reason
any of these numbers are worth believing at all.

## Building the foundation

The first seven milestones (M0 through M6) are the unglamorous,
load-bearing part: a
GGUF parser that validates every tensor actually fits inside the file
before anything trusts it; a byte-level BPE tokenizer, hand-written
from the tokenizer's own regex rather than borrowed from a library,
checked 11/11 against the HuggingFace oracle (whose one early failure
turned out to be a newline-translation bug in the test harness itself,
not the tokenizer — an early instance of a habit that never went away:
check the oracle before blaming the code under test); a full fp32
forward pass matching that oracle's top-10 logits 10 for 10; a KV
cache validated the harder way, bit-exact against a full recompute
rather than against any external reference; three quantization formats
with dequantization done inline inside the dot product instead of
expanding weights back out in memory; multithreading, AVX2, and
memory-mapped loading; and finally chat templates and sessions
persisted to disk, resumable across separate process runs.

None of it is exotic on its own. What matters is that it was there
from the start, not bolted on once things got harder — by the time
the interesting engineering began, checking every step against an
independent reference was already just how the project worked.

## Getting fast on models that fit in RAM

The first stretch of real performance work (roughly the project's
first two weeks) is a straightforward kernel campaign against
llama.cpp on models small enough to live entirely in memory — Qwen3 at
0.6B through 30B-A3B. The interesting part isn't any single kernel; it's
how many plausible ideas turned out to need remeasuring before they
counted as wins.

An early attempt at a wider SIMD register layout for the K-quant
kernels — batching more weights per instruction to amortize overhead —
actually *lost* 19% against the existing scalar path. The idea wasn't
wrong; the specific interleaving was. A second, more carefully designed
version of the same idea won 71–155% depending on the kernel. Elsewhere,
quantizing the KV cache to int8 was expected to be a straightforward
win and wasn't: measured end to end, it came in slightly *negative*
on decode throughput. The investigation built to explain that
shortfall found a different, unrelated lever instead — reorganizing
the cache into a per-head memory layout, independent of quantization
entirely, cut attention latency 18% on its own, for free. Int8 stayed
in the engine anyway, not for speed but for its real dividend: half
the KV cache's RAM footprint, which matters more as context grows.

The re-measurement habit paid for itself six days into this stretch.
Speculative decoding — a small model drafting tokens, the large one
verifying them in a single pass — had been implemented, validated as
byte-identical to plain decoding, and shelved with an honest negative
verdict: 0.97 tokens/second against 1.2 for the direct path, because a
newly-added integer kernel had just made decode compute-bound, which
defeats the whole premise of drafting ahead. The code stayed in the
tree anyway, correct and unused. Several unrelated optimizations
later, the same feature — zero new lines of code — was remeasured out
of habit rather than any specific suspicion, and it now won 1.58x.
Nothing about speculative decoding had changed; the engine underneath
it had, enough to flip the verdict. Trusting the earlier benchmark
would have left a free win sitting disabled in a drawer indefinitely
— though the 1.58x itself turned out to be an easy-case number: on
real content a couple of days later, the same setting actually lost
to plain decoding, and the project's default drafting depth was
lowered specifically because of that follow-up measurement, one more
verdict this project never let sit unchecked.

Short dense prefill was the stubbornest of these fronts, resisting six
different kernel techniques before power telemetry — not a
clock-speed guess, actual RAPL measurement — found the real cause: an
instruction mix that cost more energy per token than llama.cpp's,
independent of raw instruction count. That diagnosis pointed to a
specific fix, pairing groups of activations to halve L3 cache traffic,
which closed the front to 101.7% — a real if narrow edge that the
project's own later summary still calls essentially tied.

By the point the project called this thread closed, four of six
measured fronts against llama.cpp were ahead — long-context decode
2.1–2.4x, 30B MoE decode ~1.3x, MoE prefill up 24.8%, long dense
prefill up a more modest 2.3% — and the other two sat at parity for
different reasons: short decode hit an exact, reproducible 100.0%, the
memory-bandwidth wall both engines run into at the same point; short
dense prefill landed at the 101.7% described above, close enough that
the project's own final tally calls it tied too, but for a different
reason — compute-side kernel work closing a real gap, not the same
bandwidth ceiling. The bandwidth-bound tie was itself a finding worth
having: past a certain point, kernel cleverness stops mattering and
the two engines are burning the same physical resource at the same
rate.

## Making it big: mixture-of-experts, and a KV cache that shrinks 8.9x

Two architectural jumps had to happen before a 744B model was even a
realistic target.

First, Qwen3-30B-A3B — a real MoE model, 128 experts, 8 active per
token — arrived without needing new kernels: once the router's
top-k-then-normalize order was verified against the actual HuggingFace
source (not assumed from a library default, which turned out to be set
wrong for this specific model), the expert weights were just 3D GGUF
tensors that the engine's existing dequantization and matmul kernels
already knew how to slice.

Second, porting DeepSeek-V2-Lite's Multi-head Latent Attention (MLA)
architecture — the mechanism GLM-5.2 also uses — compressed the KV
cache from 5,120 values per token down to 576, a 8.9x reduction, by
caching a shared low-rank latent representation instead of full
per-head keys and values. The real payoff isn't decode speed at short
context (measured tied there) — it's *reach*: the same RAM budget
now holds a dramatically longer context, which is exactly the
resource GLM-5.2's disk-streaming architecture would go on to need
freed up.

## The actual problem: 744B parameters, 32GB of RAM

GLM-5.2 doesn't fit in RAM by roughly a factor of six. The starting,
completely honest number, the day the model first produced any output
at all, was 0.005 tokens per second — effectively 100% of wall-clock
time spent blocked on disk I/O, because the naive path read expert
weights through ordinary memory-mapped page faults: 4KB reads at
random offsets, the single worst access pattern a storage device can
be asked for.

The fix wasn't a smarter cache. It was accepting that the disk had to
be treated as a first-class part of the architecture, not a fallback,
and building for its actual physics:

- **Large, explicit, contiguous reads instead of page faults.**
  Routing decides which 8 of 256 experts a token needs *before* any of
  them are read, so all of a layer's misses can be issued together as
  large sequential reads instead of trickling in as 4KB faults. This
  single change moved the measured disk throughput from a 26MB/s
  floor to a stable 560MB/s — about 21x — which is close to the
  drive's real sequential ceiling.
- **Finding out the bottleneck had already moved.** Once I/O stopped
  dominating, direct CPU sampling of the running process — not a
  theory, a measurement — showed decode had quietly become
  compute-bound: roughly 96 seconds of CPU time inside a 94-second
  wall-clock token. The actual cost was dequantizing five weight
  formats that had no accelerated kernel yet. Writing fused AVX2
  kernels for the three that carried enough of the model to justify
  it brought the token down to roughly 36-37 seconds end to end, a
  2.6x wall-clock improvement from this lever alone — the other two
  formats cover so little of the file that a dedicated kernel would
  have been validation work with no real payoff.
- **Cross-layer prefetch, measured before it was built.** Predicting
  the next layer's experts from the current layer's hidden state, and
  speculatively reading them while the current layer is still
  computing, was checked offline first: an 80.8% prediction accuracy
  against what the router actually chose. Wired into the live engine,
  the accuracy reproduced to the exact same digit, and the technique
  delivered an 18% wall-clock reduction on top of the streaming
  baseline — one of the rarer cases in the diary where the measured
  number matched the projection instead of contradicting it.
- **Spending fewer bytes, not just reading them faster.** Marginal
  experts — ones ranked low enough by the router that their
  contribution barely matters — are skipped entirely when they're not
  already cached and their weight falls under a measured threshold:
  around -30% of fetch bytes at the default setting, up to -45% at a
  more aggressive one, with the top-ranked experts never sacrificed
  regardless of how aggressive the setting gets. The threshold came
  from a 330-position KL-divergence sweep against the unmodified
  baseline, not a single perplexity number — and that same discipline
  caught the aggressive setting's real limit: it later failed a
  multi-turn task the KL numbers alone hadn't flagged, and was
  downgraded to short, self-contained generations only.

Each of these was measured in isolation, gated against a bit-exact
reference with the feature disabled, and only combined once it held up
on its own. Across this whole stretch of work — contiguous reads,
fused kernels, cross-layer prefetch, and other measured refinements
along the way — **196 seconds per token came down to roughly 9.3**,
about 21x, before miss-skip shaved further bytes off on top of that
baseline. Every step still produces output
byte-identical to an unaccelerated reference run whenever the
optimizations are switched off. That's not the end of the story; the
second, larger jump came from rethinking the file format itself.

## Building a format instead of just an engine

A comparison against colibrì — the closest comparable project, another
from-scratch CPU engine with SSD expert streaming — had already
produced one useful result weeks earlier: on a smaller shared model,
the two engines, built independently with no shared code, produced
byte-identical output, a real mutual correctness check neither project
could have gotten alone. A later, direct same-laptop, same-disk
comparison on GLM-5.2 itself exposed a trade nobody should have had to
make. Running the compressed GGUF this project had been using,
pulsarforge's entire run — model load included — finished before
colibrì's decode phase alone did, roughly 7x faster per token.
But colibrì's own int4 format answered cleanly and stopped on its own,
every time; pulsarforge's more aggressively compressed weights
occasionally opened with the right answer and then failed to stop —
the same artifact-level fragility present in colibrì's own deprecated
low-bitrate format, not a flaw specific to either engine, but a real
cost of the bitrate chosen.

That comparison is what justified designing a new file format instead
of continuing to adapt GGUF. `.forge` packs each expert's three weight
matrices — gate, up, down — contiguously and sector-aligned, so
loading one expert is a single ~20MB read instead of three separate
ones scattered across the file. Experts are stored at a uniform 4.25
bits/weight (int4, group size 64) — the precision point the earlier
comparison had already shown was defensible — while dense weights,
attention, and the router stay at higher precision, informed directly
by this project's own measurements of where fidelity actually
mattered. The result kept the speed advantage and fixed the
reliability problem: clean, deterministic termination, the same
correctness guarantees as before. Every stage of the from-scratch,
~755GB streaming converter that builds `.forge` from the original FP8
weights was validated against the source at the tensor level before
being trusted.

A further pass added transparent compression on top (`.forgez`, then
`.forgezh` with a Huffman-coded variant and parallel block decode),
plus batched prefill with overlapped fetch and compute, cutting a
representative scenario from 682 seconds to roughly 300. Combined with
everything upstream of it, the full arc — from the first working but
essentially unusable 0.005 tokens/second to a reliable, cleanly
terminating **roughly 4.5 seconds per token** — is a ~44x improvement,
none of it at the cost of the bit-exact guarantees the rest of the
project runs on.

## Making the numbers mean the same thing everywhere

The last major arc ported the whole engine to Linux and made its
output bit-for-bit identical to the Windows build — not "close," not
"same architecture," identical down to the byte, on any input. That
turned out to require more than a build script: Windows's C runtime
and Linux's two common C libraries (glibc, musl) don't compute the
same transcendental math functions — exp, log, sin, cos — to the same
bit, only to the same *tolerance*. For a project whose entire
validation discipline rests on bit-exact comparison, a few ulps of
difference between platforms is a real problem, not a rounding
curiosity.

The fix was writing a small, deterministic math library from scratch
(derived from the public-domain Cephes implementations, but
re-verified independently rather than trusted), used identically on
every platform instead of each platform's own libm. The payoff: the
exact same model file, run through the exact same engine, produces
byte-identical logits on Windows/MSVC, WSL2/musl, and — validated last,
booted from a live USB stick on bare metal with no virtualization at
all — native Linux/glibc. Three independent C runtimes, one number.

## What it doesn't do

- It's tuned to one specific laptop; the techniques generalize, the
  tuned constants don't.
- It doesn't beat llama.cpp everywhere — short decode sits at the
  memory-bandwidth wall, a tie by physics, not a loss to fix.
- For a long stretch, GLM-5.2 was genuinely demo-grade — seconds per
  token, not the 1–2 tokens/second that would make it comfortable to
  use daily — and the diary states plainly that this specific hardware
  has a real disk-bandwidth ceiling that no further software work gets
  past.
- Several fully-built, fully-validated features were deliberately left
  off by default because their measured gain didn't clear the
  complexity or risk they'd add — each closed with a number attached,
  not left as an open hope.

## Where it landed

> PulsarForge is the existence proof for the bottom of the curve — a
> 744B frontier model, bit-gated quality, on a laptop that predates
> the model by seven years.

That's the project's own summary of its niche, arrived at only after
a full survey of every comparable engine — all of which assume either
GPU memory (16GB+, sometimes far more) or NVMe bandwidth well beyond
this laptop's USB SSD.

## Where to go from here

- [README.md](README.md) — what the engine does today, how to run it,
  full benchmark tables.
- [GETTING_STARTED.md](GETTING_STARTED.md) — build and setup, Windows
  and Linux.
- [STREAMING_RESEARCH.md](STREAMING_RESEARCH.md) — the research survey
  behind the streaming architecture: genealogy of SSD-based MoE
  streaming since 2021, state of the art, alternatives considered.
