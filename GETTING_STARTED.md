# Getting Started — from zero to GLM-5.2 on your laptop

**Status: research engine.** Windows is the primary, most-measured
path; Linux (native or WSL) is a validated second target — same
source, cross-platform bit-exact (see below). All defaults were
tuned by measurement on one machine — i7-8550U (4C/8T, AVX2), 32GB RAM,
Crucial X9 USB-SSD (~650MB/s rated, 580-590MB/s measured real) — and every one of them is bracketed by
documented experiments (see [RETROSPECTIVE.md](RETROSPECTIVE.md) for
the story). On different hardware the auto-config is a sane start,
not a guarantee: re-measure.

**Cross-platform bit-exactness.** Windows (MSVC/UCRT), Linux via WSL2
(gcc/musl), and native Linux (gcc/glibc) produce **byte-identical**
logits from the same `.forgezh` container — verified with a tiny
reference fixture (`tools/test_nf_math.c`, `tools/glm_tiny_forgelogits_ref.bin`)
and confirmed on the real 744B model. The transcendentals (`nf_math.h`)
use only exactly-rounded IEEE operations, no platform libm, so the
same input produces the same output everywhere `-ffp-contract=off`
(or MSVC's default `/fp:precise`) holds. To reproduce this on your own
Linux machine: `sh validate_native.sh` from the repo root — builds,
runs the tiny gate against the reference fixture, and (if a converted
`.forgezh` is present one directory up) times a chat turn on the real
model too.

## Requirements

**Windows**: Windows 10/11, Visual Studio 2022 (any edition) with C++
tools.

**Linux**: gcc with AVX2/FMA/BMI2/F16C support, pthreads. `O_DIRECT` is
used when the filesystem supports it, with an automatic buffered
fallback otherwise (e.g. WSL's 9p mounts, some NTFS/ntfs3 setups).

**Both**:
- CPU with AVX2 (BMI2 recommended — the Huff0 decoder uses it)
- 32GB RAM (less will thrash; more helps the expert cache)
- Disk: **~460GB free during conversion**, ~355GB permanently
  (351GB `.forgezh` container + workspace). An SSD is mandatory;
  sequential read speed is the single biggest performance factor.
- Python 3.10+ with `zstandard` (`pip install zstandard`) for conversion
- Internet for the one-time download (~755GB of FP8 source weights
  streamed from HuggingFace; converted down to the ~404GB `.forge`
  container on the fly, so the ~460GB disk figure above never needs
  the full FP8 copy)

## 1. Build

```bat
rem Windows (MSVC)
build.bat
```

Produces `nf.exe` (builds the vendored zstd decoder lib on first run).

```sh
# Linux (gcc) — mirrors build.bat, produces ./nf
sh build_posix.sh nf
```

The examples below use `nf.exe`; on Linux, substitute `./nf`.

## 2. Tokenizer (one command, ~20MB download)

```bat
python tools\tokenizer_from_hf.py
```

Writes `models/glm52-tokenizer.gguf` (~9MB) straight from the official
`zai-org/GLM-5.2` tokenizer.json. `nf` finds it there automatically.
Gate if you want: `nf.exe tokenize models\glm52-tokenizer.gguf "test"`.

## 3. Get the model — convert it yourself (one-time, ~day of streaming + hours of CPU)

The container is built in three stages (they exist for historical
reasons — each stage was a measured improvement; a unified converter
is future work). All stages are resumable after interruption.

```bat
rem 3a. HuggingFace -> .forge (int4-gs64, ~404GB). Streams shard by
rem     shard via range requests; safe to interrupt and rerun.
python tools\forge_convert.py

rem 3b. .forge -> .forgez (zstd frames, ~346GB). Verifies every frame
rem     bit-exactly BEFORE progressively truncating the source, so the
rem     ~460GB peak never needs both full copies at once.
python tools\forge_zip.py

rem 3c. .forgez -> .forgezh (Huffman frames, ~351GB) + index. Same
rem     verify-then-truncate discipline. build.bat doesn't build
rem     forge_rez.exe (it's a separate one-off tool, not part of the
rem     engine); build it once, from a Visual Studio Developer Command
rem     Prompt, with the command in the header of tools\forge_rez.c:
cl /nologo /O2 /W4 /I. tools\forge_rez.c ^
   zstd\common\entropy_common.c zstd\common\error_private.c ^
   zstd\common\fse_decompress.c zstd\common\xxhash.c ^
   zstd\common\zstd_common.c zstd\compress\fse_compress.c ^
   zstd\compress\hist.c zstd\compress\huf_compress.c ^
   zstd\decompress\huf_decompress.c zstd\decompress\zstd_ddict.c ^
   zstd\decompress\zstd_decompress.c ^
   zstd\decompress\zstd_decompress_block.c /DZSTD_DISABLE_ASM ^
   /Fe:tools\forge_rez.exe

python tools\forge_rez_idx.py --prep glm52.forgez
tools\forge_rez.exe glm52.forgez glm52.forgezh
python tools\forge_rez_idx.py --idx glm52.forgez glm52.forgezh
```

(Weights are GLM-5.2 by Z.ai — their model license applies to the
converted container exactly as to the original weights.)

## 4. Chat

```bat
nf.exe chat glm52.forgezh
```

That's it — fifteen measured defaults engage automatically (expert
streaming, prefetch diet, cache-aware routing, k=6 truncation, batched
prefill, int8 kernels). On the reference machine: ~300s for a
50-token prompt + 24 tokens, ~4.5 s/token decode. `NF_GLM_AUTO=0`
disables all defaults; any env you set yourself always wins.

## 5. OpenAI-compatible API

```bat
nf.exe serve glm52.forgezh --port 8080
```

`POST /v1/chat/completions`, SSE streaming supported, same defaults.

## Known limits (honest list)

- Linux I/O v1 is blocking `pread` per worker thread, not an
  OVERLAPPED-style async pipeline (no io_uring yet) — same parallel
  decode path, simpler concurrency model, not yet re-measured for
  throughput on Linux the way Windows has been over months of tuning
- `nf serve` on Linux is validated functionally (native Ubuntu +
  WSL2), not put through the same performance campaign as Windows
- One flagship model for the `.forgezh` path (GLM-5.2); Qwen3/DeepSeek-V2-Lite/OLMoE run via classic GGUF
- Multi-step arithmetic is weak at greedy — that's the int4 model, not the engine (measured: k=8 fails it too)
- Defaults tuned on one machine; see [RETROSPECTIVE.md](RETROSPECTIVE.md) for how every knob was measured, so you can re-run the experiments on yours
