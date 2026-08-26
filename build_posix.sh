#!/bin/sh
# POSIX build (Linux port project) — mirrors build.bat
set -e
cd "$(dirname "$0")"
CFLAGS="-std=gnu11 -O2 -ffp-contract=off -mavx2 -mfma -mbmi2 -mf16c -fopenmp -I."
Z="zstd/common/debug.c zstd/common/entropy_common.c zstd/common/error_private.c zstd/common/fse_decompress.c zstd/common/xxhash.c zstd/common/zstd_common.c zstd/decompress/huf_decompress.c zstd/decompress/zstd_ddict.c zstd/decompress/zstd_decompress.c zstd/decompress/zstd_decompress_block.c"
OUT="${1:-/tmp/nf}"
gcc $CFLAGS -DZSTD_DISABLE_ASM -DZSTD_LEGACY_SUPPORT=0 -w \
    nf.c nf_gguf.c nf_tokenizer.c nf_model.c $Z \
    -o "$OUT" -lm -lpthread
echo "OK: $OUT"
