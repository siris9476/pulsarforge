# Build with gcc (Linux, MSYS2/MinGW). On Windows with MSVC use build.bat.
# -mf16c is MANDATORY: the f16 KV cache uses vcvtph2ps/vcvtps2ph, which
# gcc does NOT enable with -mavx2 alone (MSVC's /arch:AVX2 does).
# HONEST NOTE: this path isn't testable on the reference machine (no
# gcc installed) — last verified at M5, before the integer kernels and
# the f16 KV cache. If it fails, suspect the flags first, not the code.
CC     = gcc
CFLAGS = -std=c11 -O2 -Wall -Wextra -D_FILE_OFFSET_BITS=64 -ffp-contract=off -mavx2 -mfma -mbmi2 -mf16c -fopenmp -I. -DZSTD_DISABLE_ASM -DZSTD_LEGACY_SUPPORT=0
ZSRC   = zstd/common/debug.c zstd/common/entropy_common.c zstd/common/error_private.c \
         zstd/common/fse_decompress.c zstd/common/xxhash.c zstd/common/zstd_common.c \
         zstd/decompress/huf_decompress.c zstd/decompress/zstd_ddict.c \
         zstd/decompress/zstd_decompress.c zstd/decompress/zstd_decompress_block.c

nf: nf.c nf_gguf.c nf_gguf.h nf_tokenizer.c nf_tokenizer.h nf_model.c nf_model.h
	$(CC) $(CFLAGS) -o nf nf.c nf_gguf.c nf_tokenizer.c nf_model.c $(ZSRC) -lm -lpthread -fopenmp

test: nf
	python tools/make_test_gguf.py tests/test.gguf
	./nf inspect tests/test.gguf --tensors

clean:
	rm -f nf nf.exe *.obj tests/test.gguf
