# Build with gcc (Linux, MSYS2/MinGW). On Windows with MSVC use build.bat.
# -mf16c is MANDATORY: the f16 KV cache uses vcvtph2ps/vcvtps2ph, which
# gcc does NOT enable with -mavx2 alone (MSVC's /arch:AVX2 does).
# -std=gnu11, NOT -std=c11: strict ISO sets __STRICT_ANSI__, glibc drops
# _DEFAULT_SOURCE and pread/setenv/posix_memalign lose their prototypes
# (hard error on gcc 14+). build_posix.sh has always used gnu11; the two
# build paths now agree. (-D_GNU_SOURCE is the other equivalent, but
# nf_os.h defines it itself too, so passing it only adds a
# "redefined" warning.)
# HONEST NOTE: this path isn't testable on the reference machine (no
# gcc installed) — verified on gcc 15.2 / glibc 2.42 (Zen 5, 8C/16T).
CC     = gcc
CFLAGS = -std=gnu11 -O2 -Wall -Wextra -D_FILE_OFFSET_BITS=64 -ffp-contract=off -mavx2 -mfma -mbmi2 -mf16c -fopenmp -I. -DZSTD_DISABLE_ASM -DZSTD_LEGACY_SUPPORT=0
# Most Linux distributions have python3 only; Windows/MSYS2 ships python.
PYTHON ?= $(shell command -v python3 2>/dev/null || echo python)
ZSRC   = zstd/common/debug.c zstd/common/entropy_common.c zstd/common/error_private.c \
         zstd/common/fse_decompress.c zstd/common/xxhash.c zstd/common/zstd_common.c \
         zstd/decompress/huf_decompress.c zstd/decompress/zstd_ddict.c \
         zstd/decompress/zstd_decompress.c zstd/decompress/zstd_decompress_block.c

nf: nf.c nf_gguf.c nf_gguf.h nf_tokenizer.c nf_tokenizer.h nf_model.c nf_model.h
	$(CC) $(CFLAGS) -o nf nf.c nf_gguf.c nf_tokenizer.c nf_model.c $(ZSRC) -lm -lpthread -fopenmp

test: nf
	$(PYTHON) tools/make_test_gguf.py tests/test.gguf
	./nf inspect tests/test.gguf --tensors

clean:
	rm -f nf nf.exe *.obj tests/test.gguf
