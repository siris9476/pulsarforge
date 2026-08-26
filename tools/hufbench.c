/* hufbench — point 0 of the rANS/Huffman workstream: how fast does
 * Huff0 (zstd) ACTUALLY decode on this i7-8550U, on a real expert
 * record? PER-FRAME table (built once, 128KB blocks), X1 and X2
 * decode, with and without BMI2. The number that decides: >=900MB/s =
 * proceed with the battle-tested codec; ~500 = need the custom rANS
 * (or close the workstream).
 *
 *   cl /O2 /arch:AVX2 /Izstd /Izstd/common tools/hufbench.c
 *      zstd/compress/huf_compress.c zstd/compress/hist.c
 *      zstd/compress/fse_compress.c zstd/common/entropy_common.c
 *      zstd/common/error_private.c zstd/common/fse_decompress.c
 *      zstd/decompress/huf_decompress.c /DZSTD_DISABLE_ASM
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "zstd/common/mem.h"
#include "zstd/common/huf.h"
#include "zstd/compress/hist.h"

static double now_sec(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

#define BLK (128 * 1024)

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "tests/forge_rec_sample.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "no sample\n"); return 1; }
    fseek(f, 0, SEEK_END);
    const size_t n = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *src = malloc(n);
    fread(src, 1, n, f);
    fclose(f);

    /* frame table: counts over the whole record */
    unsigned count[256];
    unsigned maxSym = 255;
    size_t r = HIST_count(count, &maxSym, src, n);
    if (HUF_isError(r)) { fprintf(stderr, "hist err\n"); return 1; }
    HUF_CElt ct[HUF_CTABLE_SIZE_ST(255)];
    unsigned char wksp[HUF_CTABLE_WORKSPACE_SIZE + HUF_WORKSPACE_SIZE];
    const unsigned tlog = 11;
    r = HUF_buildCTable_wksp(ct, count, maxSym, tlog, wksp, sizeof(wksp));
    if (HUF_isError(r)) { fprintf(stderr, "build err: %s\n", HUF_getErrorName(r)); return 1; }
    unsigned char hdr[512];
    const size_t hdr_sz = HUF_writeCTable_wksp(hdr, sizeof(hdr), ct, maxSym,
                                               (unsigned)r, wksp, sizeof(wksp));
    if (HUF_isError(hdr_sz)) { fprintf(stderr, "writect err\n"); return 1; }

    /* compress the blocks with the frame CTable */
    const int nblk = (int)((n + BLK - 1) / BLK);
    unsigned char **cblk = malloc((size_t)nblk * sizeof(void *));
    size_t *csz = malloc((size_t)nblk * sizeof(size_t));
    size_t tot_c = hdr_sz;
    for (int b = 0; b < nblk; b++) {
        const size_t off = (size_t)b * BLK;
        const size_t len = off + BLK <= n ? BLK : n - off;
        cblk[b] = malloc(HUF_compressBound(len));
        csz[b] = HUF_compress4X_usingCTable(cblk[b], HUF_compressBound(len),
                                            src + off, len, ct, 0);
        if (HUF_isError(csz[b]) || csz[b] == 0) {
            fprintf(stderr, "blk %d not compressible (%zu)\n", b, csz[b]);
            return 1;
        }
        tot_c += csz[b];
    }
    printf("record %zu bytes -> %zu compressed (ratio %.3fx, table %zuB)\n",
           n, tot_c, (double)n / (double)tot_c, hdr_sz);

    /* decode: DTable from the frame header, then the blocks */
    unsigned char *out = malloc(n);
    unsigned dwksp[HUF_DECOMPRESS_WORKSPACE_SIZE_U32];
    static const char *names[4] = { "X1", "X1+bmi2", "X2", "X2+bmi2" };
    for (int mode = 0; mode < 4; mode++) {
        const int flags = (mode & 1) ? HUF_flags_bmi2 : 0;
        const int x2 = mode >= 2;
        HUF_CREATE_STATIC_DTABLEX2(dt, HUF_TABLELOG_MAX);
        size_t dr = x2
            ? HUF_readDTableX2_wksp(dt, hdr, hdr_sz, dwksp, sizeof(dwksp), flags)
            : HUF_readDTableX1_wksp(dt, hdr, hdr_sz, dwksp, sizeof(dwksp), flags);
        if (HUF_isError(dr)) { fprintf(stderr, "readdt err %s\n", HUF_getErrorName(dr)); continue; }
        /* warmup + verification */
        int ok = 1;
        for (int b = 0; b < nblk; b++) {
            const size_t off = (size_t)b * BLK;
            const size_t len = off + BLK <= n ? BLK : n - off;
            const size_t d = HUF_decompress4X_usingDTable(out + off, len,
                                                          cblk[b], csz[b], dt, flags);
            if (HUF_isError(d) || d != len) { ok = 0; break; }
        }
        if (!ok || memcmp(out, src, n) != 0) {
            fprintf(stderr, "%s: VERIFICATION FAILED\n", names[mode]);
            continue;
        }
        const int reps = 6;
        const double t0 = now_sec();
        for (int rep = 0; rep < reps; rep++)
            for (int b = 0; b < nblk; b++) {
                const size_t off = (size_t)b * BLK;
                const size_t len = off + BLK <= n ? BLK : n - off;
                HUF_decompress4X_usingDTable(out + off, len, cblk[b], csz[b],
                                             dt, flags);
            }
        const double dt_s = now_sec() - t0;
        printf("%-8s decode: %.0f MB/s (1 core)\n", names[mode],
               (double)n * reps / dt_s / 1e6);
    }
    return 0;
}
