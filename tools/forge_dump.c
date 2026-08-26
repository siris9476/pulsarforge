/* forge_dump — point 0 of unknown territory B: extracts ONE decoded
 * expert record from the .forgezh (Python does not decode Huff0).
 *   forge_dump <file.forgezh> <idx> <lm> <e> <out.bin>
 * out.bin = raw record span in bytes (down|gate|up, i4gs64 layout).
 * build: cl /nologo /O2 /W4 /I. tools\forge_dump.c
 *   zstd\common\entropy_common.c zstd\common\error_private.c
 *   zstd\common\fse_decompress.c zstd\decompress\huf_decompress.c
 *   /DZSTD_DISABLE_ASM /Fe:tools\forge_dump.exe
 *   (reuses the huf_unframe logic from forge_rez).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "zstd/common/mem.h"
#include "zstd/common/huf.h"

#define BLK (128 * 1024)
#define RAWBIT 0x80000000u

static int huf_unframe(const unsigned char *fr, size_t fr_sz,
                       unsigned char *out, size_t span) {
    const unsigned char *p = fr, *end = fr + fr_sz;
    if (p + 4 > end) return 0;
    uint32_t hdr_sz; memcpy(&hdr_sz, p, 4); p += 4;
    if (p + hdr_sz + 4 > end) return 0;
    HUF_CREATE_STATIC_DTABLEX1(dt, HUF_TABLELOG_MAX);
    unsigned dwksp[HUF_DECOMPRESS_WORKSPACE_SIZE_U32];
    if (HUF_isError(HUF_readDTableX1_wksp(dt, p, hdr_sz, dwksp, sizeof(dwksp),
                                          HUF_flags_bmi2))) return 0;
    p += hdr_sz;
    uint32_t nblk; memcpy(&nblk, p, 4); p += 4;
    if (nblk != (span + BLK - 1) / BLK) return 0;
    const uint32_t *csz = (const uint32_t *)p;
    p += 4ull * nblk;
    size_t off = 0;
    for (uint32_t b = 0; b < nblk; b++) {
        const size_t len = off + BLK <= span ? BLK : span - off;
        uint32_t c; memcpy(&c, csz + b, 4);
        if (c & RAWBIT) {
            if ((c & ~RAWBIT) != len || p + len > end) return 0;
            memcpy(out + off, p, len); p += len;
        } else {
            if (p + c > end) return 0;
            const size_t d = HUF_decompress4X_usingDTable(out + off, len, p, c,
                                                          dt, HUF_flags_bmi2);
            if (HUF_isError(d) || d != len) return 0;
            p += c;
        }
        off += len;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: forge_dump fz idx lm e out\n"); return 1; }
    FILE *fi = fopen(argv[2], "rb");
    if (!fi) return 1;
    uint32_t magic, ver, nc, nd, nlm, nex;
    fread(&magic, 4, 1, fi); fread(&ver, 4, 1, fi);
    fread(&nc, 4, 1, fi);
    fseek(fi, (long)nc * 40, SEEK_CUR);
    fread(&nd, 4, 1, fi);
    fseek(fi, (long)nd * (96 + 4 + 4 + 4 + 8), SEEK_CUR);
    fread(&nlm, 4, 1, fi); fread(&nex, 4, 1, fi);
    fseek(fi, (long)nlm * 3 * 4, SEEK_CUR);
    const long lm = atol(argv[3]), e = atol(argv[4]);
    _fseeki64(fi, (long long)((size_t)lm * nex + e) * 40, SEEK_CUR);
    uint64_t zr[5];
    fread(zr, 8, 5, fi);
    fclose(fi);
    fprintf(stderr, "lm=%ld e=%ld zoff=%llu zlen=%llu span=%llu\n", lm, e,
            (unsigned long long)zr[0], (unsigned long long)zr[1],
            (unsigned long long)zr[2]);
    FILE *fz = fopen(argv[1], "rb");
    unsigned char *fr = malloc(zr[1]);
    unsigned char *out = malloc(zr[2]);
    _fseeki64(fz, (long long)zr[0], SEEK_SET);
    fread(fr, 1, zr[1], fz);
    fclose(fz);
    if (!huf_unframe(fr, zr[1], out, zr[2])) {
        fprintf(stderr, "unframe FAILED\n");
        return 1;
    }
    FILE *fo = fopen(argv[5], "wb");
    fwrite(out, 1, zr[2], fo);
    fclose(fo);
    fprintf(stderr, "ok: %llu bytes\n", (unsigned long long)zr[2]);
    return 0;
}
