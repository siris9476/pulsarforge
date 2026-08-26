/* forge_rez — reconverts .forgez (zstd frames) -> .forgezh (Huff0
 * frames): the benchmark shows 1154MB/s decode on one core against
 * zstd's 542 — the CPU ceiling that throttled the .forgezh falls.
 *
 * Huff0 frame (4096-aligned in the file):
 *   u32 hdr_sz | hdr (CTable) | u32 nblk | u32 csz[nblk] | blob...
 *   csz with the high bit set (0x80000000) = RAW block of (csz&mask)
 *   bytes (fixed frame table: a rare block can expand — never allow a
 *   compressed frame to exceed the span).
 * 128KB blocks, PER-FRAME table built on the counts of the whole
 * record (91B measured, ratio 1.129x).
 *
 * Staged with truncation of the source .forgez (there isn't enough
 * space for a full copy), bit-exact verification of EVERY frame
 * against the original zstd BEFORE each truncation, resumable json
 * manifest. The .forgez is consumed FROM THE TAIL (frames in
 * decreasing zoff order) so truncation frees real space.
 *
 * Usage: forge_rez <src.forgez> <dst.forgezh> [--keep]
 *   (the <dst>.manifest.json manifest is created/resumed automatically;
 *    the v4 idx must be generated AFTERWARDS with tools/forge_rez_idx.py)
 *
 * build: cl /nologo /O2 /W4 /I. tools\forge_rez.c
 *   zstd\common\entropy_common.c zstd\common\error_private.c
 *   zstd\common\fse_decompress.c zstd\common\xxhash.c
 *   zstd\common\zstd_common.c zstd\compress\fse_compress.c
 *   zstd\compress\hist.c zstd\compress\huf_compress.c
 *   zstd\decompress\huf_decompress.c zstd\decompress\zstd_ddict.c
 *   zstd\decompress\zstd_decompress.c
 *   zstd\decompress\zstd_decompress_block.c /DZSTD_DISABLE_ASM
 *   /Fe:tools\forge_rez.exe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>
#include "zstd/zstd.h"
#include "zstd/common/mem.h"
#include "zstd/common/huf.h"
#include "zstd/compress/hist.h"

#define BLK (128 * 1024)
#define AL4K(x) (((x) + 4095ull) & ~4095ull)
#define RAWBIT 0x80000000u

static uint64_t file_size(const char *p) {
    WIN32_FILE_ATTRIBUTE_DATA a;
    if (!GetFileAttributesExA(p, GetFileExInfoStandard, &a)) return 0;
    return ((uint64_t)a.nFileSizeHigh << 32) | a.nFileSizeLow;
}

/* Free space on the volume that holds `path` (the .forgezh output/
 * target file), not a hardcoded drive letter — derive the root
 * "X:\\" from path's own drive prefix when present; for a relative
 * path (no "X:" prefix) pass NULL so Windows resolves it against the
 * current directory's volume instead. */
static uint64_t disk_free(const char *path) {
    char root[4] = { 0 };
    const char *dir = NULL;
    if (path && path[0] && path[1] == ':') {
        root[0] = path[0]; root[1] = ':'; root[2] = '\\'; root[3] = '\0';
        dir = root;
    }
    ULARGE_INTEGER fb;
    GetDiskFreeSpaceExA(dir, &fb, NULL, NULL);
    return fb.QuadPart;
}

/* compresses a record into a Huff0 frame; returns the bytes of the
 * frame written to dst (capacity >= span + slack), 0 on error */
static size_t huf_frame(const unsigned char *rec, size_t span,
                        unsigned char *dst, size_t dst_cap) {
    unsigned count[256];
    unsigned maxSym = 255;
    if (HUF_isError(HIST_count(count, &maxSym, rec, span))) return 0;
    HUF_CElt ct[HUF_CTABLE_SIZE_ST(255)];
    static unsigned char wksp[HUF_CTABLE_WORKSPACE_SIZE + HUF_WORKSPACE_SIZE];
    size_t nbits = HUF_buildCTable_wksp(ct, count, maxSym, 11, wksp, sizeof(wksp));
    if (HUF_isError(nbits)) return 0;
    unsigned char hdr[512];
    const size_t hdr_sz = HUF_writeCTable_wksp(hdr, sizeof(hdr), ct, maxSym,
                                               (unsigned)nbits, wksp, sizeof(wksp));
    if (HUF_isError(hdr_sz)) return 0;
    const uint32_t nblk = (uint32_t)((span + BLK - 1) / BLK);
    unsigned char *p = dst;
    if (dst_cap < 4 + hdr_sz + 4 + 4ull * nblk) return 0;
    *(uint32_t *)p = (uint32_t)hdr_sz; p += 4;
    memcpy(p, hdr, hdr_sz); p += hdr_sz;
    *(uint32_t *)p = nblk; p += 4;
    uint32_t *csz = (uint32_t *)p; p += 4ull * nblk;
    for (uint32_t b = 0; b < nblk; b++) {
        const size_t off = (size_t)b * BLK;
        const size_t len = off + BLK <= span ? BLK : span - off;
        if ((size_t)(p - dst) + len + 64 > dst_cap) return 0;
        size_t c = HUF_compress4X_usingCTable(p, len - 1, rec + off, len, ct, 0);
        if (HUF_isError(c) || c == 0 || c >= len) {
            memcpy(p, rec + off, len);      /* raw fallback */
            csz[b] = RAWBIT | (uint32_t)len;
            p += len;
        } else {
            csz[b] = (uint32_t)c;
            p += c;
        }
    }
    return (size_t)(p - dst);
}

/* decodes a Huff0 frame (for VERIFICATION, same code as the engine) */
static int huf_unframe(const unsigned char *fr, size_t fr_sz,
                       unsigned char *out, size_t span) {
    const unsigned char *p = fr, *end = fr + fr_sz;
    if (p + 4 > end) return 0;
    const uint32_t hdr_sz = *(const uint32_t *)p; p += 4;
    if (p + hdr_sz + 4 > end) return 0;
    HUF_CREATE_STATIC_DTABLEX1(dt, HUF_TABLELOG_MAX);
    unsigned dwksp[HUF_DECOMPRESS_WORKSPACE_SIZE_U32];
    if (HUF_isError(HUF_readDTableX1_wksp(dt, p, hdr_sz, dwksp, sizeof(dwksp),
                                          HUF_flags_bmi2))) return 0;
    p += hdr_sz;
    const uint32_t nblk = *(const uint32_t *)p; p += 4;
    if (nblk != (span + BLK - 1) / BLK) return 0;
    const uint32_t *csz = (const uint32_t *)p; p += 4ull * nblk;
    size_t off = 0;
    for (uint32_t b = 0; b < nblk; b++) {
        const size_t len = off + BLK <= span ? BLK : span - off;
        const uint32_t c = csz[b];
        if (c & RAWBIT) {
            if ((c & ~RAWBIT) != len || p + len > end) return 0;
            memcpy(out + off, p, len);
            p += len;
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

/* minimal manifest: binary file <dst>.rezman —
 * [u64 done_from][u64 z_end][u64 n_frames] then per converted frame
 * (ORIGINAL index j): u64 j, u64 hoff, u64 hlen. Rewritten atomically. */
typedef struct { uint64_t j, hoff, hlen; } rez_ent;

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: forge_rez src.forgez dst [--keep]\n"); return 1; }
    const char *SRC = argv[1], *DST = argv[2];
    const int keep = argc > 3 && strcmp(argv[3], "--keep") == 0;
    char manp[1024];
    snprintf(manp, sizeof(manp), "%s.rezman", DST);

    /* read the forgez's json for the header + the frame list from the
     * forgez's json manifest (zoff,zlen per index) — C doesn't parse
     * json: that part is done by tools/forge_rez_idx.py, which
     * PREPARES a binary file <src>.frames: [u64 n][per frame: u64
     * j,zoff,zlen,span] sorted by INCREASING zoff. */
    char framesp[1024];
    snprintf(framesp, sizeof(framesp), "%s.frames", SRC);
    FILE *ff = fopen(framesp, "rb");
    if (!ff) { fprintf(stderr, "missing %s (generate with forge_rez_idx.py --prep)\n", framesp); return 1; }
    uint64_t nfr = 0, json_end = 0;
    fread(&json_end, 8, 1, ff);
    fread(&nfr, 8, 1, ff);
    typedef struct { uint64_t j, zoff, zlen, span; } fent;
    fent *fe = malloc(nfr * sizeof(fent));
    if (fread(fe, sizeof(fent), nfr, ff) != nfr) { fprintf(stderr, "frames file truncated\n"); return 1; }
    fclose(ff);

    /* resume state */
    uint64_t done_from = nfr;   /* first index (in fe), from the tail, NOT yet converted */
    uint64_t h_end = 0;
    rez_ent *ents = calloc(nfr, sizeof(rez_ent));
    uint64_t n_ents = 0;
    FILE *mf = fopen(manp, "rb");
    if (mf) {
        fread(&done_from, 8, 1, mf);
        fread(&h_end, 8, 1, mf);
        fread(&n_ents, 8, 1, mf);
        fread(ents, sizeof(rez_ent), n_ents, mf);
        fclose(mf);
        printf("resumed: done_from=%llu h_end=%.1fGB ents=%llu\n",
               (unsigned long long)done_from, h_end / 1e9,
               (unsigned long long)n_ents);
    }

    FILE *src = fopen(SRC, "rb");
    FILE *dst = fopen(DST, h_end ? "r+b" : "wb");
    if (!src || !dst) { fprintf(stderr, "open fail\n"); return 1; }
    if (!h_end) {
        /* head of the .forgezh: VERBATIM copy of the forgez's head
         * (magic+json+dense tensors, everything below json_end, which
         * here = end of the 4096-aligned dense zone) — the magic stays
         * NFRZ: the loader tells it apart from the v4 IDX by that,
         * not by the magic. */
        unsigned char *buf = malloc(64 << 20);
        uint64_t left = json_end;
        while (left) {
            const size_t chunk = left > (64u << 20) ? (64u << 20) : (size_t)left;
            fread(buf, 1, chunk, src);
            fwrite(buf, 1, chunk, dst);
            left -= chunk;
        }
        free(buf);
        fflush(dst);
        h_end = json_end;
        printf("head copied: %.1fGB\n", json_end / 1e9);
    }

    const size_t span_max = 32u << 20;
    unsigned char *rec = malloc(span_max), *zbuf = malloc(span_max);
    unsigned char *hbuf = malloc(span_max + (8u << 20));
    unsigned char *vbuf = malloc(span_max);

    /* from the tail: fe is sorted by increasing zoff; conversion goes
     * from the last unconverted index down to 0, truncating the
     * forgez as the margin runs low. */
    uint64_t z_trunc = file_size(SRC);
    uint64_t iter = 0;
    while (done_from > 0) {
        const fent *E = &fe[done_from - 1];
        _fseeki64(src, (long long)E->zoff, SEEK_SET);
        fread(zbuf, 1, E->zlen, src);
        const size_t un = ZSTD_decompress(rec, E->span, zbuf, E->zlen);
        if (ZSTD_isError(un) || un != E->span) {
            fprintf(stderr, "zstd err frame %llu\n", (unsigned long long)E->j);
            return 1;
        }
        const size_t hlen = huf_frame(rec, E->span, hbuf, span_max + (8u << 20));
        if (!hlen || !huf_unframe(hbuf, hlen, vbuf, E->span) ||
            memcmp(vbuf, rec, E->span) != 0) {
            fprintf(stderr, "HUF VERIFICATION FAILED frame %llu — stopping, source untouched\n",
                    (unsigned long long)E->j);
            return 1;
        }
        const uint64_t hoff = AL4K(h_end);
        _fseeki64(dst, (long long)hoff, SEEK_SET);
        fwrite(hbuf, 1, hlen, dst);
        h_end = hoff + hlen;
        ents[n_ents].j = E->j; ents[n_ents].hoff = hoff; ents[n_ents].hlen = hlen;
        n_ents++;
        done_from--;
        iter++;

        const int low = disk_free(DST) < 12ull << 30;
        if (low || done_from == 0 || (iter % 2000) == 0) {
            fflush(dst);
            FlushFileBuffers((HANDLE)_get_osfhandle(_fileno(dst)));
            /* atomic manifest BEFORE truncating */
            char tmp[1060];
            snprintf(tmp, sizeof(tmp), "%s.tmp", manp);
            FILE *m = fopen(tmp, "wb");
            fwrite(&done_from, 8, 1, m);
            fwrite(&h_end, 8, 1, m);
            fwrite(&n_ents, 8, 1, m);
            fwrite(ents, sizeof(rez_ent), n_ents, m);
            fflush(m); fclose(m);
            remove(manp); rename(tmp, manp);
            if (low && !keep) {
                fclose(src);
                HANDLE h = CreateFileA(SRC, GENERIC_WRITE, 0, NULL,
                                       OPEN_EXISTING, 0, NULL);
                LARGE_INTEGER li; li.QuadPart = (LONGLONG)E->zoff;
                SetFilePointerEx(h, li, NULL, FILE_BEGIN);
                SetEndOfFile(h);
                CloseHandle(h);
                z_trunc = E->zoff;
                src = fopen(SRC, "rb");
                printf("  truncated forgez to %.1fGB, %.1fGB free\n",
                       z_trunc / 1e9, disk_free(DST) / 1e9);
            }
            printf("  %llu/%llu frames, out %.1fGB, %.1fGB free\n",
                   (unsigned long long)(nfr - done_from),
                   (unsigned long long)nfr, h_end / 1e9, disk_free(DST) / 1e9);
            fflush(stdout);
        }
    }
    fclose(dst); fclose(src);
    printf("DONE: %llu frames, %s=%.1fGB — generate the v4 idx with "
           "forge_rez_idx.py\n", (unsigned long long)n_ents, DST, h_end / 1e9);
    return 0;
}
