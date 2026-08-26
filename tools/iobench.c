/* iobench.c — raw I/O benchmark for the model disk (standalone, in the
 * style of evict_sim).
 *
 * Question it answers: are the ~580MB/s seen from the pool the PHYSICAL
 * ceiling of the disk, or the ceiling of the pool's PARAMETERS (6
 * synchronous threads, ~3.4MB reads per tensor)? Read-size x queue-depth
 * sweep with NO_BUFFERING + OVERLAPPED on random offsets of the real
 * file (representative pattern: an expert's 3 tensors are slices of
 * large tensors far apart from each other), plus one sequential pass
 * for the absolute ceiling.
 *
 * Usage: iobench <file> [seconds-per-combo]
 */
#include <windows.h>
#include <process.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint64_t xr64(void) {
    uint64_t x = g_rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    g_rng = x; return x;
}

static double now_sec(void) {
    static LARGE_INTEGER f; LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

#define MAX_QD 64

/* one combo: reads of `rsz` bytes at `qd` depth, for `dur` seconds.
 * seq!=0: sequential offsets (from a random base), otherwise random. */
static double run_combo(HANDLE fh, uint64_t fsize, uint32_t rsz, int qd, double dur, int seq) {
    unsigned char *buf[MAX_QD];
    OVERLAPPED ov[MAX_QD];
    HANDLE ev[MAX_QD];
    uint64_t span = fsize - rsz;
    uint64_t seqoff = (xr64() % span) & ~4095ull;
    for (int i = 0; i < qd; i++) {
        buf[i] = (unsigned char *)VirtualAlloc(NULL, rsz, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        ev[i] = CreateEventA(NULL, TRUE, FALSE, NULL);
        if (!buf[i] || !ev[i]) { fprintf(stderr, "alloc failed\n"); exit(1); }
    }
    uint64_t bytes = 0;
    double t0 = now_sec();
    for (int i = 0; i < qd; i++) {
        uint64_t off;
        if (seq) { off = seqoff; seqoff += rsz; if (seqoff >= span) seqoff = 0; }
        else off = (xr64() % span) & ~4095ull;
        memset(&ov[i], 0, sizeof(ov[i]));
        ov[i].Offset = (DWORD)(off & 0xFFFFFFFFu);
        ov[i].OffsetHigh = (DWORD)(off >> 32);
        ov[i].hEvent = ev[i];
        if (!ReadFile(fh, buf[i], rsz, NULL, &ov[i]) && GetLastError() != ERROR_IO_PENDING) {
            fprintf(stderr, "ReadFile err=%lu\n", GetLastError()); exit(1);
        }
    }
    int inflight = qd;
    for (;;) {
        DWORD w = WaitForMultipleObjects((DWORD)qd, ev, FALSE, INFINITE);
        int i = (int)(w - WAIT_OBJECT_0);
        DWORD got = 0;
        if (!GetOverlappedResult(fh, &ov[i], &got, TRUE)) {
            fprintf(stderr, "GetOverlappedResult err=%lu\n", GetLastError()); exit(1);
        }
        bytes += got;
        ResetEvent(ev[i]);
        inflight--;
        if (now_sec() - t0 < dur) {
            uint64_t off;
            if (seq) { off = seqoff; seqoff += rsz; if (seqoff >= span) seqoff = 0; }
            else off = (xr64() % span) & ~4095ull;
            memset(&ov[i], 0, sizeof(ov[i]));
            ov[i].Offset = (DWORD)(off & 0xFFFFFFFFu);
            ov[i].OffsetHigh = (DWORD)(off >> 32);
            ov[i].hEvent = ev[i];
            if (!ReadFile(fh, buf[i], rsz, NULL, &ov[i]) && GetLastError() != ERROR_IO_PENDING) {
                fprintf(stderr, "ReadFile err=%lu\n", GetLastError()); exit(1);
            }
            inflight++;
        }
        if (!inflight) break;
    }
    double el = now_sec() - t0;
    for (int i = 0; i < qd; i++) { VirtualFree(buf[i], 0, MEM_RELEASE); CloseHandle(ev[i]); }
    return (double)bytes / el / (1024.0 * 1024.0);
}

/* burn mode — N threads saturating RAM bandwidth with FMA on large
 * arrays (the regime of the decode's AVX2 kernels), to measure how
 * much RAM contention steals from the USB DMA. iobench <file> <sec>
 * burn=<n> */
static volatile int g_burn_stop = 0;
static unsigned __stdcall burn_fn(void *arg) {
    (void)arg;
    const size_t N = 16u << 20;   /* 64MB of floats: outside any cache */
    float *a = (float *)malloc(N * sizeof(float));
    float *b = (float *)malloc(N * sizeof(float));
    if (!a || !b) return 0;
    for (size_t i = 0; i < N; i++) { a[i] = (float)(i & 1023); b[i] = 1.0f; }
    float acc = 0.0f;
    while (!g_burn_stop) {
        for (size_t i = 0; i < N; i++) acc += a[i] * b[i];
        b[0] = acc * 1e-30f;
    }
    free(a); free(b);
    return 0;
}

/* mode=sync — EXACT replica of the engine's pattern: N threads, one
 * NO_BUFFERING handle (without OVERLAPPED) each, SYNCHRONOUS ReadFile
 * on random 4096-aligned offsets. map=1: keeps a MapViewOfFile open on
 * the file during the sweep (the engine has the dense tensors mmapped
 * from the same file while it reads the experts). */
typedef struct {
    const char *path;
    uint32_t rsz;
    double dur;
    uint64_t fsize;
    volatile LONG64 *bytes;
} SyncArg;

static unsigned __stdcall sync_worker(void *p) {
    SyncArg *a = (SyncArg *)p;
    HANDLE fh = CreateFileA(a->path, GENERIC_READ,
                            FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                            OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, NULL);
    if (fh == INVALID_HANDLE_VALUE) return 0;
    unsigned char *buf = (unsigned char *)VirtualAlloc(NULL, a->rsz,
        MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    uint64_t span = a->fsize - a->rsz;
    double t0 = now_sec();
    while (now_sec() - t0 < a->dur) {
        uint64_t off = (xr64() % span) & ~4095ull;
        LARGE_INTEGER li; li.QuadPart = (LONGLONG)off;
        OVERLAPPED ov; memset(&ov, 0, sizeof(ov));
        ov.Offset = li.LowPart; ov.OffsetHigh = (DWORD)li.HighPart;
        DWORD got = 0;
        if (ReadFile(fh, buf, a->rsz, &got, &ov))
            InterlockedAdd64(a->bytes, got);
    }
    VirtualFree(buf, 0, MEM_RELEASE);
    CloseHandle(fh);
    return 0;
}

static void run_sync(const char *path, uint64_t fsize, double dur, int do_map) {
    HANDLE mfh = INVALID_HANDLE_VALUE, mh = NULL;
    void *view = NULL;
    if (do_map) {
        mfh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                          NULL, OPEN_EXISTING, 0, NULL);
        mh = CreateFileMappingA(mfh, NULL, PAGE_READONLY, 0, 0, NULL);
        view = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
        printf("map: mmap view active (%p)\n", view);
        /* touch a few pages, as the engine would with the dense tensors */
        volatile unsigned char sink = 0;
        for (uint64_t o = 0; view && o < (64ull << 20); o += 4096)
            sink += ((const unsigned char *)view)[o];
        (void)sink;
    }
    static const uint32_t szs[] = { 3584u * 1024, 10u << 20 };
    static const int ths[] = { 1, 6, 10 };
    printf("== SYNC per-thread (engine style)%s ==\n",
           do_map ? " + MMAP OPEN" : "");
    printf("   size\\th ");
    for (size_t t = 0; t < sizeof(ths)/sizeof(ths[0]); t++) printf("%8d", ths[t]);
    printf("\n");
    for (size_t si = 0; si < sizeof(szs)/sizeof(szs[0]); si++) {
        printf("   %5.1fM ", (double)szs[si] / (1u << 20));
        for (size_t t = 0; t < sizeof(ths)/sizeof(ths[0]); t++) {
            volatile LONG64 bytes = 0;
            SyncArg a = { path, szs[si], dur, fsize, &bytes };
            HANDLE hs[16];
            for (int i = 0; i < ths[t]; i++)
                hs[i] = (HANDLE)_beginthreadex(NULL, 0, sync_worker, &a, 0, NULL);
            WaitForMultipleObjects((DWORD)ths[t], hs, TRUE, INFINITE);
            for (int i = 0; i < ths[t]; i++) CloseHandle(hs[i]);
            printf("%8.0f", (double)bytes / dur / (1024.0 * 1024.0));
            fflush(stdout);
        }
        printf("\n");
    }
    if (view) UnmapViewOfFile(view);
    if (mh) CloseHandle(mh);
    if (mfh != INVALID_HANDLE_VALUE) CloseHandle(mfh);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: iobench <file> [sec-per-combo]\n"); return 1; }
    double dur = (argc > 2) ? atof(argv[2]) : 4.0;
    HANDLE fh = CreateFileA(argv[1], GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, NULL);
    if (fh == INVALID_HANDLE_VALUE) { fprintf(stderr, "open failed err=%lu\n", GetLastError()); return 1; }
    LARGE_INTEGER sz; GetFileSizeEx(fh, &sz);
    int burn = 0;
    for (int i = 1; i < argc; i++)
        if (sscanf(argv[i], "burn=%d", &burn) == 1) break;
    HANDLE bh[16];
    if (burn > 16) burn = 16;
    for (int i = 0; i < burn; i++)
        bh[i] = (HANDLE)_beginthreadex(NULL, 0, burn_fn, NULL, 0, NULL);
    if (burn) printf("burn: %d FMA threads on 128MB each\n", burn);
    printf("file: %s (%.1f GB), %.1fs per combo\n", argv[1], (double)sz.QuadPart / (1u << 30), dur);
    int wantsync = 0, wantmap = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "mode=sync") == 0) wantsync = 1;
        if (strcmp(argv[i], "map=1") == 0) wantmap = 1;
    }
    if (wantsync) {
        run_sync(argv[1], (uint64_t)sz.QuadPart, dur, wantmap);
        return 0;
    }

    static const uint32_t sizes[] = { 3584u * 1024, 10u << 20, 20u << 20, 40u << 20 };
    static const int qds[] = { 1, 2, 4, 6, 8, 12, 16, 32 };

    printf("\n== RANDOM reads (expert pattern) ==\n");
    printf("%10s", "size\\qd");
    for (int q = 0; q < (int)(sizeof(qds)/sizeof(*qds)); q++) printf("%8d", qds[q]);
    printf("\n");
    for (int s = 0; s < (int)(sizeof(sizes)/sizeof(*sizes)); s++) {
        printf("%9.1fM", sizes[s] / 1048576.0);
        for (int q = 0; q < (int)(sizeof(qds)/sizeof(*qds)); q++) {
            double mbs = run_combo(fh, (uint64_t)sz.QuadPart, sizes[s], qds[q], dur, 0);
            printf("%8.0f", mbs);
            fflush(stdout);
        }
        printf("\n");
    }

    printf("\n== SEQUENTIAL (absolute ceiling) ==\n");
    for (int q = 0; q < (int)(sizeof(qds)/sizeof(*qds)); q++) {
        double mbs = run_combo(fh, (uint64_t)sz.QuadPart, 10u << 20, qds[q], dur, 1);
        printf("  10.0M qd=%-2d  %8.0f MB/s\n", qds[q], mbs);
        fflush(stdout);
    }
    CloseHandle(fh);
    return 0;
}
