/* nf_os.h — Windows/POSIX portability layer (Linux port project).
 * Philosophy: THIN shims, identical semantics, zero
 * gratuitous abstraction. The Windows branch must produce binaries
 * BIT-IDENTICAL to before this header was introduced (tiny gate after
 * every port phase).
 *
 * Covers: monotonic time, sleep, 64-bit atomics, aligned alloc, env,
 * 64-bit seek, thread + priority. The QUEUE (mutex/cond) lives in
 * nf_queue.h with its own POSIX branch; direct I/O (O_DIRECT vs
 * NO_BUFFERING) lives at the call sites in nf_model.c because the
 * concurrency model differs (OVERLAPPED pipeline on Windows, blocking
 * per-thread pread on POSIX). */
#ifndef NF_OS_H
#define NF_OS_H

#ifdef _WIN32
/* ---------------------------------------------------------- Windows */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN   /* no winsock1: nf.c includes winsock2 */
#endif
#include <windows.h>
#include <process.h>
#include <malloc.h>

typedef HANDLE nf_thread_t;
typedef unsigned (__stdcall *nf_thread_fn)(void *);
#define NF_THREAD_RET unsigned __stdcall

static inline nf_thread_t nf_thread_start(nf_thread_fn fn, void *arg) {
    return (nf_thread_t)_beginthreadex(NULL, 0, fn, arg, 0, NULL);
}
static inline void nf_thread_join(nf_thread_t t) {
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
}
static inline void nf_thread_detach(nf_thread_t t) { CloseHandle(t); }

#define nf_atomic_add64(p, v)  InterlockedAdd64((volatile LONG64 *)(p), (LONG64)(v))
#define nf_atomic_inc64(p)     InterlockedIncrement64((volatile LONG64 *)(p))
#define nf_atomic_inc_long(p)    InterlockedIncrement((volatile LONG *)(p))
#define nf_atomic_dec_long(p)    InterlockedDecrement((volatile LONG *)(p))
#define nf_atomic_xchg_long(p,v) InterlockedExchange((volatile LONG *)(p), (LONG)(v))
#define nf_atomic_load_long(p)   InterlockedCompareExchange((volatile LONG *)(p), 0, 0)

#define nf_aligned_alloc(sz, al) _aligned_malloc((sz), (al))
#define nf_aligned_free(p)       _aligned_free(p)

static inline double nf_now_sec(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#define nf_sleep_ms(ms) Sleep(ms)
#define nf_yield()      Sleep(0)

static inline int nf_setenv(const char *kv) { return _putenv(kv); }
#define nf_fseek64 _fseeki64

/* file handle for direct I/O (expert streaming). On Windows the code
 * uses CreateFile/ReadFile+OVERLAPPED directly at the call sites
 * (long-standing path, never perturbed); the typedef is only needed
 * for SHARED structures. */
typedef HANDLE nf_iofh;
#define NF_IOFH_INVALID INVALID_HANDLE_VALUE
static inline void nf_io_close_rw(nf_iofh fh) { CloseHandle(fh); }
static inline void nf_io_close(nf_iofh fh) { CloseHandle(fh); }
static inline long long nf_io_pread(nf_iofh fh, void *buf, size_t len,
                                    unsigned long long off) {
    OVERLAPPED ov; memset(&ov, 0, sizeof ov);
    ov.Offset = (DWORD)(off & 0xFFFFFFFFu);
    ov.OffsetHigh = (DWORD)(off >> 32);
    DWORD got = 0;
    if (!ReadFile(fh, buf, (DWORD)len, &got, &ov)) return -1;
    return (long long)got;
}

#else
/* ------------------------------------------------------------ POSIX */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef pthread_t nf_thread_t;
typedef void *(*nf_thread_fn)(void *);
#define NF_THREAD_RET void *

static inline nf_thread_t nf_thread_start(nf_thread_fn fn, void *arg) {
    pthread_t t;
    if (pthread_create(&t, NULL, fn, arg) != 0) return (pthread_t)0;
    return t;
}
static inline void nf_thread_join(nf_thread_t t) { pthread_join(t, NULL); }
static inline void nf_thread_detach(nf_thread_t t) { pthread_detach(t); }

#define nf_atomic_add64(p, v)  atomic_fetch_add((_Atomic long long *)(p), (long long)(v))
#define nf_atomic_inc64(p)     (atomic_fetch_add((_Atomic long long *)(p), 1) + 1)
#define nf_atomic_inc_long(p)    (atomic_fetch_add((_Atomic long *)(p), 1) + 1)
#define nf_atomic_dec_long(p)    (atomic_fetch_sub((_Atomic long *)(p), 1) - 1)
#define nf_atomic_xchg_long(p,v) atomic_exchange((_Atomic long *)(p), (long)(v))
#define nf_atomic_load_long(p)   atomic_load((_Atomic long *)(p))

static inline void *nf_aligned_alloc(size_t sz, size_t al) {
    void *p = NULL;
    if (posix_memalign(&p, al, sz) != 0) return NULL;
    return p;
}
#define nf_aligned_free(p) free(p)

static inline double nf_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static inline void nf_sleep_ms(unsigned ms) {
    struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}
#define nf_yield() sched_yield()

#include <strings.h>
#define _strnicmp strncasecmp
#define _stricmp  strcasecmp

/* _putenv("K=V") / _putenv("K=") -> setenv/unsetenv (internal copy) */
static inline int nf_setenv(const char *kv) {
    const char *eq = strchr(kv, '=');
    if (!eq) return -1;
    char key[128];
    size_t kl = (size_t)(eq - kv);
    if (kl >= sizeof(key)) return -1;
    memcpy(key, kv, kl); key[kl] = 0;
    if (eq[1] == 0) return unsetenv(key);
    return setenv(key, eq + 1, 1);
}
#define nf_fseek64 fseeko

/* POSIX direct I/O: open O_DIRECT (same 4KB alignment rules as
 * Windows' NO_BUFFERING — guaranteed by the .forge* format), blocking
 * per-thread pread (disk concurrency comes from the worker pool, as
 * designed for the port: no io_uring in v1). */
#include <fcntl.h>
#include <sys/types.h>

typedef int nf_iofh;
#define NF_IOFH_INVALID (-1)

#ifndef O_DIRECT
#define O_DIRECT 0   /* filesystem without O_DIRECT (e.g. WSL's 9p): buffered */
#endif

static inline nf_iofh nf_io_open_direct(const char *path) {
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) fd = open(path, O_RDONLY);   /* buffered fallback */
    return fd;
}
static inline void nf_io_close(nf_iofh fh) { if (fh >= 0) close(fh); }
static inline void nf_io_close_rw(nf_iofh fh) { if (fh >= 0) close(fh); }
static inline long long nf_io_pread(nf_iofh fh, void *buf, size_t len,
                                    unsigned long long off) {
    return (long long)pread(fh, buf, len, (off_t)off);
}

#endif /* _WIN32 */
#endif /* NF_OS_H */
