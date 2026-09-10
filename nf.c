/* nf.c — pulsarforge CLI entry point.
 *
 * Usage:
 *   nf inspect  <model.gguf> [--tensors]     M0: inspect the file
 *   nf tokenize <model.gguf> <text>          M1: tokenize a string
 */
#include "nf_gguf.h"
#include "nf_model.h"
#include "nf_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include "nf_os.h"
#include "nf_math.h"   /* platform shim (Linux port) */
#ifdef _OPENMP
#include <omp.h>
#endif

/* High-resolution, portable timer — clock() on Windows has too coarse
 * a granularity to measure a handful of tokens. */
#ifdef _WIN32
/* winsock2 BEFORE windows.h: windows.h without LEAN_AND_MEAN pulls in
 * the old winsock.h, which conflicts with winsock2 if included after
 * (M7). */
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <process.h>   /* _beginthreadex (concurrent server, M8) */
static double now_seconds(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
/* minimal Winsock compat: the serve code uses win names, here they become BSD */
typedef int SOCKET;
#define INVALID_SOCKET (-1)
#define closesocket(s) close(s)
#define SD_SEND SHUT_WR
static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}
#endif

/* M6: reads a line from stdin in UTF-8. On a real Windows console,
 * fgets would use the system codepage (the same argv problem already
 * hit in M1) — ReadConsoleW works around it by reading wide chars
 * directly and converting to UTF-8 ourselves. If stdin isn't a
 * console (redirected from a file/pipe, e.g. for the tests) we fall
 * back to fgets: in that case the bytes are already whatever the
 * caller wrote, no codepage in the way. Returns the length, or -1 at
 * EOF. */
#ifdef _WIN32
static int read_line_utf8(char *buf, int cap) {
    HANDLE hin = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode;
    if (hin != INVALID_HANDLE_VALUE && GetConsoleMode(hin, &mode)) {
        wchar_t wbuf[8192];
        DWORD nread = 0;
        if (!ReadConsoleW(hin, wbuf, 8191, &nread, NULL) || nread == 0) return -1;
        while (nread > 0 && (wbuf[nread - 1] == L'\n' || wbuf[nread - 1] == L'\r'))
            nread--;
        wbuf[nread] = 0;
        const int n = WideCharToMultiByte(CP_UTF8, 0, wbuf, (int)nread,
                                          buf, cap - 1, NULL, NULL);
        buf[n > 0 ? n : 0] = 0;
        return n > 0 ? n : 0;
    }
    if (!fgets(buf, cap, stdin)) return -1;
    int n = (int)strlen(buf);
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = 0;
    return n;
}
#else
static int read_line_utf8(char *buf, int cap) {
    if (!fgets(buf, cap, stdin)) return -1;
    int n = (int)strlen(buf);
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = 0;
    return n;
}
#endif

static int cmd_inspect(const char *path, int all_tensors) {
    printf("== pulsarforge: GGUF inspection ==\n");
    printf("file: %s\n\n-- metadata --\n", path);

    nf_gguf g;
    if (nf_gguf_open(&g, path, /*verbose=*/1)) return 1;

    printf("\n-- header --\n");
    printf("  GGUF version:   %u\n", g.version);
    printf("  size:           %.2f MB\n", (double)g.file_size / (1024 * 1024));
    printf("  metadata:       %" PRIu64 " keys\n", g.n_kv);
    printf("  tensors:        %" PRIu64 "\n", g.n_tensors);
    printf("  alignment:      %u\n", g.alignment);
    printf("  data start:     0x%" PRIx64 "\n", g.data_offset);

    printf("\n-- tensors --\n");
    const uint64_t shown = all_tensors ? g.n_tensors
                         : (g.n_tensors < 12 ? g.n_tensors : 12);
    uint64_t tot_elems = 0, tot_bytes = 0;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        const nf_tensor_info *t = &g.tensors[i];
        tot_elems += t->n_elements;
        tot_bytes += t->n_bytes;
        if (i >= shown) continue;

        printf("  %-40s %-8s [", t->name, nf_ggml_type_name(t->type));
        for (uint32_t d = 0; d < t->n_dims; d++)
            printf("%s%" PRIu64, d ? " x " : "", t->dims[d]);
        printf("]  %.2f MB\n", (double)t->n_bytes / (1024 * 1024));
    }
    if (shown < g.n_tensors)
        printf("  ... and %" PRIu64 " more (use --tensors for the full list)\n",
               g.n_tensors - shown);

    printf("\n-- totals --\n");
    printf("  parameters:     %.2f M (%" PRIu64 ")\n",
           (double)tot_elems / 1e6, tot_elems);
    printf("  tensor weight:  %.2f MB\n", (double)tot_bytes / (1024 * 1024));
    if (tot_elems)
        printf("  bits per weight: %.2f\n",
               (double)tot_bytes * 8 / (double)tot_elems);

    nf_gguf_free(&g);
    return 0;
}

/* Reads a whole file into a NUL-terminated buffer (owned by the
 * caller). On Windows argv arrives in the system codepage, not UTF-8:
 * for non-ASCII text the only reliable channel is a file. */
static char *read_whole_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "nf: cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = n >= 0 ? malloc((size_t)n + 1) : NULL;
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[n] = 0;
    fclose(f);
    return buf;
}

static int cmd_tokenize(const char *path, const char *text) {
    nf_tokenizer *t = nf_tokenizer_load(path);
    if (!t) return 1;

    enum { MAX_TOK = 65536 };
    int32_t *ids = malloc(MAX_TOK * sizeof(int32_t));
    if (!ids) { nf_tokenizer_free(t); return 1; }

    const int64_t n = nf_tokenize(t, text, ids, MAX_TOK);
    if (n < 0) {
        fprintf(stderr, "nf: tokenization failed\n");
        free(ids);
        nf_tokenizer_free(t);
        return 1;
    }
    for (int64_t i = 0; i < n && i < MAX_TOK; i++) {
        char piece[256];
        const int w = nf_token_decode(t, ids[i], piece, sizeof(piece) - 1);
        piece[w > 0 ? w : 0] = 0;
        printf("%d\t|%s|\n", ids[i], piece);
    }
    fprintf(stderr, "%" PRId64 " tokens (vocab %" PRIu64 ")\n",
            n, nf_tokenizer_vocab_size(t));

    free(ids);
    nf_tokenizer_free(t);
    return 0;
}

/* M2: tokenizes the prompt, runs the forward pass, and prints the
 * top-10 logits; with dump_path it writes all the logits as raw
 * float32 (for check_m2). */
static int cmd_logits(const char *path, const char *text,
                      const char *dump_path) {
    nf_tokenizer *tk = nf_tokenizer_load(path);
    if (!tk) return 1;

    int32_t ids[4096];
    const int64_t n = nf_tokenize(tk, text, ids, 4096);
    if (n <= 0 || n > 4096) {
        fprintf(stderr, "nf: prompt empty or too long\n");
        nf_tokenizer_free(tk);
        return 1;
    }
    fprintf(stderr, "nf: %" PRId64 " tokens\n", n);

    nf_model *m = nf_model_load(path);
    if (!m) { nf_tokenizer_free(tk); return 1; }

    const float *logits = nf_model_forward(m, ids, (int)n);
    if (!logits) { nf_model_free(m); nf_tokenizer_free(tk); return 1; }
    const int64_t vocab = (int64_t)nf_model_vocab(m);

    if (dump_path) {
        FILE *f = fopen(dump_path, "wb");
        if (!f || fwrite(logits, sizeof(float), vocab, f) != (size_t)vocab) {
            fprintf(stderr, "nf: failed to write %s\n", dump_path);
            if (f) fclose(f);
            nf_model_free(m);
            nf_tokenizer_free(tk);
            return 1;
        }
        fclose(f);
        fprintf(stderr, "nf: logits written to %s\n", dump_path);
    }

    /* top-10 for human inspection: what does the model "want to say"? */
    printf("-- top 10 --\n");
    char seen[10 * 256];
    (void)seen;
    float best_v[10];
    int64_t best_i[10];
    int n_best = 0;
    for (int64_t v = 0; v < vocab; v++) {
        int j = n_best;
        while (j > 0 && logits[v] > best_v[j - 1]) j--;
        if (j >= 10) continue;
        if (n_best < 10) n_best++;
        for (int k = n_best - 1; k > j; k--) {
            best_v[k] = best_v[k - 1];
            best_i[k] = best_i[k - 1];
        }
        best_v[j] = logits[v];
        best_i[j] = v;
    }
    for (int j = 0; j < n_best; j++) {
        char piece[256];
        const int w = nf_token_decode(tk, (int32_t)best_i[j], piece,
                                      sizeof(piece) - 1);
        piece[w > 0 ? w : 0] = 0;
        printf("  %8" PRId64 "  %9.4f  |%s|\n", best_i[j], best_v[j], piece);
    }

    nf_model_free(m);
    nf_tokenizer_free(tk);
    return 0;
}

/* glm-dsa (GLM-5.2): validates the naive oracle (nf_model_forward,
 * nf_forward_layer_glm) against an external HF reference (transformers
 * GlmMoeDsaForCausalLM) on a tiny synthetic model — see tools/
 * make_glm_tiny_gguf.py and colibri's own c/tools/make_glm_oracle.py.
 * Bypasses the tokenizer on purpose: the test model has an arbitrary
 * vocab_size=256 (no real vocabulary in the GGUF), the token ids
 * arrive ALREADY as CSV in argv. Also bypasses nf_session (the cached
 * path isn't extended to glm-dsa yet, out of scope for this phase):
 * calls nf_model_forward for EVERY prefix 1..n (teacher
 * forcing, same scheme as cmd_selfcheck above for qwen3/deepseek2),
 * prints the argmax per position and, with dump_path, writes all the
 * logits [n x vocab] as raw float32 (one row per position) for
 * comparison in Python (cosine + argmax) against the oracle.
 *
 * Hot-path phase (glm_eval_core in nf_model.c, dispatched from
 * nf_session_eval/nf_session_eval_all): below, the SAME comparison
 * that cmd_selfcheck does for qwen3/deepseek2 ("incremental cache ==
 * full forward" + "single-shot prefill == full forward") is added, but
 * bypassing the tokenizer for the same reason as above (the test GGUF
 * doesn't have a real one) — so this isn't a branch inside
 * cmd_selfcheck (which calls nf_tokenizer_load(path) unconditionally,
 * and would fail on this GGUF), but the SAME comparison logic, made
 * repeatable via `nf debugglm`: no external HF reference is needed
 * here, the naive oracle (already validated elsewhere against
 * transformers) serves as the reference. */
static int cmd_debugglm(const char *path, const char *csv,
                        const char *dump_path) {
    nf_model *m = nf_model_load(path);
    if (!m) return 1;

    int32_t ids[4096];
    int n = 0;
    const char *p = csv;
    while (*p && n < 4096) {
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        ids[n++] = (int32_t)v;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    if (n <= 0) {
        fprintf(stderr, "nf: no valid tokens in '%s' (expected CSV, e.g. "
                "3,14,159)\n", csv);
        nf_model_free(m);
        return 1;
    }

    const int64_t vocab = (int64_t)nf_model_vocab(m);
    FILE *df = NULL;
    if (dump_path) {
        df = fopen(dump_path, "wb");
        if (!df) {
            fprintf(stderr, "nf: cannot open %s\n", dump_path);
            nf_model_free(m);
            return 1;
        }
    }

    int rc = 0;
    float *full_logits = malloc((size_t)n * (size_t)vocab * sizeof(float));
    for (int k = 1; k <= n; k++) {
        const float *logits = nf_model_forward(m, ids, k);
        if (!logits) {
            fprintf(stderr, "nf: glm-dsa: forward failed at k=%d\n", k);
            rc = 1;
            break;
        }
        int64_t am = 0;
        for (int64_t i = 1; i < vocab; i++)
            if (logits[i] > logits[am]) am = i;
        printf("k=%2d  argmax=%-4" PRId64 "  logit=%.6f\n", k, am, logits[am]);
        if (full_logits)
            memcpy(full_logits + (size_t)(k - 1) * vocab, logits,
                   (size_t)vocab * sizeof(float));
        if (df && fwrite(logits, sizeof(float), (size_t)vocab, df) != (size_t)vocab) {
            fprintf(stderr, "nf: failed to write %s\n", dump_path);
            rc = 1;
            break;
        }
    }
    if (df) {
        fclose(df);
        if (!rc)
            fprintf(stderr, "nf: logits [%d x %" PRId64 "] written to %s\n",
                    n, vocab, dump_path);
    }

    /* Comparison 1: incremental cache (glm_eval_core, n_new=1 at every
     * step) against the full oracle just computed — same scheme as
     * cmd_selfcheck (k=1 reported only here, k>1 compared against
     * full_logits[k-1] already in hand, without calling
     * nf_model_forward again). */
    int inc_ok = 1, batch_ok = 1;
    if (!rc && full_logits) {
        /* OPTIONAL dump of the SESSION's logits (hot-path
         * glm_eval_core), env NF_DEBUG_DUMP_INC=<path>: used for
         * BIT-EXACT comparisons between two runs of the session path
         * (e.g. the equivalence gate NF_GLM_MISS_SKIP=always vs
         * NF_MOE_TOPK_EFF — the legacy --dump writes the ORACLE, which
         * those env vars don't touch the same way, and the printout
         * truncates to 6 decimals). Env absent: no file, unchanged
         * behavior. */
        FILE *incf = NULL;
        {
            const char *incp = getenv("NF_DEBUG_DUMP_INC");
            if (incp && *incp) {
                incf = fopen(incp, "wb");
                if (!incf)
                    fprintf(stderr, "nf: NF_DEBUG_DUMP_INC: cannot "
                            "open %s\n", incp);
            }
        }
        nf_session *si = nf_session_create(m, n + 8);
        if (!si) {
            fprintf(stderr, "nf: glm-dsa: nf_session_create (incremental) failed\n");
            inc_ok = 0;
        } else {
            /* with the dump active the loop does NOT stop at the first
             * FAIL against the oracle: whoever is dumping is comparing
             * TWO session runs against each other (the oracle can
             * legitimately diverge, e.g. NF_GLM_MISS_SKIP=always) and
             * wants the COMPLETE file — inc_ok stays tracked and the
             * printed verdict is honest. */
            for (int k = 1; k <= n && (inc_ok || incf); k++) {
                const float *inc = nf_session_eval(si, &ids[k - 1], 1);
                const float *full = full_logits + (size_t)(k - 1) * vocab;
                if (!inc) { inc_ok = 0; break; }
                if (incf)
                    fwrite(inc, sizeof(float), (size_t)vocab, incf);
                int64_t af = 0, ai = 0;
                float maxd = 0.0f;
                for (int64_t i = 0; i < vocab; i++) {
                    if (full[i] > full[af]) af = i;
                    if (inc[i] > inc[ai]) ai = i;
                    const float d = fabsf(full[i] - inc[i]);
                    if (d > maxd) maxd = d;
                }
                const int ok = af == ai && maxd < 1e-2f;
                if (!ok) inc_ok = 0;
                printf("[glm-session] k=%2d  argmax full=%-4" PRId64
                       " inc=%-4" PRId64 "  max|d|=%.6f  %s\n",
                       k, af, ai, (double)maxd, ok ? "OK" : "FAIL");
            }
            nf_session_free(si);
        }
        if (incf) fclose(incf);
        printf("%s\n", inc_ok
               ? "GLM SESSION OK: incremental cache == full forward"
               : "GLM SESSION FAILED");

        /* Comparison 2: single-shot prefill (n_new=n, this
         * architecture's equivalent of cmd_selfcheck's "batch" check —
         * here glm_eval_core handles it with an internal sequential
         * per-position loop, see the comment on glm_eval_core in
         * nf_model.c: a deliberate fallback, not a postponement). */
        nf_session *sb = nf_session_create(m, n + 8);
        if (!sb) {
            fprintf(stderr, "nf: glm-dsa: nf_session_create (batch) failed\n");
            batch_ok = 0;
        } else {
            const float *inc_n = nf_session_eval(sb, ids, n);
            const float *full_n = full_logits + (size_t)(n - 1) * vocab;
            if (!inc_n) {
                batch_ok = 0;
            } else {
                int64_t af = 0, ai = 0;
                float maxd = 0.0f;
                for (int64_t i = 0; i < vocab; i++) {
                    if (full_n[i] > full_n[af]) af = i;
                    if (inc_n[i] > inc_n[ai]) ai = i;
                    const float d = fabsf(full_n[i] - inc_n[i]);
                    if (d > maxd) maxd = d;
                }
                batch_ok = af == ai && maxd < 1e-2f;
                printf("[glm-batch] n_new=%d  argmax full=%-4" PRId64
                       " inc=%-4" PRId64 "  max|d|=%.6f  %s\n",
                       n, af, ai, (double)maxd, batch_ok ? "OK" : "FAIL");
            }
            nf_session_free(sb);
        }
        printf("%s\n", batch_ok
               ? "GLM BATCH OK: single-shot prefill == full forward"
               : "GLM BATCH FAILED");
        if (!inc_ok || !batch_ok) rc = 1;
    }
    free(full_logits);

    nf_route_profile_report();  /* no-op unless NF_GLM_ROUTE_PROFILE — a
                                  * sanity check of the instrumentation on
                                  * the tiny models (goes through
                                  * glm_eval_core via nf_session_eval
                                  * above, not the batch path). */
    nf_debug_timing_report();   /* no-op unless NF_DEBUG_TIMING (this
                                  * lets you see the glm-stream/glm-prefetch
                                  * counters on the tiny models, which
                                  * generate never reaches — stderr only,
                                  * the validated stdout output stays
                                  * unchanged). */
    nf_model_free(m);
    return rc;
}

/* Session persistence: round-trip nf_session_save/load
 * against a "live, never saved" session oracle — generalized to ANY
 * architecture (qwen3/qwen3moe/deepseek2/glm-dsa/olmoe), used both to
 * empirically reproduce the suspected security bug on glm-dsa/deepseek2
 * (K/V layout incompatible with the generic format) and, after the
 * fix, as a correctness test. Protocol: CSV of n tokens, split at
 * k = n/2. Session A evaluates the first k tokens, then ALL n (A
 * continues in memory, the oracle). Session A2 evaluates the same
 * first k tokens, saves to `path`, reloads into a NEW session B (never
 * seen the first k tokens except through the file), then evaluates
 * the remaining n-k tokens on B. If save/load are correct, B's logits
 * for every position k+1..n must match (same tolerance as
 * debugglm/selfcheck, 1e-2) A's at the same position. */
static int cmd_debugsession(const char *path, const char *csv,
                            const char *sess_path) {
    nf_model *m = nf_model_load(path);
    if (!m) return 1;

    int32_t ids[4096];
    int n = 0;
    const char *p = csv;
    while (*p && n < 4096) {
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        ids[n++] = (int32_t)v;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    if (n < 2) {
        fprintf(stderr, "nf: debugsession: need at least 2 CSV tokens\n");
        nf_model_free(m);
        return 1;
    }
    const int kk = n / 2;
    const int64_t vocab = (int64_t)nf_model_vocab(m);

    /* Oracle: session A, NEVER saved/reloaded, evaluates all n. */
    nf_session *A = nf_session_create(m, n + 8);
    if (!A) { fprintf(stderr, "nf: debugsession: nf_session_create A failed\n"); nf_model_free(m); return 1; }
    float *oracle = malloc((size_t)n * (size_t)vocab * sizeof(float));
    int ok = 1;
    for (int i = 0; ok && i < n; i++) {
        const float *lg = nf_session_eval(A, &ids[i], 1);
        if (!lg) { ok = 0; break; }
        memcpy(oracle + (size_t)i * vocab, lg, (size_t)vocab * sizeof(float));
    }
    nf_session_free(A);
    if (!ok) {
        fprintf(stderr, "nf: debugsession: oracle eval failed\n");
        free(oracle);
        nf_model_free(m);
        return 1;
    }

    /* A2: first kk tokens, then save. */
    nf_session *A2 = nf_session_create(m, n + 8);
    if (!A2) { fprintf(stderr, "nf: debugsession: nf_session_create A2 failed\n"); free(oracle); nf_model_free(m); return 1; }
    for (int i = 0; i < kk; i++) {
        if (!nf_session_eval(A2, &ids[i], 1)) {
            fprintf(stderr, "nf: debugsession: A2 eval failed at i=%d\n", i);
            nf_session_free(A2);
            free(oracle);
            nf_model_free(m);
            return 1;
        }
    }
    const int rc_save = nf_session_save(A2, ids, sess_path);
    nf_session_free(A2);
    printf("nf_session_save -> rc=%d (path=%s, kk=%d tokens)\n", rc_save, sess_path, kk);
    if (rc_save != 0) {
        fprintf(stderr, "nf: debugsession: nf_session_save failed\n");
        free(oracle);
        nf_model_free(m);
        return 1;
    }

    /* B: NEW session, reloads from file, continues with the remaining
     * n-kk tokens. */
    int32_t hist[4096];
    int n_hist = 0;
    nf_session *B = nf_session_load(m, sess_path, n + 8, hist, 4096, &n_hist);
    if (!B) {
        fprintf(stderr, "nf: debugsession: nf_session_load failed (NULL)\n");
        free(oracle);
        nf_model_free(m);
        return 1;
    }
    printf("nf_session_load -> OK (n_hist=%d, expected=%d)\n", n_hist, kk);
    int hist_ok = (n_hist == kk);
    for (int i = 0; hist_ok && i < kk; i++)
        if (hist[i] != ids[i]) hist_ok = 0;
    printf("hist round-trip: %s\n", hist_ok ? "OK" : "FAIL");

    int round_ok = 1;
    for (int i = kk; i < n; i++) {
        const float *lg = nf_session_eval(B, &ids[i], 1);
        if (!lg) {
            printf("nf_session_eval on B failed at i=%d\n", i);
            round_ok = 0;
            break;
        }
        const float *ref = oracle + (size_t)i * vocab;
        int64_t af = 0, ab = 0;
        float maxd = 0.0f;
        for (int64_t j = 0; j < vocab; j++) {
            if (ref[j] > ref[af]) af = j;
            if (lg[j] > lg[ab]) ab = j;
            const float d = fabsf(ref[j] - lg[j]);
            if (d > maxd) maxd = d;
        }
        const int pos_ok = af == ab && maxd < 1e-2f;
        if (!pos_ok) round_ok = 0;
        printf("[session-roundtrip] i=%2d  argmax oracle=%-4" PRId64
               " reloaded=%-4" PRId64 "  max|d|=%.6f  %s\n",
               i, af, ab, (double)maxd, pos_ok ? "OK" : "FAIL");
    }
    nf_session_free(B);
    free(oracle);
    nf_model_free(m);
    printf("%s\n", (round_ok && hist_ok)
           ? "SESSION ROUNDTRIP OK"
           : "SESSION ROUNDTRIP FAILED");
    return (round_ok && hist_ok) ? 0 : 1;
}

/* Load-only: a minimal counterpart to cmd_debugsession for
 * verifying rejections — NEVER saves, only attempts nf_session_load on
 * an already-existing file (created by a previous run, maybe with a
 * different architecture) and reports success/failure. Used to prove
 * the "v5 file loaded onto a non-glm model", "v3/v4 file loaded onto a
 * glm model", "deepseek2 model" cases without cmd_debugsession
 * overwriting the file under test with its own save. */
static int cmd_debugsessionloadonly(const char *path, const char *sess_path) {
    nf_model *m = nf_model_load(path);
    if (!m) return 1;
    int32_t hist[4096];
    int n_hist = 0;
    nf_session *s = nf_session_load(m, sess_path, 4096, hist, 4096, &n_hist);
    printf("nf_session_load -> %s (n_hist=%d)\n", s ? "OK" : "NULL", n_hist);
    if (s) nf_session_free(s);
    nf_model_free(m);
    return s ? 0 : 1;
}

/* Lightweight teacher-forced replay (point 0 of the KL
 * quality protocol for NF_GLM_MISS_SKIP): evaluates a token sequence TOKEN by
 * token (n_new=1, the glm_eval_core path where miss-skip acts) on a
 * NEW session and writes the full float32 logits of EVERY position to
 * the dump file (same format as NF_DEBUG_DUMP_INC: n_vocab floats per
 * position, positions in order). NO nf_model_forward oracle:
 * cmd_debugglm computes one per prefix (O(n^2) positions), impractical
 * on the real 202GB GLM-5.2 — here only replay+dump, O(n). The dump
 * comparison itself is done downstream by tools/kl_compare.c.
 *
 * Input: --file <text> (tokenized the same way as `nf generate
 * --file`) or --csv "1,2,3" for the tiny models, which don't have a
 * real vocabulary in the GGUF (same reason for the bypass in
 * cmd_debugglm). At the end of the run, prints n_positions and n_vocab
 * to stderr for the downstream tool. */
static int cmd_debugglmreplay(const char *path, const char *input,
                              int is_csv, const char *dump_path) {
    int32_t ids[4096];
    int n = 0;
    if (is_csv) {
        const char *p = input;
        while (*p && n < 4096) {
            char *end;
            long v = strtol(p, &end, 10);
            if (end == p) break;
            ids[n++] = (int32_t)v;
            p = end;
            while (*p == ',' || *p == ' ') p++;
        }
    } else {
        nf_tokenizer *tk = nf_tokenizer_load(path);
        if (!tk) return 1;
        const int64_t nt = nf_tokenize(tk, input, ids, 4096);
        nf_tokenizer_free(tk);
        if (nt <= 0 || nt > 4096) {
            fprintf(stderr, "nf: debugglmreplay: prompt empty or too "
                    "long\n");
            return 1;
        }
        n = (int)nt;
    }
    if (n <= 0) {
        fprintf(stderr, "nf: debugglmreplay: no input tokens\n");
        return 1;
    }

    nf_model *m = nf_model_load(path);
    if (!m) return 1;
    const int64_t vocab = (int64_t)nf_model_vocab(m);

    FILE *df = fopen(dump_path, "wb");
    if (!df) {
        fprintf(stderr, "nf: debugglmreplay: cannot open %s\n",
                dump_path);
        nf_model_free(m);
        return 1;
    }

    nf_session *s = nf_session_create(m, n + 8);
    if (!s) {
        fprintf(stderr, "nf: debugglmreplay: nf_session_create failed "
                "(out of memory?)\n");
        fclose(df);
        nf_model_free(m);
        return 1;
    }

    int rc = 0;
    for (int k = 1; k <= n; k++) {
        const float *lg = nf_session_eval(s, &ids[k - 1], 1);
        if (!lg) {
            fprintf(stderr, "nf: debugglmreplay: eval failed at k=%d\n", k);
            rc = 1;
            break;
        }
        if (fwrite(lg, sizeof(float), (size_t)vocab, df) != (size_t)vocab) {
            fprintf(stderr, "nf: debugglmreplay: failed to write %s\n",
                    dump_path);
            rc = 1;
            break;
        }
        int64_t am = 0;
        for (int64_t i = 1; i < vocab; i++)
            if (lg[i] > lg[am]) am = i;
        /* progress on stderr: on the real model every position costs
         * seconds, the log is the only sign of life for a background run */
        fprintf(stderr, "[glm-replay] k=%3d/%d  argmax=%" PRId64 "\n",
                k, n, am);
    }
    nf_session_free(s);
    fclose(df);
    if (!rc)
        fprintf(stderr, "nf: debugglmreplay: n_positions=%d n_vocab=%"
                PRId64 " -> %s\n", n, vocab, dump_path);
    nf_model_free(m);
    return rc;
}

/* Embedding extraction (for RAG-style retrieval use cases): tokenizes
 * the prompt, calls nf_model_embed on `layer` and prints/writes the
 * last token's raw hidden state — see the comment on nf_model_embed
 * in nf_model.h for why it doesn't normalize. */
static int cmd_embed(const char *path, const char *text, int layer,
                     const char *dump_path) {
    nf_tokenizer *tk = nf_tokenizer_load(path);
    if (!tk) return 1;

    int32_t ids[4096];
    const int64_t n = nf_tokenize(tk, text, ids, 4096);
    if (n <= 0 || n > 4096) {
        fprintf(stderr, "nf: prompt empty or too long\n");
        nf_tokenizer_free(tk);
        return 1;
    }
    fprintf(stderr, "nf: %" PRId64 " tokens\n", n);

    nf_model *m = nf_model_load(path);
    if (!m) { nf_tokenizer_free(tk); return 1; }

    const int E = nf_model_n_embd(m);
    float *out = malloc((size_t)E * sizeof(float));
    if (!out) { nf_model_free(m); nf_tokenizer_free(tk); return 1; }

    if (nf_model_embed(m, ids, (int)n, layer, out)) {
        fprintf(stderr, "nf: embed failed (layer %d out of range?)\n", layer);
        free(out);
        nf_model_free(m);
        nf_tokenizer_free(tk);
        return 1;
    }

    if (dump_path) {
        FILE *f = fopen(dump_path, "wb");
        if (!f || fwrite(out, sizeof(float), (size_t)E, f) != (size_t)E) {
            fprintf(stderr, "nf: failed to write %s\n", dump_path);
            if (f) fclose(f);
            free(out);
            nf_model_free(m);
            nf_tokenizer_free(tk);
            return 1;
        }
        fclose(f);
        fprintf(stderr, "nf: embedding written to %s\n", dump_path);
    } else {
        /* summary for human inspection: not authoritative, debug only */
        double norm2 = 0.0;
        for (int i = 0; i < E; i++) norm2 += (double)out[i] * (double)out[i];
        printf("layer=%d  dim=%d  l2_norm=%.4f\n", layer, E, sqrt(norm2));
        printf("first values: ");
        for (int i = 0; i < E && i < 8; i++) printf("%.4f ", out[i]);
        printf("\n");
    }

    free(out);
    nf_model_free(m);
    nf_tokenizer_free(tk);
    return 0;
}

/* ---- M3: sampling ------------------------------------------------------
 *
 * xorshift64* — a minimal, deterministic generator: seeding it with
 * --seed is enough for reproducible generations, a property we want by
 * default in a project that lives on validation against an oracle. */
static uint64_t xorshift64star(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}
static float rng_uniform01(uint64_t *state) {
    return (float)((xorshift64star(state) >> 11) * (1.0 / 9007199254740992.0));
}

typedef struct { float p; int32_t id; } nf_probitem;
static int cmp_probitem_desc(const void *a, const void *b) {
    const float pa = ((const nf_probitem *)a)->p, pb = ((const nf_probitem *)b)->p;
    return pa < pb ? 1 : (pa > pb ? -1 : 0);
}

/* temp <= 0: greedy (argmax, deterministic — this is what we compare
 * against the oracle). temp > 0: softmax with temperature + nucleus
 * (top-p): sorts tokens by decreasing probability, keeps the smallest
 * prefix whose cumulative probability reaches top_p, renormalizes and
 * samples only within it — discards the long improbable tail instead
 * of letting it compete. */
/* Repetition penalty (llama.cpp/CTRL-style): for every token that
 * appeared in the recent window, if its logit is positive it's divided
 * by `penalty`, otherwise multiplied — either way, made less
 * attractive. penalty=1.0 disables it, no effect. Why this exists:
 * discovered in M6 that without it a small model can get stuck in a
 * loop that never stops on its own (an "imaginary quiz" repeated
 * forever) — making literal repetition progressively less likely
 * pushes sampling out of the loop before the whole available context
 * is exhausted. */
static void apply_repeat_penalty(float *logits, int64_t vocab,
                                 const int32_t *recent, int n_recent,
                                 float penalty) {
    if (penalty == 1.0f) return;
    for (int i = 0; i < n_recent; i++) {
        const int32_t id = recent[i];
        if (id < 0 || id >= vocab) continue;
        float *l = &logits[id];
        *l = *l > 0.0f ? *l / penalty : *l * penalty;
    }
}

/* top_k (0 = disabled): keeps only the top_k entries by probability
 * before any other filter — the classic "only consider the K most
 * likely words", independent of the SHAPE of the distribution (unlike
 * top-p, which adapts to the tail but doesn't cap how many entries
 * compete at the top).
 * min_p (0 = disabled, llama.cpp-style): keeps only entries with
 * probability >= min_p * the most likely token's probability — a
 * threshold RELATIVE to the peak, so it self-adapts both to flat
 * distributions (few entries survive anyway, if the peak is low then
 * min_p*peak is low) and to peaked ones (only the near-winners
 * survive). Applied AFTER top_k, BEFORE top_p: the three filters
 * compose by progressively narrowing the same sorted list, never
 * re-sorted twice. */
static int32_t sample_token(const float *logits, int64_t n, float temp,
                            float top_p, int32_t top_k, float min_p,
                            uint64_t *rng) {
    if (temp <= 0.0f) {
        int64_t best = 0;
        for (int64_t i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return (int32_t)best;
    }

    nf_probitem *items = malloc((size_t)n * sizeof(nf_probitem));
    if (!items) return 0;
    float mx = logits[0];
    for (int64_t i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        const float e = nf_expf((logits[i] - mx) / temp);
        items[i].p = e;
        items[i].id = (int32_t)i;
        sum += e;
    }
    for (int64_t i = 0; i < n; i++) items[i].p = (float)(items[i].p / sum);
    qsort(items, (size_t)n, sizeof(nf_probitem), cmp_probitem_desc);

    int64_t limit = n;
    if (top_k > 0 && (int64_t)top_k < limit) limit = top_k;
    if (min_p > 0.0f) {
        const float thresh = items[0].p * min_p;
        int64_t i = 1; /* the top token always survives */
        while (i < limit && items[i].p >= thresh) i++;
        limit = i;
    }

    int64_t cutoff = limit;
    if (top_p < 1.0f) {
        double cum = 0.0;
        for (int64_t i = 0; i < limit; i++) {
            cum += items[i].p;
            if (cum >= (double)top_p) { cutoff = i + 1; break; }
        }
    }
    double renorm = 0.0;
    for (int64_t i = 0; i < cutoff; i++) renorm += items[i].p;

    const double r = rng_uniform01(rng) * renorm;
    double acc = 0.0;
    int32_t chosen = items[0].id;
    for (int64_t i = 0; i < cutoff; i++) {
        acc += items[i].p;
        if (acc >= r) { chosen = items[i].id; break; }
    }
    free(items);
    return chosen;
}

/* More than one token can mark "end": Qwen3 has both <|im_end|>
 * (eos_token_id, the turn-end marker) and <|endoftext|> (bos_token_id
 * in this GGUF, but also pretraining's generic end-of-text).
 * Empirically observed: with sampling the model can generate
 * <|endoftext|> instead of <|im_end|> to close a response — if only
 * the configured eos is checked, generation degenerates into a run-on
 * that prints the special token as literal text and keeps going
 * forever. Standard practice: stop on a SET of stop tokens, not just
 * the "official" one. */
/* Cap at 6 (was 4): glm-dsa alone already carries 3 "official" stop
 * ids (endoftext/user/observation, see chat_style_init) — with the
 * generic GGUF's eos_token_id and bos_token_id (below) already taking
 * up to 2 slots, 4 total would have risked saturating the set BEFORE
 * even adding user/observation (silently dropped by
 * stopset_add_special, never a crash, but incomplete coverage). Purely
 * a sizing margin, no other architecture has ever needed it so far. */
typedef struct { int32_t ids[6]; int n; } nf_stopset;

static void build_stopset(const char *path, nf_stopset *ss) {
    ss->n = 0;
    uint64_t v;
    if (!nf_gguf_read_kv_u64(path, "tokenizer.ggml.eos_token_id", &v))
        ss->ids[ss->n++] = (int32_t)v;
    if (!nf_gguf_read_kv_u64(path, "tokenizer.ggml.bos_token_id", &v)) {
        int dup = 0;
        for (int i = 0; i < ss->n; i++) if (ss->ids[i] == (int32_t)v) dup = 1;
        if (!dup) ss->ids[ss->n++] = (int32_t)v;
    }
}
static int is_stop_token(const nf_stopset *ss, int32_t id) {
    for (int i = 0; i < ss->n; i++) if (ss->ids[i] == id) return 1;
    return 0;
}

/* Adds a special token's id to the stopset by looking it up VIA THE
 * TOKENIZER (match_special recognizes it as an atomic entry), not via
 * metadata. A real bug found during the steering experiment: unsloth's
 * GGUFs don't have `tokenizer.ggml.bos_token_id`, so <|endoftext|>
 * (151643) never entered the stopset nor — as a consequence — the
 * textual filter, and a generation could run right through it without
 * stopping. The textual lookup works on any file in the family. */
static void stopset_add_special(const nf_tokenizer *tk, nf_stopset *ss,
                                const char *text) {
    int32_t ids[4];
    if (nf_tokenize(tk, text, ids, 4) != 1) return;  /* not atomic: skip */
    if (ss->n < 6 && !is_stop_token(ss, ids[0])) ss->ids[ss->n++] = ids[0];
}

/* Textual stop: is_stop_token trusts the model to SAMPLE the special
 * token's atomic id. Empirically observed, though, that a model can
 * instead WRITE that special token's text using ordinary BPE
 * sub-words (e.g. "<", "|", "endoftext", "|", ">" as five normal
 * tokens) — same final string, but no stop id is ever sampled, so
 * is_stop_token never fires and generation continues past what should
 * have been a closed turn. The stop strings are the LITERAL text of
 * the same ids already in nf_stopset (decoded with nf_token_decode,
 * which for special tokens copies the UTF-8 bytes as-is — see
 * nf_tokenizer.c): no separate list is needed, it's the same set of
 * "turn boundaries" seen as text instead of as ids. */
#define NF_MAX_TEXTSTOPS 4
#define NF_TEXTSTOP_LEN  32

typedef struct { char s[NF_MAX_TEXTSTOPS][NF_TEXTSTOP_LEN]; int n; } nf_textstop;

static void build_textstops(const nf_tokenizer *tk, const nf_stopset *ss, nf_textstop *ts) {
    ts->n = 0;
    for (int i = 0; i < ss->n && ts->n < NF_MAX_TEXTSTOPS; i++) {
        char piece[NF_TEXTSTOP_LEN];
        const int w = nf_token_decode(tk, ss->ids[i], piece, sizeof(piece) - 1);
        if (w <= 0) continue;
        piece[w] = 0;
        memcpy(ts->s[ts->n], piece, (size_t)w + 1);
        ts->n++;
    }
}

/* Looks for the first (leftmost) occurrence of any stop string inside
 * buf[0..buflen). Returns the start offset and the match length, or
 * -1 if no stop string is present. */
static int find_stop_match(const char *buf, int buflen, const nf_textstop *ts, int *match_len) {
    for (int start = 0; start < buflen; start++) {
        for (int i = 0; i < ts->n; i++) {
            const int slen = (int)strlen(ts->s[i]);
            if (slen > 0 && start + slen <= buflen &&
                memcmp(buf + start, ts->s[i], (size_t)slen) == 0) {
                *match_len = slen;
                return start;
            }
        }
    }
    return -1;
}

/* A stop's text can form across several BPE tokens, so a just-decoded
 * piece can't be printed immediately: the tail that might be the start
 * of a still-incomplete stop must be held in a buffer (`pending`) and
 * printed only once it's certain it's no longer one. Returns how many
 * bytes at the end of buf are a proper prefix of some stop string (so
 * must be held back), 0 if no tail is a suspect prefix. */
static int pending_stop_prefix_len(const char *buf, int buflen, const nf_textstop *ts) {
    int best = 0;
    for (int i = 0; i < ts->n; i++) {
        const int slen = (int)strlen(ts->s[i]);
        for (int k = (slen - 1 < buflen ? slen - 1 : buflen); k > 0; k--) {
            if (memcmp(buf + buflen - k, ts->s[i], (size_t)k) == 0) {
                if (k > best) best = k;
                break;
            }
        }
    }
    return best;
}

/* Core of the textual stop filter: appends `piece` to `pending`, writes
 * into `out` (caller's buffer, capacity >= pending_cap+plen) the part
 * that's now certainly safe, holds back the rest. Extracted from
 * stream_piece_through_stopfilter (M7): the chat/generate path prints
 * `out` to stdout as always, `nf serve` sends it as an SSE chunk — same
 * buffering logic, not two copies. Returns 1 if a stop string just
 * completed (out contains the output up to its start; the caller must
 * stop WITHOUT feeding anything else). */
static int stopfilter_feed(const char *piece, int plen,
                           char *pending, int *n_pending, int pending_cap,
                           const nf_textstop *ts,
                           char *out, int *n_out) {
    *n_out = 0;
    if (plen <= 0) return 0;
    if (*n_pending + plen >= pending_cap) {
        memcpy(out + *n_out, pending, (size_t)*n_pending);
        *n_out += *n_pending;
        *n_pending = 0;
    }
    memcpy(pending + *n_pending, piece, (size_t)plen);
    *n_pending += plen;

    int match_len;
    const int match_at = find_stop_match(pending, *n_pending, ts, &match_len);
    if (match_at >= 0) {
        memcpy(out + *n_out, pending, (size_t)match_at);
        *n_out += match_at;
        *n_pending = 0;
        return 1;
    }
    const int keep = pending_stop_prefix_len(pending, *n_pending, ts);
    const int flush_upto = *n_pending - keep;
    memcpy(out + *n_out, pending, (size_t)flush_upto);
    *n_out += flush_upto;
    memmove(pending, pending + flush_upto, (size_t)keep);
    *n_pending = keep;
    return 0;
}

/* Legacy stdout variant, shared by cmd_generate, cmd_chat, and the
 * self-test (nf debugstop): identical behavior to before the M7
 * refactor (flush on every call). */
static int stream_piece_through_stopfilter(const char *piece, int plen,
                                            char *pending, int *n_pending,
                                            int pending_cap,
                                            const nf_textstop *ts) {
    char out[1024];
    int n_out;
    const int r = stopfilter_feed(piece, plen, pending, n_pending,
                                  pending_cap, ts, out, &n_out);
    if (n_out > 0) fwrite(out, 1, (size_t)n_out, stdout);
    fflush(stdout);
    return r;
}

/* Self-test of the textual stop algorithm (nf debugstop A|B), no model
 * needed: we build an nf_textstop by hand with the two strings we care
 * about (exactly what build_textstops would produce for Qwen3) and
 * simulate a stream of "pieces" as if they came from nf_token_decode
 * one BPE token at a time.
 *
 * Case A: the special token <|endoftext|> arrives split across several
 * ordinary BPE pieces ("<", "|", "endoftext", "|", ">") — it must be
 * fully suppressed from the output, and the caller must stop
 * immediately (here: we don't feed the next piece "SHOULD_NOT_APPEAR").
 * Expected on stdout: "Hello world" (no trailing newline).
 *
 * Case B: a prefix that LOOKS like the start of a stop ("<|im") but
 * then diverges ("_middle" instead of "_end|>") must never block the
 * output forever: once it's clear it isn't a stop, all the held-back
 * text still gets printed. Expected: the exact concatenation of every
 * piece, "Hello <|im_middle|> world". */
static int cmd_debug_textstop(const char *which) {
    nf_textstop ts;
    ts.n = 0;
    strncpy(ts.s[ts.n++], "<|im_end|>", NF_TEXTSTOP_LEN - 1);
    strncpy(ts.s[ts.n++], "<|endoftext|>", NF_TEXTSTOP_LEN - 1);

    char pending[512];
    int n_pending = 0;

    if (which && strcmp(which, "A") == 0) {
        const char *pieces[] = { "Hello", " world", "<", "|", "endoftext", "|", ">" };
        for (size_t i = 0; i < sizeof(pieces) / sizeof(pieces[0]); i++) {
            if (stream_piece_through_stopfilter(pieces[i], (int)strlen(pieces[i]),
                                                pending, &n_pending, (int)sizeof(pending), &ts))
                break;
        }
    } else if (which && strcmp(which, "B") == 0) {
        const char *pieces[] = { "Hello", " <", "|", "im", "_middle", "|", ">", " world" };
        for (size_t i = 0; i < sizeof(pieces) / sizeof(pieces[0]); i++) {
            stream_piece_through_stopfilter(pieces[i], (int)strlen(pieces[i]),
                                            pending, &n_pending, (int)sizeof(pending), &ts);
        }
        if (n_pending > 0) fwrite(pending, 1, (size_t)n_pending, stdout);
    } else {
        fprintf(stderr, "nf debugstop A|B\n");
        return 1;
    }
    fflush(stdout);
    return 0;
}

/* DEBUG: if the NF_DUMP_INIT_LOGITS environment variable is set,
 * writes the current logits there (n_vocab floats) — used to compare
 * byte-for-byte the first logit state before sampling between
 * 'nf generate' and 'nf chat' on an equivalent prompt. Not a
 * dedicated CLI flag on purpose: it's a throwaway diagnostic tool,
 * same style as the dump environment variables seen in other engines. */
static void dump_init_logits_if_requested(const float *logits, int64_t vocab) {
    const char *path = getenv("NF_DUMP_INIT_LOGITS");
    if (!path) return;
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(logits, sizeof(float), (size_t)vocab, f);
    fclose(f);
    fprintf(stderr, "nf: [debug] initial logits written to %s\n", path);
}

/* Applies the repetition penalty on a "virtual" history: hist[0..len)
 * followed by extra[0..n_extra) (the draft's proposed tokens not yet
 * accepted). Used by speculative decoding to EXACTLY replicate the
 * normal path's penalty sequence (which applies the penalty before
 * sampling EVERY token, with the window including the whole history
 * up to that point). */
static void penalty_virtual(float *logits, int64_t vocab,
                            const int32_t *hist, int len,
                            const int32_t *extra, int n_extra,
                            float rp, int rln) {
    if (rp == 1.0f) return;
    int32_t buf[1024];
    int win = len + n_extra;
    if (win > rln) win = rln;
    if (win > 1024) win = 1024;
    const int from_extra = n_extra < win ? n_extra : win;
    const int from_hist = win - from_extra;
    memcpy(buf, hist + len - from_hist, (size_t)from_hist * sizeof(int32_t));
    memcpy(buf + from_hist, extra + n_extra - from_extra,
           (size_t)from_extra * sizeof(int32_t));
    apply_repeat_penalty(logits, vocab, buf, win, rp);
}

/* Speculative decoding (greedy): a small, fast DRAFT model proposes K
 * tokens; the TARGET verifies them all in ONE batched pass over the
 * weights (nf_session_eval_all) instead of K passes — this is
 * prefill's batched matmul applied to decode. If proposed token j
 * matches the target's argmax, it's EXACTLY the token the target would
 * have generated on its own: output is guaranteed identical to
 * non-speculative (this is why it requires greedy: sampling would need
 * probabilistic verification, not implemented). On rejection, the
 * target's correction is emitted and the cache is rewound
 * (nf_session_rewind). Gain = tokens accepted per pass: depends on how
 * well the draft "guesses" the target. */
static int spec_generate(nf_tokenizer *tk, nf_model *m, nf_session *s,
                         const nf_stopset *stop, const nf_textstop *tstop,
                         const char *draft_path, int K,
                         const int32_t *ids, int n, int n_predict,
                         int max_ctx, float top_p, uint64_t seed,
                         float repeat_penalty, int repeat_last_n) {
    nf_model *dm = nf_model_load(draft_path);
    if (!dm) return 1;
    if (nf_model_vocab(dm) != nf_model_vocab(m)) {
        fprintf(stderr, "nf: draft and target have different vocabularies\n");
        nf_model_free(dm);
        return 1;
    }
    nf_session *ds = nf_session_create(dm, max_ctx);
    const int64_t vocab = (int64_t)nf_model_vocab(m);
    float *va = malloc((size_t)(K + 2) * vocab * sizeof(float));
    int32_t *hist = malloc((size_t)max_ctx * sizeof(int32_t));
    int32_t *dtok = malloc((size_t)K * sizeof(int32_t));
    int32_t *batch = malloc((size_t)(K + 2) * sizeof(int32_t));
    if (!ds || !va || !hist || !dtok || !batch) {
        free(va); free(hist); free(dtok); free(batch);
        if (ds) nf_session_free(ds);
        nf_model_free(dm);
        return 1;
    }

    memcpy(hist, ids, (size_t)n * sizeof(int32_t));
    int len = n;          /* accepted tokens (prompt included)          */
    int t_past = 0;       /* positions consumed by the target           */
    int d_past = 0;       /* positions consumed by the draft            */

    /* target prefill over the whole prompt EXCEPT the last token: the
     * loop below always feeds "pending + proposals" in a single
     * eval_all, and the logits after the last pending one act as the
     * judge for the first proposal — so the round is uniform from the
     * very first one. */
    if (n > 1) {
        if (!nf_session_eval(s, hist, n - 1)) goto fail;
        t_past = n - 1;
    }

    fputs("", stdout);
    uint64_t rng = seed;
    int n_generated = 0, emitted = 0, stopped = 0;
    int rounds = 0, drafted = 0, accepted = 0;
    char pend[512];
    int n_pend_txt = 0;
    const double t0 = now_seconds();

    while (!stopped && emitted < n_predict) {
        if (len + K + 1 > max_ctx) break;

        /* 1) the draft proposes K greedy tokens from its own state */
        const float *dl = nf_session_eval(ds, hist + d_past, len - d_past);
        if (!dl) break;
        d_past = len;
        int nd = 0;
        int32_t top2[64]; /* PROBE: the draft's second choice */
        float conf[64];   /* PROBE (NF_DEBUG_SPEC_CONF=1): the
                             draft's confidence in its own argmax,
                             logged together with the verification
                             outcome — point 0 of adaptive gating: if it
                             doesn't predict acceptance, no gating. */
        static int dbg_conf = -1;
        if (dbg_conf < 0) dbg_conf = getenv("NF_DEBUG_SPEC_CONF") != NULL;
        /* GATING (NF_SPEC_PMIN=P): point 0 measured that the
         * draft's confidence predicts acceptance with a very steep
         * curve (9-18% below 0.3, 95-100% above 0.9): below the
         * threshold the proposal is discarded before costing an eval —
         * long K on predictable stretches, near zero on hard ones.
         * 0 = off. */
        static float pmin = -1.0f;
        if (pmin < 0.0f) {
            const char *e = getenv("NF_SPEC_PMIN");
            pmin = e ? (float)atof(e) : 0.0f;
            if (pmin < 0.0f) pmin = 0.0f;
        }
        for (int j = 0; j < K; j++) {
            penalty_virtual((float *)dl, vocab, hist, len, dtok, j,
                            repeat_penalty, repeat_last_n);
            dtok[j] = sample_token(dl, vocab, 0.0f, top_p, 0, 0.0f, &rng);
            if ((dbg_conf || pmin > 0.0f) && j < 64) {
                /* PROBE: also the draft's SECOND choice — on
                 * rejection, measures how often the target's correction
                 * was exactly it (point 0 of tree speculation: if it's
                 * often, branching is worth it). */
                float mx = dl[0], mx2 = -3.4e38f;
                int32_t id1 = 0, i2 = -1;
                for (int64_t v = 1; v < vocab; v++) {
                    if (dl[v] > mx) {
                        mx2 = mx; i2 = id1;
                        mx = dl[v]; id1 = (int32_t)v;
                    } else if (dl[v] > mx2) {
                        mx2 = dl[v]; i2 = (int32_t)v;
                    }
                }
                top2[j] = i2;
                double se = 0.0;
                for (int64_t v = 0; v < vocab; v++)
                    se += nf_exp((double)dl[v] - mx);
                conf[j] = (float)(nf_exp((double)dl[dtok[j]] - mx) / se);
                if (pmin > 0.0f && conf[j] < pmin)
                    break;   /* proposal discarded: round closes here */
            }
            nd++;
            dl = nf_session_eval(ds, &dtok[j], 1);
            if (!dl) break;
        }
        d_past += nd;
        drafted += nd;

        /* 2) the target verifies pending+proposals in one pass */
        const int n_pend = len - t_past;
        const int n_batch = n_pend + nd;
        memcpy(batch, hist + t_past, (size_t)n_pend * sizeof(int32_t));
        memcpy(batch + n_pend, dtok, (size_t)nd * sizeof(int32_t));
        if (!nf_session_eval_all(s, batch, n_batch, va)) break;
        t_past += n_batch;
        const int base = n_pend - 1;

        /* 3) sequential comparison: at the first disagreement, the
         * target's correction replaces the proposal */
        int n_acc = 0;
        int32_t fix = -1;
        for (int j = 0; j < nd; j++) {
            float *Lj = va + (size_t)(base + j) * vocab;
            penalty_virtual(Lj, vocab, hist, len, dtok, j,
                            repeat_penalty, repeat_last_n);
            const int32_t g = sample_token(Lj, vocab, 0.0f, top_p, 0, 0.0f, &rng);
            if (g == dtok[j]) { n_acc++; continue; }
            fix = g;
            break;
        }
        if (fix < 0) {
            float *Lk = va + (size_t)(base + nd) * vocab;
            penalty_virtual(Lk, vocab, hist, len, dtok, nd,
                            repeat_penalty, repeat_last_n);
            fix = sample_token(Lk, vocab, 0.0f, top_p, 0, 0.0f, &rng);
        }
        accepted += n_acc;
        rounds++;
        /* PROBE: outcome per proposed token — 1 accepted, 0 the
         * first rejection (past the rejection the comparison is
         * meaningless: the trajectory has diverged, don't log it) */
        if (dbg_conf) {
            for (int j = 0; j <= n_acc && j < nd && j < 64; j++)
                fprintf(stderr, "SPECCONF %.4f %d\n", conf[j],
                        j < n_acc ? 1 : 0);
            /* PROBE: on rejection, was the target's correction
             * the draft's SECOND choice? */
            if (n_acc < nd && n_acc < 64)
                fprintf(stderr, "SPECTREE %d %.4f\n",
                        fix == top2[n_acc] ? 1 : 0, conf[n_acc]);
        }

        /* 4) emission: same steps (GENID, id-based stop, textual
         * filter) as the normal path, one token at a time */
        const int old_len = len;
        for (int e = 0; e <= n_acc && !stopped && emitted < n_predict; e++) {
            const int32_t tok = e < n_acc ? dtok[e] : fix;
            fprintf(stderr, "GENID %d\n", tok);
            if (len < max_ctx) hist[len++] = tok;
            emitted++;
            if (is_stop_token(stop, tok)) { stopped = 1; break; }
            char piece[256];
            const int w = nf_token_decode(tk, tok, piece, sizeof(piece) - 1);
            n_generated++;
            if (stream_piece_through_stopfilter(piece, w > 0 ? w : 0, pend,
                                                &n_pend_txt, (int)sizeof(pend),
                                                tstop)) {
                stopped = 1;
                break;
            }
        }

        /* 5) roll back both caches to the accepted prefix: the
         * correction (or bonus token) is NOT in the cache — it will be
         * pending for the next round for both models */
        nf_session_rewind(s, old_len + n_acc);
        nf_session_rewind(ds, old_len + n_acc);
        t_past = old_len + n_acc;
        d_past = old_len + n_acc;
    }
    if (n_pend_txt > 0) { fwrite(pend, 1, (size_t)n_pend_txt, stdout); fflush(stdout); }
    printf("\n");
    {
        const double dt = now_seconds() - t0;
        fprintf(stderr,
                "nf: speculative: %d rounds, %d/%d proposals accepted (%.0f%%), "
                "K=%d, %d tokens in %.1fs (%.2f tok/s)\n",
                rounds, accepted, drafted,
                drafted ? 100.0 * accepted / drafted : 0.0, K,
                emitted, dt, dt > 0 ? emitted / dt : 0.0);
    }
    fprintf(stderr, "nf: generated %d tokens (greedy speculative, seed=%" PRIu64 ")\n",
            n_generated, seed);

    free(va); free(hist); free(dtok); free(batch);
    nf_session_free(ds);
    nf_model_free(dm);
    return 0;

fail:
    free(va); free(hist); free(dtok); free(batch);
    nf_session_free(ds);
    nf_model_free(dm);
    return 1;
}

/* ---- M6: chat template style, from the gguf (general.architecture) --
 *
 * Qwen3 uses ChatML: "<|im_start|>role\ntext<|im_end|>\n". <|im_start|>
 * and <|im_end|> are atomic vocabulary entries (id 151644/151645, see
 * nf_tokenizer.c: match_special) — <|im_end|> is also the EOS. The
 * assistant's turn closes when the model samples it on its own.
 *
 * No generic template engine (no Jinja): exactly DwarfStar's narrowness
 * lesson applied to the chat template — one model, one format,
 * hardcoded and verified, not a generic interpreter that would have to
 * work for models we don't have.
 *
 * Second format, same philosophy. DeepSeek-V2-Lite-Chat:
 *   {system}\n\n  then  User: {u}\n\nAssistant: {a}<eos>  per turn,
 * and the turn to generate opens with "Assistant:".
 *
 * Third format (olmoe): <|system|>\n{s} then <|user|>\n{u} then
 * <|assistant|>\n{a}<eot> per turn, opening "<|assistant|>\n".
 *
 * Fourth format (glm-dsa, GLM-5.2) — see the comment on
 * append_glm_prefix_if_first below for the verified sources. */
static int g_chat_style = 0;   /* 0 = ChatML (qwen3), 1 = deepseek, 2 = olmoe,
                                 * 3 = glm-dsa (GLM-5.2) */

/* .forge container (produced by the port project, Phase C): the
 * container carries neither the tokenizer nor GGUF metadata — v1:
 * NF_TOKENIZER points at any GGUF with the SAME tokenizer (e.g.
 * glm52-tokenizer.gguf next to it), from which the stopset and template
 * are also read. A .forge's arch is glm-dsa by construction (v1). */
static int is_forge_path(const char *path) {
    const size_t n = strlen(path);
    return (n > 6 && strcmp(path + n - 6, ".forge") == 0) ||
           (n > 7 && strcmp(path + n - 7, ".forgez") == 0) ||
           (n > 8 && strcmp(path + n - 8, ".forgezh") == 0);
}

/* path to read the tokenizer/stopset from for `path`: the path itself
 * for GGUFs, NF_TOKENIZER for .forge files (NULL if missing: the
 * caller decides whether that's fatal — yes for chat/generate, no for
 * sweepgen --csv). */
static const char *tok_source_for(const char *path) {
    if (!is_forge_path(path)) return path;
    const char *t = getenv("NF_TOKENIZER");
    if (t && *t) return t;
    /* default: the tokenizer extracted from the retired
     * GGUF — 9MB of metadata only, validated token-by-token against
     * the original before retiring it. If it exists, it's the right
     * default. */
    static const char *dflt = "models/glm52-tokenizer.gguf";
    FILE *p = fopen(dflt, "rb");
    if (p) { fclose(p); return dflt; }
    fprintf(stderr, "nf: a .forge doesn't carry the tokenizer: set "
            "NF_TOKENIZER=<gguf with the same tokenizer> (e.g. "
            "models/glm52-tokenizer.gguf)\n");
    return NULL;
}

static void chat_style_init(const char *gguf_path) {
    if (is_forge_path(gguf_path)) { g_chat_style = 3; return; }
    char *arch = nf_gguf_read_kv_str(gguf_path, "general.architecture");
    g_chat_style = arch && strcmp(arch, "deepseek2") == 0 ? 1
                 : (arch && strcmp(arch, "olmoe") == 0 ? 2
                 : (arch && strcmp(arch, "glm-dsa") == 0 ? 3 : 0));
    free(arch);
}

/* AUTOMATIC glm-dsa daily-driver config ("usability"
 * project): the configuration measured as best for interactive use of
 * the real GLM (FAST: high priority, 8GB/6-thread streaming, prefetch,
 * QPRIO, overlap, miss-skip 0.3, dynamic-k 0.4 — every value tuned
 * individually) required remembering 8 environment
 * variables by hand: a NON-deliberate default was exactly the fact
 * that it had no default at all. Rules: applies ONLY to glm-dsa
 * ggufs, in chat/generate/serve alike (the benches stay explicit),
 * every variable the user has ALREADY set wins (only the gaps get
 * filled), NF_GLM_AUTO=0 turns it off entirely. Must be called BEFORE
 * nf_model_load: many gates cache getenv on first use. */
static void glm_daily_env(const char *gguf_path) {
    const char *ae = getenv("NF_GLM_AUTO");
    if (ae && atoi(ae) == 0) return;
    int is_glm = is_forge_path(gguf_path);   /* .forge = glm-dsa (v1) */
    if (!is_glm) {
        char *arch = nf_gguf_read_kv_str(gguf_path, "general.architecture");
        is_glm = arch && strcmp(arch, "glm-dsa") == 0;
        free(arch);
    }
    if (!is_glm) return;
    /* .forge: resident int4 densities ~8.5GB (not the GGUF+repack's 17)
     * — the expert cache can rise to 15GB (~750 20MB slots, above one
     * token's working set; measured: 286->200GB read,
     * 14.5->9 s/position). */
    const int forge = is_forge_path(gguf_path);
    static const char *defs[][2] = {
        { "NF_PRIORITY",       "high" },
        { "NF_GLM_STREAM_GB",  "8"    },
        { "NF_GLM_IO_THREADS", "6"    },
        { "NF_GLM_PREFETCH",   "1"    },
        { "NF_GLM_QPRIO",      "1"    },
        { "NF_GLM_OVERLAP",    "1"    },
        { "NF_GLM_MISS_SKIP",  "0.3"  },
        { "NF_MOE_DYNK",       "0.4"  },
        /* prefetch diet: only rank 0-2 predictions get
         * anticipated — the queue stays clean and mandatory misses
         * start immediately (in-window bandwidth 418->650+, wall -16%).
         * Measured ONLY on .forge (20MB span): stays off on GGUF. */
        { "NF_GLM_PF_TOPK",    "0"    },
        /* cache-aware routing:
         * residency bonus on rank >=4 near-ties — 390->343s measured,
         * quality 3/3 on the factual task at both 0.05 and 0.10. Same
         * output-changing-default class as MISS_SKIP. */
        { "NF_GLM_CACHE_BIAS", "0.05" },
        /* batched prefill + int8 kernels on by default ONLY on .forge:
         * batched = -19% mixed chat/-34% long
         * prompts (batched attention included), int8 = -15% cpu,
         * argmax-stable. Under --draft, batched is disabled with a
         * note (guard in cmd_chat/cmd_generate: the interaction was
         * never validated). Both stay off on GGUF (never measured
         * there). */
        { "NF_GLM_PREFILL_BATCH", "0"  },
        { "NF_GLM_PREFILL_CHUNK", "32" },
        { "NF_FORGE_INT8",        "0"  },
        /* the promotion (7-task battery widened + attribution
         * CHECKS: the two red items — multi-step arithmetic and
         * long-prompt artifacts — fail the SAME or WORSE on the k=8
         * series: the int4 model's fault, not the truncation's): k=6
         * (arXiv 2505.03531) and a conservative prefill bias
         * (rank>=6) on by default ONLY on .forge — 343 -> ~300s. */
        { "NF_MOE_TOPK_EFF",           "0" },
        { "NF_GLM_CACHE_BIAS_PREFILL", "0" },
    };
    int applied = 0, prio_ours = 0;
    for (size_t i = 0; i < sizeof(defs) / sizeof(defs[0]); i++) {
        if (getenv(defs[i][0])) continue;
        char kv[64];
        const char *val = defs[i][1];
        if (forge && strcmp(defs[i][0], "NF_GLM_STREAM_GB") == 0)
            val = "15";
        if (forge && strcmp(defs[i][0], "NF_GLM_PF_TOPK") == 0)
            val = "3";
        if (forge && (strcmp(defs[i][0], "NF_GLM_PREFILL_BATCH") == 0 ||
                      strcmp(defs[i][0], "NF_FORGE_INT8") == 0))
            val = "1";
        /* k=6 and prefill-bias: forge ONLY, and never putenv it outside
         * (TOPK_EFF's parse would clamp "0" to k=1: must be skipped,
         * not zeroed) */
        if (strcmp(defs[i][0], "NF_MOE_TOPK_EFF") == 0) {
            if (!forge) continue;
            val = "6";
        }
        if (strcmp(defs[i][0], "NF_GLM_CACHE_BIAS_PREFILL") == 0) {
            if (!forge) continue;
            val = "0.05";
        }
        snprintf(kv, sizeof(kv), "%s=%s", defs[i][0], val);
        nf_setenv(kv);
        applied++;
        if (strcmp(defs[i][0], "NF_PRIORITY") == 0) prio_ours = 1;
    }
#ifdef _WIN32
    /* main()'s NF_PRIORITY block has ALREADY run by the time commands
     * get here: if the priority is one of our own defaults (not the
     * user's) it needs to be applied directly, same call. */
    if (prio_ours)
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    if (applied)
        fprintf(stderr, "nf: glm daily config active (%d defaults "
                "applied; NF_GLM_AUTO=0 to disable it, a variable "
                "already set always wins)\n", applied);
}

/* glm-dsa (GLM-5.2) chat template. Strings/tokens verified against
 * primary sources, not from memory: zai-org/GLM-5.2's
 * tokenizer_config.json (fetched directly) confirms the
 * strings <|system|>/<|user|>/<|assistant|>/<|observation|>/
 * <|endoftext|>/[gMASK]/<sop>; the same repo's chat_template.jinja
 * confirms the structure ([gMASK]<sop> prefix ONLY ONCE at the head of
 * the whole conversation, no '\n' between role and content,
 * <|observation|> for tool responses). colibri.c (lines ~4664-4680
 * run_score, ~5615-5642 REPL) is a second independent cross-check
 * already used in production in colibri's own codebase: same prefix, same
 * "no newline", the same THREE stop ids (endoftext/user/observation —
 * the comment "GLM-5.2 has THREE" at colibri.c:885) and the same
 * forced/opened <think> block for the assistant turn. Every PAST turn
 * (system/user/assistant already completed) is emitted by append_turn
 * below; ONLY the very first content ever written into an empty
 * session (n_past==0) carries the prefix — see
 * append_glm_prefix_if_first, called at the sites that build a buffer
 * from scratch. */
static void append_glm_prefix_if_first(char **buf, size_t *cap, size_t *len,
                                       int is_first) {
    if (g_chat_style != 3 || !is_first) return;
    const size_t need = *len + 16;
    if (need > *cap) {
        size_t newcap = *cap ? *cap * 2 : 4096;
        while (newcap < need) newcap *= 2;
        char *nb = realloc(*buf, newcap);
        if (!nb) return;
        *buf = nb;
        *cap = newcap;
    }
    *len += (size_t)snprintf(*buf + *len, *cap - *len, "[gMASK]<sop>");
}

/* ---- glm-dsa MTP draft-verify (--mtp-draft) -----------------
 *
 * ID-level, GREEDY-only loop, shared by cmd_generate (--mtp-draft) and
 * `nf debugmtpdraft` (validation on the tiny model, which has no
 * tokenizer): a k=1 round costs ONE 2-position verify window
 * (nf_glm_eval_verify: expert union + amortized decode) + ONE MTP head
 * forward, and yields 2 tokens if the draft is accepted (the confirmed
 * one + the bonus from the draft position's logits), 1 if rejected
 * (the correction from the first position's logits). Key property
 * (greedy speculative decoding): every emitted token is ALWAYS the
 * main model's argmax — the sequence must be identical to greedy
 * generation without a draft (validated by debugmtpdraft on the tiny
 * model and by A/B on the real model; the only legitimate source of
 * divergence is float reordering noise on the verify path for
 * near-tied candidates, same class as the fused-vecdot flip
 * documented elsewhere in this file).
 *
 * The MTP head is fed teacher-forced with the RECENT history of pairs
 * (h after output_norm at position t, the true token at t+1) — only
 * already-verified tokens, never unconfirmed drafts; a sliding window
 * of HIST_MAX pairs with RoPE positions relative to the batch,
 * consistent with the mtpcheck/mtporacle validation (see
 * nf_mtp_draft_glm). On rejection: nf_session_rewind by 1 (the draft's
 * KV stays and will be overwritten; the DSA indexer holds no derived
 * state beyond per-position Ixk, selection is recomputed on every eval
 * bounded by nk=pos+1 — no stale entry is ever readable, verified by
 * the consistency test).
 *
 * stop can be NULL (debug): the loop runs to n_predict. GENID lines go
 * to stderr HERE, at the moment of decision (externally timestampable
 * for the s/token measurement); the caller emits the text.
 *
 * oracle_draft (ONLY debugmtpdraft, NULL in production): replaces the
 * MTP draft with the already-known greedy sequence — a "perfect"
 * drafter that forces 100% acceptance and so deterministically
 * exercises the accept/bonus branch, which on the tiny model (random
 * weights: the MTP head doesn't correlate with the main model,
 * acceptance ~1/vocab) would never fire on its own; the normal run on
 * the tiny model exercises the reject/rewind branch. Together, the two
 * runs cover both branches of the loop.
 * Returns 0 on success, -1 on an eval/alloc error (never silent). */
static int mtp_draft_ids(nf_model *m, nf_session *s, const int32_t *prompt,
                         int n_prompt, int n_predict, const nf_stopset *stop,
                         const int32_t *oracle_draft,
                         int32_t *out_ids, int *out_n, int *out_rounds,
                         int *out_accepted, double *out_decode_s) {
    enum { HIST_MAX = 16 };
    const int E = nf_model_n_embd(m);
    const int64_t vocab = (int64_t)nf_model_vocab(m);
    float *hid  = malloc((size_t)2 * E * sizeof(float));   /* window of 2 */
    float *hh   = malloc((size_t)HIST_MAX * E * sizeof(float));
    float *va   = malloc((size_t)2 * vocab * sizeof(float));
    float *dlog = malloc((size_t)vocab * sizeof(float));
    int32_t th[HIST_MAX];
    *out_n = 0; *out_rounds = 0; *out_accepted = 0; *out_decode_s = 0.0;
    if (!hid || !hh || !va || !dlog) {
        fprintf(stderr, "nf: [mtp-draft] scratch not allocated\n");
        free(hid); free(hh); free(va); free(dlog);
        return -1;
    }
    int rc = -1;
    nf_session_set_hidden_out(s, hid);

    const float *lg = nf_session_eval(s, prompt, n_prompt);
    if (!lg) goto done;
    const double t0 = now_seconds();
    uint64_t rng = 0;   /* never consumed at temp=0 (pure argmax) */

    int nh = 0, n_out = 0, rounds = 0, accepted = 0, stopped = 0;
    /* first token from the prefill, same as the normal path */
    int32_t pend = sample_token(lg, vocab, 0.0f, 1.0f, 0, 0.0f, &rng);
    fprintf(stderr, "GENID %d\n", pend);
    out_ids[n_out++] = pend;
    if (stop && is_stop_token(stop, pend)) stopped = 1;
    /* first MTP pair: (h of the prompt's last position, pend) */
    memcpy(hh, hid, (size_t)E * sizeof(float));
    th[0] = pend;
    nh = 1;

    while (!stopped && n_out < n_predict) {
        /* 1) draft D from the MTP head on the teacher-forced history
         * (or from the debug oracle, see the comment at the top) */
        int32_t D;
        if (oracle_draft) {
            D = oracle_draft[n_out];
        } else {
            if (nf_mtp_draft_glm(m, hh, th, nh, dlog) != 0) {
                fprintf(stderr, "nf: [mtp-draft] MTP forward failed\n");
                goto done;
            }
            D = sample_token(dlog, vocab, 0.0f, 1.0f, 0, 0.0f, &rng);
        }

        /* 2) verify window [pend, D] — logits and h for BOTH */
        const int32_t batch[2] = { pend, D };
        if (!nf_glm_eval_verify(s, batch, 2, va)) goto done;
        rounds++;

        /* 3) the true token after pend, from position 0's logits */
        const int32_t g = sample_token(va, vocab, 0.0f, 1.0f, 0, 0.0f, &rng);
        const int acc = (g == D);
        if (acc) accepted++;
        fprintf(stderr, "GENID %d\n", g);
        out_ids[n_out++] = g;
        /* pair (h at pend's position, g): valid on both branches */
        if (nh == HIST_MAX) {
            memmove(hh, hh + E, (size_t)(HIST_MAX - 1) * E * sizeof(float));
            memmove(th, th + 1, (size_t)(HIST_MAX - 1) * sizeof(int32_t));
            nh--;
        }
        memcpy(hh + (size_t)nh * E, hid, (size_t)E * sizeof(float));
        th[nh] = g;
        nh++;
        if (stop && is_stop_token(stop, g)) { stopped = 1; break; }

        if (acc) {
            /* draft confirmed: D is already in the cache, its logits
             * give the next token for FREE (bonus continuation) */
            if (n_out < n_predict) {
                const int32_t g2 = sample_token(va + vocab, vocab, 0.0f,
                                                1.0f, 0, 0.0f, &rng);
                fprintf(stderr, "GENID %d\n", g2);
                out_ids[n_out++] = g2;
                if (stop && is_stop_token(stop, g2)) { stopped = 1; break; }
                if (nh == HIST_MAX) {
                    memmove(hh, hh + E,
                            (size_t)(HIST_MAX - 1) * E * sizeof(float));
                    memmove(th, th + 1,
                            (size_t)(HIST_MAX - 1) * sizeof(int32_t));
                    nh--;
                }
                memcpy(hh + (size_t)nh * E, hid + E,
                       (size_t)E * sizeof(float));
                th[nh] = g2;
                nh++;
                pend = g2;
            } else {
                break;
            }
        } else {
            /* rejection: discard the draft's KV (the higher position),
             * the true token g becomes the next round's pending one */
            nf_session_rewind(s, nf_session_n_past(s) - 1);
            pend = g;
        }
    }

    *out_decode_s = now_seconds() - t0;
    *out_n = n_out; *out_rounds = rounds; *out_accepted = accepted;
    rc = 0;
done:
    nf_session_set_hidden_out(s, NULL);
    free(hid); free(hh); free(va); free(dlog);
    return rc;
}

/* ---- glm-dsa layer-skip draft-verify (--layerskip-draft) ----
 *
 * Same round-by-round loop as mtp_draft_ids (draft -> 2-position verify
 * window [pend, D] -> accept/bonus or reject/rewind), but draft D comes
 * from nf_glm_draft_skip instead of the MTP head: simpler than
 * mtp_draft_ids because the layer-skip drafter reads the session's
 * REAL KV (read-only, see the isolation argument on
 * glm_eval_draft_skip in nf_model.c) instead of keeping a separate
 * teacher-forced history — no hid/hh/th/nh here, a single scratch
 * (dlog) for the draft's logits. The property under test is identical:
 * every emitted token is ALWAYS the main model's argmax, so the
 * sequence must be IDENTICAL to greedy generation without a draft
 * (validated by debuglayerskipdraft on the tiny model and by A/B on
 * the real model). Returns 0 on success, -1 on an eval/alloc error. */
static int layerskip_draft_ids(nf_model *m, nf_session *s, const int32_t *prompt,
                               int n_prompt, int n_predict, const nf_stopset *stop,
                               int32_t *out_ids, int *out_n, int *out_rounds,
                               int *out_accepted, double *out_decode_s) {
    const int64_t vocab = (int64_t)nf_model_vocab(m);
    float *va   = malloc((size_t)2 * vocab * sizeof(float));
    float *dlog = malloc((size_t)vocab * sizeof(float));
    *out_n = 0; *out_rounds = 0; *out_accepted = 0; *out_decode_s = 0.0;
    if (!va || !dlog) {
        fprintf(stderr, "nf: [layerskip-draft] scratch not allocated\n");
        free(va); free(dlog);
        return -1;
    }
    int rc = -1;

    const float *lg = nf_session_eval(s, prompt, n_prompt);
    if (!lg) goto done;
    const double t0 = now_seconds();
    uint64_t rng = 0;   /* never consumed at temp=0 (pure argmax) */

    int n_out = 0, rounds = 0, accepted = 0, stopped = 0;
    int32_t pend = sample_token(lg, vocab, 0.0f, 1.0f, 0, 0.0f, &rng);
    fprintf(stderr, "GENID %d\n", pend);
    out_ids[n_out++] = pend;
    if (stop && is_stop_token(stop, pend)) stopped = 1;

    while (!stopped && n_out < n_predict) {
        /* 1) draft D from the layer-skip forward, read-only on the
         * already-confirmed KV (this call never touches s->n_past nor
         * K/V — see nf_glm_draft_skip) */
        int32_t D;
        {
            if (nf_glm_draft_skip(s, pend, dlog) != 0) {
                fprintf(stderr, "nf: [layerskip-draft] drafter forward failed\n");
                goto done;
            }
            D = sample_token(dlog, vocab, 0.0f, 1.0f, 0, 0.0f, &rng);
        }

        /* 2) verify window [pend, D] — IDENTICAL to mtp_draft_ids */
        const int32_t batch[2] = { pend, D };
        if (!nf_glm_eval_verify(s, batch, 2, va)) goto done;
        rounds++;

        const int32_t g = sample_token(va, vocab, 0.0f, 1.0f, 0, 0.0f, &rng);
        const int acc = (g == D);
        if (acc) accepted++;
        fprintf(stderr, "GENID %d\n", g);
        out_ids[n_out++] = g;
        if (stop && is_stop_token(stop, g)) { stopped = 1; break; }

        if (acc) {
            if (n_out < n_predict) {
                const int32_t g2 = sample_token(va + vocab, vocab, 0.0f,
                                                1.0f, 0, 0.0f, &rng);
                fprintf(stderr, "GENID %d\n", g2);
                out_ids[n_out++] = g2;
                if (stop && is_stop_token(stop, g2)) { stopped = 1; break; }
                pend = g2;
            } else {
                break;
            }
        } else {
            nf_session_rewind(s, nf_session_n_past(s) - 1);
            pend = g;
        }
    }

    *out_decode_s = now_seconds() - t0;
    *out_n = n_out; *out_rounds = rounds; *out_accepted = accepted;
    rc = 0;
done:
    free(va); free(dlog);
    return rc;
}

/* nf debugglmverify <gguf> <csv> (step a of the validation
 * ladder): SAME token sequence through (i) the existing sequential
 * session path (glm_eval_core, n_new=1 per position, the reference
 * already validated 32/32 against the oracle) and (ii) the new
 * layer-major path nf_glm_eval_verify in windows of 2/3/4 — comparing
 * both logits AND post-output_norm hidden state at EVERY position.
 * The two paths do the same math in the same order per position (the
 * only legitimate difference is the accumulation order of the
 * union/multi-kernel MoE): max|d| < 1e-3 is required, well under the
 * historical session tolerance — anything worse is a bug. */
static int cmd_debugglmverify(const char *path, const char *csv) {
    nf_model *m = nf_model_load(path);
    if (!m) return 1;
    if (!nf_model_is_glm(m)) {
        fprintf(stderr, "nf: debugglmverify requires a glm-dsa model\n");
        nf_model_free(m);
        return 1;
    }
    int32_t ids[4096];
    int n = 0;
    const char *p = csv;
    while (*p && n < 4096) {
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        ids[n++] = (int32_t)v;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    if (n < 3) {
        fprintf(stderr, "nf: debugglmverify: need at least 3 tokens "
                "(expected CSV, e.g. 3,14,159)\n");
        nf_model_free(m);
        return 1;
    }
    const int64_t vocab = (int64_t)nf_model_vocab(m);
    const int E = nf_model_n_embd(m);
    float *ref  = malloc((size_t)n * vocab * sizeof(float));
    float *href = malloc((size_t)n * E * sizeof(float));
    float *hbuf = malloc((size_t)4 * E * sizeof(float));
    float *va   = malloc((size_t)4 * vocab * sizeof(float));
    int rc = 1;
    nf_session *sa = NULL, *sb = NULL;
    if (!ref || !href || !hbuf || !va) {
        fprintf(stderr, "nf: debugglmverify: oom\n");
        goto out;
    }

    /* (i) reference: sequential, one position at a time */
    sa = nf_session_create(m, n + 8);
    if (!sa) { fprintf(stderr, "nf: debugglmverify: session (i) failed\n"); goto out; }
    nf_session_set_hidden_out(sa, hbuf);
    for (int k = 0; k < n; k++) {
        const float *lg = nf_session_eval(sa, &ids[k], 1);
        if (!lg) { fprintf(stderr, "nf: debugglmverify: eval (i) failed at k=%d\n", k); goto out; }
        memcpy(ref + (size_t)k * vocab, lg, (size_t)vocab * sizeof(float));
        memcpy(href + (size_t)k * E, hbuf, (size_t)E * sizeof(float));
    }
    nf_session_free(sa);
    sa = NULL;

    /* (ii) verify: first token alone (a window starts at 2 minimum),
     * then cycling 2,3,4 windows — cover EVERY supported size; any
     * size-1 leftover goes through the sequential path, same as a
     * real caller would. */
    sb = nf_session_create(m, n + 8);
    if (!sb) { fprintf(stderr, "nf: debugglmverify: session (ii) failed\n"); goto out; }
    nf_session_set_hidden_out(sb, hbuf);
    {
        int all_ok = 1, pos = 0, wi = 0;
        float gmaxd = 0.0f, gmaxh = 0.0f;
        while (pos < n) {
            int w;
            if (pos == 0) w = 1;
            else {
                static const int ws[3] = { 2, 3, 4 };
                w = ws[wi % 3];
                wi++;
                if (w > n - pos) w = n - pos;
            }
            const float *lg = NULL;
            if (w < 2) {
                lg = nf_session_eval(sb, &ids[pos], 1);
                if (lg) memcpy(va, lg, (size_t)vocab * sizeof(float));
            } else {
                lg = nf_glm_eval_verify(sb, &ids[pos], w, va);
            }
            if (!lg) {
                fprintf(stderr, "nf: debugglmverify: eval (ii) failed at "
                        "pos=%d (w=%d)\n", pos, w);
                goto out;
            }
            for (int j = 0; j < w; j++) {
                const float *lr = ref + (size_t)(pos + j) * vocab;
                const float *lv = va + (size_t)j * vocab;
                const float *hr = href + (size_t)(pos + j) * E;
                const float *hv = hbuf + (size_t)j * E;
                int64_t ar = 0, av = 0;
                float maxd = 0.0f, maxh = 0.0f;
                for (int64_t i = 0; i < vocab; i++) {
                    if (lr[i] > lr[ar]) ar = i;
                    if (lv[i] > lv[av]) av = i;
                    const float d = fabsf(lr[i] - lv[i]);
                    if (d > maxd) maxd = d;
                }
                for (int i = 0; i < E; i++) {
                    const float d = fabsf(hr[i] - hv[i]);
                    if (d > maxh) maxh = d;
                }
                const int ok = ar == av && maxd < 1e-3f && maxh < 1e-3f;
                if (!ok) all_ok = 0;
                if (maxd > gmaxd) gmaxd = maxd;
                if (maxh > gmaxh) gmaxh = maxh;
                printf("[glm-verify] pos=%2d w=%d  argmax seq=%-4lld "
                       "ver=%-4lld  max|d|=%.6f  max|dh|=%.6f  %s\n",
                       pos + j, w, (long long)ar, (long long)av,
                       (double)maxd, (double)maxh, ok ? "OK" : "FAIL");
                /* flip diagnostics (real model): when the argmax
                 * differs, the "near-tie noise" vs "bug" verdict sits
                 * in the GAP between the two candidates on BOTH paths
                 * — a legitimate flip has gap << max|d|, a bug has
                 * differences of a different scale (same criterion as
                 * the token-5 fused-vecdot flip mentioned above). */
                if (ar != av)
                    printf("[glm-verify]   flip: seq[%lld]=%.4f "
                           "seq[%lld]=%.4f (gap %.4f) | ver[%lld]=%.4f "
                           "ver[%lld]=%.4f (gap %.4f)\n",
                           (long long)ar, (double)lr[ar],
                           (long long)av, (double)lr[av],
                           (double)(lr[ar] - lr[av]),
                           (long long)ar, (double)lv[ar],
                           (long long)av, (double)lv[av],
                           (double)(lv[av] - lv[ar]));
                /* last position: top-3 of both paths even without a
                 * flip — used to read the GAP between the top
                 * candidates on the real model (how "near-tied" is the
                 * point where a flip could happen). */
                if (pos + j == n - 1) {
                    for (int side = 0; side < 2; side++) {
                        const float *L = side == 0 ? lr : lv;
                        int64_t t1 = 0, t2 = -1, t3 = -1;
                        for (int64_t i = 1; i < vocab; i++)
                            if (L[i] > L[t1]) t1 = i;
                        for (int64_t i = 0; i < vocab; i++) {
                            if (i == t1) continue;
                            if (t2 < 0 || L[i] > L[t2]) { t3 = t2; t2 = i; }
                            else if (t3 < 0 || L[i] > L[t3]) t3 = i;
                        }
                        printf("[glm-verify]   last pos top-3 %s: "
                               "[%lld]=%.4f [%lld]=%.4f [%lld]=%.4f\n",
                               side == 0 ? "seq" : "ver",
                               (long long)t1, (double)L[t1],
                               (long long)t2, (double)L[t2],
                               (long long)t3, (double)L[t3]);
                    }
                }
            }
            pos += w;
        }
        printf("global max|d| logit=%.6f  hidden=%.6f\n",
               (double)gmaxd, (double)gmaxh);
        printf("%s\n", all_ok
               ? "GLM VERIFY OK: layer-major windows == sequential path"
               : "GLM VERIFY FAILED");
        rc = all_ok ? 0 : 1;
    }
out:
    if (sa) nf_session_free(sa);
    if (sb) nf_session_free(sb);
    free(ref); free(href); free(hbuf); free(va);
    nf_model_free(m);
    return rc;
}

/* nf debugglmprefillbatch <gguf> <csv> <chunk_size> (batched
 * prefill): SAME token sequence through (i) the existing sequential
 * session path in ONE SHOT (nf_session_eval_all with n_new=n whole —
 * glm_eval_core, the already-validated reference, NOT
 * position-by-position like in debugglmverify: here the comparison is
 * "batched chunks" vs "the full n_new in one call", the same
 * equivalence chunk dispatch in nf_session_eval rests on) and (ii) the
 * new layer-major path nf_glm_eval_prefill_batched in chunks of
 * <chunk_size> positions (the last chunk shorter if n isn't a
 * multiple — the most likely edge case for an n_past/offset accounting
 * bug). WARNING: do NOT set NF_GLM_PREFILL_BATCH while running this
 * command — session (i) must go through glm_eval_core directly (the
 * reference), session (ii) calls nf_glm_eval_prefill_batched directly
 * (bypasses the dispatch/env on purpose, to control the exact chunk
 * boundary from the CLI). Compares logits AND post-output_norm hidden
 * state at EVERY position, same tolerance as debugglmverify
 * (max|d| < 1e-3). */
static int cmd_debugglmprefillbatch(const char *path, const char *csv,
                                    int chunk_size) {
    nf_model *m = nf_model_load(path);
    if (!m) return 1;
    if (!nf_model_is_glm(m)) {
        fprintf(stderr, "nf: debugglmprefillbatch requires a glm-dsa model\n");
        nf_model_free(m);
        return 1;
    }
    if (chunk_size < 1) {
        fprintf(stderr, "nf: debugglmprefillbatch: chunk_size must be >= 1\n");
        nf_model_free(m);
        return 1;
    }
    int32_t ids[4096];
    int n = 0;
    const char *p = csv;
    while (*p && n < 4096) {
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        ids[n++] = (int32_t)v;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    if (n < 2) {
        fprintf(stderr, "nf: debugglmprefillbatch: need at least 2 tokens "
                "(expected CSV, e.g. 3,14,159)\n");
        nf_model_free(m);
        return 1;
    }
    const int64_t vocab = (int64_t)nf_model_vocab(m);
    const int E = nf_model_n_embd(m);
    float *ref  = malloc((size_t)n * vocab * sizeof(float));
    float *href = malloc((size_t)n * E * sizeof(float));
    float *hbuf = malloc((size_t)chunk_size * E * sizeof(float));
    float *va   = malloc((size_t)chunk_size * vocab * sizeof(float));
    int rc = 1;
    nf_session *sa = NULL, *sb = NULL;
    if (!ref || !href || !hbuf || !va) {
        fprintf(stderr, "nf: debugglmprefillbatch: oom\n");
        goto out;
    }

    /* (i) reference: glm_eval_core over the WHOLE n_new in a single
     * call (nf_session_eval_all, assuming env NF_GLM_PREFILL_BATCH is
     * NOT set — see the comment above) */
    sa = nf_session_create(m, n + 8);
    if (!sa) { fprintf(stderr, "nf: debugglmprefillbatch: session (i) failed\n"); goto out; }
    nf_session_set_hidden_out(sa, href);
    {
        const float *lg = nf_session_eval_all(sa, ids, n, ref);
        if (!lg) { fprintf(stderr, "nf: debugglmprefillbatch: eval (i) failed\n"); goto out; }
    }
    nf_session_free(sa);
    sa = NULL;

    /* (ii) batched prefill: chunks of chunk_size, the last one shorter
     * if n isn't a multiple — DIRECT call (bypasses env) to control the
     * exact chunk boundary. */
    sb = nf_session_create(m, n + 8);
    if (!sb) { fprintf(stderr, "nf: debugglmprefillbatch: session (ii) failed\n"); goto out; }
    nf_session_set_hidden_out(sb, hbuf);
    {
        int all_ok = 1, pos = 0;
        float gmaxd = 0.0f, gmaxh = 0.0f;
        while (pos < n) {
            int w = n - pos;
            if (w > chunk_size) w = chunk_size;
            const float *lg = nf_glm_eval_prefill_batched(sb, &ids[pos], w, va);
            if (!lg) {
                fprintf(stderr, "nf: debugglmprefillbatch: eval (ii) failed at "
                        "pos=%d (w=%d)\n", pos, w);
                goto out;
            }
            for (int j = 0; j < w; j++) {
                const float *lr = ref + (size_t)(pos + j) * vocab;
                const float *lv = va + (size_t)j * vocab;
                const float *hr = href + (size_t)(pos + j) * E;
                const float *hv = hbuf + (size_t)j * E;
                int64_t ar = 0, av = 0;
                float maxd = 0.0f, maxh = 0.0f;
                for (int64_t i = 0; i < vocab; i++) {
                    if (lr[i] > lr[ar]) ar = i;
                    if (lv[i] > lv[av]) av = i;
                    const float d = fabsf(lr[i] - lv[i]);
                    if (d > maxd) maxd = d;
                }
                for (int i = 0; i < E; i++) {
                    const float d = fabsf(hr[i] - hv[i]);
                    if (d > maxh) maxh = d;
                }
                const int ok = ar == av && maxd < 1e-3f && maxh < 1e-3f;
                if (!ok) all_ok = 0;
                if (maxd > gmaxd) gmaxd = maxd;
                if (maxh > gmaxh) gmaxh = maxh;
                printf("[glm-prefillbatch] pos=%2d chunk=%d(w=%d)  argmax seq=%-4lld "
                       "batch=%-4lld  max|d|=%.6f  max|dh|=%.6f  %s\n",
                       pos + j, chunk_size, w, (long long)ar, (long long)av,
                       (double)maxd, (double)maxh, ok ? "OK" : "FAIL");
                if (ar != av)
                    printf("[glm-prefillbatch]   flip: seq[%lld]=%.4f "
                           "seq[%lld]=%.4f (gap %.4f) | batch[%lld]=%.4f "
                           "batch[%lld]=%.4f (gap %.4f)\n",
                           (long long)ar, (double)lr[ar],
                           (long long)av, (double)lr[av],
                           (double)(lr[ar] - lr[av]),
                           (long long)ar, (double)lv[ar],
                           (long long)av, (double)lv[av],
                           (double)(lv[av] - lv[ar]));
            }
            pos += w;
        }
        printf("global max|d| logit=%.6f  hidden=%.6f  (n=%d chunk_size=%d, "
               "%d chunk%s)\n", (double)gmaxd, (double)gmaxh, n, chunk_size,
               (n + chunk_size - 1) / chunk_size,
               (n + chunk_size - 1) / chunk_size == 1 ? "" : "s");
        printf("%s\n", all_ok
               ? "GLM PREFILL BATCH OK: layer-major chunks == full sequential n_new"
               : "GLM PREFILL BATCH FAILED");
        rc = all_ok ? 0 : 1;
    }
out:
    if (sa) nf_session_free(sa);
    if (sb) nf_session_free(sb);
    free(ref); free(href); free(hbuf); free(va);
    nf_model_free(m);
    return rc;
}

/* nf debugmtpdraft <gguf> <csv> <n_predict> (step b): the
 * FULL draft-verify loop (mtp_draft_ids, the same code as --mtp-draft)
 * on the MTP test model, against pure greedy generation of the same
 * prompt — the property under test is output INVARIANCE (greedy
 * speculative decoding: identical sequences, any divergence is a
 * rollback/KV bug). Needs NF_GLM_MTP=1. */
static int cmd_debugmtpdraft(const char *path, const char *csv, int n_predict) {
    nf_model *m = nf_model_load(path);
    if (!m) return 1;
    if (!nf_model_is_glm(m) || !nf_model_has_mtp(m)) {
        fprintf(stderr, "nf: debugmtpdraft requires a glm-dsa model with "
                "the MTP head loaded (NF_GLM_MTP=1)\n");
        nf_model_free(m);
        return 1;
    }
    int32_t ids[4096];
    int n = 0;
    const char *p = csv;
    while (*p && n < 4096) {
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        ids[n++] = (int32_t)v;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    if (n <= 0 || n_predict <= 0) {
        fprintf(stderr, "nf: debugmtpdraft: invalid CSV tokens or n_predict\n");
        nf_model_free(m);
        return 1;
    }
    const int64_t vocab = (int64_t)nf_model_vocab(m);
    int32_t *base_ids  = malloc((size_t)n_predict * sizeof(int32_t));
    int32_t *draft_ids = malloc((size_t)n_predict * sizeof(int32_t));
    int rc = 1;
    nf_session *sa = NULL, *sb = NULL;
    if (!base_ids || !draft_ids) { fprintf(stderr, "nf: debugmtpdraft: oom\n"); goto out; }

    /* baseline: pure greedy, one token at a time */
    sa = nf_session_create(m, n + n_predict + 8);
    if (!sa) goto out;
    {
        uint64_t rng = 0;
        const float *lg = nf_session_eval(sa, ids, n);
        if (!lg) goto out;
        for (int i = 0; i < n_predict; i++) {
            const int32_t tok = sample_token(lg, vocab, 0.0f, 1.0f, 0, 0.0f, &rng);
            base_ids[i] = tok;
            if (i + 1 < n_predict) {
                lg = nf_session_eval(sa, &tok, 1);
                if (!lg) goto out;
            }
        }
    }
    nf_session_free(sa);
    sa = NULL;

    /* draft-verify: the same loop as --mtp-draft, two runs on fresh
     * sessions — the REAL MTP draft (~0 acceptance on the tiny model:
     * covers the reject/rewind branch) and the ORACLE draft (100%
     * acceptance: covers the accept/bonus branch). Both must reproduce
     * the greedy sequence. */
    {
        int all_same = 1;
        printf("[mtp-draft] baseline: ");
        for (int i = 0; i < n_predict; i++) printf("%d ", base_ids[i]);
        printf("\n");
        for (int pass = 0; pass < 2; pass++) {
            sb = nf_session_create(m, n + n_predict + 9);
            if (!sb) goto out;
            int n_out = 0, rounds = 0, accepted = 0;
            double dec_s = 0.0;
            if (mtp_draft_ids(m, sb, ids, n, n_predict, NULL,
                              pass == 0 ? NULL : base_ids, draft_ids,
                              &n_out, &rounds, &accepted, &dec_s) != 0)
                goto out;
            nf_session_free(sb);
            sb = NULL;
            int same = n_out == n_predict;
            for (int i = 0; same && i < n_predict; i++)
                if (base_ids[i] != draft_ids[i]) same = 0;
            if (!same) all_same = 0;
            printf("[mtp-draft] %-8s: ", pass == 0 ? "mtp" : "oracle");
            for (int i = 0; i < n_out; i++) printf("%d ", draft_ids[i]);
            printf("\n[mtp-draft] %-8s: rounds=%d accepted=%d (acc=%.3f) "
                   "tokens=%d  %s\n", pass == 0 ? "mtp" : "oracle",
                   rounds, accepted,
                   rounds > 0 ? (double)accepted / rounds : 0.0, n_out,
                   same ? "OK" : "FAIL");
        }
        printf("%s\n", all_same
               ? "MTP DRAFT OK: sequences identical to greedy without a draft"
               : "MTP DRAFT FAILED: divergent sequences");
        rc = all_same ? 0 : 1;
    }
out:
    if (sa) nf_session_free(sa);
    if (sb) nf_session_free(sb);
    free(base_ids); free(draft_ids);
    nf_model_free(m);
    return rc;
}

/* nf debuglayerskip <gguf> <csv> (step a of the layer-skip
 * drafter's validation ladder): does NOT touch the round loop — isolates
 * the layer-skip forward pass alone. For every prefix [0..k) of the CSV
 * (k=1..N-1), computes (i) the TRUE logits at position k-1 via the
 * already-validated sequential path (nf_session_eval, one token at a
 * time, on session A) and (ii) the DRAFT's logits at the same position
 * via nf_glm_draft_skip on the SAME session A (which isolation promises
 * not to alter: if that promise doesn't hold, the TRUE logits at the
 * next position would silently become wrong — this is why the test
 * re-feeds A through the same sequential path at every k, not a
 * side-channel: any corruption of the session by the draft would show
 * up DIRECTLY as a divergence in the "true" logits against a control
 * run that never calls the draft). Checks: draft logits are FINITE (no
 * NaN/Inf — a broken forward would otherwise be silent) and top-5
 * overlap with the true one — no bit-exactness claim, the draft is an
 * approximation by construction. */
static int cmd_debuglayerskip(const char *path, const char *csv) {
    nf_model *m = nf_model_load(path);
    if (!m) return 1;
    if (!nf_model_is_glm(m)) {
        fprintf(stderr, "nf: debuglayerskip requires a glm-dsa model\n");
        nf_model_free(m);
        return 1;
    }
    int32_t ids[4096];
    int n = 0;
    const char *p = csv;
    while (*p && n < 4096) {
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        ids[n++] = (int32_t)v;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    if (n < 2) {
        fprintf(stderr, "nf: debuglayerskip: need at least 2 CSV tokens\n");
        nf_model_free(m);
        return 1;
    }
    const int64_t vocab = (int64_t)nf_model_vocab(m);
    /* control run: session B, NEVER touched by the draft — a reference
     * to notice if A gets silently corrupted by it. */
    nf_session *sa = nf_session_create(m, n + 8);
    nf_session *sb = nf_session_create(m, n + 8);
    float *dlog = malloc((size_t)vocab * sizeof(float));
    int rc = 1;
    if (!sa || !sb || !dlog) { fprintf(stderr, "nf: debuglayerskip: oom\n"); goto out; }
    {
        int all_finite = 1, corrupt = 0;
        double sum_overlap = 0.0;
        int n_checked = 0;
        for (int k = 1; k < n; k++) {
            const float *lgA = nf_session_eval(sa, &ids[k - 1], 1);
            const float *lgB = nf_session_eval(sb, &ids[k - 1], 1);
            if (!lgA || !lgB) { fprintf(stderr, "nf: debuglayerskip: eval failed at k=%d\n", k); goto out; }
            /* isolation check: A and B have seen the SAME sequence and
             * the same number of draft calls so far (none, they're
             * symmetric) — they must match bit for bit every time. */
            for (int64_t i = 0; i < vocab && !corrupt; i++)
                if (lgA[i] != lgB[i]) corrupt = 1;
            if (nf_glm_draft_skip(sa, ids[k], dlog) != 0) {
                fprintf(stderr, "nf: debuglayerskip: draft failed at k=%d\n", k);
                goto out;
            }
            /* session A must NOT change as a result of the call above:
             * a second eval of the NEXT true token must give the same
             * result it would on B (checked on the next loop iteration,
             * via the lgA==lgB comparison above). */
            int64_t argv_true = 0, argd = 0;
            int64_t top5t[5] = {-1,-1,-1,-1,-1}, top5d[5] = {-1,-1,-1,-1,-1};
            for (int64_t i = 0; i < vocab; i++) {
                if (lgA[i] > lgA[argv_true]) argv_true = i;
                if (dlog[i] > dlog[argd]) argd = i;
                if (!(dlog[i] == dlog[i])) all_finite = 0;   /* NaN */
            }
            /* top-5 (simple selection, small vocab on the tiny model,
             * fine on the real one too for a non-hot-path diagnostic
             * command) */
            for (int r = 0; r < 5; r++) {
                int64_t bt = -1, bd = -1;
                for (int64_t i = 0; i < vocab; i++) {
                    int used_t = 0, used_d = 0;
                    for (int j = 0; j < r; j++) { if (top5t[j]==i) used_t=1; if (top5d[j]==i) used_d=1; }
                    if (!used_t && (bt < 0 || lgA[i] > lgA[bt])) bt = i;
                    if (!used_d && (bd < 0 || dlog[i] > dlog[bd])) bd = i;
                }
                top5t[r] = bt; top5d[r] = bd;
            }
            int overlap = 0;
            for (int a = 0; a < 5; a++)
                for (int b = 0; b < 5; b++)
                    if (top5t[a] == top5d[b]) { overlap++; break; }
            sum_overlap += overlap;
            n_checked++;
            printf("[layerskip] k=%2d  true_argmax=%-6lld draft_argmax=%-6lld  "
                   "top5_overlap=%d/5  %s\n", k, (long long)argv_true,
                   (long long)argd, overlap, argv_true == argd ? "MATCH" : "");
        }
        printf("[layerskip] finite=%s  session_isolated=%s  avg_overlap=%.2f/5 over %d positions\n",
               all_finite ? "YES" : "NO (NaN!)", corrupt ? "NO (CORRUPTED!)" : "YES",
               n_checked > 0 ? sum_overlap / n_checked : 0.0, n_checked);
        rc = (all_finite && !corrupt) ? 0 : 1;
        printf("%s\n", rc == 0
               ? "LAYERSKIP DRAFT-ALONE OK: finite and session isolated"
               : "LAYERSKIP DRAFT-ALONE FAILED");
    }
out:
    if (sa) nf_session_free(sa);
    if (sb) nf_session_free(sb);
    free(dlog);
    nf_model_free(m);
    return rc;
}

/* nf debuglayerskipdraft <gguf> <csv> <n_predict> (step b):
 * the FULL draft-verify loop (layerskip_draft_ids, the same code as
 * --layerskip-draft) against pure greedy generation of the same
 * prompt — same property under test as debugmtpdraft: output
 * INVARIANCE. A single pass (unlike debugmtpdraft, which used two to
 * deliberately force both the accept and the reject branch with a
 * random-weight drafter): the layer-skip drafter reads the model's
 * REAL weights (even on the tiny model), so with a non-trivial
 * n_predict it naturally exercises both branches — the report below
 * counts accepted vs rejected explicitly, so a run that failed to
 * exercise one branch would be immediately visible. */
static int cmd_debuglayerskipdraft(const char *path, const char *csv, int n_predict) {
    nf_model *m = nf_model_load(path);
    if (!m) return 1;
    if (!nf_model_is_glm(m)) {
        fprintf(stderr, "nf: debuglayerskipdraft requires a glm-dsa model\n");
        nf_model_free(m);
        return 1;
    }
    int32_t ids[4096];
    int n = 0;
    const char *p = csv;
    while (*p && n < 4096) {
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        ids[n++] = (int32_t)v;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    if (n <= 0 || n_predict <= 0) {
        fprintf(stderr, "nf: debuglayerskipdraft: invalid CSV tokens or n_predict\n");
        nf_model_free(m);
        return 1;
    }
    const int64_t vocab = (int64_t)nf_model_vocab(m);
    int32_t *base_ids  = malloc((size_t)n_predict * sizeof(int32_t));
    int32_t *draft_ids = malloc((size_t)n_predict * sizeof(int32_t));
    int rc = 1;
    nf_session *sa = NULL, *sb = NULL;
    if (!base_ids || !draft_ids) { fprintf(stderr, "nf: debuglayerskipdraft: oom\n"); goto out; }

    sa = nf_session_create(m, n + n_predict + 8);
    if (!sa) goto out;
    {
        uint64_t rng = 0;
        const float *lg = nf_session_eval(sa, ids, n);
        if (!lg) goto out;
        for (int i = 0; i < n_predict; i++) {
            const int32_t tok = sample_token(lg, vocab, 0.0f, 1.0f, 0, 0.0f, &rng);
            base_ids[i] = tok;
            if (i + 1 < n_predict) {
                lg = nf_session_eval(sa, &tok, 1);
                if (!lg) goto out;
            }
        }
    }
    nf_session_free(sa);
    sa = NULL;

    sb = nf_session_create(m, n + n_predict + 9);
    if (!sb) goto out;
    {
        int n_out = 0, rounds = 0, accepted = 0;
        double dec_s = 0.0;
        printf("[layerskip-draft] baseline: ");
        for (int i = 0; i < n_predict; i++) printf("%d ", base_ids[i]);
        printf("\n");
        if (layerskip_draft_ids(m, sb, ids, n, n_predict, NULL,
                                draft_ids, &n_out, &rounds, &accepted,
                                &dec_s) != 0)
            goto out;
        int same = n_out == n_predict;
        for (int i = 0; same && i < n_predict; i++)
            if (base_ids[i] != draft_ids[i]) same = 0;
        printf("[layerskip-draft] draft   : ");
        for (int i = 0; i < n_out; i++) printf("%d ", draft_ids[i]);
        printf("\n[layerskip-draft] rounds=%d accepted=%d (acc=%.3f) "
               "tokens=%d  both_branches=%s  %s\n",
               rounds, accepted, rounds > 0 ? (double)accepted / rounds : 0.0,
               n_out, (accepted > 0 && accepted < rounds) ? "YES" : "NO (only one branch exercised)",
               same ? "OK" : "FAIL");
        printf("%s\n", same
               ? "LAYERSKIP DRAFT OK: sequence identical to greedy without a draft"
               : "LAYERSKIP DRAFT FAILED: divergent sequences");
        rc = same ? 0 : 1;
    }
out:
    if (sa) nf_session_free(sa);
    if (sb) nf_session_free(sb);
    free(base_ids); free(draft_ids);
    nf_model_free(m);
    return rc;
}

/* M3: prompt prefill + autoregressive generation loop on the KV cache.
 * Emits text to stdout; every generated id also goes to stderr as
 * "GENID <id>" (a clean channel for the tests, which would otherwise
 * have to re-tokenize the output — ambiguous, since decode/encode
 * isn't guaranteed to round-trip exactly). */
/* ---- sweep project Phase A: token-major harness -----------
 * N independent sessions on the same prompt, TOKEN-major greedy decode
 * (stream s completes token t, then s+1) via public API only. Phase
 * gate: every stream produces EXACTLY the sequence a single
 * `nf generate` would (same math, same argmax) — the scaffolding
 * (N sessions, per-stream state, per-stream stop, aggregate
 * measurement) is validated here before Phases B-E touch the engine. */
/* accumulates a token's decoded piece into the stream's text (grow-only;
 * a failed malloc turns off accumulation for that stream, the bench
 * continues — the text is diagnostic, not the result) */
static void sweep_txt_append(char **txt, size_t *len, size_t *cap,
                             nf_tokenizer *tk, int32_t id) {
    if (!tk) return;
    char piece[256];
    const int w = nf_token_decode(tk, id, piece, sizeof(piece) - 1);
    if (w <= 0) return;
    if (*len + (size_t)w + 1 > *cap) {
        size_t nc = *cap ? *cap * 2 : 512;
        while (nc < *len + (size_t)w + 1) nc *= 2;
        char *nt = realloc(*txt, nc);
        if (!nt) return;
        *txt = nt; *cap = nc;
    }
    memcpy(*txt + *len, piece, (size_t)w);
    *len += (size_t)w;
    (*txt)[*len] = 0;
}

/* nf doctor <model> — readiness check (product project, Phase D): the
 * REAL failure points seen in this project, not a generic checklist.
 * Read-only, no model loading. */
/* nf forgelogits <model> <csv-ids> <out.bin> — dumps the EVAL PATH's
 * logits (session, streaming, production kernels) for every position,
 * teacher-forced. File format: [n u32][vocab u64] then n x vocab f32.
 * Feeds the oracle gate (tools/forge_oracle.py): the comparison is
 * production-engine against numpy-derived-from-HF. */
static int cmd_forgelogits(const char *path, const char *csv, const char *out) {
    nf_model *m = nf_model_load(path);
    if (!m) return 1;
    int32_t ids[512];
    int n = 0;
    const char *p = csv;
    while (*p && n < 512) {
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        ids[n++] = (int32_t)v;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    if (n <= 0) { nf_model_free(m); return 1; }
    const int64_t vocab = (int64_t)nf_model_vocab(m);
    nf_session *s = nf_session_create(m, n + 8);
    float *all = malloc((size_t)n * vocab * sizeof(float));
    if (!s || !all) { fprintf(stderr, "nf: oom/session\n"); return 1; }
    if (!nf_session_eval_all(s, ids, n, all)) {
        fprintf(stderr, "nf: eval_all failed\n");
        return 1;
    }
    FILE *f = fopen(out, "wb");
    if (!f) return 1;
    uint32_t n32 = (uint32_t)n;
    uint64_t v64 = (uint64_t)vocab;
    fwrite(&n32, 4, 1, f);
    fwrite(&v64, 8, 1, f);
    fwrite(all, sizeof(float), (size_t)n * vocab, f);
    fclose(f);
    fprintf(stderr, "nf: [forgelogits] %d positions x %lld logits -> %s\n",
            n, (long long)vocab, out);
    nf_debug_timing_report();
    free(all);
    nf_session_free(s);
    nf_model_free(m);
    return 0;
}

static int cmd_doctor(const char *path) {
    int warn = 0, fail = 0;
    printf("nf doctor - %s\n", path);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("[FAIL] file not readable\n"); return 1; }
    char magic[4] = {0};
    fread(magic, 1, 4, f);
    fclose(f);
    const int forge = is_forge_path(path);
    if (forge) {
        if (memcmp(magic, "NFRG", 4) != 0) {
            printf("[FAIL] wrong .forge magic (corrupt or incomplete container)\n");
            return 1;
        }
        printf("[ ok ] .forge container (magic NFRG)\n");
        char ipath[1100];
        snprintf(ipath, sizeof(ipath), "%s.idx", path);
        FILE *fi = fopen(ipath, "rb");
        if (!fi) {
            printf("[FAIL] missing index: python tools/forge_idx.py %s\n", path);
            fail++;
        } else {
            uint32_t im = 0, iv = 0;
            fread(&im, 4, 1, fi); fread(&iv, 4, 1, fi);
            fclose(fi);
            if (im != 0x5849464e || iv != 2) {
                printf("[FAIL] corrupt index or version %u != 2: regenerate it\n", iv);
                fail++;
            } else printf("[ ok ] .idx index v2\n");
        }
        const char *tok = getenv("NF_TOKENIZER");
        if (!tok || !*tok) {
            printf("[FAIL] NF_TOKENIZER not set (a .forge doesn't carry the "
                   "tokenizer): NF_TOKENIZER=<gguf with the same tokenizer>\n");
            fail++;
        } else {
            char *arch = nf_gguf_read_kv_str(tok, "general.architecture");
            if (!arch) {
                printf("[FAIL] NF_TOKENIZER is not a readable GGUF: %s\n", tok);
                fail++;
            } else {
                printf("[ ok ] tokenizer from %s (arch %s)\n", tok, arch);
                if (strcmp(arch, "glm-dsa") != 0) {
                    printf("[warn] the tokenizer's arch isn't glm-dsa: "
                           "the vocabulary is probably wrong\n");
                    warn++;
                }
                free(arch);
            }
        }
        const char *sg = getenv("NF_GLM_STREAM_GB");
        printf("[ ok ] expert streaming: NF_GLM_STREAM_GB=%s%s\n",
               sg && *sg ? sg : "15",
               sg && *sg ? "" : " (auto-config default for .forge)");
    } else {
        if (memcmp(magic, "GGUF", 4) != 0) {
            printf("[FAIL] neither a GGUF nor a .forge\n");
            return 1;
        }
        char *arch = nf_gguf_read_kv_str(path, "general.architecture");
        printf("[ ok ] GGUF (arch %s)\n", arch ? arch : "?");
        free(arch);
    }
#ifdef _WIN32
    {
        MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) {
            const double free_gb = (double)ms.ullAvailPhys / 1e9;
            printf("[%s] RAM available NOW: %.1f GB", free_gb >= 22.0 ? " ok " : "warn", free_gb);
            if (free_gb < 22.0) {
                printf("  (a 744B GLM wants ~22GB between resident densities "
                       "and cache: close heavy applications)");
                warn++;
            }
            printf("\n");
        }
    }
#endif
    printf(fail ? "RESULT: FAIL (%d)\n" : warn ?
           "RESULT: ok with %d warnings\n" : "RESULT: ok\n",
           fail ? fail : warn);
    return fail ? 1 : 0;
}

static int cmd_sweepgen(const char *path, const char *text, int n_streams,
                        int n_predict, int max_ctx, int csv,
                        float temp, float top_p, int32_t top_k,
                        uint64_t seed) {
    if (n_streams < 1) n_streams = 1;
    if (n_streams > 64) n_streams = 64;
    /* --csv (tiny, no tokenizer): no tokenizer, no special stopset —
     * the ids are already numeric and the bench never stops on
     * textual stops. */
    const char *toksrc = csv ? NULL : tok_source_for(path);
    nf_tokenizer *tk = toksrc ? nf_tokenizer_load(toksrc) : NULL;
    if (!csv && !tk) return 1;
    int32_t ids[4096];
    int64_t n;
    if (csv) {
        /* --csv: comma-separated numeric ids (the tiny glm models don't
         * have a textual tokenizer — same scheme as debugglm) */
        n = 0;
        const char *p = text;
        while (*p && n < 4096) {
            char *end;
            long v = strtol(p, &end, 10);
            if (end == p) break;
            ids[n++] = (int32_t)v;
            p = *end == ',' ? end + 1 : end;
        }
    } else {
        n = nf_tokenize(tk, text, ids, 4096);
    }
    if (n <= 0 || n > 4096) {
        fprintf(stderr, "nf: prompt empty or too long\n");
        nf_tokenizer_free(tk);
        return 1;
    }
    nf_model *m = nf_model_load(path);
    if (!m) { nf_tokenizer_free(tk); return 1; }
    if (max_ctx < (int)n + n_predict) max_ctx = (int)n + n_predict;
    chat_style_init(path);
    nf_stopset stop;
    build_stopset(path, &stop);
    if (tk) stopset_add_special(tk, &stop, "<|endoftext|>");

    nf_session **ss = calloc((size_t)n_streams, sizeof(nf_session *));
    int32_t *last = malloc((size_t)n_streams * sizeof(int32_t));
    int *done = calloc((size_t)n_streams, sizeof(int));
    /* best-of-N: with --temp > 0 each stream samples with its own RNG
     * (base seed + per-stream splitmix offset) — same prompt, N
     * DIFFERENT completions at the sweep's aggregate cost. With temp=0
     * sample_token is the production argmax (identical to the old one). */
    uint64_t *rngs = malloc((size_t)n_streams * sizeof(uint64_t));
    /* per-stream text (only with a tokenizer): needed for the
     * multi-stream QA gate — GENIDs alone can't be judged by eye. */
    char **txt = calloc((size_t)n_streams, sizeof(char *));
    size_t *txt_len = calloc((size_t)n_streams, sizeof(size_t));
    size_t *txt_cap = calloc((size_t)n_streams, sizeof(size_t));
    if (!ss || !last || !done || !rngs || !txt || !txt_len || !txt_cap)
        goto oom;
    for (int s = 0; s < n_streams; s++)
        rngs[s] = seed + (uint64_t)s * 0x9E3779B97F4A7C15ull;
    for (int s = 0; s < n_streams; s++) {
        ss[s] = nf_session_create(m, max_ctx);
        if (!ss[s]) {
            fprintf(stderr, "nf: session %d failed (RAM? a smaller "
                    "--ctx, or NF_GLM_MLA_ABSORB=1 on glm)\n", s);
            goto oom;
        }
    }
    const int64_t vocab = (int64_t)nf_model_vocab(m);
    const double t_start = (double)time(NULL);
    /* Phase A = token-major (reference); NF_SWEEP_PHASE=B = LAYER-MAJOR
     * via nf_glm_eval_sweep (glm+absorb only) — the phase gate is an
     * IDENTICAL sequence between the two. Note for B: "done" streams
     * stay in lockstep without emitting. */
    const char *phase_e = getenv("NF_SWEEP_PHASE");
    const int use_b = phase_e && phase_e[0] == 'B' && nf_model_is_glm(m);
    float *lgbuf = use_b
        ? malloc((size_t)n_streams * (size_t)vocab * sizeof(float)) : NULL;
    if (use_b && !lgbuf) goto oom;
    if (use_b) fprintf(stderr, "nf: [sweepgen] PHASE B: layer-major "
                       "(prefill included)\n");
    /* prefill: in B the PROMPT goes through the sweep too — one
     * layer-major call per POSITION (all streams together), so the
     * dense dequant amortizes over prefill too (it was 60% of wall
     * time on the short bench, 8 sequential prefills). In A the
     * per-stream reference path stays as is. */
    if (use_b) {
        for (int p = 0; p < (int)n; p++) {
            for (int s = 0; s < n_streams; s++) last[s] = ids[p];
            if (nf_glm_eval_sweep(ss, n_streams, last, lgbuf) != 0)
                goto fail;
        }
        for (int s = 0; s < n_streams; s++) {
            const float *lg = lgbuf + (size_t)s * vocab;
            const int32_t best = sample_token(lg, vocab, temp, top_p, top_k,
                                              0.0f, &rngs[s]);
            last[s] = best;
            fprintf(stderr, "SW%d GENID %d\n", s, best);
            sweep_txt_append(&txt[s], &txt_len[s], &txt_cap[s], tk, best);
            if (is_stop_token(&stop, best)) done[s] = 1;
        }
    } else {
        for (int s = 0; s < n_streams; s++) {
            const float *lg = nf_session_eval(ss[s], ids, (int)n);
            if (!lg) goto fail;
            const int32_t best = sample_token(lg, vocab, temp, top_p, top_k,
                                              0.0f, &rngs[s]);
            last[s] = best;
            fprintf(stderr, "SW%d GENID %d\n", s, best);
            sweep_txt_append(&txt[s], &txt_len[s], &txt_cap[s], tk, best);
            if (is_stop_token(&stop, best)) done[s] = 1;
        }
    }
    int emitted = n_streams;
    for (int t = 1; t < n_predict; t++) {
        if (use_b) {
            if (nf_glm_eval_sweep(ss, n_streams, last, lgbuf) != 0)
                goto fail;
            for (int s = 0; s < n_streams; s++) {
                if (done[s]) continue;
                const float *lg = lgbuf + (size_t)s * vocab;
                const int32_t best = sample_token(lg, vocab, temp, top_p, top_k,
                                                  0.0f, &rngs[s]);
                if (s == 0 && getenv("NF_SWEEP_LOGIT_DBG")) {
                    int32_t top = 0, second = -1;
                    for (int64_t v = 1; v < vocab; v++)
                        if (lg[v] > lg[top]) top = (int32_t)v;
                    for (int64_t v = 0; v < vocab; v++)
                        if ((int32_t)v != top &&
                            (second < 0 || lg[v] > lg[second]))
                            second = (int32_t)v;
                    fprintf(stderr, "LGDBG t=%d best=%d %.6f second=%d %.6f "
                            "margin=%.6f\n", t, top, lg[top], second,
                            lg[second], lg[top] - lg[second]);
                }
                last[s] = best;
                emitted++;
                fprintf(stderr, "SW%d GENID %d\n", s, best);
                sweep_txt_append(&txt[s], &txt_len[s], &txt_cap[s], tk, best);
                if (is_stop_token(&stop, best)) done[s] = 1;
            }
            continue;
        }
        for (int s = 0; s < n_streams; s++) {
            if (done[s]) continue;
            const float *lg = nf_session_eval(ss[s], &last[s], 1);
            if (!lg) goto fail;
            const int32_t best = sample_token(lg, vocab, temp, top_p, top_k,
                                              0.0f, &rngs[s]);
            if (s == 0 && getenv("NF_SWEEP_LOGIT_DBG")) {
                int32_t top = 0, second = -1;
                for (int64_t v = 1; v < vocab; v++)
                    if (lg[v] > lg[top]) top = (int32_t)v;
                for (int64_t v = 0; v < vocab; v++)
                    if ((int32_t)v != top &&
                        (second < 0 || lg[v] > lg[second]))
                        second = (int32_t)v;
                fprintf(stderr, "LGDBG t=%d best=%d %.6f second=%d %.6f "
                        "margin=%.6f\n", t, top, lg[top], second,
                        lg[second], lg[top] - lg[second]);
            }
            last[s] = best;
            emitted++;
            fprintf(stderr, "SW%d GENID %d\n", s, best);
            sweep_txt_append(&txt[s], &txt_len[s], &txt_cap[s], tk, best);
            if (is_stop_token(&stop, best)) done[s] = 1;
        }
    }
    free(lgbuf);
    {
        const double dt = (double)time(NULL) - t_start;
        fprintf(stderr, "nf: [sweepgen] %d streams, %d total tokens in %.0fs "
                "(%.3f tok/s aggregate)\n",
                n_streams, emitted, dt, dt > 0 ? emitted / dt : 0.0);
    }
    nf_debug_timing_report();   /* no-op unless NF_DEBUG_TIMING */
    /* per-stream text to stdout: the material for the multi-stream QA
     * gate (and the useful output of best-of-N with --temp) */
    if (tk) for (int s = 0; s < n_streams; s++)
        printf("--- stream %d ---\n%s\n", s, txt[s] ? txt[s] : "");
    for (int s = 0; s < n_streams; s++) nf_session_free(ss[s]);
    for (int s = 0; s < n_streams; s++) if (txt) free(txt[s]);
    free(ss); free(last); free(done);
    free(rngs); free(txt); free(txt_len); free(txt_cap);
    nf_model_free(m); nf_tokenizer_free(tk);
    return 0;
fail:
    fprintf(stderr, "nf: [sweepgen] eval failed\n");
oom:
    if (ss) for (int s = 0; s < n_streams; s++) if (ss[s]) nf_session_free(ss[s]);
    if (txt) for (int s = 0; s < n_streams; s++) free(txt[s]);
    free(ss); free(last); free(done);
    free(rngs); free(txt); free(txt_len); free(txt_cap);
    nf_model_free(m); nf_tokenizer_free(tk);
    return 1;
}

static int cmd_generate(const char *path, const char *text, int n_predict,
                        float temp, float top_p, int32_t top_k, float min_p,
                        int max_ctx, uint64_t seed,
                        float repeat_penalty, int repeat_last_n,
                        const char *draft_path, int draft_k, int mtp_draft,
                        int layerskip_draft) {
    glm_daily_env(path);   /* before any load/gate, see the comment there */
    /* batched+draft guard: same as cmd_chat. */
    if ((draft_path || mtp_draft || layerskip_draft) &&
        getenv("NF_GLM_PREFILL_BATCH")) {
        nf_setenv("NF_GLM_PREFILL_BATCH=0");
        fprintf(stderr, "nf: batched prefill DISABLED under draft "
                "(interaction not validated)\n");
    }
    /* Phase 2a: same incompatibility as cmd_chat — see the comment
     * there for why (speculative decoding wants a batch, disk-backing
     * always decomposes it into single calls). */
    if (draft_path && getenv("NF_LONGMEM_DISK")) {
        fprintf(stderr, "nf: --draft and NF_LONGMEM_DISK are incompatible "
                "(speculative decoding wants a batch, disk-backing "
                "always decomposes it into single calls) — pick one "
                "of the two\n");
        return 1;
    }
    /* --mtp-draft: same declared restrictions as --draft
     * (greedy-only), plus its own — v1 deliberately narrow. */
    if (mtp_draft) {
        if (draft_path) {
            fprintf(stderr, "nf: --mtp-draft and --draft are incompatible "
                    "(two speculative mechanisms on the same session)\n");
            return 1;
        }
        if (temp > 0.0f) {
            fprintf(stderr, "nf: --mtp-draft requires greedy (--temp 0)\n");
            return 1;
        }
        if (repeat_penalty != 1.0f) {
            fprintf(stderr, "nf: --mtp-draft doesn't support "
                    "--repeat-penalty (v1 pure greedy: the penalty would "
                    "change the verified argmax)\n");
            return 1;
        }
        /* NF_GLM_PREFETCH: v1 restriction lifted — back then the
         * cross-layer prefetch covered ONLY
         * sequential n_new==1 decode (glm_eval_verify never queued it),
         * so the interaction with the draft round's rewind was truly
         * untested. Today glm_eval_verify queues the per-position
         * prefetch itself (DraftExpert, see the comment at the call
         * site inside the routing loop) and the combination is
         * validated: `nf debugmtpdraft` with
         * NF_GLM_STREAM_GB+NF_GLM_PREFETCH=1 active produces the SAME
         * sequence as pure greedy on all three tiny models (rewind
         * included). */
    }
    /* --layerskip-draft: same declared restrictions as
     * --mtp-draft (greedy-only, no repeat-penalty, only one speculative
     * mechanism at a time) — BUT compatible with NF_GLM_PREFETCH
     * (unlike --mtp-draft): the layer-skip drafter never touches the
     * shared fetch/cache in a new way, it uses nf_glm_stream_fetch
     * exactly like glm_eval_core (drains the prefetch queue the same
     * way by construction) — v1 still doesn't exercise it together
     * with prefetch in validation, an incompatibility with --mtp-draft
     * itself remains (two speculative mechanisms on the same session,
     * never tested together). */
    if (layerskip_draft) {
        if (draft_path) {
            fprintf(stderr, "nf: --layerskip-draft and --draft are "
                    "incompatible (two speculative mechanisms on the "
                    "same session)\n");
            return 1;
        }
        if (mtp_draft) {
            fprintf(stderr, "nf: --layerskip-draft and --mtp-draft are "
                    "incompatible (v1, two drafters on the same "
                    "session) — pick one of the two\n");
            return 1;
        }
        if (temp > 0.0f) {
            fprintf(stderr, "nf: --layerskip-draft requires greedy (--temp 0)\n");
            return 1;
        }
        if (repeat_penalty != 1.0f) {
            fprintf(stderr, "nf: --layerskip-draft doesn't support "
                    "--repeat-penalty (v1 pure greedy: the penalty would "
                    "change the verified argmax)\n");
            return 1;
        }
    }
    const char *toksrc = tok_source_for(path);   /* .forge: NF_TOKENIZER */
    if (!toksrc) return 1;
    nf_tokenizer *tk = nf_tokenizer_load(toksrc);
    if (!tk) return 1;

    int32_t ids[4096];
    const int64_t n = nf_tokenize(tk, text, ids, 4096);
    if (n <= 0 || n > 4096) {
        fprintf(stderr, "nf: prompt empty or too long\n");
        nf_tokenizer_free(tk);
        return 1;
    }

    nf_model *m = nf_model_load(path);
    if (!m) { nf_tokenizer_free(tk); return 1; }

    if (mtp_draft && !nf_model_is_glm(m)) {
        fprintf(stderr, "nf: --mtp-draft requires a glm-dsa model\n");
        nf_model_free(m); nf_tokenizer_free(tk);
        return 1;
    }
    if (mtp_draft && !nf_model_has_mtp(m)) {
        fprintf(stderr, "nf: --mtp-draft requires the MTP head loaded: "
                "set NF_GLM_MTP=1 (and a GGUF with nextn tensors)\n");
        nf_model_free(m); nf_tokenizer_free(tk);
        return 1;
    }
    if (layerskip_draft && !nf_model_is_glm(m)) {
        fprintf(stderr, "nf: --layerskip-draft requires a glm-dsa model\n");
        nf_model_free(m); nf_tokenizer_free(tk);
        return 1;
    }

    if (max_ctx < (int)n + n_predict) max_ctx = (int)n + n_predict;
    if (mtp_draft || layerskip_draft) max_ctx += 1;   /* the rejected
                                    * draft occupies (then reuses) one
                                    * position beyond the last accepted
                                    * token — same pattern for both
                                    * drafters */
    nf_session *s = nf_session_create(m, max_ctx);
    if (!s) {
        /* Failure made LOUD — on glm-dsa the session cache
         * weighs ~4.4MB/position (f16 Gkv reconstructed over 78
         * layers): with the --ctx 2048 default that's ~9GB, and under
         * commit pressure the process used to exit here with code 1
         * without ONE line (actually observed on real GLM-5.2, looked
         * like a load crash). A smaller --ctx reduces the requirement. */
        fprintf(stderr, "nf: session creation failed (max_ctx=%d — "
                "out of memory? try a smaller --ctx)\n",
                max_ctx);
        nf_model_free(m); nf_tokenizer_free(tk);
        return 1;
    }

    /* g_chat_style isn't needed elsewhere here (nf generate doesn't
     * template turns), but it's the cheapest way to recognize glm-dsa
     * without duplicating the general.architecture read just for the
     * stopset below — see the comment on chat_style_init/nf_stopset. */
    chat_style_init(path);
    nf_stopset stop;
    build_stopset(toksrc, &stop);
    stopset_add_special(tk, &stop, "<|endoftext|>");
    if (g_chat_style == 3) {
        stopset_add_special(tk, &stop, "<|user|>");
        stopset_add_special(tk, &stop, "<|observation|>");
        /* see the twin comment in cmd_chat: an <|assistant|> emitted by
         * the model is post-response degeneration, effectively a turn end. */
        stopset_add_special(tk, &stop, "<|assistant|>");
    }
    nf_textstop tstop;
    build_textstops(tk, &stop, &tstop);

    if (draft_path) {
        if (temp > 0.0f) {
            fprintf(stderr, "nf: --draft requires greedy (--temp 0)\n");
            nf_session_free(s); nf_model_free(m); nf_tokenizer_free(tk);
            return 1;
        }
        fputs(text, stdout);
        fflush(stdout);
        const int rc = spec_generate(tk, m, s, &stop, &tstop, draft_path,
                                     draft_k, ids, (int)n, n_predict,
                                     max_ctx, top_p, seed,
                                     repeat_penalty, repeat_last_n);
        nf_session_free(s); nf_model_free(m); nf_tokenizer_free(tk);
        return rc;
    }

    if (mtp_draft) {
        /* The ID loop (live GENID, for external timestamping) and then
         * text emission with the same steps as the normal path
         * (id-based stop, textual filter). Declared v1 limit: a
         * TEXTUAL stop (not id-based) truncates the already-generated
         * output but doesn't interrupt generation upstream — for
         * `nf generate` on glm-dsa the real stops are ids
         * (endoftext/user/observation), handled inside the loop. */
        int32_t *oids = malloc((size_t)(n_predict > 0 ? n_predict : 1)
                               * sizeof(int32_t));
        int n_out = 0, rounds = 0, accepted = 0;
        double dec_s = 0.0;
        int rc = 1;
        if (oids) {
            fputs(text, stdout);
            fflush(stdout);
            rc = mtp_draft_ids(m, s, ids, (int)n, n_predict, &stop, NULL,
                               oids, &n_out, &rounds, &accepted,
                               &dec_s) == 0 ? 0 : 1;
        } else {
            fprintf(stderr, "nf: [mtp-draft] oom\n");
        }
        if (rc == 0) {
            char pend[512];
            int n_pending = 0, n_generated = 0;
            for (int i = 0; i < n_out; i++) {
                const int32_t tok = oids[i];
                if (is_stop_token(&stop, tok)) break;
                char piece[256];
                const int w = nf_token_decode(tk, tok, piece, sizeof(piece) - 1);
                n_generated++;
                if (stream_piece_through_stopfilter(piece, w > 0 ? w : 0,
                                                    pend, &n_pending,
                                                    (int)sizeof(pend), &tstop))
                    break;
            }
            if (n_pending > 0) {
                fwrite(pend, 1, (size_t)n_pending, stdout);
                fflush(stdout);
            }
            printf("\n");
            fprintf(stderr, "nf: generated %d tokens (temp=%.2f top_p=%.2f "
                    "seed=%" PRIu64 ")\n", n_generated, temp, top_p, seed);
            fprintf(stderr, "nf: [mtp-draft] rounds=%d accepted=%d "
                    "(acc=%.3f) tokens=%d decode=%.1fs (%.2f s/token)\n",
                    rounds, accepted,
                    rounds > 0 ? (double)accepted / rounds : 0.0,
                    n_out, dec_s, n_out > 0 ? dec_s / n_out : 0.0);
        }
        nf_debug_timing_report();
        free(oids);
        nf_session_free(s); nf_model_free(m); nf_tokenizer_free(tk);
        return rc;
    }

    if (layerskip_draft) {
        /* Same scheme as the --mtp-draft block above, with
         * layerskip_draft_ids instead of mtp_draft_ids. */
        int32_t *oids = malloc((size_t)(n_predict > 0 ? n_predict : 1)
                               * sizeof(int32_t));
        int n_out = 0, rounds = 0, accepted = 0;
        double dec_s = 0.0;
        int rc = 1;
        if (oids) {
            fputs(text, stdout);
            fflush(stdout);
            rc = layerskip_draft_ids(m, s, ids, (int)n, n_predict, &stop,
                                     oids, &n_out, &rounds, &accepted,
                                     &dec_s) == 0 ? 0 : 1;
        } else {
            fprintf(stderr, "nf: [layerskip-draft] oom\n");
        }
        if (rc == 0) {
            char pend[512];
            int n_pending = 0, n_generated = 0;
            for (int i = 0; i < n_out; i++) {
                const int32_t tok = oids[i];
                if (is_stop_token(&stop, tok)) break;
                char piece[256];
                const int w = nf_token_decode(tk, tok, piece, sizeof(piece) - 1);
                n_generated++;
                if (stream_piece_through_stopfilter(piece, w > 0 ? w : 0,
                                                    pend, &n_pending,
                                                    (int)sizeof(pend), &tstop))
                    break;
            }
            if (n_pending > 0) {
                fwrite(pend, 1, (size_t)n_pending, stdout);
                fflush(stdout);
            }
            printf("\n");
            fprintf(stderr, "nf: generated %d tokens (temp=%.2f top_p=%.2f "
                    "seed=%" PRIu64 ")\n", n_generated, temp, top_p, seed);
            fprintf(stderr, "nf: [layerskip-draft] rounds=%d accepted=%d "
                    "(acc=%.3f) tokens=%d decode=%.1fs (%.2f s/token)\n",
                    rounds, accepted,
                    rounds > 0 ? (double)accepted / rounds : 0.0,
                    n_out, dec_s, n_out > 0 ? dec_s / n_out : 0.0);
        }
        nf_debug_timing_report();
        free(oids);
        nf_session_free(s); nf_model_free(m); nf_tokenizer_free(tk);
        return rc;
    }

    const float *logits = nf_session_eval(s, ids, (int)n);
    if (!logits) {
        nf_session_free(s); nf_model_free(m); nf_tokenizer_free(tk);
        return 1;
    }
    dump_init_logits_if_requested(logits, (int64_t)nf_model_vocab(m));

    fputs(text, stdout);
    fflush(stdout);

    /* full history (prompt + generated) for the repetition penalty
     * window — repetition counts even if the repeated text came from
     * the prompt, not only from what the model has already generated
     * this turn. */
    int32_t *hist = malloc((size_t)max_ctx * sizeof(int32_t));
    int n_hist = (int)n;
    if (hist) memcpy(hist, ids, (size_t)n * sizeof(int32_t));

    uint64_t rng = seed;
    const int64_t vocab = (int64_t)nf_model_vocab(m);
    int n_generated = 0;
    char pending[512];
    int n_pending = 0;
    for (int i = 0; i < n_predict; i++) {
        if (hist) {
            const int win = n_hist < repeat_last_n ? n_hist : repeat_last_n;
            apply_repeat_penalty((float *)logits, vocab,
                                hist + n_hist - win, win, repeat_penalty);
        }
        const int32_t next = sample_token(logits, vocab, temp, top_p,
                                          top_k, min_p, &rng);
        fprintf(stderr, "GENID %d\n", next);
        if (hist && n_hist < max_ctx) hist[n_hist++] = next;
        if (is_stop_token(&stop, next)) break;

        char piece[256];
        const int w = nf_token_decode(tk, next, piece, sizeof(piece) - 1);
        const int plen = w > 0 ? w : 0;
        n_generated++;

        if (stream_piece_through_stopfilter(piece, plen, pending, &n_pending,
                                            (int)sizeof(pending), &tstop))
            break;

        logits = nf_session_eval(s, &next, 1);
        if (!logits) break;
    }
    if (n_pending > 0) { fwrite(pending, 1, (size_t)n_pending, stdout); fflush(stdout); }
    printf("\n");
    fprintf(stderr, "nf: generated %d tokens (temp=%.2f top_p=%.2f seed=%" PRIu64 ")\n",
            n_generated, temp, top_p, seed);

    nf_debug_timing_report();   /* profile from generate too */
    nf_route_profile_report();  /* no-op unless NF_GLM_ROUTE_PROFILE */
    free(hist);
    nf_session_free(s);
    nf_model_free(m);
    nf_tokenizer_free(tk);
    return 0;
}

/* M3: the real correctness check for this milestone. For every prefix
 * of the prompt, compares nf_model_forward (full recompute, validated
 * in M2) against nf_session_eval fed one token at a time (the cached
 * path that generation relies on): same math, same operation order —
 * they must match almost bit for bit. If they don't, the bug is in the
 * cache indexing or the position passed to RoPE, not in the model's
 * math (already verified against the external oracle in M2). */
static int cmd_selfcheck(const char *path, const char *text) {
    nf_tokenizer *tk = nf_tokenizer_load(path);
    if (!tk) return 1;

    int32_t ids[256];
    const int64_t n = nf_tokenize(tk, text, ids, 256);
    if (n <= 0 || n > 256) {
        fprintf(stderr, "nf: prompt empty or too long for selfcheck\n");
        nf_tokenizer_free(tk);
        return 1;
    }

    nf_model *m = nf_model_load(path);
    if (!m) { nf_tokenizer_free(tk); return 1; }
    nf_session *s = nf_session_create(m, (int)n + 8);
    if (!s) { nf_model_free(m); nf_tokenizer_free(tk); return 1; }

    const int64_t vocab = (int64_t)nf_model_vocab(m);
    int all_ok = 1;
    for (int64_t k = 1; k <= n; k++) {
        const float *full = nf_model_forward(m, ids, (int)k);
        const float *inc = nf_session_eval(s, &ids[k - 1], 1);
        if (!full || !inc) { all_ok = 0; break; }

        int64_t argmax_f = 0, argmax_i = 0;
        float maxd = 0.0f;
        for (int64_t i = 0; i < vocab; i++) {
            if (full[i] > full[argmax_f]) argmax_f = i;
            if (inc[i] > inc[argmax_i]) argmax_i = i;
            const float d = fabsf(full[i] - inc[i]);
            if (d > maxd) maxd = d;
        }
        const int ok = argmax_f == argmax_i && maxd < 1e-2f;
        if (!ok) all_ok = 0;
        printf("k=%2" PRId64 "  argmax full=%-7" PRId64 " inc=%-7" PRId64
               "  max|d|=%.6f  %s\n", k, argmax_f, argmax_i, (double)maxd,
               ok ? "OK" : "FAIL");
    }
    printf("\n%s\n", all_ok
           ? "SELFCHECK OK: incremental cache == full forward"
           : "SELFCHECK FAILED");

    /* The loop above calls nf_session_eval with n_new=1 at every step:
     * it NEVER exercises the batched path (n_new>1, dense matmul,
     * per-expert MoE dispatch) — a whole prompt prefilled in ONE call,
     * the way `generate`/`chat` actually do it. A clean session, the
     * whole prompt at once, compared against the same full forward
     * above (k=n, already computed). */
    nf_session *s2 = nf_session_create(m, (int)n + 8);
    int batch_ok = 0;
    if (s2) {
        const float *full_n = nf_model_forward(m, ids, (int)n);
        const float *inc_n = nf_session_eval(s2, ids, (int)n);
        if (full_n && inc_n) {
            int64_t argmax_f = 0, argmax_i = 0;
            float maxd = 0.0f;
            for (int64_t i = 0; i < vocab; i++) {
                if (full_n[i] > full_n[argmax_f]) argmax_f = i;
                if (inc_n[i] > inc_n[argmax_i]) argmax_i = i;
                const float d = fabsf(full_n[i] - inc_n[i]);
                if (d > maxd) maxd = d;
            }
            batch_ok = argmax_f == argmax_i && maxd < 1e-2f;
            printf("batch (n_new=%" PRId64 ")  argmax full=%-7" PRId64
                   " inc=%-7" PRId64 "  max|d|=%.6f  %s\n",
                   n, argmax_f, argmax_i, (double)maxd,
                   batch_ok ? "OK" : "FAIL");
        }
        nf_session_free(s2);
    }
    printf("%s\n", batch_ok
           ? "SELFCHECK BATCH OK: single-shot prefill == full forward"
           : "SELFCHECK BATCH FAILED");

    nf_session_free(s);
    nf_model_free(m);
    nf_tokenizer_free(tk);
    return (all_ok && batch_ok) ? 0 : 1;
}

/* Perplexity over a corpus — the tool that turns quality choices
 * (weight quant, kv-q8 vs f16, draft quant) into comparable numbers
 * instead of impressions.
 *
 * FIXED protocol (the numbers are only comparable to each other, not
 * to llama-perplexity, which uses a different protocol): NON-overlapping
 * windows of `win` tokens, every position j in [0, win-2] contributes
 * -log p(token[j+1] | token[0..j]); PPL = exp(mean NLL). Every window
 * starts from an empty cache (nf_session_rewind(s, 0) — the remaining
 * K/V rows get overwritten). The window is evaluated in 64-position
 * sub-blocks via nf_session_eval_all (the logits of EVERY position are
 * needed here, not just the last one): the buffer stays 64 x vocab
 * floats (~39MB on Qwen3), never win x vocab (~311MB at 512) — a RAM
 * constraint. The last position of each sub-block predicts the FIRST
 * token of the next one: the true token is still inside the window, no
 * special stitching; only the very last position of the very last
 * window is skipped. Per-position logsumexp in parallel (OpenMP,
 * independent positions, reduction on nll) with the max trick for
 * stability. */
static int cmd_perplexity(const char *path, const char *text,
                          int win, int max_windows) {
    nf_tokenizer *tk = nf_tokenizer_load(path);
    if (!tk) return 1;

    enum { MAX_CORPUS = 1 << 16, PPL_CHUNK = 64 };
    int32_t *ids = malloc(MAX_CORPUS * sizeof(int32_t));
    if (!ids) { nf_tokenizer_free(tk); return 1; }
    const int64_t n = nf_tokenize(tk, text, ids, MAX_CORPUS);
    if (n < win + 1) {
        fprintf(stderr, "nf: corpus too short (%" PRId64 " tokens, "
                "need at least %d)\n", n, win + 1);
        free(ids); nf_tokenizer_free(tk);
        return 1;
    }

    nf_model *m = nf_model_load(path);
    if (!m) { free(ids); nf_tokenizer_free(tk); return 1; }
    nf_session *s = nf_session_create(m, win + 8);
    const int64_t vocab = (int64_t)nf_model_vocab(m);
    float *va = malloc((size_t)PPL_CHUNK * vocab * sizeof(float));
    if (!s || !va) {
        free(va); if (s) nf_session_free(s);
        nf_model_free(m); free(ids); nf_tokenizer_free(tk);
        return 1;
    }

    int n_win = (int)(n / win);
    if (max_windows > 0 && n_win > max_windows) n_win = max_windows;
    fprintf(stderr, "nf: perplexity over %d windows of %d tokens "
            "(corpus of %" PRId64 ")\n", n_win, win, n);

    /* NF_PPL_ROWS: evaluates through the ROWS path (mixed ubatch)
     * instead of nf_session_eval_all — i.e. chunked prefill MIXED with
     * a "decoy" session's decode. This is the only way to make PPL
     * judge the path that actually runs in production with mixed
     * batching: a clean chunking, with no other sessions' rows in the
     * batch, wouldn't exercise the property under test (the result
     * depends on the batch's composition). */
    const int ppl_rows = getenv("NF_PPL_ROWS") != NULL
                       && nf_model_kind_of(m) != NF_MODEL_PLAIN;
    nf_session *decoy = NULL;
    if (ppl_rows) {
        decoy = nf_session_create(m, win + 8);
        if (!decoy || !nf_session_eval(decoy, ids, 8)) {
            fprintf(stderr, "nf: [ppl-rows] decoy failed\n");
            goto fail;
        }
        fprintf(stderr, "nf: [ppl-rows] evaluating via dsk_eval_batched_rows "
                "(chunk 16, MIXED with a second session's decode)\n");
    }

    const double t0 = now_seconds();
    double nll = 0.0;
    int64_t counted = 0;
    for (int w = 0; w < n_win; w++) {
        const int32_t *wt = ids + (size_t)w * win;
        nf_session_rewind(s, 0);
        for (int c0 = 0; c0 < win; c0 += PPL_CHUNK) {
            const int cn = win - c0 < PPL_CHUNK ? win - c0 : PPL_CHUNK;
            const int ok = ppl_rows
                ? nf_ppl_eval_rows(s, decoy, wt + c0, cn, 16, va) == 0
                : nf_session_eval_all(s, wt + c0, cn, va) != NULL;
            if (!ok) {
                fprintf(stderr, "nf: eval failed (window %d)\n", w);
                goto fail;
            }
            double cnll = 0.0;
            int i;
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) private(i) reduction(+:cnll)
            #endif
            for (i = 0; i < cn; i++) {
                const int j = c0 + i;          /* position within the window */
                if (j + 1 >= win) continue;    /* the last one has no target */
                const float *lg = va + (size_t)i * vocab;
                float mx = lg[0];
                for (int64_t v = 1; v < vocab; v++) if (lg[v] > mx) mx = lg[v];
                double se = 0.0;
                for (int64_t v = 0; v < vocab; v++)
                    se += nf_exp((double)lg[v] - mx);
                cnll += (mx + nf_log(se)) - (double)lg[wt[j + 1]];
            }
            nll += cnll;
            counted += (c0 + cn >= win) ? cn - 1 : cn;
        }
        printf("window %2d/%d  partial ppl %.4f\n", w + 1, n_win,
               nf_exp(nll / (double)counted));
        fflush(stdout);
    }
    printf("PPL = %.4f  (%" PRId64 " positions, windows %d x %d, %.0fs)\n",
           nf_exp(nll / (double)counted), counted, n_win, win,
           now_seconds() - t0);

    free(va); nf_session_free(s); nf_model_free(m);
    free(ids); nf_tokenizer_free(tk);
    return 0;
fail:
    free(va); nf_session_free(s); nf_model_free(m);
    free(ids); nf_tokenizer_free(tk);
    return 1;
}

/* M5: ds4-bench-style benchmark harness. Walks the prompt up to
 * ctx_max in geometric frontiers (x step_mul), measures only the
 * incremental prefill from the last frontier (reuses the KV cache,
 * doesn't recompute from scratch), then greedily decodes gen_tokens
 * tokens to also measure generation speed at that context length. No
 * optimization yet: this is the starting number to beat. */
static int cmd_bench(const char *path, const char *text, int ctx_start,
                     int ctx_max, double step_mul, int gen_tokens) {
    nf_tokenizer *tk = nf_tokenizer_load(path);
    if (!tk) return 1;

    enum { MAX_PROMPT = 1 << 16 };
    int32_t *ids = malloc(MAX_PROMPT * sizeof(int32_t));
    if (!ids) { nf_tokenizer_free(tk); return 1; }
    const int64_t n = nf_tokenize(tk, text, ids, MAX_PROMPT);
    if (n <= 0) {
        fprintf(stderr, "nf: empty prompt\n");
        free(ids); nf_tokenizer_free(tk);
        return 1;
    }
    if (ctx_max > n) ctx_max = (int)n;
    fprintf(stderr, "nf: %" PRId64 "-token prompt, benchmarking up to %d\n",
            n, ctx_max);

    const double t_load0 = now_seconds();
    nf_model *m = nf_model_load(path);
    if (!m) { free(ids); nf_tokenizer_free(tk); return 1; }
    fprintf(stderr, "nf: model loaded in %.2fs\n",
            now_seconds() - t_load0);

    /* the session has to hold every frontier PLUS the tokens generated
     * on each round (which stay in context for the next round, pushing
     * it past the next "nominal" frontier) — a generous margin instead
     * of computing the exact round count */
    nf_session *s = nf_session_create(m, ctx_max + gen_tokens * 32 + 64);
    if (!s) { nf_model_free(m); free(ids); nf_tokenizer_free(tk); return 1; }

    const int64_t vocab = (int64_t)nf_model_vocab(m);
    printf("ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps\n");
    fflush(stdout);

    int frontier = ctx_start;
    while (frontier <= ctx_max) {
        /* nf_session_n_past() is the source of truth: after the
         * previous round the session has also advanced by the
         * gen_tokens generated, not just the prompt tokens prefilled */
        const int cur = nf_session_n_past(s);
        if (cur >= frontier) {
            const int next = (int)((double)frontier * step_mul);
            frontier = next > frontier ? next : frontier + 1;
            continue;
        }
        int to_feed = frontier - cur;
        if (cur + to_feed > (int)n) to_feed = (int)n - cur;
        if (to_feed <= 0) break;

        const double p0 = now_seconds();
        const float *logits = nf_session_eval(s, ids + cur, to_feed);
        const double prefill_sec = now_seconds() - p0;
        if (!logits) break;

        int32_t tok = 0;
        for (int64_t i = 1; i < vocab; i++) if (logits[i] > logits[tok]) tok = (int32_t)i;

        const double g0 = now_seconds();
        for (int i = 0; i < gen_tokens; i++) {
            logits = nf_session_eval(s, &tok, 1);
            if (!logits) break;
            tok = 0;
            for (int64_t j = 1; j < vocab; j++) if (logits[j] > logits[tok]) tok = (int32_t)j;
        }
        const double gen_sec = now_seconds() - g0;

        printf("%d,%d,%.2f,%d,%.2f\n", frontier, to_feed,
               to_feed / prefill_sec, gen_tokens, gen_tokens / gen_sec);
        fflush(stdout);

        const int next = (int)((double)frontier * step_mul);
        frontier = next > frontier ? next : frontier + 1;
    }

    nf_debug_timing_report();
    nf_route_profile_report();  /* no-op unless NF_GLM_ROUTE_PROFILE */
    nf_session_free(s);
    nf_model_free(m);
    free(ids);
    nf_tokenizer_free(tk);
    return 0;
}

/* DEBUG M4: prints the dequantized floats of blk.<layer>.attn_q or
 * attn_v, row `row`, to compare against an external reference
 * (gguf.quants). */
static int cmd_debug_wq(const char *path, int layer, uint64_t row, int which_v) {
    nf_model *m = nf_model_load(path);
    if (!m) return 1;
    /* out must hold n_embd elements (tensor_row_to_f32 dequantizes a
     * whole row n_embd wide) — a fixed [1024] buffer only worked for
     * Qwen3-0.6B (n_embd=1024); on Qwen3-8B (n_embd=4096) it was a
     * silent stack overflow (STATUS_STACK_BUFFER_OVERRUN), found while
     * validating dequant on the 8B. */
    const int n_embd = nf_model_n_embd(m);
    float *out = malloc((size_t)n_embd * sizeof(float));
    if (!out) { nf_model_free(m); return 1; }
    const int rc = which_v ? nf_debug_dequant_wv_row(m, layer, row, out)
                           : nf_debug_dequant_wq_row(m, layer, row, out);
    if (rc) {
        fprintf(stderr, "nf: invalid layer/row\n");
        free(out);
        nf_model_free(m);
        return 1;
    }
    for (int i = 0; i < n_embd; i++) printf("%.17g\n", (double)out[i]);
    free(out);
    nf_model_free(m);
    return 0;
}

/* ---- M6: chat template, REPL, on-disk session ------------------------
 *
 * The 4 styles (g_chat_style) are documented above, near
 * chat_style_init: here just the implementation (append_turn for an
 * already-complete turn, append_assistant_open to open the one to
 * generate). No generic template engine (no Jinja): one model, one
 * format, hardcoded and verified, not a generic interpreter that would
 * have to work for models we don't have. */
static int append_turn(char **buf, size_t *cap, size_t *len,
                       const char *role, const char *text) {
    /* +96, not the minimum (+24 of the template): after the LAST
     * append_turn the caller still appends the assistant turn opening
     * (22 bytes) and possibly --no-think's empty think block (20
     * bytes) without coming back through here — the margin must cover
     * them. */
    const size_t need = *len + strlen(role) + strlen(text) + 96;
    if (need > *cap) {
        size_t newcap = *cap ? *cap * 2 : 4096;
        while (newcap < need) newcap *= 2;
        char *nb = realloc(*buf, newcap);
        if (!nb) return -1;
        *buf = nb;
        *cap = newcap;
    }
    if (g_chat_style == 1) {
        if (strcmp(role, "system") == 0)
            *len += (size_t)snprintf(*buf + *len, *cap - *len,
                                     "%s\n\n", text);
        else if (strcmp(role, "user") == 0)
            *len += (size_t)snprintf(*buf + *len, *cap - *len,
                                     "User: %s\n\n", text);
        else
            *len += (size_t)snprintf(*buf + *len, *cap - *len,
                                     "Assistant: %s<\xef\xbd\x9c"
                                     "end\xe2\x96\x81of\xe2\x96\x81sentence"
                                     "\xef\xbd\x9c>", text);
        return 0;
    }
    if (g_chat_style == 2) {
        if (strcmp(role, "system") == 0)
            *len += (size_t)snprintf(*buf + *len, *cap - *len,
                                     "<|system|>\n%s", text);
        else if (strcmp(role, "user") == 0)
            *len += (size_t)snprintf(*buf + *len, *cap - *len,
                                     "<|user|>\n%s", text);
        else
            *len += (size_t)snprintf(*buf + *len, *cap - *len,
                                     "<|assistant|>\n%s<|endoftext|>", text);
        return 0;
    }
    if (g_chat_style == 3) {
        /* glm-dsa: NO '\n' between role and content (unlike style
         * 2/olmoe above, which looks similar at a glance but isn't) —
         * verified against both the official chat_template.jinja and
         * colibri.c, see the comment on chat_style_init. The
         * [gMASK]<sop> prefix is NOT here: it's the caller's
         * responsibility (append_glm_prefix_if_first), once per
         * conversation. */
        if (strcmp(role, "system") == 0)
            *len += (size_t)snprintf(*buf + *len, *cap - *len,
                                     "<|system|>%s", text);
        else if (strcmp(role, "user") == 0)
            *len += (size_t)snprintf(*buf + *len, *cap - *len,
                                     "<|user|>%s", text);
        else
            *len += (size_t)snprintf(*buf + *len, *cap - *len,
                                     "<|assistant|>%s<|endoftext|>", text);
        return 0;
    }
    *len += (size_t)snprintf(*buf + *len, *cap - *len,
                             "<|im_start|>%s\n%s<|im_end|>\n", role, text);
    return 0;
}

/* Opening of the turn to generate — the other piece of the template */
static size_t append_assistant_open(char *buf, size_t cap, size_t len,
                                    int no_think) {
    if (g_chat_style == 1)
        return len + (size_t)snprintf(buf + len, cap - len, "Assistant:");
    if (g_chat_style == 2)
        return len + (size_t)snprintf(buf + len, cap - len, "<|assistant|>\n");
    if (g_chat_style == 3) {
        /* Empty <think></think> if --no-think, otherwise the turn
         * stays open (the model chooses whether to open its own
         * <think>...</think> or answer directly) — the same "soft"
         * convention already used above for style 0/ChatML (qwen3).
         * NOTE (not validated against a real GGUF): colibri.c observes
         * the OPPOSITE behavior as GLM-5.2's de facto default (without
         * THINK=1, thinking is ALWAYS disabled, never left open) — here
         * the choice was to stay consistent with this file's existing
         * --no-think convention rather than silently flip the default;
         * to revisit once a real GGUF is available to measure against. */
        len += (size_t)snprintf(buf + len, cap - len, "<|assistant|>");
        /* Default FLIPPED once the measurement the note
         * above was waiting for was done on the real GGUF: open turn =
         * "42" followed by arithmetic degeneration for all 128 tokens
         * (~21 min); empty think = "17 plus 25 is 42." in ~25 tokens,
         * clean. The sources (Unsloth/Z.ai) confirm GLM-5.2 hybrid
         * reasoning with thinking on by default: without an explicit
         * close, the open turn is no-man's-land. Empty think is now the
         * DEFAULT for glm; NF_GLM_THINK=1 reopens the turn for
         * reasoning tasks (the --no-think convention stays valid and
         * redundant on glm). */
        if (no_think || !getenv("NF_GLM_THINK"))
            len += (size_t)snprintf(buf + len, cap - len, "<think></think>");
        return len;
    }
    len += (size_t)snprintf(buf + len, cap - len, "<|im_start|>assistant\n");
    if (no_think)
        len += (size_t)snprintf(buf + len, cap - len,
                                "<think>\n\n</think>\n\n");
    return len;
}

/* Context shifting (opt-in --ctx-shift): if the next write
 * of `need` tokens would exceed max_ctx, discards HALF of the usable
 * window beyond n_keep (standard heuristic, llama.cpp: discarding a
 * lot in one go amortizes the shift over many future turns instead of
 * redoing it at every token) instead of stopping. hist[] follows the
 * session's shift (same cut, same n_keep index). Returns 1 if after
 * any shift there's room for `need` tokens, 0 if there wasn't enough
 * discardable state (context genuinely exhausted even after
 * discarding everything past n_keep). */
static int chat_ctx_shift_if_needed(nf_session *s, int32_t *hist,
                                    int *pn_hist, int max_ctx, int n_keep,
                                    int need) {
    if (*pn_hist + need <= max_ctx) return 1;
    const int usable = max_ctx - n_keep;
    if (usable <= 0) return 0;
    int n_discard = usable / 2;
    if (n_discard <= 0) n_discard = 1;
    if (n_keep + n_discard > *pn_hist) n_discard = *pn_hist - n_keep;
    if (n_discard <= 0) return 0;
    const int applied = nf_session_shift(s, n_keep, n_discard);
    if (applied <= 0) return 0;
    memmove(hist + n_keep, hist + n_keep + applied,
            (size_t)(*pn_hist - n_keep - applied) * sizeof(int32_t));
    *pn_hist -= applied;
    fprintf(stderr, "\nnf: [ctx-shift] discarded %d tokens (kept the first "
            "%d), free context: %d/%d\n", applied, n_keep, *pn_hist,
            max_ctx);
    return *pn_hist + need <= max_ctx;
}

/* A chat turn under speculative decoding — the same propose-verify
 * loop as spec_generate (see its comment for the mechanics and why it
 * requires greedy), adapted for the REPL:
 *   - t_past/d_past are PERSISTENT across turns (both caches live for
 *     the whole conversation, not for a single generate call);
 *   - the turn's pending part (the user tokens just added to hist,
 *     plus any correction left over from the previous turn) is
 *     prefilled "everything but the last one" BEFORE the loop: the
 *     verify logits buffer is sized K+2 rows, it can't hold a whole
 *     user turn — and batched prefill is the fast path anyway;
 *   - at the end of the turn a FLUSH brings the target cache back in
 *     sync with hist: the speculative rollback leaves the correction
 *     (often exactly the <|im_end|> that closes the turn) out of the
 *     cache, and without a flush nf_session_save would save a
 *     malformed chat format (same invariant as the comment on EOS in
 *     the normal path).
 * Returns 0 (turn completed) or -1 (hard error: the REPL exits). */
static int chat_spec_turn(nf_tokenizer *tk, nf_session *s, nf_session *ds,
                          const nf_stopset *stop, const nf_textstop *tstop,
                          int32_t *hist, int *pn_hist, int *pt_past,
                          int *pd_past, int max_ctx, int K, int64_t vocab,
                          float *va, int32_t *dtok, int32_t *batch,
                          float top_p, uint64_t *rng,
                          float repeat_penalty, int repeat_last_n,
                          int max_turn_tokens) {
    int len = *pn_hist, t_past = *pt_past, d_past = *pd_past;
    const int genid = getenv("NF_DEBUG_GENID") != NULL;

    if (len - t_past > 1) {
        if (!nf_session_eval(s, hist + t_past, len - t_past - 1)) return -1;
        t_past = len - 1;
    }

    int emitted = 0, stopped = 0;
    int rounds = 0, drafted = 0, accepted = 0;
    char pend[512];
    int n_pend_txt = 0;
    const double t0 = now_seconds();

    while (!stopped && emitted < max_turn_tokens) {
        if (len + K + 1 > max_ctx) {
            fprintf(stderr, "\nnf: context exhausted\n");
            break;
        }

        /* 1) the draft proposes K greedy tokens from its own state */
        const float *dl = nf_session_eval(ds, hist + d_past, len - d_past);
        if (!dl) break;
        d_past = len;
        int nd = 0;
        for (int j = 0; j < K; j++) {
            penalty_virtual((float *)dl, vocab, hist, len, dtok, j,
                            repeat_penalty, repeat_last_n);
            dtok[j] = sample_token(dl, vocab, 0.0f, top_p, 0, 0.0f, rng);
            nd++;
            dl = nf_session_eval(ds, &dtok[j], 1);
            if (!dl) break;
        }
        d_past += nd;
        drafted += nd;

        /* 2) the target verifies pending+proposals in one pass */
        const int n_pend = len - t_past;
        const int n_batch = n_pend + nd;
        memcpy(batch, hist + t_past, (size_t)n_pend * sizeof(int32_t));
        memcpy(batch + n_pend, dtok, (size_t)nd * sizeof(int32_t));
        if (!nf_session_eval_all(s, batch, n_batch, va)) break;
        t_past += n_batch;
        const int base = n_pend - 1;

        /* 3) sequential comparison, correction at the first disagreement */
        int n_acc = 0;
        int32_t fix = -1;
        for (int j = 0; j < nd; j++) {
            float *Lj = va + (size_t)(base + j) * vocab;
            penalty_virtual(Lj, vocab, hist, len, dtok, j,
                            repeat_penalty, repeat_last_n);
            const int32_t g = sample_token(Lj, vocab, 0.0f, top_p, 0, 0.0f, rng);
            if (g == dtok[j]) { n_acc++; continue; }
            fix = g;
            break;
        }
        if (fix < 0) {
            float *Lk = va + (size_t)(base + nd) * vocab;
            penalty_virtual(Lk, vocab, hist, len, dtok, nd,
                            repeat_penalty, repeat_last_n);
            fix = sample_token(Lk, vocab, 0.0f, top_p, 0, 0.0f, rng);
        }
        accepted += n_acc;
        rounds++;

        /* 4) emission with chat's conventions (GENID only on request,
         * id-based stop, textual filter) */
        const int old_len = len;
        for (int e = 0; e <= n_acc && !stopped && emitted < max_turn_tokens; e++) {
            const int32_t tok = e < n_acc ? dtok[e] : fix;
            if (genid) fprintf(stderr, "GENID %d\n", tok);
            if (len >= max_ctx) { stopped = 1; break; }
            hist[len++] = tok;
            emitted++;
            if (is_stop_token(stop, tok)) { stopped = 1; break; }
            char piece[256];
            const int w = nf_token_decode(tk, tok, piece, sizeof(piece) - 1);
            if (stream_piece_through_stopfilter(piece, w > 0 ? w : 0, pend,
                                                &n_pend_txt, (int)sizeof(pend),
                                                tstop)) {
                stopped = 1;
                break;
            }
        }

        /* 5) roll back both caches to the prefix actually kept:
         * min(accepted, emitted) — if the stop arrived INSIDE the
         * accepted proposals, hist stopped before n_acc and the cache
         * must not contain tokens hist doesn't have (unlike
         * spec_generate, here the session survives the turn). */
        int keep = old_len + n_acc;
        if (keep > len) keep = len;
        nf_session_rewind(s, keep);
        nf_session_rewind(ds, keep);
        t_past = keep;
        d_past = keep;
    }
    if (n_pend_txt > 0) { fwrite(pend, 1, (size_t)n_pend_txt, stdout); fflush(stdout); }
    printf("\n");

    /* end-of-turn flush: target cache back in sync with hist (the
     * EOS-in-cache invariant of the chat format + correct
     * nf_session_save) */
    if (len > t_past) {
        if (!nf_session_eval(s, hist + t_past, len - t_past)) return -1;
        t_past = len;
    }

    {
        const double dt = now_seconds() - t0;
        fprintf(stderr,
                "nf: speculative: %d rounds, %d/%d accepted (%.0f%%), "
                "%d tokens in %.1fs (%.2f tok/s)\n",
                rounds, accepted, drafted,
                drafted ? 100.0 * accepted / drafted : 0.0,
                emitted, dt, dt > 0 ? emitted / dt : 0.0);
    }

    *pn_hist = len;
    *pt_past = t_past;
    *pd_past = d_past;
    return 0;
}

static int cmd_chat(const char *path, const char *system_text, int max_ctx,
                    float temp, float top_p, int32_t top_k, float min_p,
                    uint64_t seed,
                    const char *session_path, int max_turn_tokens,
                    float repeat_penalty, int repeat_last_n, int no_think,
                    const char *draft_path, int draft_k, int ctx_shift) {
    glm_daily_env(path);   /* before any load/gate, see the comment there */
    /* batched+draft guard: interaction never validated —
     * under draft, batched prefill turns off, with a note. */
    if (draft_path && getenv("NF_GLM_PREFILL_BATCH")) {
        nf_setenv("NF_GLM_PREFILL_BATCH=0");
        fprintf(stderr, "nf: batched prefill DISABLED under --draft "
                "(interaction not validated)\n");
    }
    /* long-memory (Phase 1, and Phase 2a disk-backing which requires it
     * as a prerequisite — NF_LONGMEM_DISK without NF_LONGMEM is a
     * no-op, see nf_session_create) and --ctx-shift are in a CONFLICT
     * OF PURPOSE, not just a mechanical one: the shift PERMANENTLY
     * ERASES exactly the middle block that long-memory exists to keep
     * recoverable. Unlike --draft+--ctx-shift (warning + silent
     * runtime guard, a secondary combination), here it's rejected
     * immediately: the two features would silently defeat each other's
     * purpose if left to coexist. */
    if (ctx_shift && getenv("NF_LONGMEM")) {
        fprintf(stderr, "nf: --ctx-shift and NF_LONGMEM are incompatible in "
                "purpose (the shift permanently discards what long-memory "
                "is supposed to keep recoverable) — pick one of the two\n");
        return 1;
    }
    /* Phase 2a: speculative decoding WANTS an n_new>1 batch to verify
     * several tokens in one pass — disk-backing always decomposes it
     * into N single calls (see lm_disk_decompose_wanted in
     * nf_model.c), which would only make speculative verification
     * SLOWER than the non-speculative path, never faster. Rejected
     * immediately, same pattern as --ctx-shift+NF_LONGMEM above. */
    if (draft_path && getenv("NF_LONGMEM_DISK")) {
        fprintf(stderr, "nf: --draft and NF_LONGMEM_DISK are incompatible "
                "(speculative decoding wants a batch, disk-backing "
                "always decomposes it into single calls) — pick one "
                "of the two\n");
        return 1;
    }
    chat_style_init(path);   /* ChatML or deepseek, from the gguf */
    const double t_dbg0 = now_seconds();
    const int dbg_load = getenv("NF_DEBUG_LOAD") != NULL;
    const char *toksrc = tok_source_for(path);   /* .forge: NF_TOKENIZER */
    if (!toksrc) return 1;
    nf_tokenizer *tk = nf_tokenizer_load(toksrc);
    if (!tk) return 1;
    if (dbg_load)
        fprintf(stderr, "nf: [load] tokenizer: %.1fs\n", now_seconds() - t_dbg0);
    nf_model *m = nf_model_load(path);
    if (!m) { nf_tokenizer_free(tk); return 1; }
    if (dbg_load)
        fprintf(stderr, "nf: [load] model (cumulative): %.1fs\n",
                now_seconds() - t_dbg0);

    nf_stopset stop;
    build_stopset(toksrc, &stop);
    stopset_add_special(tk, &stop, "<|endoftext|>");
    if (g_chat_style == 3) {   /* glm-dsa: 3 stop ids, see chat_style_init */
        stopset_add_special(tk, &stop, "<|user|>");
        stopset_add_special(tk, &stop, "<|observation|>");
        /* fourth stop (termination bench): at FULL QUALITY
         * the 2.17-bit GGUF can degenerate after the response and emit
         * a new <|assistant|> as text — an assistant turn opened by
         * the MODEL is effectively an end-of-turn (the only legitimate
         * <|assistant|> is written by the engine in the template,
         * never by the model). */
        stopset_add_special(tk, &stop, "<|assistant|>");
    }
    nf_textstop tstop;
    build_textstops(tk, &stop, &tstop);
    const int64_t vocab = (int64_t)nf_model_vocab(m);
    /* Cost-AWARE per-turn safety cap: the historical
     * default of 512 is sized for models running at tens of tok/s; on
     * the real glm (9-13 s/token) a runaway answer would cost 77-110
     * MINUTES, and the termination bench shows that at this bitrate the
     * model often does NOT stop on its own after the answer (at every
     * quality level, dynk/miss-skip cleared with byte-identical A/B).
     * -1 = the user didn't pass --max-tokens. */
    if (max_turn_tokens < 0) {
        max_turn_tokens = g_chat_style == 3 ? 96 : 512;
        if (g_chat_style == 3)
            fprintf(stderr, "nf: per-turn cap = 96 tokens (~15 min max "
                    "on this hardware; --max-tokens to change it)\n");
    }

    /* token history for the whole conversation: only needed to
     * save/reload the session (nf_session already holds K/V, but not
     * the ids — kept here alongside for the same reason DwarfStar
     * always pairs the KV payload with the exact token prefix that
     * produced it) */
    int32_t *hist = malloc((size_t)max_ctx * sizeof(int32_t));
    int n_hist = 0;
    nf_session *s = NULL;

    if (session_path) {
        s = nf_session_load(m, session_path, max_ctx, hist, max_ctx, &n_hist);
        if (s) fprintf(stderr, "nf: conversation resumed (%d tokens)\n", n_hist);
    }
    if (!s) {
        s = nf_session_create(m, max_ctx);
        if (dbg_load)
            fprintf(stderr, "nf: [load] session (cumulative): %.1fs\n",
                    now_seconds() - t_dbg0);
        if (s && system_text && *system_text) {
            char *buf = NULL; size_t cap = 0, len = 0;
            append_glm_prefix_if_first(&buf, &cap, &len, nf_session_n_past(s) == 0);
            if (append_turn(&buf, &cap, &len, "system", system_text)) {
                free(buf); nf_session_free(s); s = NULL;
            } else {
                const int64_t nt = nf_tokenize(tk, buf, hist, max_ctx);
                free(buf);
                if (nt > 0 && nt <= max_ctx) {
                    if (!nf_session_eval(s, hist, (int)nt)) { nf_session_free(s); s = NULL; }
                    else n_hist = (int)nt;
                }
            }
        }
    }
    if (!s || !hist) {
        free(hist); if (s) nf_session_free(s);
        nf_model_free(m); nf_tokenizer_free(tk);
        return 1;
    }
    /* n_keep = how much is already in cache at opening (system prompt,
     * or the whole conversation resumed from --session): never
     * discarded by the shift, only what gets added from here on.
     * --draft (speculative) isn't covered: chat_spec_turn has two
     * sessions and its own t_past/d_past state, out of scope for this
     * integration */
    const int n_keep = ctx_shift ? n_hist : 0;
    if (ctx_shift && draft_path)
        fprintf(stderr, "nf: --ctx-shift isn't supported with --draft "
                "(ignored)\n");

    /* speculative decoding (--draft): setup ONCE, before the REPL. Same
     * guards as cmd_generate (greedy mandatory, matching vocabularies).
     * The draft always starts from an empty cache: if --session resumed
     * a conversation, the first speculative round catches it up on its
     * own (small model prefill, cheap). */
    nf_model *dm = NULL;
    nf_session *ds = NULL;
    float *sva = NULL;
    int32_t *sdtok = NULL, *sbatch = NULL;
    int t_past = n_hist, d_past = 0;
    if (draft_path) {
        int ok_draft = 0;
        if (temp > 0.0f) {
            fprintf(stderr, "nf: --draft requires greedy (--temp 0)\n");
        } else if ((dm = nf_model_load(draft_path)) == NULL) {
            /* message already emitted by the load */
        } else if (nf_model_vocab(dm) != nf_model_vocab(m)) {
            fprintf(stderr, "nf: draft and target have different vocabularies\n");
        } else {
            ds = nf_session_create(dm, max_ctx);
            sva = malloc((size_t)(draft_k + 2) * vocab * sizeof(float));
            sdtok = malloc((size_t)draft_k * sizeof(int32_t));
            sbatch = malloc((size_t)(draft_k + 2) * sizeof(int32_t));
            ok_draft = ds && sva && sdtok && sbatch;
        }
        if (!ok_draft) {
            free(sva); free(sdtok); free(sbatch);
            if (ds) nf_session_free(ds);
            if (dm) nf_model_free(dm);
            free(hist); nf_session_free(s);
            nf_model_free(m); nf_tokenizer_free(tk);
            return 1;
        }
        fprintf(stderr, "nf: speculative active (draft, K=%d)\n", draft_k);
    }

    fprintf(stderr,
            "nf chat — /quit or /exit to leave%s. Context: %d/%d tokens.\n",
            session_path ? " (the session will be saved on exit)" : "",
            n_hist, max_ctx);

    uint64_t rng = seed;
    char line[8192];
    for (;;) {
        printf("\nyou> ");
        fflush(stdout);
        if (read_line_utf8(line, sizeof(line)) < 0) break;
        if (strcmp(line, "/quit") == 0 || strcmp(line, "/exit") == 0) break;
        if (!line[0]) continue;

        char *buf = NULL; size_t cap = 0, len = 0;
        append_glm_prefix_if_first(&buf, &cap, &len, nf_session_n_past(s) == 0);
        if (append_turn(&buf, &cap, &len, "user", line)) { free(buf); break; }
        /* opens the assistant's turn but doesn't close it: the text
         * ends here, so the model continues from where the prompt
         * leaves it — exactly as the template intends */
        /* --no-think (inside append_assistant_open): the "soft" switch
         * Qwen3 was trained with — a think block ALREADY EMPTY at the
         * start of the assistant turn (this is what the official
         * template produces with enable_thinking=False). The model
         * skips explicit reasoning and answers directly. Found
         * necessary during the second real interactive test: at
         * Q4_K_M extended reasoning collapses into a loop that never
         * converges, but the "direct answer" route survives 4-bit
         * quantization just fine (same prompt: 900 tokens burned with
         * no answer with think, correct answer in 32 tokens without).
         * deepseek: opens "Assistant:", never a think block. */
        len = append_assistant_open(buf, cap, len, no_think);

        int32_t new_ids[4096];
        const int64_t n_new = nf_tokenize(tk, buf, new_ids, 4096);
        free(buf);
        if (n_new <= 0) {
            fprintf(stderr, "nf: invalid input\n");
            break;
        }
        if (ctx_shift && !ds)
            chat_ctx_shift_if_needed(s, hist, &n_hist, max_ctx, n_keep,
                                     (int)n_new);
        if (n_hist + n_new > max_ctx) {
            fprintf(stderr, "nf: context exhausted%s\n",
                    ctx_shift ? " (even after the shift: the turn alone "
                                "exceeds the window)" : "");
            break;
        }
        memcpy(hist + n_hist, new_ids, (size_t)n_new * sizeof(int32_t));

        if (ds) {
            /* speculative: the new turn is NOT prefilled here — the
             * tokens just added are the "pending" ones that
             * chat_spec_turn consumes (prefill n-1 + first batched
             * verify, see its comment) */
            n_hist += (int)n_new;
            printf("bot> ");
            fflush(stdout);
            if (chat_spec_turn(tk, s, ds, &stop, &tstop, hist, &n_hist,
                               &t_past, &d_past, max_ctx, draft_k, vocab,
                               sva, sdtok, sbatch, top_p, &rng,
                               repeat_penalty, repeat_last_n,
                               max_turn_tokens))
                break;
            continue;
        }

        const float *logits = nf_session_eval(s, new_ids, (int)n_new);
        if (!logits) break;
        n_hist += (int)n_new;
        dump_init_logits_if_requested(logits, vocab);

        printf("bot> ");
        fflush(stdout);
        /* per-turn safety cap: without a repetition penalty, a small
         * model can degenerate into a loop that never re-emits a stop
         * token (empirically observed: imaginary quizzes repeated
         * forever) — every real chat interface has a limit like this,
         * it isn't a patch specific to this bug */
        char pending[512];
        int n_pending = 0;
        int turn_first_piece = 1;   /* spurious </think> filter, see below */
        for (int n_this_turn = 0; n_this_turn < max_turn_tokens; n_this_turn++) {
            const int win = n_hist < repeat_last_n ? n_hist : repeat_last_n;
            apply_repeat_penalty((float *)logits, vocab,
                                hist + n_hist - win, win, repeat_penalty);
            int32_t next = sample_token(logits, vocab, temp, top_p,
                                        top_k, min_p, &rng);
            /* empty-turn guard: a TERMINAL as the turn's
             * first token = an empty response, never useful. Real
             * case: with batched prefill the argmax after the empty
             * think block landed on <|user|> from a near-tie (33/34
             * argmax identical to sequential on the real logits, a
             * documented f32 reordering class). The terminal is
             * discarded and the best non-terminal is resampled, with a
             * note — never silently. */
            if (n_this_turn == 0 && is_stop_token(&stop, next)) {
                float *lg_ = (float *)logits;
                for (int si_ = 0; si_ < stop.n; si_++)
                    lg_[stop.ids[si_]] = -1e30f;
                next = sample_token(logits, vocab, temp, top_p,
                                    top_k, min_p, &rng);
                fprintf(stderr, "nf: [chat] the turn's first token was a "
                        "terminal (empty turn): resampled the best "
                        "non-terminal one\n");
            }
            if (getenv("NF_DEBUG_GENID")) fprintf(stderr, "GENID %d\n", next);
            if (ctx_shift && !ds)
                chat_ctx_shift_if_needed(s, hist, &n_hist, max_ctx, n_keep, 1);
            if (n_hist >= max_ctx) { fprintf(stderr, "\nnf: context exhausted\n"); break; }

            /* the EOS (<|im_end|>) still gets fed to the session: it
             * closes the turn in the KV cache exactly like the
             * "<|im_end|>\n" of system/user turns does. Without this,
             * the next user turn would find itself glued to the
             * previous response with no separator in the cache — a
             * malformed chat format from the model's point of view,
             * even though invisible just looking at the text printed
             * on screen. */
            hist[n_hist++] = next;
            logits = nf_session_eval(s, &next, 1);
            if (is_stop_token(&stop, next)) break;
            if (!logits) break;

            char piece[256];
            const int w = nf_token_decode(tk, next, piece, sizeof(piece) - 1);
            const int plen = w > 0 ? w : 0;

            /* glm: spurious "</think>" at turn start (first
             * .forge bench): with the empty-think template the model
             * can emit ANOTHER </think> as the very first token — it's
             * a special token, arrives as an atomic piece. ONLY the
             * turn's first piece is swallowed, only if it's exactly
             * that: the rest of the stream is never touched. */
            if (g_chat_style == 3 && turn_first_piece && plen == 8 &&
                memcmp(piece, "</think>", 8) == 0) {
                turn_first_piece = 0;
                continue;
            }
            turn_first_piece = 0;

            /* textual stop: see the comment on build_textstops. A
             * just-decoded piece isn't printed right away — it stays
             * in `pending` until it's certain it isn't the start of a
             * still-incomplete stop (e.g. the model is writing
             * "<|endoftext|>" one BPE piece at a time). */
            if (stream_piece_through_stopfilter(piece, plen, pending, &n_pending,
                                                (int)sizeof(pending), &tstop))
                break;
        }
        if (n_pending > 0) { fwrite(pending, 1, (size_t)n_pending, stdout); fflush(stdout); }
        printf("\n");
    }

    if (session_path) nf_session_save(s, hist, session_path);
    nf_debug_timing_report();   /* no-op unless NF_DEBUG_TIMING — even
                                 * on EOF exit (a gap seen in the 8/12
                                 * .forge benches: runs via a pipe never
                                 * printed fetch/decode) */

    free(sva); free(sdtok); free(sbatch);
    if (ds) nf_session_free(ds);
    if (dm) nf_model_free(dm);
    nf_session_free(s);
    nf_model_free(m);
    nf_tokenizer_free(tk);
    free(hist);
    return 0;
}

/* ==== M7: nf serve — OpenAI-compatible API over HTTP =====================
 *
 * The engine learns to speak the protocol that open-source GUIs (Open
 * WebUI, Jan, ...) already speak: POST /v1/chat/completions with SSE
 * streaming. This way the GUI isn't written: it's chosen. Plus a
 * minimal built-in HTML page on GET / to have something visual right
 * away with nothing to install.
 *
 * The design crux: the API is STATELESS (every request carries the
 * whole history) but re-prefilling everything on every message would
 * be unusable on the 30B. Solution: PREFIX CACHING — the server keeps a
 * session and the token history; every request tokenizes the whole
 * prompt, finds the common prefix with the history, nf_session_rewind
 * to the prefix, and evaluates only the new suffix. Same principle as
 * llama-server, with bricks we already had (rewind was born for
 * speculative decoding). No authentication: trusted network. Windows/
 * Winsock only (the project's gcc build is MSYS2 anyway).
 *
 * M8 (continuous batching, Phase 1): the "one client at a time"
 * constraint above was deliberate ("the engine saturates the machine
 * anyway") but the PMU profile showed that decode is
 * memory-bandwidth-bound, not compute-bound — several concurrent
 * sessions can share the expert weight reads. Architecture:
 * CONCURRENCY ONLY IN I/O, compute
 * strictly serial — a single thread ever touches nf_model/nf_session
 * (same funnel as today: expert_cache_get, the matmul scratch pools
 * etc. stay serial by construction, zero new locks there);
 * thread-per-connection ONLY for reading/writing the socket, they
 * communicate with the compute thread via queues. */

#define NF_SERVE_MAX_MSGS 64

/* ---- generic thread-safe queue (fixed-capacity ring buffer) ----------
 * Extracted into nf_queue.h: nf_model.c needs it too
 * (Phase 1 async prefetch in moe_ffn_xsess). Verbatim move, see
 * nf_queue.h for the primitive's original comments. */
#include "nf_queue.h"

/* Isolated bench of the queue (M8 Phase 1, BEFORE hooking it up to the
 * server): NPROD producers each push NITEMS values (disjoint ranges,
 * so a lost/duplicated item shows up immediately), NCONS consumers
 * pop until close. Check: the total count and sum extracted must
 * match exactly what was pushed — no loss, no duplication, no
 * deadlock (capacity deliberately small, so both the "full" and
 * "empty" waits get exercised). */
#define NF_DBGQ_NPROD 4
#define NF_DBGQ_NCONS 3
#define NF_DBGQ_NITEMS 20000

typedef struct { nf_queue *q; int id; } nf_dbgq_prod_arg;
typedef struct { nf_queue *q; long long count; long long sum; } nf_dbgq_cons_arg;

static NF_THREAD_RET nf_dbgq_producer(void *p) {
    nf_dbgq_prod_arg *a = (nf_dbgq_prod_arg *)p;
    const long long base = (long long)a->id * NF_DBGQ_NITEMS;
    for (int i = 0; i < NF_DBGQ_NITEMS; i++) {
        long long v = base + i;
        nf_queue_push(a->q, &v);
    }
    return 0;
}

static NF_THREAD_RET nf_dbgq_consumer(void *p) {
    nf_dbgq_cons_arg *a = (nf_dbgq_cons_arg *)p;
    long long v;
    while (nf_queue_pop(a->q, &v) == 0) { a->count++; a->sum += v; }
    return 0;
}

void nf_debug_queue_bench(void) {
    nf_queue q;
    nf_queue_init(&q, sizeof(long long), 8);  /* deliberately small: forces full/empty waits */

    nf_thread_t prod_h[NF_DBGQ_NPROD];
    nf_dbgq_prod_arg prod_a[NF_DBGQ_NPROD];
    nf_thread_t cons_h[NF_DBGQ_NCONS];
    nf_dbgq_cons_arg cons_a[NF_DBGQ_NCONS] = {0};

    const double t0 = now_seconds();
    for (int i = 0; i < NF_DBGQ_NCONS; i++) {
        cons_a[i].q = &q;
        cons_h[i] = nf_thread_start(nf_dbgq_consumer, &cons_a[i]);
    }
    for (int i = 0; i < NF_DBGQ_NPROD; i++) {
        prod_a[i].q = &q;
        prod_a[i].id = i;
        prod_h[i] = nf_thread_start(nf_dbgq_producer, &prod_a[i]);
    }
    for (int i = 0; i < NF_DBGQ_NPROD; i++) nf_thread_join(prod_h[i]);
    nf_queue_close(&q);   /* producers are done: unblocks the waiting consumers */
    for (int i = 0; i < NF_DBGQ_NCONS; i++) nf_thread_join(cons_h[i]);
    const double dt = now_seconds() - t0;

    long long total_count = 0, total_sum = 0;
    for (int i = 0; i < NF_DBGQ_NCONS; i++) {
        total_count += cons_a[i].count;
        total_sum += cons_a[i].sum;
        fprintf(stderr, "nf: [debugqueue] consumer %d: %lld items, sum=%lld\n",
                i, cons_a[i].count, cons_a[i].sum);
    }
    const long long expect_count = (long long)NF_DBGQ_NPROD * NF_DBGQ_NITEMS;
    long long expect_sum = 0;
    for (int p = 0; p < NF_DBGQ_NPROD; p++) {
        const long long base = (long long)p * NF_DBGQ_NITEMS;
        expect_sum += (long long)NF_DBGQ_NITEMS * base
                    + (long long)NF_DBGQ_NITEMS * (NF_DBGQ_NITEMS - 1) / 2;
    }
    const int ok = total_count == expect_count && total_sum == expect_sum;
    fprintf(stderr, "nf: [debugqueue] TOTAL: %lld/%lld items, sum %lld/%lld expected "
            "-> %s (%.3fs, %.0f items/s)\n",
            total_count, expect_count, total_sum, expect_sum,
            ok ? "OK" : "FAILED (loss or duplication)", dt,
            (double)total_count / dt);
}

/* ---- mini-JSON: targeted body parsing ---------------------------------
 * Not a general JSON parser: reads only the keys that matter
 * (messages/role/content, stream, temperature, top_p, max_tokens) and
 * skips everything else. String unescaping is complete, though (\"
 * \\ \/ \b \f \n \r \t \uXXXX with surrogate pairs): non-ASCII text
 * arrives often as escaped codepoints, emoji as surrogate pairs. */
static const char *json_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    return p;
}

static int json_hex4(const char *p, unsigned *out) {
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        const char c = p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
        else return -1;
    }
    *out = v;
    return 0;
}

static size_t utf8_put(unsigned cp, char *o) {
    if (cp < 0x80) { o[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        o[0] = (char)(0xC0 | (cp >> 6));
        o[1] = (char)(0x80 | (cp & 63));
        return 2;
    }
    if (cp < 0x10000) {
        o[0] = (char)(0xE0 | (cp >> 12));
        o[1] = (char)(0x80 | ((cp >> 6) & 63));
        o[2] = (char)(0x80 | (cp & 63));
        return 3;
    }
    o[0] = (char)(0xF0 | (cp >> 18));
    o[1] = (char)(0x80 | ((cp >> 12) & 63));
    o[2] = (char)(0x80 | ((cp >> 6) & 63));
    o[3] = (char)(0x80 | (cp & 63));
    return 4;
}

/* p points at the opening quote; returns the pointer AFTER the closing
 * one and puts an already-unescaped malloc'd copy (NUL-terminated)
 * into *out. NULL on error. */
static const char *json_string(const char *p, char **out) {
    if (*p != '"') return NULL;
    p++;
    const char *q = p;
    size_t cap = 64, len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    while (*q && *q != '"') {
        if (len + 8 >= cap) {
            cap *= 2;
            char *nb = realloc(buf, cap);
            if (!nb) { free(buf); return NULL; }
            buf = nb;
        }
        if (*q == '\\') {
            q++;
            switch (*q) {
            case '"':  buf[len++] = '"';  q++; break;
            case '\\': buf[len++] = '\\'; q++; break;
            case '/':  buf[len++] = '/';  q++; break;
            case 'b':  buf[len++] = '\b'; q++; break;
            case 'f':  buf[len++] = '\f'; q++; break;
            case 'n':  buf[len++] = '\n'; q++; break;
            case 'r':  buf[len++] = '\r'; q++; break;
            case 't':  buf[len++] = '\t'; q++; break;
            case 'u': {
                unsigned cp;
                if (json_hex4(q + 1, &cp)) { free(buf); return NULL; }
                q += 5;
                if (cp >= 0xD800 && cp <= 0xDBFF && q[0] == '\\' && q[1] == 'u') {
                    unsigned lo;
                    if (json_hex4(q + 2, &lo)) { free(buf); return NULL; }
                    if (lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        q += 6;
                    }
                }
                len += utf8_put(cp, buf + len);
                break;
            }
            default: free(buf); return NULL;
            }
        } else {
            buf[len++] = *q++;
        }
    }
    if (*q != '"') { free(buf); return NULL; }
    buf[len] = 0;
    *out = buf;
    return q + 1;
}

/* Skips any JSON value (for keys we don't care about). */
static const char *json_skip(const char *p) {
    p = json_ws(p);
    if (*p == '"') {
        char *tmp;
        p = json_string(p, &tmp);
        if (p) free(tmp);
        return p;
    }
    if (*p == '{' || *p == '[') {
        const char open = *p, close = (open == '{') ? '}' : ']';
        int depth = 0;
        do {
            if (*p == '"') {
                char *tmp;
                p = json_string(p, &tmp);
                if (!p) return NULL;
                free(tmp);
                continue;
            }
            if (*p == open) depth++;
            else if (*p == close) depth--;
            else if (*p == 0) return NULL;
            p++;
        } while (depth > 0);
        return p;
    }
    while (*p && *p != ',' && *p != '}' && *p != ']') p++;
    return p;
}

typedef struct {
    char *role, *content;          /* malloc'd by the parser */
} nf_msg;

typedef struct {
    nf_msg msgs[NF_SERVE_MAX_MSGS];
    int n_msgs;
    int stream;
    float temperature, top_p;      /* <0 = absent from the body */
    int top_k;                     /* <0 = absent (0 is a valid "off") */
    float min_p;                   /* <0 = absent (0 is a valid "off") */
    int max_tokens;                /* <0 = absent */
} nf_chat_req;

static void chat_req_free(nf_chat_req *r) {
    for (int i = 0; i < r->n_msgs; i++) { free(r->msgs[i].role); free(r->msgs[i].content); }
    r->n_msgs = 0;
}

static int parse_chat_request(const char *body, nf_chat_req *r) {
    memset(r, 0, sizeof(*r));
    r->temperature = -1.0f;
    r->top_p = -1.0f;
    r->top_k = -1;
    r->min_p = -1.0f;
    r->max_tokens = -1;
    const char *p = json_ws(body);
    if (*p != '{') return -1;
    p = json_ws(p + 1);
    while (*p && *p != '}') {
        char *key;
        p = json_string(p, &key);
        if (!p) goto fail;
        p = json_ws(p);
        if (*p != ':') { free(key); goto fail; }
        p = json_ws(p + 1);
        if (strcmp(key, "messages") == 0 && *p == '[') {
            p = json_ws(p + 1);
            while (*p && *p != ']') {
                if (*p != '{') { free(key); goto fail; }
                char *role = NULL, *content = NULL;
                p = json_ws(p + 1);
                while (*p && *p != '}') {
                    char *mk;
                    p = json_string(p, &mk);
                    if (!p) { free(role); free(content); free(key); goto fail; }
                    p = json_ws(p);
                    if (*p != ':') { free(mk); free(role); free(content); free(key); goto fail; }
                    p = json_ws(p + 1);
                    if (strcmp(mk, "role") == 0 && *p == '"') {
                        free(role);
                        p = json_string(p, &role);
                    } else if (strcmp(mk, "content") == 0 && *p == '"') {
                        free(content);
                        p = json_string(p, &content);
                    } else {
                        p = json_skip(p);   /* non-string content (a
                                               multimodal array etc.): ignored */
                    }
                    free(mk);
                    if (!p) { free(role); free(content); free(key); goto fail; }
                    p = json_ws(p);
                    if (*p == ',') p = json_ws(p + 1);
                }
                if (*p != '}') { free(role); free(content); free(key); goto fail; }
                p = json_ws(p + 1);
                if (role && content && r->n_msgs < NF_SERVE_MAX_MSGS) {
                    r->msgs[r->n_msgs].role = role;
                    r->msgs[r->n_msgs].content = content;
                    r->n_msgs++;
                } else {
                    free(role);
                    free(content);
                }
                if (*p == ',') p = json_ws(p + 1);
            }
            if (*p != ']') { free(key); goto fail; }
            p++;
        } else if (strcmp(key, "stream") == 0) {
            r->stream = (strncmp(p, "true", 4) == 0);
            p = json_skip(p);
        } else if (strcmp(key, "temperature") == 0) {
            r->temperature = (float)strtod(p, NULL);
            p = json_skip(p);
        } else if (strcmp(key, "top_p") == 0) {
            r->top_p = (float)strtod(p, NULL);
            p = json_skip(p);
        } else if (strcmp(key, "top_k") == 0) {
            r->top_k = (int)strtol(p, NULL, 10);
            p = json_skip(p);
        } else if (strcmp(key, "min_p") == 0) {
            r->min_p = (float)strtod(p, NULL);
            p = json_skip(p);
        } else if (strcmp(key, "max_tokens") == 0 ||
                   strcmp(key, "max_completion_tokens") == 0) {
            r->max_tokens = (int)strtol(p, NULL, 10);
            p = json_skip(p);
        } else {
            p = json_skip(p);
        }
        free(key);
        if (!p) goto fail;
        p = json_ws(p);
        if (*p == ',') p = json_ws(p + 1);
    }
    return (*p == '}' && r->n_msgs > 0) ? 0 : -1;
fail:
    chat_req_free(r);
    return -1;
}

/* Symmetric JSON escape for output (SSE chunks and the full response). */
static size_t json_escape(const char *in, size_t n, char *out, size_t cap) {
    size_t o = 0;
    for (size_t i = 0; i < n && o + 8 < cap; i++) {
        const unsigned char c = (unsigned char)in[i];
        switch (c) {
        case '"':  out[o++] = '\\'; out[o++] = '"';  break;
        case '\\': out[o++] = '\\'; out[o++] = '\\'; break;
        case '\b': out[o++] = '\\'; out[o++] = 'b';  break;
        case '\f': out[o++] = '\\'; out[o++] = 'f';  break;
        case '\n': out[o++] = '\\'; out[o++] = 'n';  break;
        case '\r': out[o++] = '\\'; out[o++] = 'r';  break;
        case '\t': out[o++] = '\\'; out[o++] = 't';  break;
        default:
            if (c < 0x20)
                o += (size_t)snprintf(out + o, cap - o, "\\u%04x", c);
            else
                out[o++] = (char)c;
        }
    }
    out[o] = 0;
    return o;
}

/* ---- minimal HTTP ------------------------------------------------------ */
static int send_all(SOCKET c, const char *buf, size_t n) {
    while (n > 0) {
        const int w = send(c, buf, (int)(n > 1 << 20 ? 1 << 20 : n), 0);
        if (w <= 0) return -1;
        buf += w;
        n -= (size_t)w;
    }
    return 0;
}

static int send_str(SOCKET c, const char *s) { return send_all(c, s, strlen(s)); }

static const char NF_CORS[] =
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
    "Access-Control-Allow-Headers: *\r\n";

static int send_simple(SOCKET c, const char *status, const char *ctype,
                       const char *body) {
    char hdr[512];
    snprintf(hdr, sizeof(hdr),
             "HTTP/1.1 %s\r\n%sContent-Type: %s\r\n"
             "Content-Length: %zu\r\nConnection: close\r\n\r\n",
             status, NF_CORS, ctype, strlen(body));
    if (send_str(c, hdr)) return -1;
    return send_str(c, body);
}

/* An SSE chunk in OpenAI's format. `content` NULL = final chunk with
 * finish_reason; role!=NULL = first chunk (delta carrying the role). */
static int send_sse_chunk(SOCKET c, const char *model, const char *role,
                          const char *content, int content_len,
                          const char *finish) {
    char esc[3072];
    char line[4096];
    if (content) {
        json_escape(content, (size_t)content_len, esc, sizeof(esc));
        snprintf(line, sizeof(line),
                 "data: {\"id\":\"chatcmpl-nf\",\"object\":\"chat.completion.chunk\","
                 "\"created\":0,\"model\":\"%s\",\"choices\":[{\"index\":0,"
                 "\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
                 model, esc);
    } else if (role) {
        snprintf(line, sizeof(line),
                 "data: {\"id\":\"chatcmpl-nf\",\"object\":\"chat.completion.chunk\","
                 "\"created\":0,\"model\":\"%s\",\"choices\":[{\"index\":0,"
                 "\"delta\":{\"role\":\"%s\"},\"finish_reason\":null}]}\n\n",
                 model, role);
    } else {
        snprintf(line, sizeof(line),
                 "data: {\"id\":\"chatcmpl-nf\",\"object\":\"chat.completion.chunk\","
                 "\"created\":0,\"model\":\"%s\",\"choices\":[{\"index\":0,"
                 "\"delta\":{},\"finish_reason\":\"%s\"}]}\n\n",
                 model, finish);
    }
    return send_str(c, line);
}

/* ---- the built-in page (GET /) ---------------------------------------- */
static const char NF_SERVE_PAGE[] =
"<!doctype html><html><head><meta charset=utf-8>"
"<meta name=viewport content='width=device-width,initial-scale=1'>"
"<title>pulsarforge</title><style>"
"body{font-family:system-ui,sans-serif;max-width:720px;margin:0 auto;"
"padding:12px;background:#111;color:#eee;display:flex;flex-direction:column;"
"height:100vh;box-sizing:border-box}"
"#log{flex:1;overflow-y:auto;padding:8px 0}"
".m{margin:6px 0;padding:10px 14px;border-radius:12px;white-space:pre-wrap;"
"word-wrap:break-word;line-height:1.4}"
".u{background:#2a4d69;margin-left:15%}"
".b{background:#222;margin-right:15%}"
"#bar{display:flex;gap:8px;padding-top:8px}"
"#in{flex:1;padding:10px;border-radius:8px;border:1px solid #444;"
"background:#1a1a1a;color:#eee;font-size:16px}"
"button{padding:10px 18px;border-radius:8px;border:0;background:#2a4d69;"
"color:#eee;font-size:16px;cursor:pointer}"
"h3{margin:4px 0;color:#888;font-weight:normal;font-size:14px}"
"</style></head><body>"
"<h3>pulsarforge &mdash; local engine, no data leaves this machine</h3>"
"<div id=log></div>"
"<div id=bar><input id=in placeholder='Type here...' autofocus>"
"<button id=go>Send</button></div>"
"<script>\n"
"const log=document.getElementById('log'),inp=document.getElementById('in');\n"
"let msgs=[];\n"
"function bubble(cls,txt){const d=document.createElement('div');"
"d.className='m '+cls;d.textContent=txt;log.appendChild(d);"
"log.scrollTop=log.scrollHeight;return d}\n"
"async function go(){const q=inp.value.trim();if(!q)return;inp.value='';\n"
"msgs.push({role:'user',content:q});bubble('u',q);\n"
"const d=bubble('b','');let full='';\n"
"const r=await fetch('/v1/chat/completions',{method:'POST',"
"headers:{'Content-Type':'application/json'},"
"body:JSON.stringify({stream:true,messages:msgs})});\n"
"const rd=r.body.getReader(),dec=new TextDecoder();let buf='';\n"
"for(;;){const{done,value}=await rd.read();if(done)break;\n"
"buf+=dec.decode(value,{stream:true});\n"
"let i;while((i=buf.indexOf('\\n\\n'))>=0){const ln=buf.slice(0,i);"
"buf=buf.slice(i+2);\n"
"if(!ln.startsWith('data: '))continue;const p=ln.slice(6);"
"if(p==='[DONE]')continue;\n"
"try{const j=JSON.parse(p);const t=j.choices[0].delta.content;"
"if(t){full+=t;d.textContent=full;log.scrollTop=log.scrollHeight}}catch(e){}}}\n"
"msgs.push({role:'assistant',content:full})}\n"
"document.getElementById('go').onclick=go;\n"
"inp.addEventListener('keydown',e=>{if(e.key==='Enter')go()});\n"
"</script></body></html>";

/* ---- server state and handler ------------------------------------------ */

/* Session slot (point 5): one conversation = one KV cache.
 * With ONE slot, alternating two conversations in Open WebUI threw
 * away the entire prefill on every switch; with a small LRU, the right
 * cache is found again via longest-prefix match. Sessions are created
 * LAZILY (RAM only when actually needed — the budget is declared at
 * startup) and eviction is LRU.
 *
 * 16 (after fused MoE + mixed ubatch): measured in
 * isolation (no other process, to avoid confusing it with the RAM
 * contention discovered by comparing against llama-server kept alive
 * alongside) — 16 slots vs 8 with the SAME queue of 16 concurrent
 * users: +10.3% (15.31 vs 13.88 tok/s), 4 rounds out of 4, no
 * ambiguity. RAM: at ctx=2048 (the default dropped from 4096, see
 * cmd_serve) 347MB/slot x 16 = 5.55GB — IDENTICAL to Phase 0's budget
 * for 8 slots at ctx=4096 (694MB x 8 = 5.55GB), the same 1.69GB margin
 * beyond the OS reserve. Sessions stay lazy: no RAM cost until they're
 * actually needed. */
#define NF_SRV_SLOTS 16
typedef struct {
    nf_session *s;      /* NULL until the slot is needed */
    int32_t *hist;
    int n_hist;
    uint64_t last_used;
    nf_session *ds;     /* the DRAFT's session (--draft only) */
    int d_past;         /* how many hist tokens the draft has cached */
    /* M8 Phase 3b: with several jobs active AT THE SAME TIME (no longer
     * one at a time like in Phase 1-2), server_pick_slot must NEVER
     * pick a slot for a new request that an in-progress job is still
     * using — its LRU heuristic relies only on last_used, which
     * doesn't distinguish "free for a while" from "held by a long
     * stream that hasn't touched hist recently". Set by the scheduler
     * when a job starts (job_start), cleared when it finishes
     * (job_finish) — NEVER picked either as a reuse candidate or as an
     * eviction victim while it's 1. */
    int pinned;
} nf_srv_slot;

typedef struct {
    nf_tokenizer *tk;
    nf_model *m;
    nf_model *dm;       /* draft model (--draft), NULL = no
                           speculative decoding in the server */
    int draft_k;
    nf_srv_slot slot[NF_SRV_SLOTS];
    uint64_t tick;
    int max_ctx, max_tokens_def, repeat_last_n, no_think;
    int top_k_def;
    float temp_def, top_p_def, min_p_def, repeat_penalty;
    uint64_t rng;
    nf_stopset stop;
    nf_textstop tstop;
    char model_name[160];
    /* M8 Phase 2: Phase 1's critical section is replaced by a real
     * single compute thread (compute_thread_fn) that pulls jobs from
     * this queue — same property (at most one thread in the engine),
     * but now it's a dedicated PERSISTENT thread, not "whichever
     * connection holds the lock": this is the shape Phase 3 will reuse
     * to interleave several sessions per tick. */
    nf_queue reqq;
} nf_server;

/* Picks the slot for an n-token request: the longest common prefix
 * wins; if nothing reuses anything, an empty slot (created now) or the
 * least recently used one. Returns NULL only on OOM.
 *
 * GRAFT: when no slot passes the 3/4 rule (a NEW chat) but
 * someone shares a substantial prefix (a webui's system prompt: ~280
 * tokens), that prefix's KV rows get COPIED into the chosen slot
 * (nf_session_copy_prefix, bit-exact) — a memcpy of tens of MB instead
 * of tens of seconds of prefill on the 30B. NF_SRV_GRAFT_MIN threshold:
 * below it, the memcpy doesn't pay for the orchestration and prefill
 * happens as usual. */
#define NF_SRV_GRAFT_MIN 32
static nf_srv_slot *server_pick_slot(nf_server *sv, const int32_t *ids,
                                     int n, int *out_pfx) {
    nf_srv_slot *best = NULL;
    int best_pfx = 0;
    nf_srv_slot *graft = NULL;
    int graft_pfx = 0;
    for (int i = 0; i < NF_SRV_SLOTS; i++) {
        nf_srv_slot *sl = &sv->slot[i];
        if (!sl->s || sl->pinned) continue;   /* Phase 3b: never a slot in use */
        int pfx = 0;
        while (pfx < sl->n_hist && pfx < n - 1 && sl->hist[pfx] == ids[pfx])
            pfx++;
        if (pfx > graft_pfx) { graft = sl; graft_pfx = pfx; }
        /* it's the SAME conversation only if the prefix covers (almost)
         * the slot's entire history: a continuation fully contains the
         * previous turns (the "almost": regenerations/edits of the
         * last message). A short match is just shared boilerplate
         * (ChatML headers, common phrases): taking that slot would
         * DESTROY the conversation living in it — measured on the
         * first test: two chats with the same 12-token opening ended
         * up in the same slot. */
        if (pfx < (sl->n_hist * 3) / 4) continue;
        if (pfx > best_pfx || (pfx == best_pfx && best && sl->last_used >
                               best->last_used)) {
            best = sl;
            best_pfx = pfx;
        }
    }
    if (best_pfx == 0) {
        /* nothing to reuse: first a never-used slot, then the LRU one */
        best = NULL;
        for (int i = 0; i < NF_SRV_SLOTS && !best; i++)
            if (!sv->slot[i].s) best = &sv->slot[i];
        if (best) {
            best->s = nf_session_create(sv->m, sv->max_ctx);
            best->hist = malloc((size_t)sv->max_ctx * sizeof(int32_t));
            /* The draft's session is born and dies with the slot */
            best->ds = sv->dm ? nf_session_create(sv->dm, sv->max_ctx)
                              : NULL;
            best->d_past = 0;
            if (!best->s || !best->hist || (sv->dm && !best->ds)) {
                nf_session_free(best->s);
                nf_session_free(best->ds);
                free(best->hist);
                best->s = NULL;
                best->hist = NULL;
                best->ds = NULL;
                best = NULL;
            } else {
                fprintf(stderr, "nf: [serve] slot %d: session created "
                        "(%.0fMB of KV at ctx %d%s)\n",
                        (int)(best - sv->slot),
                        nf_session_kv_bytes(sv->m, sv->max_ctx) / 1e6,
                        sv->max_ctx, sv->dm ? ", + draft" : "");
            }
        }
        if (!best) {
            for (int i = 0; i < NF_SRV_SLOTS; i++) {
                nf_srv_slot *sl = &sv->slot[i];
                if (sl->s && !sl->pinned &&
                    (!best || sl->last_used < best->last_used))
                    best = sl;
            }
            if (best) {
                best->n_hist = 0; /* eviction: starts over from scratch */
                best->d_past = 0;
                if (best->ds) nf_session_rewind(best->ds, 0);
            }
        }
        /* Graft the shared prefix (typically the system prompt)
         * from the slot that owns it — never from ourselves (LRU
         * eviction might have picked the very donor). NOTE (a bug
         * found and fixed in nf_session_copy_prefix): can
         * also fire when the "donor" is actually the remains of the
         * SAME conversation in another slot (prompt reused after an
         * eviction) — legitimate, not a special case to exclude: the
         * graft stays correct as long as the function copies the right
         * layout. */
        if (best && graft && graft != best &&
            graft_pfx >= NF_SRV_GRAFT_MIN &&
            nf_session_copy_prefix(best->s, graft->s, graft_pfx) == 0) {
            memcpy(best->hist, graft->hist,
                   (size_t)graft_pfx * sizeof(int32_t));
            best->n_hist = graft_pfx;
            best_pfx = graft_pfx;
            /* The draft's cache is grafted too, if the donor has
             * it far enough along; otherwise the draft starts over
             * from scratch and lazily prefills on the first
             * speculative round (it's a 0.6B: cheap). */
            best->d_past = 0;
            if (best->ds && graft->ds && graft->d_past >= graft_pfx &&
                nf_session_copy_prefix(best->ds, graft->ds, graft_pfx) == 0)
                best->d_past = graft_pfx;
            fprintf(stderr, "nf: [serve] slot %d: GRAFTED %d tokens "
                    "from slot %d (shared system prompt)\n",
                    (int)(best - sv->slot), graft_pfx,
                    (int)(graft - sv->slot));
        }
    }
    if (best) best->last_used = ++sv->tick;
    *out_pfx = best_pfx;
    return best;
}

/* The core: ChatML prompt from messages[], prefix caching on the
 * session, generation with the usual loop, SSE or full-JSON output. */
/* M8 Phase 2: generation splits into start() (once: picks a slot,
 * rewinds the KV cache to the common prefix, prefills the suffix —
 * the ChatML prompt and tokenization were already done by the I/O
 * thread, which never touches the model) and step() (ONE piece of
 * work at a time: one token on the smooth path, one WHOLE round on
 * the speculative path — it produces several tokens per round by
 * construction, and that's why it stays out of Phase 3's
 * cross-session batching). The scheduler
 * (compute_thread_fn) calls step() repeatedly until it returns 0 — in
 * this phase ONE request at a time, but the shape is what Phase 3 will
 * reuse to interleave several sessions. All the fields are the state
 * that used to live in handle_completions's local variables: now it
 * has to survive between one step() call and the next. */
typedef struct {
    nf_server *sv;
    SOCKET c;
    int32_t *ids;
    int n;
    int stream;
    float temp, top_p, min_p;
    int32_t top_k;
    int max_new_req;

    nf_srv_slot *sl;
    const float *logits;
    int pfx;
    double t_prefill, t1;
    int max_new;
    int client_ok, stopped, emitted;
    char pending[512];
    int n_pending;
    char *full;
    size_t full_cap, full_len;

    int speculative;
    int spec_guard_ok;
    int64_t vocab;
    int K;
    float *va;
    int32_t *dtok;
    int32_t *batch;
    int rounds, drafted, accepted;
    int t_past;

    /* MIXED ubatch: prefill is no longer a blocking call
     * inside job_start — it advances in CHUNKS, one piece per tick,
     * with its rows mixed into other sessions' decode in the SAME
     * round (measured: monolithic prefill starved whoever was already
     * active). prefilling=1 while n_prefilled < n. */
    int prefilling;
    int n_prefilled;
} nf_srv_job;

/* How many prefill rows fit in one tick. 16: with the decode of 8
 * sessions that makes ~24 rows per pass, the zone where the sweep
 * (nf debugbatchscale) measures the best GMAC/s (36.6/16.7 vs 32.3/14.2
 * at 8 rows) — and these are rows obtained WITHOUT paying for other
 * concurrent users' RAM, which is why the B=24-user experiment had
 * failed. */
#define NF_PREFILL_CHUNK 16

/* DEBUG ("head-of-line bottleneck"): time relative to
 * first use, to see whether the prefills of several simultaneous
 * arrivals queue up (no decode tick for ANYONE until the LAST one has
 * finished its own prefill) instead of interleaving. Zero cost if
 * NF_DEBUG_CASCADE isn't set (getenv() is the only expense). */
static double dbg_cascade_t(void) {
    static double t0 = -1.0;
    if (t0 < 0.0) t0 = now_seconds();
    return now_seconds() - t0;
}

/* Emission of ONE token (shared between the smooth and speculative
 * paths): append to hist, id-based stop, decode, textual filter, SSE
 * send or JSON accumulation — used to be the NF_SRV_EMIT macro, now a
 * function because the state lives in the job instead of the call's
 * local variables. */
static void job_emit(nf_srv_job *jb, int32_t tok) {
    nf_server *sv = jb->sv;
    jb->sl->hist[jb->sl->n_hist++] = tok;
    jb->emitted++;
    if (jb->emitted == 1 && getenv("NF_DEBUG_CASCADE"))
        fprintf(stderr, "nf: [cascade] t=%.3fs FIRST-TOKEN socket=%d\n",
                dbg_cascade_t(), (int)jb->c);
    if (is_stop_token(&sv->stop, tok)) { jb->stopped = 1; return; }
    char piece[256];
    const int w = nf_token_decode(sv->tk, tok, piece, sizeof(piece) - 1);
    /* spurious </think> filter at turn start: parity with
     * the CLI (turn_first_piece) — seen live via the API ("content":
     * "</think>Paris."). Only the turn's FIRST piece, glm only. */
    if (g_chat_style == 3 && jb->emitted == 1 && w == 8 &&
        memcmp(piece, "</think>", 8) == 0)
        return;
    char out[1024];
    int n_out;
    if (stopfilter_feed(piece, w > 0 ? w : 0, jb->pending, &jb->n_pending,
                        (int)sizeof(jb->pending), &sv->tstop, out, &n_out))
        jb->stopped = 1;
    if (n_out > 0) {
        if (jb->stream) {
            if (send_sse_chunk(jb->c, sv->model_name, NULL, out, n_out, NULL))
                jb->client_ok = 0;
        } else {
            if (jb->full_len + (size_t)n_out + 1 > jb->full_cap) {
                jb->full_cap = jb->full_cap ? jb->full_cap * 2 : 4096;
                while (jb->full_cap < jb->full_len + (size_t)n_out + 1)
                    jb->full_cap *= 2;
                char *nf_ = realloc(jb->full, jb->full_cap);
                if (!nf_) { jb->client_ok = 0; return; }
                jb->full = nf_;
            }
            memcpy(jb->full + jb->full_len, out, (size_t)n_out);
            jb->full_len += (size_t)n_out;
        }
    }
}

/* "start" phase (once, ONLY here does the model get touched): picks
 * the slot, rewinds to the common prefix, prefills the new suffix.
 * 0 = can proceed with step(); -1 = error already answered to the
 * client. */
static int job_start(nf_srv_job *jb) {
    nf_server *sv = jb->sv;
    int pfx = 0;
    nf_srv_slot *sl = server_pick_slot(sv, jb->ids, jb->n, &pfx);
    if (!sl) {
        send_simple(jb->c, "500 Internal Server Error", "application/json",
                    "{\"error\":\"session oom\"}");
        return -1;
    }
    sl->pinned = 1;   /* Phase 3b: from here on no other job can touch it */
    jb->sl = sl;      /* assigned RIGHT AWAY: if something below fails,
                        * the caller still knows which slot to unpin */
    fprintf(stderr, "nf: [serve] slot %d: reused prefix %d/%d tokens\n",
            (int)(sl - sv->slot), pfx, jb->n);
    nf_session_rewind(sl->s, pfx);
    /* The draft follows — its cache can't contain tokens hist no
     * longer has (the rest gets lazily prefilled on the first
     * speculative round) */
    if (sl->ds && sl->d_past > pfx) {
        nf_session_rewind(sl->ds, pfx);
        sl->d_past = pfx;
    }
    const int dbg_casc = getenv("NF_DEBUG_CASCADE") != NULL;
    if (dbg_casc)
        fprintf(stderr, "nf: [cascade] t=%.3fs prefill-START socket=%d "
                "slot=%d n_new=%d\n", dbg_cascade_t(), (int)jb->c,
                (int)(sl - sv->slot), jb->n - pfx);

    /* MIXED ubatch: prefill is DEFERRED to the scheduler, which
     * advances it in chunks mixed with the decode of other sessions.
     * Not chunkable (-> blocking prefill as before, unchanged
     * behavior): speculative decoding (variable-width rounds, excluded
     * from cross-session batching by construction) and models without
     * row kernels (NF_MODEL_PLAIN — neither deepseek2 nor qwen3/moe). */
    jb->prefilling = 0;
    jb->n_prefilled = pfx;
    const int spec_req = sv->dm && jb->temp == 0.0f;
    const int can_chunk = nf_model_kind_of(sv->m) != NF_MODEL_PLAIN && !spec_req
                       && getenv("NF_MIXBATCH_OFF") == NULL;
    if (can_chunk && jb->n - pfx > 0) {
        memcpy(sl->hist, jb->ids, (size_t)jb->n * sizeof(int32_t));
        sl->n_hist = jb->n;
        jb->t_prefill = 0.0;
        jb->logits = NULL;
        jb->pfx = pfx;
        int max_new = jb->max_new_req;
        if (max_new > sv->max_ctx - sl->n_hist) max_new = sv->max_ctx - sl->n_hist;
        jb->max_new = max_new;
        jb->client_ok = 1;
        if (jb->stream) {
            char hdr[512];
            snprintf(hdr, sizeof(hdr),
                     "HTTP/1.1 200 OK\r\n%sContent-Type: text/event-stream\r\n"
                     "Cache-Control: no-cache\r\nConnection: close\r\n\r\n",
                     NF_CORS);
            jb->client_ok = send_str(jb->c, hdr) == 0 &&
                            send_sse_chunk(jb->c, sv->model_name, "assistant",
                                           NULL, 0, NULL) == 0;
        }
        jb->emitted = 0;
        jb->stopped = 0;
        jb->n_pending = 0;
        jb->full = NULL;
        jb->full_cap = 0;
        jb->full_len = 0;
        /* t1 acts as the prefill-start marker: when the last chunk
         * finishes it becomes the start of generation, so the log
         * (prefill in Xs, generated in Ys) stays truthful even chunked. */
        jb->t1 = now_seconds();
        jb->speculative = 0;
        jb->prefilling = 1;   /* the scheduler will advance it in chunks */
        return 0;
    }

    const double t0 = now_seconds();
    const float *logits = nf_session_eval(sl->s, jb->ids + pfx, jb->n - pfx);
    if (dbg_casc)
        fprintf(stderr, "nf: [cascade] t=%.3fs prefill-END socket=%d "
                "slot=%d (%.3fs)\n", dbg_cascade_t(), (int)jb->c,
                (int)(sl - sv->slot), now_seconds() - t0);
    if (!logits) {
        send_simple(jb->c, "500 Internal Server Error", "application/json",
                    "{\"error\":\"eval failed\"}");
        sl->pinned = 0;   /* failed: no job_finish will come to unpin it */
        return -1;
    }
    memcpy(sl->hist, jb->ids, (size_t)jb->n * sizeof(int32_t));
    sl->n_hist = jb->n;
    jb->t_prefill = now_seconds() - t0;
    jb->logits = logits;
    jb->pfx = pfx;

    int max_new = jb->max_new_req;
    if (max_new > sv->max_ctx - sl->n_hist) max_new = sv->max_ctx - sl->n_hist;
    jb->max_new = max_new;

    jb->client_ok = 1;
    if (jb->stream) {
        char hdr[512];
        snprintf(hdr, sizeof(hdr),
                 "HTTP/1.1 200 OK\r\n%sContent-Type: text/event-stream\r\n"
                 "Cache-Control: no-cache\r\nConnection: close\r\n\r\n",
                 NF_CORS);
        jb->client_ok = send_str(jb->c, hdr) == 0 &&
                        send_sse_chunk(jb->c, sv->model_name, "assistant",
                                       NULL, 0, NULL) == 0;
    }

    jb->emitted = 0;
    jb->stopped = 0;
    jb->n_pending = 0;
    jb->full = NULL;
    jb->full_cap = 0;
    jb->full_len = 0;
    jb->t1 = now_seconds();

    /* SPECULATIVE generation (greedy only — the identity guarantee
     * with direct decode holds there). The first token comes from the
     * prefill's logits; from there on the target has ONE pending token
     * per round (base = n_pend-1). */
    jb->speculative = sv->dm && jb->temp == 0.0f;
    if (jb->speculative) {
        jb->vocab = (int64_t)nf_model_vocab(sv->m);
        jb->K = sv->draft_k;
        jb->va = malloc((size_t)(jb->K + 2) * jb->vocab * sizeof(float));
        jb->dtok = malloc((size_t)jb->K * sizeof(int32_t));
        jb->batch = malloc((size_t)(jb->K + 2) * sizeof(int32_t));
        jb->rounds = jb->drafted = jb->accepted = 0;
        jb->t_past = sl->n_hist;
        jb->spec_guard_ok = jb->va && jb->dtok && jb->batch &&
                            sl->n_hist < sv->max_ctx && jb->max_new > 0;
        if (jb->spec_guard_ok) {
            const int win = sl->n_hist < sv->repeat_last_n
                          ? sl->n_hist : sv->repeat_last_n;
            apply_repeat_penalty((float *)jb->logits, jb->vocab,
                                 sl->hist + sl->n_hist - win, win,
                                 sv->repeat_penalty);
            int32_t first = sample_token(jb->logits, jb->vocab, 0.0f,
                                         jb->top_p, 0, 0.0f, &sv->rng);
            if (is_stop_token(&sv->stop, first)) {   /* empty-turn guard */
                float *lg_ = (float *)jb->logits;
                for (int si_ = 0; si_ < sv->stop.n; si_++)
                    lg_[sv->stop.ids[si_]] = -1e30f;
                first = sample_token(jb->logits, jb->vocab, 0.0f,
                                     jb->top_p, 0, 0.0f, &sv->rng);
                fprintf(stderr, "nf: [serve] terminal first token "
                        "discarded (empty turn)\n");
            }
            job_emit(jb, first);
        }
    } else if (sv->dm) {
        fprintf(stderr, "nf: [serve] temp>0: smooth path (speculative "
                "is greedy only)\n");
    }
    return 0;
}

/* M8 Phase 3b: samples the smooth path's next token WITHOUT calling
 * the model — the call (nf_session_eval for a single session) becomes
 * a tick grouped with other ready sessions (dsk_eval_batched_sessions),
 * done by the scheduler right after collecting the tokens of EVERY
 * job ready this round. Same logic and order as the old
 * job_step_smooth, just split in two: this half (samples, no effect
 * on the cache) and job_emit (called after the grouped tick, with the
 * outcome already in sl->s->logits). Returns 1 and writes *out_tok if
 * the job enters the next tick, 0 if it has no more work to do (the
 * caller considers it finished). */
static int job_presample(nf_srv_job *jb, int32_t *out_tok) {
    nf_server *sv = jb->sv;
    nf_srv_slot *sl = jb->sl;
    if (!(jb->client_ok && !jb->stopped && jb->emitted < jb->max_new))
        return 0;
    const int64_t vocab = (int64_t)nf_model_vocab(sv->m);
    const int win = sl->n_hist < sv->repeat_last_n ? sl->n_hist
                                                   : sv->repeat_last_n;
    apply_repeat_penalty((float *)jb->logits, vocab,
                         sl->hist + sl->n_hist - win, win, sv->repeat_penalty);
    int32_t next = sample_token(jb->logits, vocab, jb->temp, jb->top_p,
                                jb->top_k, jb->min_p, &sv->rng);
    /* empty-turn guard (parity with the chat CLI): a
     * TERMINAL as the first token = an empty response, never useful.
     * Seen live on serve (completion_tokens=1, content=""). */
    if (jb->emitted == 0 && is_stop_token(&sv->stop, next)) {
        float *lg_ = (float *)jb->logits;
        for (int si_ = 0; si_ < sv->stop.n; si_++)
            lg_[sv->stop.ids[si_]] = -1e30f;
        next = sample_token(jb->logits, vocab, jb->temp, jb->top_p,
                            jb->top_k, jb->min_p, &sv->rng);
        fprintf(stderr, "nf: [serve] terminal first token discarded "
                "(empty turn): resampled the best non-terminal one\n");
    }
    if (sl->n_hist >= sv->max_ctx) return 0;   /* finish_reason "length" */
    *out_tok = next;
    return 1;
}

/* step() speculative path: ONE whole round (propose-verify-
 * emit-rewind) per call — produces several tokens per round by
 * construction, stays out of cross-session batching (Phase 3). */
static int job_step_spec(nf_srv_job *jb) {
    nf_server *sv = jb->sv;
    nf_srv_slot *sl = jb->sl;
    const int64_t vocab = jb->vocab;
    const int K = jb->K;
    if (!(jb->client_ok && !jb->stopped && jb->emitted < jb->max_new))
        return 0;

    const int len = sl->n_hist;
    if (len + K + 1 > sv->max_ctx) return 0;
    /* 1) the draft proposes K greedy tokens from its own state */
    const float *dl = nf_session_eval(sl->ds, sl->hist + sl->d_past,
                                      len - sl->d_past);
    if (!dl) return 0;
    sl->d_past = len;
    int nd = 0;
    for (int j = 0; j < K; j++) {
        penalty_virtual((float *)dl, vocab, sl->hist, len, jb->dtok, j,
                        sv->repeat_penalty, sv->repeat_last_n);
        jb->dtok[j] = sample_token(dl, vocab, 0.0f, jb->top_p, 0, 0.0f,
                                   &sv->rng);
        nd++;
        dl = nf_session_eval(sl->ds, &jb->dtok[j], 1);
        if (!dl) break;
    }
    sl->d_past += nd;
    jb->drafted += nd;
    /* 2) batched verification of pending + proposals */
    const int n_pend = len - jb->t_past;
    const int n_batch = n_pend + nd;
    memcpy(jb->batch, sl->hist + jb->t_past, (size_t)n_pend * sizeof(int32_t));
    memcpy(jb->batch + n_pend, jb->dtok, (size_t)nd * sizeof(int32_t));
    if (!nf_session_eval_all(sl->s, jb->batch, n_batch, jb->va)) return 0;
    jb->t_past += n_batch;
    const int base = n_pend - 1;
    /* 3) accepts up to the first disagreement, then a correction */
    int n_acc = 0;
    int32_t fix = -1;
    for (int j = 0; j < nd; j++) {
        float *Lj = jb->va + (size_t)(base + j) * vocab;
        penalty_virtual(Lj, vocab, sl->hist, len, jb->dtok, j,
                        sv->repeat_penalty, sv->repeat_last_n);
        const int32_t g = sample_token(Lj, vocab, 0.0f, jb->top_p, 0, 0.0f,
                                       &sv->rng);
        if (g == jb->dtok[j]) { n_acc++; continue; }
        fix = g;
        break;
    }
    if (fix < 0) {
        float *Lk = jb->va + (size_t)(base + nd) * vocab;
        penalty_virtual(Lk, vocab, sl->hist, len, jb->dtok, nd,
                        sv->repeat_penalty, sv->repeat_last_n);
        fix = sample_token(Lk, vocab, 0.0f, jb->top_p, 0, 0.0f, &sv->rng);
    }
    jb->accepted += n_acc;
    jb->rounds++;
    /* 4) emit the kept tokens + the correction */
    const int old_len = len;
    for (int e = 0; e <= n_acc && jb->client_ok && !jb->stopped &&
                    jb->emitted < jb->max_new; e++) {
        if (sl->n_hist >= sv->max_ctx) { jb->stopped = 1; break; }
        const int32_t tok = e < n_acc ? jb->dtok[e] : fix;
        job_emit(jb, tok);
    }
    /* 5) roll back the caches to the prefix actually kept */
    int keep = old_len + n_acc;
    if (keep > sl->n_hist) keep = sl->n_hist;
    nf_session_rewind(sl->s, keep);
    nf_session_rewind(sl->ds, keep);
    jb->t_past = keep;
    sl->d_past = keep;
    return 1;
}

/* "finish" phase (once, called by the scheduler when a job runs out of
 * work): resyncs the draft's cache, flushes the textual filter, final
 * response, log, socket close. */
static void job_finish(nf_srv_job *jb) {
    nf_server *sv = jb->sv;
    nf_srv_slot *sl = jb->sl;
    sl->pinned = 0;   /* Phase 3b: from here on the slot is available again */
    if (jb->speculative) {
        if (jb->spec_guard_ok) {
            /* target cache back in sync with hist: this is the
             * invariant that keeps the next turn's prefix caching exact */
            if (sl->n_hist > jb->t_past &&
                nf_session_eval(sl->s, sl->hist + jb->t_past,
                                sl->n_hist - jb->t_past))
                jb->t_past = sl->n_hist;
            fprintf(stderr, "nf: [serve] speculative: %d rounds, %d/%d "
                    "accepted (%.0f%%), K=%d\n", jb->rounds, jb->accepted,
                    jb->drafted,
                    jb->drafted ? 100.0 * jb->accepted / jb->drafted : 0.0,
                    jb->K);
        }
        free(jb->va);
        free(jb->dtok);
        free(jb->batch);
    }

    /* filter's tail (held-back text that wasn't a stop) */
    if (jb->client_ok && !jb->stopped && jb->n_pending > 0) {
        if (jb->stream) {
            if (send_sse_chunk(jb->c, sv->model_name, NULL, jb->pending,
                               jb->n_pending, NULL))
                jb->client_ok = 0;
        } else if (jb->full_len + (size_t)jb->n_pending + 1 <= jb->full_cap ||
                   (jb->full = realloc(jb->full, jb->full_cap =
                        jb->full_len + (size_t)jb->n_pending + 1)) != NULL) {
            memcpy(jb->full + jb->full_len, jb->pending, (size_t)jb->n_pending);
            jb->full_len += (size_t)jb->n_pending;
        }
    }
    const double t_gen = now_seconds() - jb->t1;
    const char *finish = jb->stopped ? "stop" : "length";

    if (jb->client_ok) {
        if (jb->stream) {
            send_sse_chunk(jb->c, sv->model_name, NULL, NULL, 0, finish);
            send_str(jb->c, "data: [DONE]\n\n");
        } else {
            char esc[16384];
            json_escape(jb->full ? jb->full : "", jb->full_len, esc,
                       sizeof(esc));
            char *resp = malloc(strlen(esc) + 1024);
            if (resp) {
                const int rl = snprintf(resp, strlen(esc) + 1024,
                    "{\"id\":\"chatcmpl-nf\",\"object\":\"chat.completion\","
                    "\"created\":0,\"model\":\"%s\",\"choices\":[{\"index\":0,"
                    "\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
                    "\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,"
                    "\"completion_tokens\":%d,\"total_tokens\":%d}}",
                    sv->model_name, esc, finish, jb->n, jb->emitted,
                    jb->n + jb->emitted);
                char hdr[512];
                snprintf(hdr, sizeof(hdr),
                         "HTTP/1.1 200 OK\r\n%sContent-Type: application/json\r\n"
                         "Content-Length: %d\r\nConnection: close\r\n\r\n",
                         NF_CORS, rl);
                if (send_str(jb->c, hdr) == 0) send_str(jb->c, resp);
                free(resp);
            }
        }
    }
    free(jb->full);

    fprintf(stderr, "nf: [serve] prompt=%d (reused %d, evaluated %d in %.1fs) "
            "generated=%d in %.1fs (%.2f tok/s)%s\n",
            jb->n, jb->pfx, jb->n - jb->pfx, jb->t_prefill, jb->emitted,
            t_gen, t_gen > 0 ? jb->emitted / t_gen : 0.0,
            jb->client_ok ? "" : " [client disconnected]");
    nf_debug_timing_report();   /* no-op unless NF_DEBUG_TIMING */

    shutdown(jb->c, SD_SEND);
    closesocket(jb->c);
}

/* Shutdown state: defined further below alongside the Ctrl+C handler
 * (tentative definitions, C11) — needed here because the POST
 * /shutdown route raises it too. */
static volatile int g_srv_stop;
static SOCKET g_srv_sock;

/* Compares a header's value against `want` (for the
 * /shutdown token). Looks for the "Name:" line in the header section
 * and compares its value stripped of leading spaces and trailing
 * CR/LF. Doesn't allocate, nothing to free, and doesn't modify the
 * buffer (unlike the first-line parser, which plants terminators
 * there): can be called after http_read_request without disturbing
 * method/path/body.
 * 0 = matches, -1 = header absent or value different. */
static int http_header_equals(const char *buf, const char *end,
                              const char *name, const char *want) {
    if (!buf || !end || !name || !want || !*want) return -1;
    const size_t nlen = strlen(name), wlen = strlen(want);
    /* Scan with EXPLICIT BOUNDS, never string-based: by the time this
     * helper is called, http_read_request has already planted
     * terminators inside the first line (to separate method and path),
     * so any strstr/strchr starting from the beginning would stop
     * there without ever seeing the headers. */
    const char *h = buf;
    while (h < end) {
        const char *eol = h;
        while (eol + 1 < end && !(eol[0] == '\r' && eol[1] == '\n')) eol++;
        if (eol + 1 >= end) break;
        if ((size_t)(eol - h) > nlen + 1 &&
            _strnicmp(h, name, nlen) == 0 && h[nlen] == ':') {
            const char *v = h + nlen + 1;
            while (v < eol && (*v == ' ' || *v == '\t')) v++;
            return ((size_t)(eol - v) == wlen && memcmp(v, want, wlen) == 0)
                   ? 0 : -1;
        }
        h = eol + 2;
    }
    return -1;
}

/* Reads a complete HTTP request (headers + body from Content-Length). */
static int http_read_request(SOCKET c, char **method, char **path,
                             char **body, char *buf, int buf_cap) {
    int used = 0;
    char *hdr_end = NULL;
    while (used < buf_cap - 1) {
        const int r = recv(c, buf + used, buf_cap - 1 - used, 0);
        if (r <= 0) return -1;
        used += r;
        buf[used] = 0;
        hdr_end = strstr(buf, "\r\n\r\n");
        if (hdr_end) break;
    }
    if (!hdr_end) return -1;
    const int hdr_len = (int)(hdr_end - buf) + 4;

    *method = buf;
    char *sp = strchr(buf, ' ');
    if (!sp) return -1;
    *sp = 0;
    *path = sp + 1;
    sp = strchr(*path, ' ');
    if (!sp) return -1;
    *sp = 0;
    char *qm = strchr(*path, '?');
    if (qm) *qm = 0;

    int clen = 0;
    for (char *h = sp + 1; h < hdr_end; h++)
        if (_strnicmp(h, "Content-Length:", 15) == 0) {
            clen = atoi(h + 15);
            break;
        }
    if (clen < 0 || clen > 4 << 20) return -1;
    if (hdr_len + clen > buf_cap - 1) return -1;
    while (used < hdr_len + clen) {
        const int r = recv(c, buf + used, buf_cap - 1 - used, 0);
        if (r <= 0) return -1;
        used += r;
    }
    buf[hdr_len + clen] = 0;
    *body = buf + hdr_len;
    return 0;
}

/* M8 Phase 1: counter of still-alive connection threads — an accept()
 * spawns a thread and increments BEFORE _beginthreadex, conn_thread_fn
 * decrements as the last thing before returning. Shutdown waits for it
 * to drop to zero BEFORE touching sv.slot[] to save it: the compute
 * critical section alone guarantees that one thread at a time touches
 * the model, but it isn't enough on its own to know when the LAST
 * thread has finished (there could be others still queued for the
 * critical section, or still reading/tokenizing before reaching it). */
static volatile long g_srv_inflight = 0;

typedef struct { nf_server *sv; SOCKET c; } nf_conn_arg;

/* Body of ONE connection (used to be the body of cmd_serve's
 * while(accept)): reads/tokenizes WITHOUT a lock (safe, never touches
 * the model), then only the part that touches nf_model/nf_session
 * (server_pick_slot inside handle_completions, prefill, the generation
 * loop) is wrapped by the server's critical section — one thread at a
 * time, exactly like the old serial accept(). */
static NF_THREAD_RET conn_thread_fn(void *arg_) {
    nf_conn_arg *arg = (nf_conn_arg *)arg_;
    nf_server *sv = arg->sv;
    const SOCKET c = arg->c;
    free(arg);
    /* M8 Phase 2: if a job gets created and queued, the socket and the
     * "in flight" count move to the compute thread (it closes the
     * socket in job_finish, it decrements g_srv_inflight) — this
     * thread must NOT touch them anymore in the epilogue. */
    int handed_off = 0;
    const int dbg = getenv("NF_DEBUG_CONN") != NULL;
    if (dbg) fprintf(stderr, "nf: [conn] accepted (socket %d)\n", (int)c);

    const int REQ_CAP = 8 << 20;
    char *reqbuf = malloc((size_t)REQ_CAP);
    if (!reqbuf && dbg)
        fprintf(stderr, "nf: [conn] reqbuf malloc failed (socket %d)\n", (int)c);
    if (reqbuf) {
        char *method, *path, *body;
        const int hrr = http_read_request(c, &method, &path, &body, reqbuf, REQ_CAP);
        if (dbg) fprintf(stderr, "nf: [conn] http_read_request=%d (socket %d)\n",
                        hrr, (int)c);
        if (hrr == 0) {
            if (strcmp(method, "OPTIONS") == 0) {
                char hdr[256];
                snprintf(hdr, sizeof(hdr),
                         "HTTP/1.1 204 No Content\r\n%s"
                         "Connection: close\r\n\r\n", NF_CORS);
                send_str(c, hdr);
            } else if (strcmp(method, "GET") == 0 && strcmp(path, "/") == 0) {
                send_simple(c, "200 OK", "text/html; charset=utf-8",
                            NF_SERVE_PAGE);
            } else if (strcmp(method, "GET") == 0 &&
                       strcmp(path, "/v1/models") == 0) {
                char mbody[512];
                snprintf(mbody, sizeof(mbody),
                         "{\"object\":\"list\",\"data\":[{\"id\":\"%s\","
                         "\"object\":\"model\",\"owned_by\":\"pulsarforge\"}]}",
                         sv->model_name);
                send_simple(c, "200 OK", "application/json", mbody);
            } else if (strcmp(method, "POST") == 0 &&
                       strcmp(path, "/shutdown") == 0) {
                /* Clean shutdown WITHOUT a console.
                 *
                 * Why it's needed: session saving (--sessions-dir) is
                 * hooked ONLY to the Ctrl+C handler, which requires an
                 * interactive console. A server started from a script,
                 * as a service, or in the background doesn't have one:
                 * on termination the conversations get lost in
                 * SILENCE. Discovered while trying to validate
                 * persistence end-to-end — `taskkill` without /F on a
                 * console-less process never reaches the handler and
                 * the process even stays alive.
                 *
                 * DISABLED by default: this server already declares it
                 * has no authentication, and a remote shutdown open to
                 * anyone would be a gift. Enabled ONLY by setting
                 * NF_SERVE_SHUTDOWN_TOKEN, and the request must carry
                 * the same value in the X-Shutdown-Token header.
                 * Without the variable the route answers 404 like any
                 * unknown path: no extra surface for whoever doesn't
                 * use it, and no hint of its existence. */
                const char *want = getenv("NF_SERVE_SHUTDOWN_TOKEN");
                if (want && *want &&
                    http_header_equals(reqbuf, body, "X-Shutdown-Token",
                                       want) == 0) {
                    fprintf(stderr, "nf: [serve] shutdown requested via "
                            "/shutdown — saving sessions and exiting\n");
                    send_simple(c, "200 OK", "application/json",
                                "{\"status\":\"shutting down\"}");
                    g_srv_stop = 1;
                    if (g_srv_sock != INVALID_SOCKET) {
                        closesocket(g_srv_sock);
                        g_srv_sock = INVALID_SOCKET;
                    }
                } else {
                    send_simple(c, "404 Not Found", "text/plain", "not found");
                }
            } else if (strcmp(method, "POST") == 0 &&
                       strcmp(path, "/v1/chat/completions") == 0) {
                nf_chat_req req;
                const int pcr = parse_chat_request(body, &req);
                if (dbg) fprintf(stderr, "nf: [conn] parse_chat_request=%d "
                                "(socket %d)\n", pcr, (int)c);
                if (pcr == 0) {
                    /* ChatML prompt + tokenization: safe outside the
                     * compute thread (strings only, verified in the
                     * design investigation) — from here on ONLY
                     * job_start/job_step/job_finish touch the model,
                     * and it's the scheduler that calls them. */
                    char *cbuf = NULL;
                    size_t ccap = 0, clen = 0;
                    int cml_ok = 1;
                    /* glm-dsa: cbuf ALWAYS rebuilds the whole
                     * conversation from scratch starting from req.msgs
                     * (it's not incremental like nf chat) — is_first is
                     * therefore ALWAYS true here, not conditioned on any
                     * session state (the prefix match against the
                     * existing KV cache happens later, inside
                     * job_start, on the same logical prompt that still
                     * has to start with [gMASK]<sop> at position 0). */
                    append_glm_prefix_if_first(&cbuf, &ccap, &clen, 1);
                    for (int mi = 0; mi < req.n_msgs && cml_ok; mi++)
                        if (append_turn(&cbuf, &ccap, &clen,
                                       req.msgs[mi].role,
                                       req.msgs[mi].content))
                            cml_ok = 0;
                    if (cml_ok)
                        clen = append_assistant_open(cbuf, ccap, clen,
                                                     sv->no_think);

                    int32_t *ids = cml_ok
                        ? malloc((size_t)sv->max_ctx * sizeof(int32_t))
                        : NULL;
                    const int64_t n64 = ids
                        ? nf_tokenize(sv->tk, cbuf, ids, sv->max_ctx) : -1;
                    if (dbg) fprintf(stderr, "nf: [conn] cml_ok=%d n64=%lld "
                                    "(socket %d)\n", cml_ok,
                                    (long long)n64, (int)c);
                    free(cbuf);

                    if (!ids) {
                        send_simple(c, "500 Internal Server Error",
                                   "application/json", "{\"error\":\"oom\"}");
                    } else if (n64 <= 0 || n64 >= sv->max_ctx) {
                        send_simple(c, "400 Bad Request", "application/json",
                                    "{\"error\":\"prompt empty or beyond "
                                    "the context\"}");
                        free(ids);
                    } else {
                        nf_srv_job *jb = calloc(1, sizeof(nf_srv_job));
                        if (!jb) {
                            send_simple(c, "500 Internal Server Error",
                                       "application/json",
                                       "{\"error\":\"oom\"}");
                            free(ids);
                        } else {
                            jb->sv = sv;
                            jb->c = c;
                            jb->ids = ids;
                            jb->n = (int)n64;
                            jb->stream = req.stream;
                            jb->temp = req.temperature >= 0.0f
                                     ? req.temperature : sv->temp_def;
                            jb->top_p = req.top_p >= 0.0f
                                      ? req.top_p : sv->top_p_def;
                            jb->top_k = req.top_k >= 0
                                      ? req.top_k : sv->top_k_def;
                            jb->min_p = req.min_p >= 0.0f
                                      ? req.min_p : sv->min_p_def;
                            jb->max_new_req = req.max_tokens > 0
                                            ? req.max_tokens
                                            : sv->max_tokens_def;
                            if (dbg) fprintf(stderr, "nf: [conn] job ready, "
                                            "pushing to queue (socket %d)\n", (int)c);
                            if (nf_queue_push(&sv->reqq, &jb) == 0) {
                                handed_off = 1;
                                if (dbg) fprintf(stderr, "nf: [conn] job "
                                                "pushed (socket %d)\n", (int)c);
                            } else {
                                /* queue closed: the server is shutting down */
                                send_simple(c, "503 Service Unavailable",
                                           "application/json",
                                           "{\"error\":\"server "
                                           "shutting down\"}");
                                free(jb->ids);
                                free(jb);
                            }
                        }
                    }
                    chat_req_free(&req);
                } else {
                    send_simple(c, "400 Bad Request", "application/json",
                                "{\"error\":\"invalid json or missing "
                                "messages\"}");
                }
            } else {
                send_simple(c, "404 Not Found", "application/json",
                            "{\"error\":\"unknown route\"}");
            }
        }
        free(reqbuf);
    }
    if (!handed_off) {
        shutdown(c, SD_SEND);
        closesocket(c);
        nf_atomic_dec_long(&g_srv_inflight);
    }
    return 0;
}

/* Closes a job (job_finish if it started, otherwise just the socket —
 * job_start already answered the error to the client on its own) and
 * frees everything. Shared by every exit point of the scheduler. */
static void job_dispose(nf_srv_job *jb, int started) {
    if (started) job_finish(jb);
    else { shutdown(jb->c, SD_SEND); closesocket(jb->c); }
    free(jb->ids);
    free(jb);
    nf_atomic_dec_long(&g_srv_inflight);
}

/* The single compute thread (M8 Phase 3b, continuous batching): every
 * round, collects up to NF_SRV_SLOTS ACTIVE jobs (already started,
 * not yet finished) and advances EACH ONE by one piece:
 *  - speculative (--draft): one WHOLE round, inline, NEVER grouped
 *    (produces a variable number of tokens per round — doesn't lend
 *    itself to single-token ticks);
 *  - smooth: samples the next token WITHOUT calling the model
 *    (job_presample), then EVERY session ready this round passes
 *    together in ONE call to the model's row kernel
 *    (dsk_eval_batched_rows or qwen_eval_batched_rows) — shares the
 *    expert weight reads when the router makes them converge on the
 *    same expert (Phase 3's gain).
 *    On NF_MODEL_PLAIN models (no cross-session kernel) it falls back
 *    to nf_session_eval one session at a time: correct, zero gain, no
 *    regression.
 * Never more than NF_SRV_SLOTS active jobs at once: server_pick_slot
 * never picks a "pinned" slot, so as long as active jobs
 * are fewer than the total slots there's ALWAYS a free slot for a new
 * one — avoids the "two jobs on the same slot" race. */
/* Row cap per tick: in the worst case every slot decodes (1 row) and
 * one of them is instead prefilling a whole chunk. Kept generous — the
 * cost is just a few stack arrays. */
#define NF_SRV_ROWS (NF_SRV_SLOTS + NF_PREFILL_CHUNK)

/* Defined next to physical_cores(), just before main(): threads that run
 * parallel regions re-apply the thread count chosen there. */
static void nf_omp_threads_apply(void);

static NF_THREAD_RET compute_thread_fn(void *arg) {
    nf_server *sv = (nf_server *)arg;
    nf_omp_threads_apply();   /* this thread has its OWN data environment */
    const nf_model_kind mk = nf_model_kind_of(sv->m);
    nf_srv_job *active[NF_SRV_SLOTS];
    int n_active = 0;
    int32_t batch_tok[NF_SRV_SLOTS];
    nf_srv_job *batch_jobs[NF_SRV_SLOTS];
    nf_srv_job *keep[NF_SRV_SLOTS];
    /* mixed ubatch rows (prefill + decode together) */
    nf_session *row_sess[NF_SRV_ROWS];
    int32_t     row_tok[NF_SRV_ROWS];
    int         row_pos[NF_SRV_ROWS];
    int         row_want[NF_SRV_ROWS];
    nf_srv_job *row_job[NF_SRV_ROWS];
    /* Bug found (extends to Qwen, never seen before
     * because masked on deepseek2): row_is_prefill
     * marks PER ROW whether it's a piece of the prompt, instead of
     * re-checking j->prefilling (a JOB flag) inside the emission loop
     * — a job whose last chunk has MORE than one row (almost always,
     * with NF_PREFILL_CHUNK=16) made j->prefilling=0 fire on the FIRST
     * row of that job, and the SAME job's later rows in the SAME tick
     * (still prompt, not generation) fell into the job_emit branch —
     * the prompt's tail was shipped to the client as if it were
     * output. A per-row marker, immutable during the loop, removes the
     * ambiguity at the root. */
    int         row_is_prefill[NF_SRV_ROWS];

    for (;;) {
        /* 1) admits AT MOST ONE new request per round (no longer
         * "while there's room"): measured (
         * NF_DEBUG_CASCADE) that draining the WHOLE queue before the
         * first tick starves every already-ready session until the
         * LAST new arrival has finished its own prefill — with 8
         * near-simultaneous arrivals, 21s out of 69s of test (30%)
         * with not a single token emitted for ANYONE. One job admitted
         * per round, then it ALWAYS falls through to the tick below:
         * whoever is already ready advances immediately, new arrivals
         * are admitted one at a time interleaved with the ticks — no
         * splitting a prefill into pieces (riskier, not measured as
         * necessary): it's enough not to block the ALREADY-ready ones
         * behind someone else's prefill. Blocking only if there's
         * NOTHING to do (n_active==0). */
        if (n_active < NF_SRV_SLOTS) {
            nf_srv_job *jb;
            const int got = n_active > 0 ? nf_queue_try_pop(&sv->reqq, &jb)
                                         : nf_queue_pop(&sv->reqq, &jb);
            if (got != 0) {
                if (n_active == 0) goto shutdown_scheduler;   /* queue closed and empty */
                /* nothing in the queue RIGHT NOW: no problem, fall through to the tick */
            } else {
                if (getenv("NF_DEBUG_CONN"))
                    fprintf(stderr, "nf: [sched] job pulled from the queue "
                            "(socket %d), n_active before=%d\n", (int)jb->c, n_active);
                const int js = job_start(jb);
                if (getenv("NF_DEBUG_CONN"))
                    fprintf(stderr, "nf: [sched] job_start=%d (socket %d)\n",
                            js, (int)jb->c);
                if (js == 0) active[n_active++] = jb;
                else job_dispose(jb, 0);
            }
        }

        /* 2) a tick: every active job advances by ONE piece. The
         * "finished" verdict is decided IMMEDIATELY from the return
         * value of the step just taken (never re-derived later from
         * state flags: some exits — full context, draft exhausted —
         * don't touch client_ok/stopped/emitted, re-deriving it there
         * would call the same job forever without ever disposing it).
         * Smooth jobs included in the batch stay "alive" for this
         * round even if the emit below stops them: the NEXT round
         * notices it (job_presample returns 0 immediately) — a cheap
         * waste, not a bug, avoids duplicating the stop logic here. */
        int n_batch = 0, n_keep = 0;
        /* MIXED ubatch: this round's rows are of TWO kinds —
         *  - PREFILL chunks (several rows per job, consecutive
         *    positions, logits requested only on the prompt's last one)
         *  - DECODE (one row per job, logits always requested)
         * travel together in the same kernel call. Rows of the same
         * session are by construction in increasing pos order (the
         * chunk is built forward), as required. */
        int n_rows = 0;
        for (int i = 0; i < n_active; i++) {
            nf_srv_job *j = active[i];
            if (j->speculative) {
                const int alive = j->spec_guard_ok && job_step_spec(j);
                if (alive) keep[n_keep++] = j; else job_dispose(j, 1);
                continue;
            }
            if (j->prefilling) {
                /* the client might already be dead: don't spend compute */
                if (!j->client_ok) { job_dispose(j, 1); continue; }
                const int left = j->n - j->n_prefilled;
                const int room = NF_SRV_ROWS - n_rows;
                if (room <= 0) { keep[n_keep++] = j; continue; }   /* next round */
                int take = left < NF_PREFILL_CHUNK ? left : NF_PREFILL_CHUNK;
                if (take > room) take = room;
                for (int q = 0; q < take; q++) {
                    const int p = j->n_prefilled + q;
                    row_sess[n_rows] = j->sl->s;
                    row_tok[n_rows]  = j->ids[p];
                    row_pos[n_rows]  = p;
                    /* logits only on the prompt's LAST token: the
                     * intermediate rows are never sampled, and the head
                     * (vocab x embd) is compute-bound, not free */
                    row_want[n_rows] = (p == j->n - 1);
                    row_job[n_rows]  = j;
                    row_is_prefill[n_rows] = 1;
                    n_rows++;
                }
                j->n_prefilled += take;
                keep[n_keep++] = j;
                continue;
            }
            int32_t tok;
            if (job_presample(j, &tok)) {
                if (n_rows >= NF_SRV_ROWS) { keep[n_keep++] = j; continue; }
                row_sess[n_rows] = j->sl->s;
                row_tok[n_rows]  = tok;
                row_pos[n_rows]  = nf_session_n_past(j->sl->s);
                row_want[n_rows] = 1;
                row_job[n_rows]  = j;
                row_is_prefill[n_rows] = 0;
                n_rows++;
                batch_jobs[n_batch] = j;
                batch_tok[n_batch] = tok;
                n_batch++;
                keep[n_keep++] = j;
            } else {
                job_dispose(j, 1);
            }
        }
        /* Phase 3b diagnostics (live regression, startup-ramp
         * hypothesis): histogram of the REAL group size per tick, only
         * under NF_DEBUG_BATCH — zero cost otherwise. */
        if (getenv("NF_DEBUG_BATCH")) {
            static uint64_t hist[NF_SRV_SLOTS + 1] = {0};
            static uint64_t ticks_ = 0;
            hist[n_batch]++;
            if (++ticks_ % 20 == 0) {
                fprintf(stderr, "nf: [batch-hist] %llu tick: ",
                        (unsigned long long)ticks_);
                for (int b = 0; b <= NF_SRV_SLOTS; b++)
                    fprintf(stderr, "%d=%llu ", b, (unsigned long long)hist[b]);
                fprintf(stderr, "\n");
            }
        }
        if (n_rows > 0) {
            /* NF_NO_XSESS: turns off cross-session batching (every
             * session goes on its own, like in Phase 2) — used for the
             * A/B that answers "does batching help or hurt?" without
             * switching binaries. */
            static int no_xsess = -1;
            if (no_xsess < 0) no_xsess = getenv("NF_NO_XSESS") != NULL;
            /* A single decode row and no prefill: the row kernel (which
             * inherits moe_ffn_*, built to amortize over many rows)
             * would have nothing to share and would only pay its own
             * bookkeeping — measured, nearly double on down. Falls back
             * to nf_session_eval (dsk_eval_core/session_eval_core,
             * already great for bn=1), like on models with no row
             * kernel. */
            const int use_rows = mk != NF_MODEL_PLAIN && !no_xsess
                              && (n_rows >= 2 || n_batch != n_rows);
            int rows_ok = 1;
            if (use_rows) {
                rows_ok = (mk == NF_MODEL_DSK
                    ? dsk_eval_batched_rows(row_sess, row_tok, row_pos,
                                            row_want, n_rows, NULL)
                    : qwen_eval_batched_rows(row_sess, row_tok, row_pos,
                                             row_want, n_rows, NULL)) == 0;
                if (!rows_ok)
                    for (int i = 0; i < n_rows; i++) row_job[i]->client_ok = 0;
            } else {
                for (int i = 0; i < n_rows; i++) {
                    /* non-batched path: one row at a time. Prefill
                     * chunks DON'T exist here (can_chunk is false on
                     * models with no row kernel), so every row is a
                     * decode: nf_session_eval advances n_past on its own. */
                    const float *lg = nf_session_eval(row_sess[i],
                                                      &row_tok[i], 1);
                    if (!lg) row_job[i]->client_ok = 0;
                    else row_job[i]->logits = lg;
                }
            }
            /* emission: ONLY decode rows produce a token. A prefill row
             * NEVER emits (row_is_prefill[i], a PER-ROW marker — never
             * j->prefilling, a JOB flag that once cleared on the job's
             * first row would "contaminate" that same job's later rows
             * in the same tick, see the comment above row_is_prefill).
             * A prefilling job that just finished its LAST chunk starts
             * decoding from the next round (its logits are already in
             * the session, row_want[i] was 1 on the prompt's last row:
             * the "completed" signal is read from the row, not from
             * j->n_prefilled/j->n, which at this point in the loop are
             * already updated for EVERY row of the job). */
            for (int i = 0; i < n_rows; i++) {
                nf_srv_job *j = row_job[i];
                if (!j->client_ok) continue;
                if (row_is_prefill[i]) {
                    if (row_want[i]) {   /* prompt's last row: prefill COMPLETED */
                        j->prefilling = 0;
                        j->logits = nf_session_logits(j->sl->s);
                        const double tn = now_seconds();
                        j->t_prefill = tn - j->t1;   /* t1 was the start */
                        j->t1 = tn;                  /* decode starts now */
                        if (getenv("NF_DEBUG_CASCADE"))
                            fprintf(stderr, "nf: [cascade] t=%.3fs "
                                    "prefill-END socket=%d (chunked, %.3fs)\n",
                                    dbg_cascade_t(), (int)j->c, j->t_prefill);
                    }
                    continue;   /* no token to emit either way */
                }
                if (use_rows) j->logits = nf_session_logits(j->sl->s);
                job_emit(j, row_tok[i]);
            }
        }
        memcpy(active, keep, (size_t)n_keep * sizeof(*keep));
        n_active = n_keep;
    }
shutdown_scheduler:
    return 0;
}

/* Clean shutdown for session persistence. The Ctrl+C handler
 * (another thread) raises the flag and closes the listening socket:
 * the in-progress accept() fails, the loop exits, and the slots get
 * saved to disk before exiting. If a generation is in progress, that
 * request finishes and THEN it exits (the handler returns TRUE: the
 * process isn't killed mid-write). */
static volatile int g_srv_stop = 0;
static SOCKET g_srv_sock = INVALID_SOCKET;
#ifdef _WIN32
static BOOL WINAPI srv_ctrl_handler(DWORD type) {
    (void)type;
    g_srv_stop = 1;
    if (g_srv_sock != INVALID_SOCKET) closesocket(g_srv_sock);
    return TRUE;
}
#else
/* SIGINT/SIGTERM: shutdown() wakes up the blocked accept (a plain
 * close on Linux does NOT unblock it), then close. Both are
 * async-signal-safe. */
static void srv_ctrl_handler_posix(int sig) {
    (void)sig;
    g_srv_stop = 1;
    if (g_srv_sock != INVALID_SOCKET) {
        shutdown(g_srv_sock, SHUT_RDWR);
        closesocket(g_srv_sock);
    }
}
#endif

static int cmd_serve(const char *model_path, const char *host, int port,
                     int max_ctx, float temp, float top_p, int32_t top_k,
                     float min_p, uint64_t seed,
                     float repeat_penalty, int repeat_last_n,
                     int max_tokens, int no_think,
                     const char *warmup_system,
                     const char *draft_path, int draft_k,
                     const char *sessions_dir) {
    chat_style_init(model_path);
    /* Serve inherits the auto-config (15 measured defaults)
     * and the tokenizer via tok_source_for — on .forgezh it used to be
     * BROKEN (tokenizer looked for inside the container, which doesn't
     * carry it) and ran with the slow config regardless. Draft guard
     * same as generate. */
    glm_daily_env(model_path);
    if (draft_path && getenv("NF_GLM_PREFILL_BATCH")) {
        nf_setenv("NF_GLM_PREFILL_BATCH=0");
        fprintf(stderr, "nf: batched prefill DISABLED under draft "
                "(interaction not validated)\n");
    }
    nf_server sv = {0};
    const char *sv_toksrc = tok_source_for(model_path);
    if (!sv_toksrc) return 1;
    sv.tk = nf_tokenizer_load(sv_toksrc);
    if (!sv.tk) return 1;
    sv.m = nf_model_load(model_path);
    if (!sv.m) { nf_tokenizer_free(sv.tk); return 1; }
    /* Draft model for speculative generation (greedy). Same
     * guards as generate/chat: matching vocabularies or nothing. */
    if (draft_path) {
        sv.dm = nf_model_load(draft_path);
        if (!sv.dm) { nf_model_free(sv.m); nf_tokenizer_free(sv.tk); return 1; }
        if (nf_model_vocab(sv.dm) != nf_model_vocab(sv.m)) {
            fprintf(stderr, "nf: draft and target have different vocabularies\n");
            nf_model_free(sv.dm); nf_model_free(sv.m);
            nf_tokenizer_free(sv.tk);
            return 1;
        }
        sv.draft_k = draft_k > 0 ? draft_k : 3;
        fprintf(stderr, "nf: [serve] speculative active (draft %s, K=%d, "
                "greedy only)\n", draft_path, sv.draft_k);
    }
    /* session slots are created lazily on the first request
     * (server_pick_slot) — only the RAM budget is declared here */
    fprintf(stderr, "nf: [serve] up to %d cached conversations "
            "(%.0fMB of KV each at ctx %d, allocated on demand)\n",
            NF_SRV_SLOTS, nf_session_kv_bytes(sv.m, max_ctx) / 1e6,
            max_ctx);
    sv.max_ctx = max_ctx;
    sv.temp_def = temp;
    sv.top_p_def = top_p;
    sv.top_k_def = top_k;
    sv.min_p_def = min_p;
    sv.rng = seed;
    sv.repeat_penalty = repeat_penalty;
    sv.repeat_last_n = repeat_last_n;
    sv.max_tokens_def = max_tokens;
    sv.no_think = no_think;
    build_stopset(sv_toksrc, &sv.stop);   /* .forgezh: from the extracted tokenizer */
    stopset_add_special(sv.tk, &sv.stop, "<|endoftext|>");
    if (g_chat_style == 3) {   /* glm-dsa: 3 stop ids, see chat_style_init */
        stopset_add_special(sv.tk, &sv.stop, "<|user|>");
        stopset_add_special(sv.tk, &sv.stop, "<|observation|>");
        stopset_add_special(sv.tk, &sv.stop, "<|assistant|>");   /* parity with chat */
    }
    build_textstops(sv.tk, &sv.stop, &sv.tstop);
    const char *base = strrchr(model_path, '/');
    const char *base2 = strrchr(model_path, '\\');
    if (base2 > base) base = base2;
    snprintf(sv.model_name, sizeof(sv.model_name), "%s",
             base ? base + 1 : model_path);

    /* System prompt prefill at STARTUP (--warmup-system file). The
     * system turn (same ChatML template as requests) is evaluated in
     * slot 0 BEFORE opening the port: the first chat finds it warm (its
     * prompt contains this history as a prefix: direct reuse),
     * subsequent ones inherit it via the graft. The prefill is
     * paid once, at boot — while the user is still opening the browser
     * — instead of on the first request. The file must contain EXACTLY
     * the system prompt the webui sends (byte for byte: one different
     * character and the prefix diverges there). */
    /* Restoring saved sessions (--sessions-dir) BEFORE warmup —
     * conversations that survived a restart go back into their slots
     * with history and cache; warmup will go to the first free slot.
     * A cache mode mismatch between save and boot: the load rejects it
     * with a message and the slot starts over from scratch (nf_session_
     * load's behavior). The draft, if active, starts empty and lazily
     * prefills on the first speculative round. */
    if (sessions_dir) {
        for (int i = 0; i < NF_SRV_SLOTS; i++) {
            char sp[512];
            snprintf(sp, sizeof(sp), "%s/slot%d.nfs", sessions_dir, i);
            int32_t *toks = malloc((size_t)max_ctx * sizeof(int32_t));
            int ntk = 0;
            nf_session *ls = toks ? nf_session_load(sv.m, sp, max_ctx,
                                                    toks, max_ctx, &ntk)
                                  : NULL;
            if (ls && sv.dm) {
                sv.slot[i].ds = nf_session_create(sv.dm, max_ctx);
                if (!sv.slot[i].ds) { nf_session_free(ls); ls = NULL; }
            }
            if (ls) {
                sv.slot[i].s = ls;
                sv.slot[i].hist = toks;
                sv.slot[i].n_hist = ntk;
                sv.slot[i].d_past = 0;
                sv.slot[i].last_used = ++sv.tick;
                fprintf(stderr, "nf: [serve] slot %d restored "
                        "(%d tokens)\n", i, ntk);
            } else
                free(toks);
        }
    }

    if (warmup_system) {
        char *wtxt = read_whole_file(warmup_system);
        if (wtxt) {
            char *wbuf = NULL;
            size_t wcap = 0, wlen = 0;
            int32_t *wids = NULL;
            /* glm-dsa: always the first content of a new slot (sl->s
             * gets created right after, see below) — unconditional
             * is_first=1, same reasoning as the system warmup. */
            append_glm_prefix_if_first(&wbuf, &wcap, &wlen, 1);
            if (!append_turn(&wbuf, &wcap, &wlen, "system", wtxt) &&
                (wids = malloc((size_t)max_ctx * sizeof(int32_t))) != NULL) {
                const int64_t wn = nf_tokenize(sv.tk, wbuf, wids, max_ctx);
                if (wn > 0 && wn < max_ctx) {
                    /* first free slot (restore might have taken
                     * slot 0); all full = warmup is moot */
                    nf_srv_slot *sl = NULL;
                    for (int i = 0; i < NF_SRV_SLOTS && !sl; i++)
                        if (!sv.slot[i].s) sl = &sv.slot[i];
                    if (!sl) goto warmup_done;
                    sl->s = nf_session_create(sv.m, max_ctx);
                    sl->hist = malloc((size_t)max_ctx * sizeof(int32_t));
                    sl->ds = sv.dm ? nf_session_create(sv.dm, max_ctx)
                                   : NULL;
                    if (sl->s && sl->hist && (!sv.dm || sl->ds)) {
                        const double w0 = now_seconds();
                        if (nf_session_eval(sl->s, wids, (int)wn)) {
                            memcpy(sl->hist, wids,
                                   (size_t)wn * sizeof(int32_t));
                            sl->n_hist = (int)wn;
                            sl->last_used = ++sv.tick;
                            /* The draft gets warmed up too (0.6B,
                             * cheap) so the first speculative round
                             * prefills nothing */
                            if (sl->ds &&
                                nf_session_eval(sl->ds, wids, (int)wn))
                                sl->d_past = (int)wn;
                            fprintf(stderr, "nf: [serve] slot 0 "
                                    "warmed up: %d tokens of system "
                                    "prompt in %.1fs%s\n",
                                    (int)wn, now_seconds() - w0,
                                    sl->d_past ? " (draft too)" : "");
                        } else {
                            nf_session_free(sl->s); free(sl->hist);
                            nf_session_free(sl->ds);
                            sl->s = NULL; sl->hist = NULL; sl->ds = NULL;
                        }
                    }
                }
            }
warmup_done:
            free(wids); free(wbuf); free(wtxt);
        } else
            fprintf(stderr, "nf: [serve] warmup: cannot read "
                    "%s\n", warmup_system);
    }

#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa)) return 1;
#endif
    SOCKET srv = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        fprintf(stderr, "nf: invalid --host address: %s\n", host);
        return 1;
    }
    const int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));
    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) ||
        listen(srv, 8)) {
        fprintf(stderr, "nf: bind/listen on %s:%d failed (port busy?)\n",
                host, port);
        return 1;
    }
    fprintf(stderr,
            "nf: listening on http://%s:%d — built-in page at /, OpenAI "
            "API at /v1/chat/completions (Ctrl+C to exit)\n"
            "nf: WARNING: no authentication — trusted network only\n",
            host, port);

    /* M8 Phase 2: one thread per connection for I/O (accept stays
     * serial, it's the only truly cheap thing to do in a loop); a
     * single PERSISTENT compute thread pulls jobs from the queue — at
     * most one thread in the engine, like in Phase 1, but now it's a
     * real dedicated scheduler instead of "whichever connection holds
     * the lock". No external dependency (native _beginthreadex). */
    nf_queue_init(&sv.reqq, sizeof(nf_srv_job *), 64);
    const nf_thread_t compute_th = nf_thread_start(compute_thread_fn, &sv);

    /* Ctrl+C = clean shutdown with session saving */
    g_srv_sock = srv;
#ifdef _WIN32
    SetConsoleCtrlHandler(srv_ctrl_handler, TRUE);
#else
    signal(SIGINT, srv_ctrl_handler_posix);
    signal(SIGTERM, srv_ctrl_handler_posix);
    signal(SIGPIPE, SIG_IGN);   /* client gone: send fails, doesn't kill us */
#endif

    while (!g_srv_stop) {
        SOCKET c = accept(srv, NULL, NULL);
        if (c == INVALID_SOCKET) {
            if (g_srv_stop) break;
            continue;
        }
        nf_conn_arg *carg = malloc(sizeof(nf_conn_arg));
        if (!carg) { shutdown(c, SD_SEND); closesocket(c); continue; }
        carg->sv = &sv;
        carg->c = c;
        nf_atomic_inc_long(&g_srv_inflight);
        const nf_thread_t th = nf_thread_start(conn_thread_fn, carg);
        if (!th) {
            /* spawn failed (resources exhausted): handle it right here,
             * never lose the request nor the in-flight count */
            conn_thread_fn(carg);
        } else {
            nf_thread_detach(th); /* fire-and-forget: the thread cleans
                                 * itself up, the handle isn't needed after */
        }
    }

    /* M8 Phase 2: close the queue (unblocks the compute thread if
     * waiting; jobs ALREADY in the queue still get drained) and wait
     * for it to finish — only then will no thread touch sv.slot[]
     * anymore. Then also wait for every connection still reading/
     * parsing (hasn't pushed a job yet) to finish, before saving the
     * sessions. */
    nf_queue_close(&sv.reqq);
    if (compute_th) {
        nf_thread_join(compute_th);
    }
    while (nf_atomic_load_long(&g_srv_inflight) != 0)
        nf_sleep_ms(20);
    nf_queue_free(&sv.reqq);

    /* Clean shutdown — every live slot gets saved to disk and
     * will come back on the next boot (same format as chat --session,
     * versioned by cache mode) */
    if (sessions_dir) {
        int saved = 0;
        for (int i = 0; i < NF_SRV_SLOTS; i++) {
            nf_srv_slot *sl = &sv.slot[i];
            if (!sl->s || sl->n_hist <= 0) continue;
            char sp[512];
            snprintf(sp, sizeof(sp), "%s/slot%d.nfs", sessions_dir, i);
            if (nf_session_save(sl->s, sl->hist, sp) == 0) saved++;
        }
        fprintf(stderr, "nf: [serve] shutdown: %d sessions saved to "
                "%s\n", saved, sessions_dir);
    } else
        fprintf(stderr, "nf: [serve] shutdown\n");
    return 0;
}

/* Looks for "--flag value" in argv; NULL if absent. */
static const char *arg_value(int argc, char **argv, const char *flag) {
    for (int i = 0; i < argc - 1; i++)
        if (strcmp(argv[i], flag) == 0) return argv[i + 1];
    return NULL;
}

/* --long-memory (Phase 1, RAM-only, qwen3/qwen3moe, decode only):
 * enables NF_LONGMEM for the session via the same env
 * vars nf_session_create already reads, same scheme
 * expcache/convertiq4xs use to set NF_NO_EXPW. --long-memory-block/
 * -sink/-window/-topk are optional (defaults already measured in
 * nf_session_create: 128/64/512/16); without --long-memory the rest is
 * ignored, zero behavior change. */
static void apply_longmem_cli_args(int argc, char **argv) {
    int on = 0;
    for (int i = 3; i < argc; i++)
        if (strcmp(argv[i], "--long-memory") == 0) on = 1;
    if (!on) return;
    nf_setenv("NF_LONGMEM=1");
    const char *b = arg_value(argc, argv, "--long-memory-block");
    const char *s = arg_value(argc, argv, "--long-memory-sink");
    const char *w = arg_value(argc, argv, "--long-memory-window");
    const char *k = arg_value(argc, argv, "--long-memory-topk");
    char eb[48], es[48], ew[48], ek[48];
    if (b) { snprintf(eb, sizeof eb, "NF_LONGMEM_BLOCK=%s", b);  nf_setenv(eb); }
    if (s) { snprintf(es, sizeof es, "NF_LONGMEM_SINK=%s", s);   nf_setenv(es); }
    if (w) { snprintf(ew, sizeof ew, "NF_LONGMEM_WINDOW=%s", w); nf_setenv(ew); }
    if (k) { snprintf(ek, sizeof ek, "NF_LONGMEM_TOPK=%s", k);   nf_setenv(ek); }
    /* Phase 2a (disk-backing): opt-in ON TOP OF --long-memory, same
     * env-var scheme. --long-memory-disk-path/-pool are optional
     * (defaults: a per-process temp file / topk+8, see
     * nf_session_create). */
    int disk = 0;
    for (int i = 3; i < argc; i++)
        if (strcmp(argv[i], "--long-memory-disk") == 0) disk = 1;
    if (disk) {
        nf_setenv("NF_LONGMEM_DISK=1");
        const char *dp = arg_value(argc, argv, "--long-memory-disk-path");
        const char *dc = arg_value(argc, argv, "--long-memory-disk-pool");
        const char *dm = arg_value(argc, argv, "--long-memory-disk-margin");
        char edp[1080], edc[48], edm[48];
        if (dp) { snprintf(edp, sizeof edp, "NF_LONGMEM_DISK_PATH=%s", dp); nf_setenv(edp); }
        if (dc) { snprintf(edc, sizeof edc, "NF_LONGMEM_DISK_POOL=%s", dc); nf_setenv(edc); }
        if (dm) { snprintf(edm, sizeof edm, "NF_LONGMEM_DISK_MARGIN=%s", dm); nf_setenv(edm); }
    }
}

/* Counts PHYSICAL cores (not logical threads). On bandwidth-bound
 * kernels hyperthreading adds contention, not bandwidth — measured:
 * 4 threads beat the default 8 by +6% on
 * dense decode and never lose on the 30B MoE. 0 = undeterminable (the
 * OpenMP default is left alone).
 *
 * POSIX NOTE: the Linux branch was added later — before it, this
 * function returned 0 on Linux, so omp_set_num_threads() was never
 * called and libgomp defaulted to EVERY logical CPU. On an 8C/16T
 * machine that cost 32-59% of decode throughput (0.6B 80.0 -> 54.4,
 * 4B 19.5 -> 8.0, 8B 10.7 -> 5.0, 30B MoE 24.5 -> 11.6 tok/s). */
#ifdef _WIN32
static int physical_cores(void) {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0) return 0;
    unsigned char *buf = malloc(len);
    if (!buf) return 0;
    int n = 0;
    if (GetLogicalProcessorInformationEx(RelationProcessorCore,
            (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf, &len)) {
        DWORD off = 0;
        while (off < len) {
            const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *info =
                (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(buf + off);
            if (info->Relationship == RelationProcessorCore) n++;
            off += info->Size;
        }
    }
    free(buf);
    return n;
}
#else
/* Linux: /proc/cpuinfo carries (physical id, core id) per logical CPU;
 * counting the distinct pairs yields PHYSICAL cores, so an SMT machine
 * does not get one OpenMP thread per logical CPU (measured: 24.5 vs
 * 11.6 tok/s decode on a 8C/16T Ryzen with the 30B MoE). If the
 * topology is unreadable (no /proc, ARM cpuinfo without core id) the
 * number of ONLINE cpus is a safe last resort: single-CPU-per-core and
 * one-logical-per-physical both make it correct, and it only
 * overshoots on SMT machines whose /proc is unreadable. */
static int physical_cores(void) {
    int pids[512], cids[512];
    int i, n = 0, phys = -1, core = -1;
    char line[256];
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        while (fgets(line, sizeof line, f)) {
            if (strncmp(line, "physical id", 11) == 0) {
                const char *c = strchr(line, ':');
                if (c) phys = atoi(c + 1);
            } else if (strncmp(line, "core id", 7) == 0) {
                const char *c = strchr(line, ':');
                if (c) core = atoi(c + 1);
            } else if (line[0] == '\n') {   /* end of a processor block */
                if (phys >= 0 && core >= 0) {
                    int dup = 0;
                    for (i = 0; i < n; i++)
                        if (pids[i] == phys && cids[i] == core) { dup = 1; break; }
                    if (!dup && n < 512) { pids[n] = phys; cids[n] = core; n++; }
                }
                phys = core = -1;
            }
        }
        fclose(f);
    }
    if (n <= 0) {
        const long c = sysconf(_SC_NPROCESSORS_ONLN);
        n = c > 0 ? (int)c : 0;
    }
    return n;
}
#endif

#ifdef _OPENMP
/* Thread count chosen by main (0 = leave the runtime alone).
 *
 * Why this exists: omp_set_num_threads() only sets the CALLING thread's
 * data environment. A thread created later (the persistent compute
 * thread of `nf serve`, conn_thread_fn/compute_thread_fn) starts from
 * libgomp's GLOBAL ICV — the environment value, or ALL logical CPUs
 * when OMP_NUM_THREADS is unset — so the physical-core pinning above
 * silently evaporates the moment the work moves off the main thread.
 * Measured on an 8C/16T Zen 5, Qwen3-4B, `nf serve` N=1: 7.7 tok/s
 * with the auto-config vs 18.8 with OMP_NUM_THREADS=8 (and 18.8 again
 * once this is applied). Setting the environment from main does NOT
 * work: libgomp reads it at library init, before main. */
static int g_omp_threads = 0;
static void nf_omp_threads_apply(void) {
    if (g_omp_threads > 0) omp_set_num_threads(g_omp_threads);
}
#else
static void nf_omp_threads_apply(void) {}
#endif

int main(int argc, char **argv) {
#ifdef _WIN32
    /* Process priority (NF_PRIORITY=high|above, opt-in).
     *
     * Why it exists: the kernel bench (nf debugkernelbench) measured
     * swings of up to 2x between identical runs, and the cause was NOT
     * the code but the machine — this laptop stably runs Docker,
     * several VS Code instances, and a remote desktop, with a 12-32%
     * background load even "at rest". At normal priority, inference
     * gets preempted continuously: raising it takes back what the
     * background steals.
     *
     * Measured interleaved A/B/A on the real model (16 tokens, gold
     * config, IDENTICAL output across the three runs): normal 332s and
     * 324s, HIGH 287s — wall -12.5%, and the hot path's CPU time from
     * ~153s to 110s (-28%). Zero effect on the math: it's purely
     * scheduling.
     *
     * Stays OPT-IN and isn't the default for an honest reason: the gain
     * is proportional to how disturbed the machine is (on a quiesced
     * machine it would be much smaller), and HIGH_PRIORITY_CLASS across
     * all cores can make the system less responsive while it runs.
     * "above" (ABOVE_NORMAL) is the compromise for someone who wants to
     * use the PC in the meantime. */
    {
        const char *pr = getenv("NF_PRIORITY");
        if (pr && *pr) {
            DWORD cls = 0;
            if (strcmp(pr, "high") == 0)  cls = HIGH_PRIORITY_CLASS;
            else if (strcmp(pr, "above") == 0) cls = ABOVE_NORMAL_PRIORITY_CLASS;
            else fprintf(stderr, "nf: NF_PRIORITY='%s' not recognized "
                         "(expected 'high' or 'above') — ignored\n", pr);
            if (cls && !SetPriorityClass(GetCurrentProcess(), cls))
                fprintf(stderr, "nf: SetPriorityClass failed (err %lu)\n",
                        (unsigned long)GetLastError());
            else if (cls)
                fprintf(stderr, "nf: process priority = %s (measured "
                        "-12.5%% wall time on this disturbed machine; "
                        "the gain is smaller on a quiesced one)\n", pr);
        }
    }
#endif
#ifdef _OPENMP
    /* default: one thread per physical core. OMP_NUM_THREADS, if set by
     * the user, wins (omp_set_num_threads doesn't get called). */
    if (!getenv("OMP_NUM_THREADS")) {
        const int pc = physical_cores();
        if (pc > 0) {
            omp_set_num_threads(pc);
            g_omp_threads = pc;
#ifdef __linux__
            /* On Linux the batched prefill path would otherwise still
             * ask omp_get_num_procs() for ALL logical CPUs (see
             * batch_num_threads in nf_model.c): physical-core regions
             * interleaved with a logical-CPU-width batch region cost
             * 3.6x on prefill (252 vs 911 tok/s, 0.6B). Pin the batch
             * path to the same count. NF_BATCH_THREADS wins if the
             * user already chose one. */
            if (!getenv("NF_BATCH_THREADS")) {
                char b[32];
                snprintf(b, sizeof b, "NF_BATCH_THREADS=%d", pc);
                nf_setenv(b);
            }
#endif
        }
    }
#endif
    /* K/V cache: the default is per-head int8 (kv-q8) — see
     * kv_q8_on for the full dossier. --kv-f16 (or NF_KV_F16=1) restores
     * f16; --kv-q8 stays accepted for compatibility with scripts and
     * historical notes. Applies to any command that creates sessions. */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--kv-q8") == 0) nf_setenv("NF_KV_Q8=1");
        if (strcmp(argv[i], "--kv-f16") == 0) nf_setenv("NF_KV_F16=1");
    }
    if (argc >= 3 && strcmp(argv[1], "inspect") == 0) {
        const int all = argc > 3 && strcmp(argv[3], "--tensors") == 0;
        return cmd_inspect(argv[2], all);
    }
    if (argc >= 4 && strcmp(argv[1], "tokenize") == 0) {
        if (argc >= 5 && strcmp(argv[3], "--file") == 0) {
            char *text = read_whole_file(argv[4]);
            if (!text) return 1;
            const int rc = cmd_tokenize(argv[2], text);
            free(text);
            return rc;
        }
        return cmd_tokenize(argv[2], argv[3]);
    }

    if (argc >= 5 && strcmp(argv[1], "logits") == 0 &&
        strcmp(argv[3], "--file") == 0) {
        char *text = read_whole_file(argv[4]);
        if (!text) return 1;
        const char *dump = NULL;
        if (argc >= 7 && strcmp(argv[5], "--dump") == 0) dump = argv[6];
        const int rc = cmd_logits(argv[2], text, dump);
        free(text);
        return rc;
    }

    if (argc >= 5 && strcmp(argv[1], "embed") == 0 &&
        strcmp(argv[3], "--file") == 0) {
        char *text = read_whole_file(argv[4]);
        if (!text) return 1;
        const char *layer_s = arg_value(argc, argv, "--layer");
        if (!layer_s) {
            fprintf(stderr, "nf: embed requires --layer N (no default)\n");
            free(text);
            return 1;
        }
        const char *dump = arg_value(argc, argv, "--dump");
        const int rc = cmd_embed(argv[2], text, atoi(layer_s), dump);
        free(text);
        return rc;
    }

    if (argc >= 5 && strcmp(argv[1], "generate") == 0 &&
        strcmp(argv[3], "--file") == 0) {
        apply_longmem_cli_args(argc, argv);
        char *text = read_whole_file(argv[4]);
        if (!text) return 1;
        const char *np_s   = arg_value(argc, argv, "--n-predict");
        const char *temp_s = arg_value(argc, argv, "--temp");
        const char *topp_s = arg_value(argc, argv, "--top-p");
        const char *topk_s = arg_value(argc, argv, "--top-k");
        const char *minp_s = arg_value(argc, argv, "--min-p");
        const char *seed_s = arg_value(argc, argv, "--seed");
        const char *ctx_s  = arg_value(argc, argv, "--ctx");
        const char *rp_s   = arg_value(argc, argv, "--repeat-penalty");
        const char *rln_s  = arg_value(argc, argv, "--repeat-last-n");
        const char *draft_s = arg_value(argc, argv, "--draft");
        const char *dk_s    = arg_value(argc, argv, "--draft-k");
        int mtp_draft = 0;   /* boolean flag, same scheme as --kv-q8 */
        int layerskip_draft = 0;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--mtp-draft") == 0) mtp_draft = 1;
            if (strcmp(argv[i], "--layerskip-draft") == 0) layerskip_draft = 1;
        }
        const int rc = cmd_generate(argv[2], text,
                                    np_s   ? atoi(np_s)        : 64,
                                    temp_s ? (float)atof(temp_s) : 0.0f,
                                    topp_s ? (float)atof(topp_s) : 1.0f,
                                    topk_s ? atoi(topk_s)      : 0,
                                    minp_s ? (float)atof(minp_s) : 0.0f,
                                    ctx_s  ? atoi(ctx_s)        : 2048,
                                    seed_s ? strtoull(seed_s, NULL, 10) : 42ULL,
                                    /* default 1.0 (off): check_m3.py compares
                                     * greedy against the HF oracle, which
                                     * applies no penalty — a default other
                                     * than 1.0 would silently break that
                                     * comparison */
                                    rp_s   ? (float)atof(rp_s)  : 1.0f,
                                    rln_s  ? atoi(rln_s)        : 64,
                                    draft_s,
                                    dk_s   ? atoi(dk_s)         : 6,
                                    mtp_draft, layerskip_draft);
        free(text);
        return rc;
    }

    if (argc >= 3 && strcmp(argv[1], "chat") == 0) {
        apply_longmem_cli_args(argc, argv);
        const char *sys_file = arg_value(argc, argv, "--system-file");
        const char *sys_txt  = arg_value(argc, argv, "--system");
        char *system_text_owned = sys_file ? read_whole_file(sys_file) : NULL;
        const char *system_text = sys_file ? system_text_owned : sys_txt;
        if (sys_file && !system_text_owned) return 1;

        const char *ctx_s    = arg_value(argc, argv, "--ctx");
        const char *temp_s   = arg_value(argc, argv, "--temp");
        const char *topp_s   = arg_value(argc, argv, "--top-p");
        const char *topk_s   = arg_value(argc, argv, "--top-k");
        const char *minp_s   = arg_value(argc, argv, "--min-p");
        const char *seed_s   = arg_value(argc, argv, "--seed");
        const char *sess_s   = arg_value(argc, argv, "--session");
        const char *maxtok_s = arg_value(argc, argv, "--max-tokens");
        const char *rp_s     = arg_value(argc, argv, "--repeat-penalty");
        const char *rln_s    = arg_value(argc, argv, "--repeat-last-n");
        const char *draft_s  = arg_value(argc, argv, "--draft");
        const char *dk_s     = arg_value(argc, argv, "--draft-k");
        int no_think = 0, ctx_shift = 0;
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--no-think") == 0) no_think = 1;
            if (strcmp(argv[i], "--ctx-shift") == 0) ctx_shift = 1;
        }
        const int rc = cmd_chat(argv[2], system_text,
                                ctx_s  ? atoi(ctx_s)         : 4096,
                                /* default 0 = greedy, not 0.7 ("who
                                 * decided it should be 0.7?") — on a
                                 * 0.6B model sampling can derail
                                 * otherwise-correct reasoning (verified
                                 * against the HF oracle in greedy).
                                 * Whoever wants variety turns it on
                                 * explicitly. */
                                temp_s ? (float)atof(temp_s) : 0.0f,
                                topp_s ? (float)atof(topp_s) : 0.9f,
                                topk_s ? atoi(topk_s)        : 0,
                                minp_s ? (float)atof(minp_s) : 0.0f,
                                seed_s ? strtoull(seed_s, NULL, 10) : 42ULL,
                                sess_s,
                                /* -1 = not passed: cmd_chat resolves the
                                 * model-aware default (96 on glm, 512
                                 * elsewhere) — see the comment there. */
                                maxtok_s ? atoi(maxtok_s) : -1,
                                rp_s   ? (float)atof(rp_s) : 1.1f,
                                rln_s  ? atoi(rln_s)       : 64,
                                no_think,
                                draft_s,
                                dk_s   ? atoi(dk_s)        : 6,
                                ctx_shift);
        free(system_text_owned);
        return rc;
    }

    if (argc >= 2 && strcmp(argv[1], "debugqueue") == 0) {
        nf_debug_queue_bench();
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "serve") == 0) {
        /* Long-memory under the concurrent server: explicit opt-in
         * ONLY with --long-memory on THIS command line (same env-var
         * scheme as apply_longmem_cli_args, shared with generate/chat)
         * — every slot (one nf_session_create per client) reads the
         * same env vars, so all slots share the same long-memory
         * config, consistent with the rest of the server-wide options
         * (--ctx/--kv-q8/etc). Without --long-memory explicit here, any
         * NF_LONGMEM/NF_LONGMEM_DISK inherited from the environment
         * (e.g. from an earlier nf chat session in the same shell) gets
         * explicitly turned off, not silently ignored — never left to
         * chance under N concurrent slots. The per-session .lmdisk path
         * (no longer just per-process, see the fix in
         * nf_session_create) makes activating disk-backing from
         * several slots in parallel safe. */
        int longmem_cli = 0;
        for (int i = 3; i < argc; i++)
            if (strcmp(argv[i], "--long-memory") == 0) longmem_cli = 1;
        if (!longmem_cli && (getenv("NF_LONGMEM") || getenv("NF_LONGMEM_DISK"))) {
            fprintf(stderr, "nf: NF_LONGMEM/NF_LONGMEM_DISK active "
                    "in the environment but ignored for 'nf serve' "
                    "without --long-memory explicit on this command line\n");
            nf_setenv("NF_LONGMEM=");
            nf_setenv("NF_LONGMEM_DISK=");
        }
        apply_longmem_cli_args(argc, argv);
        const char *draft_check_s = arg_value(argc, argv, "--draft");
        if (draft_check_s && getenv("NF_LONGMEM_DISK")) {
            fprintf(stderr, "nf: --draft and --long-memory-disk are "
                    "incompatible under 'nf serve' (same reason as "
                    "generate/chat: batched prefill vs decode-only "
                    "decomposition)\n");
            return 1;
        }
        const char *host_s = arg_value(argc, argv, "--host");
        const char *port_s = arg_value(argc, argv, "--port");
        const char *ctx_s  = arg_value(argc, argv, "--ctx");
        const char *temp_s = arg_value(argc, argv, "--temp");
        const char *topp_s = arg_value(argc, argv, "--top-p");
        const char *topk_s = arg_value(argc, argv, "--top-k");
        const char *minp_s = arg_value(argc, argv, "--min-p");
        const char *seed_s = arg_value(argc, argv, "--seed");
        const char *rp_s   = arg_value(argc, argv, "--repeat-penalty");
        const char *rln_s  = arg_value(argc, argv, "--repeat-last-n");
        const char *mt_s   = arg_value(argc, argv, "--max-tokens");
        const char *ws_s   = arg_value(argc, argv, "--warmup-system");
        const char *dr_s   = arg_value(argc, argv, "--draft");
        const char *dk_s   = arg_value(argc, argv, "--draft-k");
        const char *sd_s   = arg_value(argc, argv, "--sessions-dir");
        int no_think = 0;
        for (int i = 3; i < argc; i++)
            if (strcmp(argv[i], "--no-think") == 0) no_think = 1;
        /* Default dropped to 2048 (along with
         * NF_SRV_SLOTS=16): at ctx=4096, 16 slots would use 11.1GB of
         * KV, beyond Phase 0's budget. At 2048 it stays 5.55GB —
         * IDENTICAL to the original budget (694MB x 8 slots at
         * ctx=4096), same margin. An explicit --ctx always wins, as
         * always. */
        return cmd_serve(argv[2],
                         host_s ? host_s : "127.0.0.1",
                         port_s ? atoi(port_s) : 8080,
                         ctx_s  ? atoi(ctx_s)  : 2048,
                         temp_s ? (float)atof(temp_s) : 0.0f,
                         topp_s ? (float)atof(topp_s) : 0.9f,
                         topk_s ? atoi(topk_s)        : 0,
                         minp_s ? (float)atof(minp_s) : 0.0f,
                         seed_s ? strtoull(seed_s, NULL, 10) : 42ULL,
                         rp_s   ? (float)atof(rp_s) : 1.1f,
                         rln_s  ? atoi(rln_s)       : 64,
                         mt_s   ? atoi(mt_s)        : 1024,
                         no_think, ws_s, dr_s,
                         dk_s ? atoi(dk_s) : 3, sd_s);
    }

    if (argc >= 5 && strcmp(argv[1], "selfcheck") == 0 &&
        strcmp(argv[3], "--file") == 0) {
        char *text = read_whole_file(argv[4]);
        if (!text) return 1;
        const int rc = cmd_selfcheck(argv[2], text);
        free(text);
        return rc;
    }

    if (argc >= 5 && strcmp(argv[1], "perplexity") == 0 &&
        strcmp(argv[3], "--file") == 0) {
        char *text = read_whole_file(argv[4]);
        if (!text) return 1;
        const char *ctx_s = arg_value(argc, argv, "--ctx");
        const char *nw_s  = arg_value(argc, argv, "--windows");
        const int rc = cmd_perplexity(argv[2], text,
                                      ctx_s ? atoi(ctx_s) : 512,
                                      nw_s ? atoi(nw_s) : 8);
        free(text);
        return rc;
    }

    if (argc >= 4 && strcmp(argv[1], "bench") == 0 && strcmp(argv[3], "--file") == 0) {
        char *text = read_whole_file(argv[4]);
        if (!text) return 1;
        const char *cs = arg_value(argc, argv, "--ctx-start");
        const char *cm = arg_value(argc, argv, "--ctx-max");
        const char *sm = arg_value(argc, argv, "--step-mul");
        const char *gt = arg_value(argc, argv, "--gen-tokens");
        const int rc = cmd_bench(argv[2], text,
                                 cs ? atoi(cs) : 128,
                                 cm ? atoi(cm) : 4096,
                                 sm ? atof(sm) : 2.0,
                                 gt ? atoi(gt) : 16);
        free(text);
        return rc;
    }

    if (argc >= 5 && strcmp(argv[1], "debugwq") == 0)
        return cmd_debug_wq(argv[2], atoi(argv[3]), strtoull(argv[4], NULL, 10),
                            argc >= 6 && strcmp(argv[5], "--v") == 0);

    if (argc >= 3 && strcmp(argv[1], "debugstop") == 0)
        return cmd_debug_textstop(argv[2]);

    if (argc >= 2 && strcmp(argv[1], "debugompcost") == 0) {
        nf_debug_ompcost_bench();
        return 0;
    }

#ifdef _WIN32
    if (argc >= 3 && strcmp(argv[1], "debugexpwio") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_expwio_bench(m);
        nf_model_free(m);
        return 0;
    }
#endif

    if (argc >= 3 && strcmp(argv[1], "debugdot") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_vecdot(m);
        nf_model_free(m);
        return 0;
    }

    /* DEBUG (MoE validation): isolates layer 0's router+experts on one
     * token, dumps xb (input) and out (weighted output) — see nf_debug_moe. */
    if (argc >= 6 && strcmp(argv[1], "debugmoe") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        const uint64_t token = strtoull(argv[3], NULL, 10);
        const int rc = nf_debug_moe(m, token, argv[4], argv[5]);
        nf_model_free(m);
        return rc;
    }

    /* DEBUG (measurement): the horizontal-reduction (hsum) share of a
     * superblock's integer dot product — see nf_debug_simd_bench. */
    if (argc >= 3 && strcmp(argv[1], "debugsimd") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_simd_bench(m);
        nf_model_free(m);
        return 0;
    }

    /* DEBUG (wide kernel, gate 0): repack v2 + kernel correctness on an
     * 8-row group (debugpack), kernel-level A/B streaming over a whole
     * tensor (debugwide) — see nf_debug_pack / nf_debug_wide_bench. */
    if (argc >= 3 && strcmp(argv[1], "debugpack") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_pack(m);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugwide") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_wide_bench(m);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debuggemm") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_gemm_bench(m);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugdskdown") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_dsk_down_bench(m);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugq6k") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_q6k(m);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugq2k") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_q2k(m);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugiq4xs") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_iq4xs(m);
        nf_model_free(m);
        return 0;
    }
    /* nf debugdequant5 <gguf> <tensor_name> <row0> <nrows> <out_prefix>
     * (real GLM-5.2): generic dump to validate Q2_K/Q3_K/
     * IQ2_XXS/IQ3_XXS/IQ1_S against the Python oracle on REAL tensors —
     * does NOT load the whole model (see the comment on
     * nf_debug_dequant5 in nf_model.c for why). */
    if (argc >= 7 && strcmp(argv[1], "debugdequant5") == 0) {
        const uint64_t row0 = strtoull(argv[4], NULL, 10);
        const int nrows = atoi(argv[5]);
        return nf_debug_dequant5(argv[2], argv[3], row0, nrows, argv[6]);
    }
    /* nf debugiqdot <gguf> <tensor_name> <nrows>: fused
     * AVX2 vecdot IQ1_S/IQ2_XXS/IQ3_XXS against scalar dequant+dotf on
     * real rows — see nf_debug_iqdot in nf_model.c. */
    if (argc >= 5 && strcmp(argv[1], "debugiqdot") == 0)
        return nf_debug_iqdot(argv[2], argv[3], atoi(argv[4]));
    if (argc >= 3 && strcmp(argv[1], "debugscale1") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_scale1_bench(m);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugnogather") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_nogather_bench(m);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugheadbatch") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_headbatch_bench(m);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugtailwide") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_tailwide_bench(m);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugdownq6k") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_downq6k_bench(m);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugprefetch") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_prefetch_bench(m);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugcrosssess") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_crosssess_bench(m);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugcrosssessqwen") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_crosssess_qwen_bench(m);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugbatchscale") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_batchscale_bench(m);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugbatchscaleqwen") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_batchscale_qwen_bench(m);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugmoecostqwen") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_moecost_qwen_bench(m);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugmixed") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_mixedbatch_bench(m);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugmixedqwen") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_mixedbatch_qwen_bench(m);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 5 && strcmp(argv[1], "convertiq4xs") == 0) {
        nf_setenv("NF_NO_EXPW=1");
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        const int rc = nf_convert_iq4xs_model(m, argv[3], argv[4]);
        nf_model_free(m);
        return rc ? 1 : 0;
    }
    if (argc >= 4 && strcmp(argv[1], "debugglm") == 0) {
        const char *dump = arg_value(argc, argv, "--dump");
        return cmd_debugglm(argv[2], argv[3], dump);
    }
    /* lightweight teacher-forced replay + logit dump (
     * KL miss-skip protocol) — see cmd_debugglmreplay. Usage:
     *   nf debugglmreplay <model.gguf> --file text.txt <dump.bin>
     *   nf debugglmreplay <model.gguf> --csv "1,2,3"   <dump.bin>   */
    if (argc >= 6 && strcmp(argv[1], "debugglmreplay") == 0 &&
        (strcmp(argv[3], "--file") == 0 || strcmp(argv[3], "--csv") == 0)) {
        const int is_csv = strcmp(argv[3], "--csv") == 0;
        char *text = NULL;
        if (!is_csv) {
            text = read_whole_file(argv[4]);
            if (!text) return 1;
        }
        const int rc = cmd_debugglmreplay(argv[2], is_csv ? argv[4] : text,
                                          is_csv, argv[5]);
        free(text);
        return rc;
    }
    /* session persistence: round-trip save/load, see
     * cmd_debugsession. Usage: nf debugsession <model.gguf> <csv> <path.nfs> */
    if (argc >= 5 && strcmp(argv[1], "debugsession") == 0)
        return cmd_debugsession(argv[2], argv[3], argv[4]);
    if (argc >= 4 && strcmp(argv[1], "debugsessionloadonly") == 0)
        return cmd_debugsessionloadonly(argv[2], argv[3]);
    /* expert kernel bench: no model to load, synthetic
     * data, two regimes (hot/cold) — see nf_debug_kernel_bench. Usage:
     * nf debugkernelbench [iterations] */
    if (argc >= 2 && strcmp(argv[1], "debugkernelbench") == 0) {
        nf_debug_kernel_bench(argc >= 3 ? atoi(argv[2]) : 20);
        return 0;
    }
    /* near-twin expert census:
     * nf debugtwinscan <gguf> <layer> — see nf_debug_twinscan. */
    /* sweep project point 0: amortization factors on real
     * weights. Usage: nf debugsweepbench <gguf> [N] */
    /* sweep project Phase A: nf sweepgen <gguf> --file p.txt --streams N
     * --n-predict K [--ctx C] [--temp T] [--top-p P] [--top-k K] [--seed S]
     * — with --temp > 0 it's best-of-N: same prompt, N different
     * completions (per-stream RNG) at the sweep's aggregate cost. The
     * truncation defaults (0.95/40) are the ones measured as necessary:
     * at top_p=1.0 on the 154k vocabulary, sampling picks up spurious
     * bytes and degenerates (mojibake) — seen on the real GGUF. */
    if (argc >= 5 && strcmp(argv[1], "forgelogits") == 0)
        return cmd_forgelogits(argv[2], argv[3], argv[4]);
    if (argc >= 3 && strcmp(argv[1], "doctor") == 0)
        return cmd_doctor(argv[2]);
    if (argc >= 4 && strcmp(argv[1], "sweepgen") == 0) {
        const char *file_s = arg_value(argc, argv, "--file");
        const char *st_s = arg_value(argc, argv, "--streams");
        const char *np_s = arg_value(argc, argv, "--n-predict");
        const char *cx_s = arg_value(argc, argv, "--ctx");
        const char *tp_s = arg_value(argc, argv, "--temp");
        const char *pp_s = arg_value(argc, argv, "--top-p");
        const char *pk_s = arg_value(argc, argv, "--top-k");
        const char *sd_s = arg_value(argc, argv, "--seed");
        if (!file_s) { fprintf(stderr, "nf: sweepgen requires --file\n"); return 1; }
        char *text = read_whole_file(file_s);
        if (!text) return 1;
        int csv = 0;
        for (int i = 1; i < argc; i++)
            if (strcmp(argv[i], "--csv") == 0) csv = 1;
        const int rc = cmd_sweepgen(argv[2], text,
                                    st_s ? atoi(st_s) : 4,
                                    np_s ? atoi(np_s) : 16,
                                    cx_s ? atoi(cx_s) : 0, csv,
                                    tp_s ? (float)atof(tp_s) : 0.0f,
                                    pp_s ? (float)atof(pp_s) : 0.95f,
                                    pk_s ? (int32_t)atoi(pk_s) : 40,
                                    sd_s ? (uint64_t)strtoull(sd_s, NULL, 10)
                                         : 42ull);
        free(text);
        return rc;
    }
    if (argc >= 3 && strcmp(argv[1], "debugforgekern") == 0)
        return nf_debug_forgekern(argv[2]);
    if (argc >= 3 && strcmp(argv[1], "debugsweepbench") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_sweepbench(m, argc >= 4 ? atoi(argv[3]) : 16);
        nf_model_free(m);
        return 0;
    }
    if (argc >= 4 && strcmp(argv[1], "debugtwinscan") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_twinscan(m, atoi(argv[3]));
        nf_model_free(m);
        return 0;
    }
    /* MTP draft-verify: equivalence of the layer-major
     * verify path against the sequential one (step a) and output
     * invariance of the full draft loop (step b, needs NF_GLM_MTP=1) —
     * see the comments on the two cmd_ functions. */
    if (argc >= 4 && strcmp(argv[1], "debugglmverify") == 0)
        return cmd_debugglmverify(argv[2], argv[3]);
    /* batched prefill: equivalence of layer-major chunks
     * vs the full sequential n_new (glm_eval_core) — see cmd_. */
    if (argc >= 5 && strcmp(argv[1], "debugglmprefillbatch") == 0)
        return cmd_debugglmprefillbatch(argv[2], argv[3], atoi(argv[4]));
    if (argc >= 5 && strcmp(argv[1], "debugmtpdraft") == 0)
        return cmd_debugmtpdraft(argv[2], argv[3], atoi(argv[4]));
    /* layer-skip drafter: standalone correctness (step a)
     * and output invariance of the full draft loop (step b) — see the
     * comments on the two cmd_ functions. */
    if (argc >= 4 && strcmp(argv[1], "debuglayerskip") == 0)
        return cmd_debuglayerskip(argv[2], argv[3]);
    if (argc >= 5 && strcmp(argv[1], "debuglayerskipdraft") == 0)
        return cmd_debuglayerskipdraft(argv[2], argv[3], atoi(argv[4]));
    /* MTP ("amortization via MTP"): offline
     * accuracy report for the MTP head against a trace written by
     * NF_GLM_MTP_TRACE during a real generation — see
     * nf_debug_mtp_report. Usage: nf mtpcheck <model.gguf> <trace.bin> */
    if (argc >= 4 && strcmp(argv[1], "mtpcheck") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        const int rc = nf_debug_mtp_report(m, argv[3]);
        nf_model_free(m);
        return rc ? 1 : 0;
    }
    /* MTP: validation against the HYBRID HF oracle (see
     * nf_debug_mtp_oracle) on the tools/glm_tiny_mtp.gguf test model.
     * Usage: nf mtporacle
     * <model.gguf> <trace.bin> <ref.bin (tools/convert_mtp_ref.py)> */
    if (argc >= 5 && strcmp(argv[1], "mtporacle") == 0) {
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        const int rc = nf_debug_mtp_oracle(m, argv[3], argv[4]);
        nf_model_free(m);
        return rc ? 1 : 0;
    }
#ifdef _WIN32
    /* Explicit glm-dsa expert streaming: byte-identity of
     * streaming vs mmap over n_pairs (layer, expert) pairs — see
     * nf_debug_glm_stream_check. Run with NF_GLM_STREAM_GB set (small,
     * e.g. 0.2, to also exercise eviction). */
    if (argc >= 3 && strcmp(argv[1], "debugglmstream") == 0) {
        const int n_pairs = argc >= 4 ? atoi(argv[3]) : 40;
        if (n_pairs <= 0) {
            fprintf(stderr, "nf: debugglmstream: invalid n_pairs\n");
            return 1;
        }
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        const int rc = nf_debug_glm_stream_check(m, n_pairs);
        nf_model_free(m);
        return rc ? 1 : 0;
    }
#endif
    if (argc >= 3 && strcmp(argv[1], "expcache") == 0) {
        /* the builder rewrites <gguf>.expw: load must NOT mmap any
         * existing file (a sharing violation on fwrite otherwise) */
        nf_setenv("NF_NO_EXPW=1");
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        const int rc = nf_expw_build(m, argv[2]);
        nf_model_free(m);
        return rc ? 1 : 0;
    }
    if (argc >= 3 && strcmp(argv[1], "densecache") == 0) {
        /* same reason as expcache: the builder REWRITES <gguf>.dpk,
         * load must not keep an old copy mmapped (a sharing violation
         * on fwrite otherwise) — and above all it must actually repack,
         * or there'd be nothing to write. */
        nf_setenv("NF_NO_DPK=1");
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        const int rc = nf_dpk_build(m, argv[2]);
        nf_model_free(m);
        return rc ? 1 : 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugdensecache") == 0) {
        /* here the sidecar MUST be active: verifies the mmapped bytes
         * are identical to what the repack would build */
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        const int rc = nf_dpk_verify(m);
        nf_model_free(m);
        return rc ? 1 : 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugshift") == 0) {
        const char *nk_s = arg_value(argc, argv, "--n-keep");
        const char *nd_s = arg_value(argc, argv, "--n-discard");
        const char *nt_s = arg_value(argc, argv, "--n-tail");
        const char *file_s = arg_value(argc, argv, "--file");
        int32_t *toks = NULL;
        int64_t n_toks = 0;
        char *text = NULL;
        if (file_s) {
            text = read_whole_file(file_s);
            if (!text) return 1;
            nf_tokenizer *tk = nf_tokenizer_load(argv[2]);
            if (!tk) { free(text); return 1; }
            toks = malloc(65536 * sizeof(int32_t));
            n_toks = toks ? nf_tokenize(tk, text, toks, 65536) : -1;
            nf_tokenizer_free(tk);
            if (n_toks <= 0) {
                fprintf(stderr, "nf: tokenization failed\n");
                free(toks); free(text);
                return 1;
            }
        }
        nf_model *m = nf_model_load(argv[2]);
        if (!m) { free(toks); free(text); return 1; }
        nf_debug_shift_bench(m, toks, (int)n_toks,
                             nk_s ? atoi(nk_s) : 8,
                             nd_s ? atoi(nd_s) : 16,
                             nt_s ? atoi(nt_s) : 24);
        nf_model_free(m);
        free(toks); free(text);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debuglongmem") == 0) {
        const char *file_s = arg_value(argc, argv, "--file");
        const char *blk_s = arg_value(argc, argv, "--block");
        const char *sink_s = arg_value(argc, argv, "--sink");
        const char *win_s = arg_value(argc, argv, "--window");
        const char *topk_s = arg_value(argc, argv, "--topk");
        const char *needlepos_s = arg_value(argc, argv, "--needle-pos");
        const char *maxtok_s = arg_value(argc, argv, "--max-tokens");
        const char *needletext_s = arg_value(argc, argv, "--needle-text");
        const char *questiontext_s = arg_value(argc, argv, "--question-text");
        if (!file_s) {
            fprintf(stderr, "nf: debuglongmem requires --file (filler "
                    "text, the longer the better)\n");
            return 1;
        }
        const int block = blk_s ? atoi(blk_s) : 128;
        const int sink = sink_s ? atoi(sink_s) : 64;
        const int window = win_s ? atoi(win_s) : 512;
        const int topk = topk_s ? atoi(topk_s) : 16; /* see nf_session_create */
        const int needle_pos = needlepos_s ? atoi(needlepos_s) : 4000;
        /* full O(n^2) attention: the bench does 2-3 long prefills (the
         * oracle ALWAYS does them in full, that's the point) — a cap
         * keeps the bench fast by default; --max-tokens for more
         * realistic corpora when you actually want to tune topk/window */
        const int max_tokens = maxtok_s ? atoi(maxtok_s) : 3000;

        char *text = read_whole_file(file_s);
        if (!text) return 1;
        nf_tokenizer *tk = nf_tokenizer_load(argv[2]);
        if (!tk) { free(text); return 1; }
        int32_t *filler = malloc(200000 * sizeof(int32_t));
        int64_t n_filler = filler ? nf_tokenize(tk, text, filler, 200000) : -1;
        if (n_filler > max_tokens) n_filler = max_tokens;
        /* by default a language-independent synthetic fact: an
         * induction test ("have you seen this string before? complete
         * it") — measures whether attention still finds information at
         * a distance, without depending on the model's linguistic
         * ability to "reason" about a real fact. --needle-text/
         * --question-text let you substitute it with a natural-language
         * fact for a true semantic test (the comparison stays ON vs
         * OFF-oracle, not "absolute right answer": it is always and
         * only a test of fidelity to the exact path, never of the
         * model's linguistic correctness). */
        static const char needle_txt_default[] = "\nQOZY-REFERENCE-CODE: 87231\n";
        static const char question_txt_default[] = "\nQOZY-REFERENCE-CODE:";
        const char *needle_txt = needletext_s ? needletext_s : needle_txt_default;
        const char *question_txt = questiontext_s ? questiontext_s : question_txt_default;
        int32_t needle_tok[64], question_tok[64];
        const int64_t n_needle = nf_tokenize(tk, needle_txt, needle_tok, 64);
        const int64_t n_question = nf_tokenize(tk, question_txt, question_tok, 64);
        nf_tokenizer_free(tk);
        free(text);
        if (n_filler <= 0 || n_needle <= 0 || n_question <= 0) {
            fprintf(stderr, "nf: tokenization failed\n");
            free(filler);
            return 1;
        }
        if ((int64_t)needle_pos + n_needle + n_question >= n_filler) {
            fprintf(stderr, "nf: --file too short for --needle-pos %d "
                    "(need at least %lld tokens, have %lld)\n", needle_pos,
                    (long long)(needle_pos + n_needle + n_question + 1),
                    (long long)n_filler);
            free(filler);
            return 1;
        }

        /* seq_below: just the start of the filler, short enough to stay
         * BELOW threshold (num_closed_blocks <= topk) */
        const int n_below_body = sink + window + block + 8;
        int32_t *seq_below = malloc((size_t)(n_below_body + n_question) * sizeof(int32_t));
        int nb = 0;
        for (int i = 0; i < n_below_body && i < (int)n_filler; i++) seq_below[nb++] = filler[i];
        for (int i = 0; i < (int)n_question; i++) seq_below[nb++] = question_tok[i];

        /* seq_far: needle at needle_pos inside the filler, then the
         * rest of the filler, then the question at the end — the
         * distance between needle and question is almost the whole
         * filler */
        int32_t *seq_far = malloc((size_t)(n_filler + n_needle + n_question) * sizeof(int32_t));
        int nfp = 0;
        for (int i = 0; i < needle_pos; i++) seq_far[nfp++] = filler[i];
        for (int i = 0; i < (int)n_needle; i++) seq_far[nfp++] = needle_tok[i];
        for (int i = needle_pos; i < (int)n_filler; i++) seq_far[nfp++] = filler[i];
        for (int i = 0; i < (int)n_question; i++) seq_far[nfp++] = question_tok[i];

        /* seq_near: same fact, but right before the question — inside
         * sink/window, a control independent of block selection */
        int32_t *seq_near = malloc((size_t)(n_filler + n_needle + n_question) * sizeof(int32_t));
        int nnp = 0;
        for (int i = 0; i < (int)n_filler; i++) seq_near[nnp++] = filler[i];
        for (int i = 0; i < (int)n_needle; i++) seq_near[nnp++] = needle_tok[i];
        for (int i = 0; i < (int)n_question; i++) seq_near[nnp++] = question_tok[i];

        nf_model *m = nf_model_load(argv[2]);
        if (!m) {
            free(filler); free(seq_below); free(seq_far); free(seq_near);
            return 1;
        }
        nf_debug_longmem_bench(m, seq_below, nb, seq_far, nfp, seq_near, nnp,
                               block, sink, window, topk);
        nf_model_free(m);
        free(filler); free(seq_below); free(seq_far); free(seq_near);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debuglongmemdisk") == 0) {
        const char *file_s = arg_value(argc, argv, "--file");
        const char *blk_s = arg_value(argc, argv, "--block");
        const char *sink_s = arg_value(argc, argv, "--sink");
        const char *win_s = arg_value(argc, argv, "--window");
        const char *topk_s = arg_value(argc, argv, "--topk");
        const char *maxtok_s = arg_value(argc, argv, "--max-tokens");
        const char *needlepos_s = arg_value(argc, argv, "--needle-pos");
        const char *needletext_s = arg_value(argc, argv, "--needle-text");
        const char *questiontext_s = arg_value(argc, argv, "--question-text");
        if (!file_s) {
            fprintf(stderr, "nf: debuglongmemdisk requires --file\n");
            return 1;
        }
        const int block = blk_s ? atoi(blk_s) : 128;
        const int sink = sink_s ? atoi(sink_s) : 64;
        const int window = win_s ? atoi(win_s) : 512;
        const int topk = topk_s ? atoi(topk_s) : 16;
        const int max_tokens = maxtok_s ? atoi(maxtok_s) : 3000;
        const int needle_pos = needlepos_s ? atoi(needlepos_s) : 4000;

        char *text = read_whole_file(file_s);
        if (!text) return 1;
        nf_tokenizer *tk = nf_tokenizer_load(argv[2]);
        if (!tk) { free(text); return 1; }
        int32_t *filler = malloc(200000 * sizeof(int32_t));
        int64_t n_filler = filler ? nf_tokenize(tk, text, filler, 200000) : -1;
        if (n_filler > max_tokens) n_filler = max_tokens;
        static const char needle_txt_default[] = "\nQOZY-REFERENCE-CODE: 87231\n";
        static const char question_txt_default[] = "\nQOZY-REFERENCE-CODE:";
        const char *needle_txt = needletext_s ? needletext_s : needle_txt_default;
        const char *question_txt = questiontext_s ? questiontext_s : question_txt_default;
        int32_t needle_tok[64], question_tok[64];
        const int64_t n_needle = nf_tokenize(tk, needle_txt, needle_tok, 64);
        const int64_t n_question = nf_tokenize(tk, question_txt, question_tok, 64);
        nf_tokenizer_free(tk);
        free(text);
        if (n_filler <= 0 || n_needle <= 0 || n_question <= 0) {
            fprintf(stderr, "nf: tokenization failed\n");
            free(filler);
            return 1;
        }
        if ((int64_t)needle_pos + n_needle + n_question >= n_filler) {
            fprintf(stderr, "nf: --file too short for --needle-pos %d "
                    "(need at least %lld tokens, have %lld)\n", needle_pos,
                    (long long)(needle_pos + n_needle + n_question + 1),
                    (long long)n_filler);
            free(filler);
            return 1;
        }

        int32_t *seq_far = malloc((size_t)(n_filler + n_needle + n_question) * sizeof(int32_t));
        int nfp = 0;
        for (int i = 0; i < needle_pos; i++) seq_far[nfp++] = filler[i];
        for (int i = 0; i < (int)n_needle; i++) seq_far[nfp++] = needle_tok[i];
        for (int i = needle_pos; i < (int)n_filler; i++) seq_far[nfp++] = filler[i];
        for (int i = 0; i < (int)n_question; i++) seq_far[nfp++] = question_tok[i];

        int32_t *seq_near = malloc((size_t)(n_filler + n_needle + n_question) * sizeof(int32_t));
        int nnp = 0;
        for (int i = 0; i < (int)n_filler; i++) seq_near[nnp++] = filler[i];
        for (int i = 0; i < (int)n_needle; i++) seq_near[nnp++] = needle_tok[i];
        for (int i = 0; i < (int)n_question; i++) seq_near[nnp++] = question_tok[i];

        nf_model *m = nf_model_load(argv[2]);
        if (!m) { free(filler); free(seq_far); free(seq_near); return 1; }
        nf_debug_longmemdisk_bench(m, filler, (int)n_filler, block, sink, window, topk);
        nf_debug_longmemdisk_recall_bench(m, seq_far, nfp, seq_near, nnp,
                                          block, sink, window, topk);
        nf_model_free(m);
        free(filler); free(seq_far); free(seq_near);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "debugattn") == 0) {
        const char *cx = arg_value(argc, argv, "--ctx");
        const char *itv = arg_value(argc, argv, "--iters");
        nf_model *m = nf_model_load(argv[2]);
        if (!m) return 1;
        nf_debug_attn_bench(m, cx ? atoi(cx) : 4096, itv ? atoi(itv) : 50);
        nf_model_free(m);
        return 0;
    }

    fprintf(stderr,
            "usage: nf inspect   <model.gguf> [--tensors]\n"
            "     nf tokenize  <model.gguf> <text | --file file.txt>\n"
            "     nf logits    <model.gguf> --file prompt.txt [--dump out.f32]\n"
            "     nf embed     <model.gguf> --file prompt.txt --layer N\n"
            "                  [--dump out.f32]  (last token's raw hidden\n"
            "                   state, layer 0-indexed — for RAG-style\n"
            "                   retrieval use cases)\n"
            "     nf generate  <model.gguf> --file prompt.txt [--n-predict N]\n"
            "                  [--temp T] [--top-p P] [--top-k K] [--min-p P]\n"
            "                  [--seed S] [--ctx C]\n"
            "                  [--repeat-penalty P] [--repeat-last-n N]\n"
            "                  [--draft small.gguf] [--draft-k K]\n"
            "                  (greedy speculative decoding: identical output,\n"
            "                   the small model proposes, the big one verifies)\n"
            "     nf selfcheck <model.gguf> --file prompt.txt\n"
            "     nf perplexity <model.gguf> --file corpus.txt [--ctx N]\n"
            "                  [--windows N] [--kv-q8]  (N=0: the whole corpus;\n"
            "                   fixed protocol, numbers comparable ONLY to each other)\n"
            "     nf bench     <model.gguf> --file prompt.txt [--ctx-start N]\n"
            "                  [--ctx-max N] [--step-mul F] [--gen-tokens N]\n"
            "     nf chat      <model.gguf> [--system TEXT | --system-file FILE]\n"
            "                  [--ctx N] [--temp T] [--top-p P] [--top-k K]\n"
            "                  [--min-p P] [--seed S]\n"
            "                  [--ctx-shift]  (sliding window: when the context\n"
            "                   fills up, discards half the history beyond the\n"
            "                   system prompt instead of stopping — RoPE K-shift,\n"
            "                   numerical deviation in the same class as\n"
            "                   quantization, not supported with --draft)\n"
            "                  [--session file.bin] [--max-tokens N] [--no-think]\n"
            "                  [--repeat-penalty P] [--repeat-last-n N]\n"
            "                  [--draft small.gguf] [--draft-k K]\n"
            "                  (greedy speculative decoding in chat too:\n"
            "                   identical output, faster turns)\n"
            "     nf serve     <model.gguf> [--host 127.0.0.1] [--port 8080]\n"
            "                  [--ctx N] [--temp T] [--top-p P] [--top-k K]\n"
            "                  [--min-p P] [--seed S]\n"
            "                  [--repeat-penalty P] [--repeat-last-n N]\n"
            "                  [--warmup-system file.txt]  (prefills the system\n"
            "                   prompt at startup: the first chat starts warm —\n"
            "                   the file must match byte-for-byte what the\n"
            "                   webui sends)\n"
            "                  [--draft small.gguf] [--draft-k 3]\n"
            "                   (Speculative decoding in the server —\n"
            "                    greedy only; K=3 is robust on real content,\n"
            "                    high K only pays off on repetitive text)\n"
            "                  [--sessions-dir dir]  (Slots save\n"
            "                   themselves on Ctrl+C shutdown and come back\n"
            "                   on boot — conversations survive a restart)\n"
            "                  [--max-tokens N] [--no-think]  (skips the\n"
            "                   <think> block: more reliable on aggressive\n"
            "                   quants, e.g. Q4_K_M)\n"
            "                  (OpenAI API /v1/chat/completions + built-in\n"
            "                   page at / — for Open WebUI and similar)\n"
            "     --kv-f16     (K/V cache in f16 — the DEFAULT is\n"
            "                   per-head int8: half the RAM, +0.04%% perplexity,\n"
            "                   faster decode at every context; --kv-q8\n"
            "                   stays accepted and is now a no-op)\n"
            "     nf expcache  <model.gguf>  (MoE: writes <gguf>.expw with\n"
            "                   every expert repacked — subsequent loads get\n"
            "                   an mmapped cache and zero warmup; NF_NO_EXPW=1\n"
            "                   to skip it)\n"
            "     nf densecache <model.gguf> (writes <gguf>.dpk with the dense/\n"
            "                   attention tensors already repacked wide — from\n"
            "                   then on loads mmap them instead of rebuilding\n"
            "                   them (GLM-5.2: ~46s less I/O and 11.7GB less\n"
            "                   malloc on every startup); NF_NO_DPK=1 to skip\n"
            "                   it, nf debugdensecache to verify byte-for-byte\n"
            "                   identity with the repack)\n"
            "     env NF_MOE_PREFILL_REPACK_MIN=8  (MoE: transient expert\n"
            "                   repack in batched prefill — 30B: 2.4x on\n"
            "                   prefill, ~3MB of scratch; ~1e-8 deviation per\n"
            "                   matmul, the batch selfcheck is no longer\n"
            "                   bit-exact this way)\n");
    return 1;
}
