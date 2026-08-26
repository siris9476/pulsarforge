/* Point 0 of the KL quality protocol for NF_GLM_MISS_SKIP — OFFLINE
 * comparison between two logit dumps written by `nf debugglmreplay`
 * (or by NF_DEBUG_DUMP_INC: same format, n_vocab float32 per position,
 * positions in order). Methodology from llama.cpp --kl-divergence;
 * arXiv 2605.02404: in the near-lossless regime accuracy degrades
 * ~linearly with KL.
 *
 * For each position: softmax of both vectors in DOUBLE with max
 * subtraction (numerical stability), then
 *   KL(P_base || Q_var) = sum_i p_i * log(p_i / q_i)
 * and top-1 agreement (argmax base == argmax variant). Key sanity
 * property: on two BIT-IDENTICAL dumps p_i == q_i exactly,
 * log(p/q) == log(1) == 0.0 exactly, so KL == 0.0 EXACTLY — if it
 * isn't, there's a bug in the dump or here.
 *
 * Prints: one line per position (index, KL, top1_match), then the
 * summary (KL mean/median/p99, top-1 agreement %, worst 5 positions).
 * p99 over 30-50 positions degenerates to the max: reported anyway,
 * it's the ceiling of the tail.
 *
 * Build:  cl /nologo /O2 /W4 tools\kl_compare.c /Fe:tools\kl_compare.exe
 * Usage:  tools\kl_compare.exe <base.bin> <variant.bin> <n_vocab> <n_pos>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <base.bin> <variant.bin> <n_vocab> "
                "<n_pos>\n", argv[0]);
        return 1;
    }
    const long long V = atoll(argv[3]);
    const int N = atoi(argv[4]);
    if (V <= 0 || N <= 0) {
        fprintf(stderr, "kl_compare: invalid n_vocab/n_pos\n");
        return 1;
    }

    FILE *fb = fopen(argv[1], "rb");
    if (!fb) { fprintf(stderr, "kl_compare: unable to open %s\n", argv[1]); return 1; }
    FILE *fv = fopen(argv[2], "rb");
    if (!fv) { fprintf(stderr, "kl_compare: unable to open %s\n", argv[2]); fclose(fb); return 1; }

    float *lb = malloc((size_t)V * sizeof(float));
    float *lv = malloc((size_t)V * sizeof(float));
    double *pb = malloc((size_t)V * sizeof(double));
    double *pv = malloc((size_t)V * sizeof(double));
    double *kls = malloc((size_t)N * sizeof(double));
    if (!lb || !lv || !pb || !pv || !kls) {
        fprintf(stderr, "kl_compare: oom\n");
        return 1;
    }

    int n_match = 0;
    for (int k = 0; k < N; k++) {
        if (fread(lb, sizeof(float), (size_t)V, fb) != (size_t)V ||
            fread(lv, sizeof(float), (size_t)V, fv) != (size_t)V) {
            fprintf(stderr, "kl_compare: truncated read at pos=%d "
                    "(wrong n_pos or n_vocab?)\n", k);
            return 1;
        }
        /* softmax in double with max-subtraction, for both */
        long long ab = 0, av = 0;
        for (long long i = 1; i < V; i++) {
            if (lb[i] > lb[ab]) ab = i;
            if (lv[i] > lv[av]) av = i;
        }
        double sb = 0.0, sv = 0.0;
        const double mb = (double)lb[ab], mv = (double)lv[av];
        for (long long i = 0; i < V; i++) {
            pb[i] = exp((double)lb[i] - mb);  sb += pb[i];
            pv[i] = exp((double)lv[i] - mv);  sv += pv[i];
        }
        double kl = 0.0;
        for (long long i = 0; i < V; i++) {
            const double p = pb[i] / sb, q = pv[i] / sv;
            /* p==0 exact contributes 0 by convention (lim x*log x);
             * q==0 with p>0 would give +inf — left to emerge, an inf
             * in the output IS the right signal of disjoint support. */
            if (p > 0.0 && p != q) kl += p * log(p / q);
        }
        kls[k] = kl;
        const int match = ab == av;
        n_match += match;
        printf("pos=%3d  KL=%.9f  top1=%s\n", k, kl, match ? "yes" : "NO");
    }
    fclose(fb); fclose(fv);

    /* summary: mean/median/p99 on a sorted copy, worst 5 */
    double mean = 0.0;
    for (int k = 0; k < N; k++) mean += kls[k];
    mean /= (double)N;
    double *sorted = malloc((size_t)N * sizeof(double));
    memcpy(sorted, kls, (size_t)N * sizeof(double));
    qsort(sorted, (size_t)N, sizeof(double), cmp_double);
    const double median = N % 2 ? sorted[N / 2]
                                : 0.5 * (sorted[N / 2 - 1] + sorted[N / 2]);
    int i99 = (int)ceil(0.99 * N) - 1;
    if (i99 < 0) i99 = 0;
    if (i99 >= N) i99 = N - 1;
    const double p99 = sorted[i99];

    printf("---\n");
    printf("n_pos=%d  n_vocab=%lld\n", N, V);
    printf("KL mean=%.9f  median=%.9f  p99=%.9f  max=%.9f\n",
           mean, median, p99, sorted[N - 1]);
    printf("top-1 agreement: %d/%d (%.1f%%)\n",
           n_match, N, 100.0 * n_match / N);
    printf("worst 5 positions:");
    for (int r = 0; r < 5 && r < N; r++) {
        /* linear scan: N is small */
        int wi = -1; double wv = -1.0;
        for (int k = 0; k < N; k++)
            if (kls[k] > wv) { wv = kls[k]; wi = k; }
        printf("  pos=%d KL=%.6f", wi, wv);
        kls[wi] = -2.0;  /* exclude it from the next round */
    }
    printf("\n");
    return 0;
}
