/* Point 0 of the "REAP-style pruning" track — analysis of the weighted
 * routing trace (NF_GLM_GATE_TRACE: uint32 layer, uint32 expert,
 * float32 sigmoid weight without bias triples). The deciding question
 * (from the literature pass, REAP arXiv 2510.13999): does the
 * bottom-25% of experts per layer carry a NEGLIGIBLE share of the
 * selected gate mass? Declared gate: >10-15% of the mass -> track
 * KILLED (the redistribution would be too big); <5% -> promoted to the
 * quality test (runtime mask + KL protocol).
 *
 * Two rankings for choosing the bottom-25% (64 of 256 experts), so as
 * not to depend on a single proxy:
 *   by MASS:      sum of weights when selected (the proxy closest to
 *                 REAP's saliency without the activation norms, which
 *                 would require calibration);
 *   by FREQUENCY: number of selections (the proxy REAP shows to be
 *                 WRONG — reported for comparison).
 * Reports for each: bottom-64 mass share per layer (min/median/max
 * over the MoE layers) + how many experts per layer are NEVER
 * selected in the trace.
 *
 * Build:  cl /nologo /O2 /W4 tools\gate_mass.c /Fe:tools\gate_mass.exe
 * Usage:  tools\gate_mass.exe <gate_trace.bin>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define NLAYER 128
#define NEXP 256

typedef struct { double mass[NEXP]; uint64_t cnt[NEXP]; int seen; } layer_t;

static int cmp_dbl_idx(const void *a, const void *b) {
    const double *x = (const double *)a, *y = (const double *)b;
    return x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <gate_trace.bin>\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "unable to open %s\n", argv[1]); return 1; }
    layer_t *L = calloc(NLAYER, sizeof(layer_t));
    uint64_t nrec = 0;
    for (;;) {
        uint32_t li, ei; float w;
        if (fread(&li, 4, 1, f) != 1 || fread(&ei, 4, 1, f) != 1 ||
            fread(&w, 4, 1, f) != 1) break;
        if (li >= NLAYER || ei >= NEXP) { fprintf(stderr, "record out of range\n"); return 1; }
        L[li].mass[ei] += (double)w;
        L[li].cnt[ei]++;
        L[li].seen = 1;
        nrec++;
    }
    fclose(f);
    printf("record: %llu\n", (unsigned long long)nrec);

    double q_mass[NLAYER], q_freq[NLAYER];
    int never[NLAYER];
    int nl = 0;
    for (int li = 0; li < NLAYER; li++) {
        if (!L[li].seen) continue;
        double tot = 0.0;
        int nev = 0;
        for (int e = 0; e < NEXP; e++) {
            tot += L[li].mass[e];
            if (L[li].cnt[e] == 0) nev++;
        }
        /* bottom-64 by MASS */
        double pairs[NEXP][2];
        for (int e = 0; e < NEXP; e++) { pairs[e][0] = L[li].mass[e]; pairs[e][1] = e; }
        qsort(pairs, NEXP, sizeof(pairs[0]), cmp_dbl_idx);
        double bm = 0.0;
        for (int i = 0; i < 64; i++) bm += pairs[i][0];
        /* bottom-64 by FREQUENCY (tie-break on mass) */
        for (int e = 0; e < NEXP; e++) {
            pairs[e][0] = (double)L[li].cnt[e] * 1e6 + L[li].mass[e];
            pairs[e][1] = e;
        }
        qsort(pairs, NEXP, sizeof(pairs[0]), cmp_dbl_idx);
        double bf = 0.0;
        for (int i = 0; i < 64; i++) {
            const int e = (int)pairs[i][1];
            bf += L[li].mass[e];
        }
        q_mass[nl] = tot > 0 ? 100.0 * bm / tot : 0.0;
        q_freq[nl] = tot > 0 ? 100.0 * bf / tot : 0.0;
        never[nl] = nev;
        printf("layer %3d: bottom-64 mass by-mass %.2f%%  by-freq %.2f%%  "
               "never-selected %d/256\n", li, q_mass[nl], q_freq[nl], nev);
        nl++;
    }
    if (nl == 0) { printf("no layer in the trace\n"); return 1; }
    /* summary: min/median/max over the layers */
    double tmp[NLAYER];
    memcpy(tmp, q_mass, sizeof(double) * (size_t)nl);
    qsort(tmp, (size_t)nl, sizeof(double), cmp_dbl_idx);
    printf("\nSUMMARY (%d MoE layers):\n", nl);
    printf("  bottom-64 by-mass:  min %.2f%%  median %.2f%%  max %.2f%%\n",
           tmp[0], tmp[nl / 2], tmp[nl - 1]);
    memcpy(tmp, q_freq, sizeof(double) * (size_t)nl);
    qsort(tmp, (size_t)nl, sizeof(double), cmp_dbl_idx);
    printf("  bottom-64 by-freq:   min %.2f%%  median %.2f%%  max %.2f%%\n",
           tmp[0], tmp[nl / 2], tmp[nl - 1]);
    long nv = 0;
    for (int i = 0; i < nl; i++) nv += never[i];
    printf("  never-selected experts: average %.1f/256 per layer\n",
           (double)nv / nl);
    printf("\nGATE (from the literature pass): >10-15%% -> track KILLED; "
           "<5%% -> promoted to the quality test.\n");
    free(L);
    return 0;
}
