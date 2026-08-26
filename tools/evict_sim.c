/* Point 0 of the "activation-aware eviction" track —
 * OFFLINE simulator of eviction policies on the real access trace
 * (layer, expert) written by NF_GLM_ACCESS_TRACE (uint32 pairs).
 * Compares, at equal slot count:
 *
 *   LRU      — the policy in production (nf_glm_stream_fetch);
 *   LFU-decay — frequency with periodic halving (MoE-Infinity/
 *              HybriMoE claim it beats LRU for MoE experts);
 *   BELADY   — the theoretical optimum (evicts whoever has the
 *              farthest next use in the future): unrealizable online,
 *              serves ONLY as a ceiling. If LRU is already close to
 *              the ceiling, no online policy can buy enough to justify
 *              the risk of touching the hot path — the track dies here.
 *
 * Declared simplifications vs. the real cache: uniform-size slots (the
 * real one is grow-only with variable bytes per layer, ~10.3MB average
 * — the simulation counts EXPERTS, not bytes); no call_floor pinning
 * (the victims of the current call are not protected: identical effect
 * on every policy, so the COMPARISON stays clean); no prefetch (the
 * trace is the TRUE demand of the fetch, prefetch changes who is
 * already in, not who is needed). The number that matters is the DELTA
 * between policies, not the absolute value.
 *
 * Build:  cl /nologo /O2 /W4 tools\evict_sim.c /Fe:tools\evict_sim.exe
 * Usage:  tools\evict_sim.exe <trace.bin> <n_slot> [decay_every]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct { uint32_t layer, expert; } acc_t;

static int key_of(const acc_t *a) { return (int)(a->layer * 4096u + a->expert); }

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <trace.bin> <n_slot> [decay_every=2048]\n",
                argv[0]);
        return 1;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "unable to open %s\n", argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    const long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    const int n = (int)(fsz / (long)sizeof(acc_t));
    acc_t *tr = malloc((size_t)n * sizeof(acc_t));
    if (!tr || (int)fread(tr, sizeof(acc_t), (size_t)n, f) != n) {
        fprintf(stderr, "trace read failed\n");
        return 1;
    }
    fclose(f);
    const int nslot = atoi(argv[2]);
    const int decay_every = argc > 3 ? atoi(argv[3]) : 2048;
    printf("trace: %d accesses, %d simulated slots\n", n, nslot);

    /* key space: layer*4096+expert — dense map via a large array */
    const int KEYSPACE = 4096 * 256;
    int *pos = malloc((size_t)KEYSPACE * sizeof(int));      /* key -> slot */
    /* --- LRU --- */
    {
        int *owner = malloc((size_t)nslot * sizeof(int));
        uint64_t *last = malloc((size_t)nslot * sizeof(uint64_t));
        memset(pos, -1, (size_t)KEYSPACE * sizeof(int));
        for (int s = 0; s < nslot; s++) { owner[s] = -1; last[s] = 0; }
        uint64_t tick = 0, hit = 0, miss = 0;
        int used = 0;
        for (int i = 0; i < n; i++) {
            const int key = key_of(&tr[i]);
            tick++;
            if (pos[key] >= 0) { hit++; last[pos[key]] = tick; continue; }
            miss++;
            int victim;
            if (used < nslot) victim = used++;
            else {
                victim = 0;
                for (int s = 1; s < nslot; s++)
                    if (last[s] < last[victim]) victim = s;
                pos[owner[victim]] = -1;
            }
            owner[victim] = key; last[victim] = tick; pos[key] = victim;
        }
        printf("LRU       : hit=%llu miss=%llu  hit-rate %.2f%%\n",
               (unsigned long long)hit, (unsigned long long)miss,
               100.0 * (double)hit / (double)(hit + miss));
        free(owner); free(last);
    }
    /* --- LFU with periodic decay --- */
    {
        int *owner = malloc((size_t)nslot * sizeof(int));
        uint64_t *freq = malloc((size_t)nslot * sizeof(uint64_t));
        uint64_t *last = malloc((size_t)nslot * sizeof(uint64_t));
        memset(pos, -1, (size_t)KEYSPACE * sizeof(int));
        for (int s = 0; s < nslot; s++) { owner[s] = -1; freq[s] = 0; last[s] = 0; }
        uint64_t tick = 0, hit = 0, miss = 0;
        int used = 0;
        for (int i = 0; i < n; i++) {
            const int key = key_of(&tr[i]);
            tick++;
            if (decay_every > 0 && (i % decay_every) == decay_every - 1)
                for (int s = 0; s < nslot; s++) freq[s] >>= 1;
            if (pos[key] >= 0) { hit++; freq[pos[key]]++; last[pos[key]] = tick; continue; }
            miss++;
            int victim;
            if (used < nslot) victim = used++;
            else {
                victim = 0;   /* minimum freq, tie-break on LRU */
                for (int s = 1; s < nslot; s++)
                    if (freq[s] < freq[victim] ||
                        (freq[s] == freq[victim] && last[s] < last[victim]))
                        victim = s;
                pos[owner[victim]] = -1;
            }
            owner[victim] = key; freq[victim] = 1; last[victim] = tick;
            pos[key] = victim;
        }
        printf("LFU-decay : hit=%llu miss=%llu  hit-rate %.2f%%  (decay every %d)\n",
               (unsigned long long)hit, (unsigned long long)miss,
               100.0 * (double)hit / (double)(hit + miss), decay_every);
        free(owner); free(freq); free(last);
    }
    /* --- BELADY (optimum, knows the future) --- */
    {
        /* next_use[i] = next index > i with the same key (or n) —
         * precomputed backwards with a key -> last-seen map. */
        int *next_use = malloc((size_t)n * sizeof(int));
        int *seen = malloc((size_t)KEYSPACE * sizeof(int));
        for (int k = 0; k < KEYSPACE; k++) seen[k] = n;
        for (int i = n - 1; i >= 0; i--) {
            const int key = key_of(&tr[i]);
            next_use[i] = seen[key];
            seen[key] = i;
        }
        int *owner = malloc((size_t)nslot * sizeof(int));
        int *nxt = malloc((size_t)nslot * sizeof(int));   /* next use of the slot */
        memset(pos, -1, (size_t)KEYSPACE * sizeof(int));
        for (int s = 0; s < nslot; s++) { owner[s] = -1; nxt[s] = n; }
        uint64_t hit = 0, miss = 0;
        int used = 0;
        for (int i = 0; i < n; i++) {
            const int key = key_of(&tr[i]);
            if (pos[key] >= 0) { hit++; nxt[pos[key]] = next_use[i]; continue; }
            miss++;
            int victim;
            if (used < nslot) victim = used++;
            else {
                victim = 0;   /* FARTHEST next use */
                for (int s = 1; s < nslot; s++)
                    if (nxt[s] > nxt[victim]) victim = s;
                pos[owner[victim]] = -1;
            }
            owner[victim] = key; nxt[victim] = next_use[i]; pos[key] = victim;
        }
        printf("BELADY    : hit=%llu miss=%llu  hit-rate %.2f%%  (theoretical ceiling)\n",
               (unsigned long long)hit, (unsigned long long)miss,
               100.0 * (double)hit / (double)(hit + miss));
        free(next_use); free(seen); free(owner); free(nxt);
    }
    /* --- LRU + PREDICTIVE PROTECTION (point 0 of track B, second
     * literature pass): like LRU, but when choosing a victim the
     * experts of the NEXT request group (the following layer — in the
     * real trace each fetch is a group of K=8 consecutive accesses)
     * are protected, each with probability (1-err) to mimic the real
     * predictor's error rate (~12-19% measured live). Pseudo-
     * randomness is DETERMINISTIC (LCG on group^key). Models the
     * mechanism "prediction used ONLY for victim selection, never for
     * reads": zero I/O cost in the real engine. If it doesn't beat LRU
     * by >=2-3 points even HERE (where the prediction is near-oracle),
     * the track dies. */
    {
        const int GROUP = 8;   /* K=8: one fetch = 8 consecutive accesses */
        const int errpct = argc > 4 ? atoi(argv[4]) : 15;
        int *owner = malloc((size_t)nslot * sizeof(int));
        uint64_t *last = malloc((size_t)nslot * sizeof(uint64_t));
        unsigned char *prot = malloc((size_t)nslot);
        memset(pos, -1, (size_t)KEYSPACE * sizeof(int));
        for (int s = 0; s < nslot; s++) { owner[s] = -1; last[s] = 0; }
        uint64_t tick = 0, hit = 0, miss = 0;
        int used = 0;
        for (int i = 0; i < n; i++) {
            const int key = key_of(&tr[i]);
            tick++;
            if (pos[key] >= 0) { hit++; last[pos[key]] = tick; continue; }
            miss++;
            int victim;
            if (used < nslot) victim = used++;
            else {
                /* protected set: keys of the next group, with
                 * deterministic noise from the LCG */
                const int lookahead = argc > 5 ? atoi(argv[5]) : 1;
                const int g = i / GROUP;
                const int nga = (g + 1) * GROUP;
                const int ngb0 = nga + lookahead * GROUP;
                const int ngb = ngb0 > n ? n : ngb0;
                memset(prot, 0, (size_t)nslot);
                for (int j = nga; j < ngb; j++) {
                    const int pk = key_of(&tr[j]);
                    uint32_t r = (uint32_t)(g * 2654435761u) ^ (uint32_t)pk;
                    r = r * 1664525u + 1013904223u;
                    if ((int)(r % 100u) < errpct) continue;   /* mispredicted */
                    if (pos[pk] >= 0) prot[pos[pk]] = 1;
                }
                victim = -1;
                for (int s = 0; s < nslot; s++) {
                    if (prot[s]) continue;
                    if (victim < 0 || last[s] < last[victim]) victim = s;
                }
                if (victim < 0) {   /* all protected: pure LRU */
                    victim = 0;
                    for (int s = 1; s < nslot; s++)
                        if (last[s] < last[victim]) victim = s;
                }
                pos[owner[victim]] = -1;
            }
            owner[victim] = key; last[victim] = tick; pos[key] = victim;
        }
        printf("LRU+prot  : hit=%llu miss=%llu  hit-rate %.2f%%  (next-group "
               "protection, err=%d%%)\n",
               (unsigned long long)hit, (unsigned long long)miss,
               100.0 * (double)hit / (double)(hit + miss), errpct);
        free(owner); free(last); free(prot);
    }
    free(pos); free(tr);
    return 0;
}
