/* membench — measures the machine's actual RAM read bandwidth.
 *
 * Why it exists: an LLM's decode reads (almost) all the weights once
 * per token, so the theoretical tok/s ceiling is
 * RAM_bandwidth / weight_size. Before optimizing the code we need to
 * know whether we're already up against that wall (then only smaller
 * weights or better hardware help) or below it (then compute is the
 * bottleneck and optimizing the kernels is worth it). AVX2 sum over a
 * buffer much larger than any cache, multithreaded with OpenMP — the
 * same access pattern (sequential streaming) as the matvecs.
 *
 * Usage: membench [MB]   (default 2048)
 */
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char **argv) {
    const size_t mb = argc > 1 ? strtoull(argv[1], NULL, 10) : 2048;
    const size_t n = mb * 1024 * 1024 / sizeof(float);
    float *buf = malloc(n * sizeof(float));
    if (!buf) { fprintf(stderr, "malloc %zu MB failed\n", mb); return 1; }
    for (size_t i = 0; i < n; i++) buf[i] = 1.0f;  /* touch every page */

    double best = 0.0;
    float sink = 0.0f;
    for (int rep = 0; rep < 5; rep++) {
        const double t0 = omp_get_wtime();
        float sum = 0.0f;
        #pragma omp parallel reduction(+:sum)
        {
            const int tid = omp_get_thread_num();
            const int nt  = omp_get_num_threads();
            const size_t chunk = n / (size_t)nt;
            const size_t start = (size_t)tid * chunk;
            const size_t end   = tid == nt - 1 ? n : start + chunk;
            __m256 acc = _mm256_setzero_ps();
            size_t i;
            for (i = start; i + 8 <= end; i += 8)
                acc = _mm256_add_ps(acc, _mm256_loadu_ps(buf + i));
            float t[8];
            _mm256_storeu_ps(t, acc);
            sum += t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
        }
        const double dt = omp_get_wtime() - t0;
        const double gbs = (double)(n * sizeof(float)) / dt / 1e9;
        if (gbs > best) best = gbs;
        sink += sum;
        fprintf(stderr, "rep %d: %.2f GB/s\n", rep, gbs);
    }
    printf("read bandwidth (best of 5): %.2f GB/s  [checksum %g]\n", best, sink);
    return 0;
}
