/* nf_math.h benchmark: precision vs. libm double + bit dump for
 * cross-compiler comparison (same grid, same bits expected everywhere). */
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "nf_math.h"

static uint32_t fbits(float f) { union { float f; uint32_t u; } v; v.f = f; return v.u; }
static uint64_t dbits(double d) { union { double d; uint64_t u; } v; v.d = d; return v.u; }

int main(void) {
    /* --- precision vs. libm's double reference --- */
    double emax_expf = 0, emax_exp = 0, emax_log = 0, emax_sin = 0, emax_cos = 0, emax_pow = 0;
    for (int i = 0; i <= 200000; i++) {
        const double x = -87.0 + 175.7 * i / 200000.0;
        const double ref = exp(x);
        const double got = (double)nf_expf((float)x);
        const double reff = (double)expf((float)x);
        double e = fabs(got - reff) / (fabs(reff) > 1e-30 ? fabs(reff) : 1e-30);
        if (e > emax_expf) emax_expf = e;
        const double gd = nf_exp(x);
        e = fabs(gd - ref) / (fabs(ref) > 1e-300 ? fabs(ref) : 1e-300);
        if (e > emax_exp) emax_exp = e;
    }
    for (int i = 1; i <= 200000; i++) {
        const double x = 1e-6 + 2.0e6 * i / 200000.0;
        const double e = fabs(nf_log(x) - log(x)) / (fabs(log(x)) > 1e-12 ? fabs(log(x)) : 1e-12);
        if (e > emax_log) emax_log = e;
    }
    for (int i = 0; i <= 200000; i++) {
        const float x = (float)(200000.0 * i / 200000.0);
        float s, c; nf_sincosf(x, &s, &c);
        double es = fabs((double)s - sin((double)x));
        double ec = fabs((double)c - cos((double)x));
        if (es > emax_sin) emax_sin = es;
        if (ec > emax_cos) emax_cos = ec;
    }
    for (int i = 0; i <= 1000; i++) {
        const float b = -1.0f + 1.0f * i / 1000.0f;
        const double refs[3] = { pow(10000.0, b), pow(1e6, b), pow(5e6, b) };
        const float bases[3] = { 10000.0f, 1e6f, 5e6f };
        for (int j = 0; j < 3; j++) {
            double e = fabs((double)nf_powf(bases[j], b) - refs[j]) / refs[j];
            if (e > emax_pow) emax_pow = e;
        }
    }
    printf("err rel max: nf_expf=%.2e (vs expf libm) nf_exp=%.2e nf_log=%.2e nf_powf=%.2e\n",
           emax_expf, emax_exp, emax_log, emax_pow);
    printf("err abs max: nf_sinf=%.2e nf_cosf=%.2e (up to x=200000)\n", emax_sin, emax_cos);

    /* --- bit dump on a fixed grid: MUST be identical on MSVC and gcc --- */
    uint64_t h = 1469598103934665603ull;   /* FNV-1a over the bits */
    for (int i = 0; i <= 9973; i++) {
        const float xf = -87.0f + 175.7f * (float)i / 9973.0f;
        const double xd = -700.0 + 1400.0 * (double)i / 9973.0;
        const float xs = 200000.0f * (float)i / 9973.0f;
        float s, c; nf_sincosf(xs, &s, &c);
        const uint64_t w[5] = {
            (uint64_t)fbits(nf_expf(xf)), dbits(nf_exp(xd)),
            dbits(nf_log(1e-6 + (double)i)), (uint64_t)fbits(s),
            (uint64_t)fbits(c)
        };
        for (int j = 0; j < 5; j++) {
            for (int b = 0; b < 8; b++) {
                h ^= (w[j] >> (8 * b)) & 0xFF;
                h *= 1099511628211ull;
            }
        }
    }
    printf("grid bit fingerprint: %016llx\n", (unsigned long long)h);
    return 0;
}
