/* nf_math.h — cross-platform deterministic transcendentals
 * (Linux port project).
 *
 * WHY THIS EXISTS: expf/logf/powf/sinf/cosf from the platform libm
 * (UCRT on Windows, musl/glibc on Linux) round DIFFERENTLY on some
 * arguments — on the port's tiny gate, position 2 diverged by 1-6 ulp
 * because of them. These implementations use ONLY exactly-rounded IEEE
 * operations (+ - * / sqrt floor, identical on every compiler/OS given
 * the same operation order) plus bit manipulation: same input -> same
 * output bits, everywhere. Requires the compiler to NOT fuse mul+add
 * into FMA (MSVC's /fp:precise doesn't fuse; gcc/clang need
 * -ffp-contract=off, already set in build_posix.sh).
 *
 * Cores taken from Cephes (Stephen L. Moshier, public domain), the same
 * family used by musl/ggml. Precision: nf_expf ~2 ulp; nf_exp/nf_log
 * double ~1 ulp; nf_sincosf reduces the argument in double (absolute
 * error ~1e-16*|x|, invisible in float up to |x|~1e9 — RoPE angles top
 * out at n_ctx, orders of magnitude below that). */
#ifndef NF_MATH_H
#define NF_MATH_H

#include <stdint.h>

/* ------------------------------------------------ exp double (Cephes) */
static inline double nf_exp(double x) {
    if (x != x) return x;                     /* NaN */
    if (x > 709.782712893383996843) {         /* overflow -> +inf */
        union { uint64_t u; double d; } inf = { 0x7FF0000000000000ull };
        return inf.d;
    }
    if (x < -708.396418532264106224) return 0.0;   /* underflow */
    /* n = round(x/ln2); r = x - n*ln2 split into two parts */
    double n = x * 1.4426950408889634073599246810019;
    n = (n >= 0.0) ? (double)(long long)(n + 0.5)
                   : (double)(long long)(n - 0.5);
    double r = x - n * 6.93145751953125e-1;
    r -= n * 1.42860682030941723212e-6;
    /* Pade: exp(r) = 1 + 2*px/(q-px), px = r*P(r^2), q = Q(r^2) */
    const double z = r * r;
    double px = 1.26177193074810590878e-4;
    px = px * z + 3.02994407707441961300e-2;
    px = px * z + 9.99999999999999999910e-1;
    px *= r;
    double q = 3.00198505138664455042e-6;
    q = q * z + 2.52448340349684104192e-3;
    q = q * z + 2.27265548208155028766e-1;
    q = q * z + 2.00000000000000000005e0;
    double e = 1.0 + 2.0 * (px / (q - px));
    /* scale by 2^n via bits, in two steps (n up to +-1024) */
    long long ni = (long long)n;
    long long h = ni / 2;
    union { uint64_t u; double d; } s1, s2;
    s1.u = (uint64_t)(h + 1023) << 52;
    s2.u = (uint64_t)(ni - h + 1023) << 52;
    return e * s1.d * s2.d;
}

/* ------------------------------------------------ log double (Cephes) */
static inline double nf_log(double x) {
    union { uint64_t u; double d; } v = { 0 };
    v.d = x;
    if (x != x) return x;                                    /* NaN */
    if (x < 0.0) { v.u = 0x7FF8000000000000ull; return v.d; } /* NaN */
    if (x == 0.0) { v.u = 0xFFF0000000000000ull; return v.d; } /* -inf */
    if (v.u >= 0x7FF0000000000000ull) return x;               /* +inf */
    int e = 0;
    if (v.u < 0x0010000000000000ull) {   /* denormal: normalize */
        v.d = x * 1.8014398509481984e16;  /* 2^54 */
        e -= 54;
    }
    /* frexp: mantissa in [0.5,1) */
    e += (int)(v.u >> 52) - 1022;
    v.u = (v.u & 0x000FFFFFFFFFFFFFull) | 0x3FE0000000000000ull;
    double m = v.d;
    if (m < 0.70710678118654752440) { m += m; e -= 1; }
    m -= 1.0;
    /* Cephes rational: log(1+m) = m - m^2/2 + m^3*P(m)/Q(m) + corr(e) */
    double p = 1.01875663804580931796e-4;
    p = p * m + 4.97494994976747001425e-1;
    p = p * m + 4.70579119878881725854e0;
    p = p * m + 1.44989225341610930846e1;
    p = p * m + 1.79368678507819816313e1;
    p = p * m + 7.70838733755885391666e0;
    double q = m + 1.12873587189167450590e1;
    q = q * m + 4.52279145837532221105e1;
    q = q * m + 8.29875266912776603211e1;
    q = q * m + 7.11544750618563894466e1;
    q = q * m + 2.31251620126765340583e1;
    const double z = m * m;
    double y = m * (z * p / q);
    const double ed = (double)e;
    y -= ed * 2.121944400546905827679e-4;
    y -= 0.5 * z;
    double r = m + y;
    r += ed * 0.693359375;
    return r;
}

/* --------------------------------------------- expf float (Cephes) */
/* pure polynomial, no divisions: this sits in the hot loops (softmax
 * over the vocabulary/heads, silu per expert, router sigmoid). */
static inline float nf_expf(float x) {
    if (x != x) return x;                     /* NaN */
    if (x > 88.72283f) {                      /* overflow -> +inf */
        union { uint32_t u; float f; } inf = { 0x7F800000u };
        return inf.f;
    }
    if (x < -87.33654f) return 0.0f;          /* underflow (flush) */
    float n = x * 1.44269504088896341f;
    n = (n >= 0.0f) ? (float)(int32_t)(n + 0.5f)
                    : (float)(int32_t)(n - 0.5f);
    float r = x - n * 0.693359375f;
    r = r - n * -2.12194440e-4f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * r * r + r + 1.0f;
    int32_t ni = (int32_t)n;
    int32_t h = ni / 2;
    union { uint32_t u; float f; } s1, s2;
    s1.u = (uint32_t)(h + 127) << 23;
    s2.u = (uint32_t)(ni - h + 127) << 23;
    return p * s1.f * s2.f;
}

/* ------------------------------- logf/powf float (cold sites: RoPE) */
static inline float nf_logf(float x) { return (float)nf_log((double)x); }
static inline float nf_powf(float a, float b) {
    return (float)nf_exp((double)b * nf_log((double)a));
}

/* --------------------------------------- sin/cos float (RoPE tables) */
/* Argument reduction mod pi/2 in double (round, mul, sub: exact),
 * quadrant from k mod 4, Cephes polynomials evaluated in double, cast
 * to float at the end. RoPE angles are >= 0 and bounded by n_ctx. */
static inline void nf_sincosf(float xf, float *so, float *co) {
    const double x = (double)xf;
    double k = x * 0.63661977236758134308;    /* 2/pi */
    k = (k >= 0.0) ? (double)(long long)(k + 0.5)
                   : (double)(long long)(k - 0.5);
    double r = x - k * 1.5707963267948966;    /* high pi/2 */
    r -= k * 6.123233995736766e-17;           /* low pi/2 */
    const int q = (int)((long long)k & 3);    /* two's complement: mod 4 */
    const double z = r * r;
    double ps = -1.9515295891e-4;
    ps = ps * z + 8.3321608736e-3;
    ps = ps * z - 1.6666654611e-1;
    const double sr = r + r * z * ps;
    double pc = 2.443315711809948e-5;
    pc = pc * z - 1.388731625493765e-3;
    pc = pc * z + 4.166664568298827e-2;
    const double cr = 1.0 - 0.5 * z + z * z * pc;
    double s, c;
    switch (q) {
    case 0:  s = sr;  c = cr;  break;
    case 1:  s = cr;  c = -sr; break;
    case 2:  s = -sr; c = -cr; break;
    default: s = -cr; c = sr;  break;
    }
    *so = (float)s;
    *co = (float)c;
}
static inline float nf_sinf(float x) { float s, c; nf_sincosf(x, &s, &c); return s; }
static inline float nf_cosf(float x) { float s, c; nf_sincosf(x, &s, &c); return c; }

#endif /* NF_MATH_H */
