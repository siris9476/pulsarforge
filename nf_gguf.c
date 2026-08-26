/* nf_gguf.c — M0: GGUF v3 format reader.
 *
 * File layout (all integers are little-endian; we assume we're running
 * on a little-endian machine, so fields are read without byte-swap —
 * true for modern x86 and ARM):
 *
 *   magic "GGUF" | u32 version | u64 n_tensors | u64 n_kv
 *   n_kv times:      key string | u32 value type | value
 *   n_tensors times: name string | u32 n_dims | u64 dims[n_dims]
 *                    | u32 ggml type | u64 offset (relative to data)
 *   padding up to general.alignment (default 32)
 *   tensor data, each aligned
 *
 * Strings are: u64 length + bytes (no terminator).
 */
#include "nf_gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

/* 64-bit fseek/ftell: real GGUFs easily exceed 2GB. */
#ifdef _WIN32
#define nf_fseek _fseeki64
#define nf_ftell _ftelli64
#else
#define nf_fseek fseeko
#define nf_ftell ftello
#endif

/* Metadata value types (from the GGUF spec). */
enum {
    GGUF_U8 = 0, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32,
    GGUF_F32, GGUF_BOOL, GGUF_STR, GGUF_ARR, GGUF_U64, GGUF_I64, GGUF_F64,
    GGUF_VTYPE_COUNT
};
static const int gguf_scalar_size[GGUF_VTYPE_COUNT] =
    {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
static const char *gguf_vtype_name[GGUF_VTYPE_COUNT] = {
    "u8", "i8", "u16", "i16", "u32", "i32",
    "f32", "bool", "str", "arr", "u64", "i64", "f64"};

/* ggml tensor types: elements per block and bytes per block. Same
 * layouts seen in DwarfStar/llama.cpp (e.g. Q4_K: 256 weights in 144
 * bytes). Only the common ones are listed; the rest come back
 * "unknown". */
typedef struct { const char *name; int blck; int bytes; } ggml_traits;
static const ggml_traits k_ggml_types[] = {
    [0]  = {"F32",     1,   4},   [1]  = {"F16",     1,   2},
    [2]  = {"Q4_0",    32,  18},  [3]  = {"Q4_1",    32,  20},
    [6]  = {"Q5_0",    32,  22},  [7]  = {"Q5_1",    32,  24},
    [8]  = {"Q8_0",    32,  34},  [9]  = {"Q8_1",    32,  36},
    [10] = {"Q2_K",    256, 84},  [11] = {"Q3_K",    256, 110},
    [12] = {"Q4_K",    256, 144}, [13] = {"Q5_K",    256, 176},
    [14] = {"Q6_K",    256, 210}, [15] = {"Q8_K",    256, 292},
    [16] = {"IQ2_XXS", 256, 66},  [17] = {"IQ2_XS",  256, 74},
    [18] = {"IQ3_XXS", 256, 98},  [19] = {"IQ1_S",   256, 50},
    [20] = {"IQ4_NL",  32,  18},  [21] = {"IQ3_S",   256, 110},
    [22] = {"IQ2_S",   256, 82},  [23] = {"IQ4_XS",  256, 136},
    [24] = {"I8",      1,   1},   [25] = {"I16",     1,   2},
    [26] = {"I32",     1,   4},   [27] = {"I64",     1,   8},
    [28] = {"F64",     1,   8},   [29] = {"IQ1_M",   256, 56},
    [30] = {"BF16",    1,   2},
    /* TERNARY formats (BitNet b1.58) — 34/35 in the ggml enum
     * (31-33 were the removed Q4_0_x_x, the ids stay reserved) */
    [34] = {"TQ1_0",   256, 54},  [35] = {"TQ2_0",   256, 66},
};
#define N_GGML_TYPES (sizeof(k_ggml_types) / sizeof(k_ggml_types[0]))

const char *nf_ggml_type_name(uint32_t t) {
    if (t < N_GGML_TYPES && k_ggml_types[t].name) return k_ggml_types[t].name;
    return "?";
}
int nf_ggml_type_block_elems(uint32_t t) {
    if (t < N_GGML_TYPES && k_ggml_types[t].name) return k_ggml_types[t].blck;
    return 0;
}
int nf_ggml_type_block_bytes(uint32_t t) {
    if (t < N_GGML_TYPES && k_ggml_types[t].name) return k_ggml_types[t].bytes;
    return 0;
}

/* ---- read helpers with error checking --------------------------------- */

static int rd(FILE *f, void *p, size_t n) {
    return fread(p, 1, n, f) == n ? 0 : -1;
}
static int rd_u32(FILE *f, uint32_t *v) { return rd(f, v, 4); }
static int rd_u64(FILE *f, uint64_t *v) { return rd(f, v, 8); }

/* Reads a GGUF string (u64 len + bytes), returns a null-terminated
 * copy. max guards against corrupt files with absurd lengths. */
static char *rd_str(FILE *f, uint64_t max) {
    uint64_t len;
    if (rd_u64(f, &len) || len > max) return NULL;
    char *s = malloc((size_t)len + 1);
    if (!s) return NULL;
    if (rd(f, s, (size_t)len)) { free(s); return NULL; }
    s[len] = 0;
    return s;
}

/* Prints a string truncated to one line (for long metadata like the
 * chat template). */
static void print_str_preview(const char *s) {
    int n = 0;
    putchar('"');
    for (; s[n] && n < 60; n++)
        putchar(s[n] == '\n' || s[n] == '\r' ? ' ' : s[n]);
    printf(s[n] ? "\"..." : "\"");
}

/* Reads (and optionally prints) a metadata value. For scalar integers
 * it also deposits the value into *out_u64, so the caller can capture
 * general.alignment. Recursive for arrays. */
static int handle_value(FILE *f, uint32_t vtype, int verbose, int depth,
                        uint64_t *out_u64) {
    if (depth > 4) return -1; /* nested arrays beyond any reasonable depth */

    if (vtype < GGUF_VTYPE_COUNT && gguf_scalar_size[vtype] > 0) {
        union { uint8_t u8; int8_t i8; uint16_t u16; int16_t i16;
                uint32_t u32; int32_t i32; float f32;
                uint64_t u64; int64_t i64; double f64; } v = {0};
        if (rd(f, &v, gguf_scalar_size[vtype])) return -1;
        if (out_u64) {
            switch (vtype) {
            case GGUF_U8:  *out_u64 = v.u8;  break;
            case GGUF_BOOL: *out_u64 = v.u8; break; /* add_bos_token */
            case GGUF_U16: *out_u64 = v.u16; break;
            case GGUF_U32: *out_u64 = v.u32; break;
            case GGUF_U64: *out_u64 = v.u64; break;
            default: break;
            }
        }
        if (verbose) {
            switch (vtype) {
            case GGUF_U8:   printf("%u",  v.u8);  break;
            case GGUF_I8:   printf("%d",  v.i8);  break;
            case GGUF_U16:  printf("%u",  v.u16); break;
            case GGUF_I16:  printf("%d",  v.i16); break;
            case GGUF_U32:  printf("%u",  v.u32); break;
            case GGUF_I32:  printf("%d",  v.i32); break;
            case GGUF_F32:  printf("%g",  v.f32); break;
            case GGUF_BOOL: printf(v.u8 ? "true" : "false"); break;
            case GGUF_U64:  printf("%" PRIu64, v.u64); break;
            case GGUF_I64:  printf("%" PRId64, v.i64); break;
            case GGUF_F64:  printf("%g",  v.f64); break;
            }
        }
        return 0;
    }

    if (vtype == GGUF_STR) {
        char *s = rd_str(f, 64u * 1024 * 1024);
        if (!s) return -1;
        if (verbose) print_str_preview(s);
        free(s);
        return 0;
    }

    if (vtype == GGUF_ARR) {
        uint32_t etype;
        uint64_t count;
        if (rd_u32(f, &etype) || rd_u64(f, &count)) return -1;
        if (verbose)
            printf("[%s x %" PRIu64 "] ",
                   etype < GGUF_VTYPE_COUNT ? gguf_vtype_name[etype] : "?", count);

        /* Show the first few elements, then skip the rest: a
         * tokenizer's vocabulary is hundreds of thousands of strings. */
        const uint64_t shown = count < 4 ? count : 4;
        if (etype < GGUF_VTYPE_COUNT && gguf_scalar_size[etype] > 0) {
            for (uint64_t i = 0; i < shown; i++) {
                if (handle_value(f, etype, verbose, depth + 1, NULL)) return -1;
                if (verbose && i + 1 < shown) printf(", ");
            }
            if (count > shown) {
                if (verbose) printf(", ...");
                if (nf_fseek(f, (int64_t)(count - shown) * gguf_scalar_size[etype],
                             SEEK_CUR)) return -1;
            }
        } else if (etype == GGUF_STR) {
            for (uint64_t i = 0; i < count; i++) {
                if (i < shown) {
                    if (handle_value(f, GGUF_STR, verbose, depth + 1, NULL)) return -1;
                    if (verbose && i + 1 < shown) printf(", ");
                } else {
                    uint64_t len; /* skip: u64 len + bytes */
                    if (rd_u64(f, &len) || nf_fseek(f, (int64_t)len, SEEK_CUR))
                        return -1;
                }
            }
            if (verbose && count > shown) printf(", ...");
        } else if (etype == GGUF_ARR) {
            for (uint64_t i = 0; i < count; i++)
                if (handle_value(f, GGUF_ARR, 0, depth + 1, NULL)) return -1;
        } else {
            return -1; /* unknown element type: can't skip it */
        }
        return 0;
    }

    return -1; /* unknown value type */
}

/* ---- file parsing ------------------------------------------------------ */

int nf_gguf_open(nf_gguf *g, const char *path, int verbose) {
    memset(g, 0, sizeof(*g));
    g->alignment = 32;

    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "nf: cannot open %s\n", path); return -1; }

    if (nf_fseek(f, 0, SEEK_END)) goto corrupt;
    g->file_size = (uint64_t)nf_ftell(f);
    if (nf_fseek(f, 0, SEEK_SET)) goto corrupt;

    char magic[4];
    if (rd(f, magic, 4) || memcmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "nf: %s is not a GGUF file\n", path);
        goto err;
    }
    if (rd_u32(f, &g->version)) goto corrupt;
    if (g->version != 3) {
        /* Like DwarfStar: we only accept the version we know and fail
         * fast, instead of trying to interpret it by guesswork. */
        fprintf(stderr, "nf: GGUF version %u not supported (v3 only)\n",
                g->version);
        goto err;
    }
    if (rd_u64(f, &g->n_tensors) || rd_u64(f, &g->n_kv)) goto corrupt;
    if (g->n_tensors > (1u << 20) || g->n_kv > (1u << 20)) goto corrupt;

    /* --- key-value metadata --- */
    for (uint64_t i = 0; i < g->n_kv; i++) {
        char *key = rd_str(f, 4096);
        uint32_t vtype;
        if (!key || rd_u32(f, &vtype)) { free(key); goto corrupt; }
        if (verbose) printf("  %-42s ", key);

        uint64_t uval = 0;
        int rc = handle_value(f, vtype, verbose, 0, &uval);
        if (verbose) putchar('\n');
        if (rc) {
            fprintf(stderr, "nf: unreadable metadata value for '%s'\n", key);
            free(key);
            goto err;
        }
        if (strcmp(key, "general.alignment") == 0 && uval)
            g->alignment = (uint32_t)uval;
        free(key);
    }

    /* --- tensor directory --- */
    g->tensors = calloc(g->n_tensors ? (size_t)g->n_tensors : 1,
                        sizeof(nf_tensor_info));
    if (!g->tensors) goto corrupt;

    for (uint64_t i = 0; i < g->n_tensors; i++) {
        nf_tensor_info *t = &g->tensors[i];
        t->name = rd_str(f, 4096);
        if (!t->name || rd_u32(f, &t->n_dims)) goto corrupt;
        if (t->n_dims == 0 || t->n_dims > NF_GGUF_MAX_DIMS) goto corrupt;

        t->n_elements = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (rd_u64(f, &t->dims[d])) goto corrupt;
            t->n_elements *= t->dims[d];
        }
        if (rd_u32(f, &t->type) || rd_u64(f, &t->offset)) goto corrupt;

        const int be = nf_ggml_type_block_elems(t->type);
        const int bb = nf_ggml_type_block_bytes(t->type);
        if (be > 0 && t->n_elements % (uint64_t)be == 0)
            t->n_bytes = t->n_elements / (uint64_t)be * (uint64_t)bb;
    }

    /* The data section starts at the first aligned offset after the
     * directory. */
    const uint64_t here = (uint64_t)nf_ftell(f);
    g->data_offset = (here + g->alignment - 1) / g->alignment * g->alignment;

    /* Validation: every tensor must fit inside the file. This is the
     * check that prevents out-of-bounds reads on corrupt or truncated
     * files. */
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const nf_tensor_info *t = &g->tensors[i];
        if (t->n_bytes &&
            (g->data_offset + t->offset + t->n_bytes > g->file_size)) {
            fprintf(stderr, "nf: tensor '%s' outside the file (corrupt?)\n",
                    t->name);
            goto err;
        }
    }

    fclose(f);
    return 0;

corrupt:
    fprintf(stderr, "nf: corrupt or truncated GGUF file: %s\n", path);
err:
    fclose(f);
    nf_gguf_free(g);
    return -1;
}

void nf_gguf_free(nf_gguf *g) {
    if (g->tensors) {
        for (uint64_t i = 0; i < g->n_tensors; i++) free(g->tensors[i].name);
        free(g->tensors);
    }
    memset(g, 0, sizeof(*g));
}

/* ---- targeted metadata extraction --------------------------------------- */

/* Opens the file, skips the header and scans the kv pairs until it
 * finds `key`; leaves the file positioned at the start of the value
 * and deposits the type in *vtype. NULL if the key doesn't exist or
 * the file is malformed. */
static FILE *kv_seek(const char *path, const char *key, uint32_t *vtype) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    char magic[4];
    uint32_t version;
    uint64_t n_tensors, n_kv;
    if (rd(f, magic, 4) || memcmp(magic, "GGUF", 4) != 0 ||
        rd_u32(f, &version) || rd_u64(f, &n_tensors) || rd_u64(f, &n_kv)) {
        fclose(f);
        return NULL;
    }
    for (uint64_t i = 0; i < n_kv; i++) {
        char *k = rd_str(f, 4096);
        if (!k || rd_u32(f, vtype)) { free(k); fclose(f); return NULL; }
        const int found = strcmp(k, key) == 0;
        free(k);
        if (found) return f;
        if (handle_value(f, *vtype, 0, 0, NULL)) { fclose(f); return NULL; }
    }
    fclose(f);
    return NULL;
}

char **nf_gguf_read_str_array(const char *path, const char *key,
                              uint64_t *count) {
    *count = 0;
    uint32_t vtype;
    FILE *f = kv_seek(path, key, &vtype);
    if (!f) return NULL;

    uint32_t etype;
    uint64_t n;
    if (vtype != GGUF_ARR || rd_u32(f, &etype) || etype != GGUF_STR ||
        rd_u64(f, &n) || n > (1u << 24)) {
        fclose(f);
        return NULL;
    }
    char **arr = calloc(n ? (size_t)n : 1, sizeof(char *));
    if (!arr) { fclose(f); return NULL; }
    for (uint64_t i = 0; i < n; i++) {
        arr[i] = rd_str(f, 1u << 20);
        if (!arr[i]) { nf_str_array_free(arr, i); fclose(f); return NULL; }
    }
    fclose(f);
    *count = n;
    return arr;
}

void nf_str_array_free(char **arr, uint64_t count) {
    if (!arr) return;
    for (uint64_t i = 0; i < count; i++) free(arr[i]);
    free(arr);
}

int32_t *nf_gguf_read_i32_array(const char *path, const char *key,
                                uint64_t *count) {
    *count = 0;
    uint32_t vtype;
    FILE *f = kv_seek(path, key, &vtype);
    if (!f) return NULL;

    uint32_t etype;
    uint64_t n;
    /* i32 or u32: GGUF writers aren't consistent on this field */
    if (vtype != GGUF_ARR || rd_u32(f, &etype) ||
        (etype != GGUF_I32 && etype != GGUF_U32) ||
        rd_u64(f, &n) || n > (1u << 24)) {
        fclose(f);
        return NULL;
    }
    int32_t *arr = malloc(n ? (size_t)n * 4 : 4);
    if (!arr || rd(f, arr, (size_t)n * 4)) {
        free(arr);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *count = n;
    return arr;
}

int nf_gguf_read_kv_u64(const char *path, const char *key, uint64_t *out) {
    uint32_t vtype;
    FILE *f = kv_seek(path, key, &vtype);
    if (!f) return -1;
    *out = 0;
    const int rc = handle_value(f, vtype, 0, 0, out);
    fclose(f);
    return rc;
}

char *nf_gguf_read_kv_str(const char *path, const char *key) {
    uint32_t vtype;
    FILE *f = kv_seek(path, key, &vtype);
    if (!f) return NULL;
    char *s = vtype == GGUF_STR ? rd_str(f, 64u * 1024 * 1024) : NULL;
    fclose(f);
    return s;
}

int nf_gguf_read_kv_f32(const char *path, const char *key, float *out) {
    uint32_t vtype;
    FILE *f = kv_seek(path, key, &vtype);
    if (!f) return -1;
    int rc = -1;
    if (vtype == GGUF_F32) {
        rc = rd(f, out, 4);
    } else if (vtype == GGUF_F64) {
        double d;
        if (!rd(f, &d, 8)) { *out = (float)d; rc = 0; }
    }
    fclose(f);
    return rc;
}

const nf_tensor_info *nf_gguf_find_tensor(const nf_gguf *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    return NULL;
}
