/* nf_gguf.h — public API of the GGUF reader (milestone M0).
 *
 * GGUF is a self-describing container: header, key-value metadata,
 * tensor directory, then aligned tensor data. This module parses the
 * header/metadata/directory and computes tensor offsets and sizes, but
 * does NOT read the data: that comes in later milestones.
 */
#ifndef NF_GGUF_H
#define NF_GGUF_H

#include <stdint.h>

#define NF_GGUF_MAX_DIMS 4

typedef struct {
    char    *name;                   /* owned, freed by nf_gguf_free */
    uint32_t n_dims;
    uint64_t dims[NF_GGUF_MAX_DIMS]; /* dims[0] = fastest-varying dimension */
    uint32_t type;                   /* ggml type id (F32=0, F16=1, Q4_K=12, ...) */
    uint64_t offset;                 /* relative to the start of the data section */
    uint64_t n_elements;
    uint64_t n_bytes;                /* 0 if the type is unknown */
} nf_tensor_info;

typedef struct {
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    uint32_t alignment;   /* from general.alignment in the metadata, default 32 */
    uint64_t data_offset; /* absolute file offset of the data section */
    uint64_t file_size;
    nf_tensor_info *tensors;
} nf_gguf;

/* Opens and parses the header + metadata + tensor directory. Does not
 * load the data. verbose != 0 prints the metadata as it's encountered.
 * Returns 0 if ok, -1 on error (message on stderr). */
int nf_gguf_open(nf_gguf *g, const char *path, int verbose);
void nf_gguf_free(nf_gguf *g);

const char *nf_ggml_type_name(uint32_t type);
/* Elements per block / bytes per block of the quantized type; 0 if unknown. */
int nf_ggml_type_block_elems(uint32_t type);
int nf_ggml_type_block_bytes(uint32_t type);

/* Targeted extraction of individual metadata entries (used by the
 * tokenizer, M1). They reopen the file and scan the kv pairs up to the
 * key: simple and stateless, perfectly fine for one-off loads like the
 * vocabulary. */
char **nf_gguf_read_str_array(const char *path, const char *key, uint64_t *count);
void   nf_str_array_free(char **arr, uint64_t count);
int    nf_gguf_read_kv_u64(const char *path, const char *key, uint64_t *out);
int    nf_gguf_read_kv_f32(const char *path, const char *key, float *out);
char  *nf_gguf_read_kv_str(const char *path, const char *key);
/* Array of i32 (used for tokenizer.ggml.token_type — M6, special token
 * recognition). Caller-owned, must be freed with free(). */
int32_t *nf_gguf_read_i32_array(const char *path, const char *key, uint64_t *count);

/* Looks up a tensor by name in the directory. NULL if absent. */
const nf_tensor_info *nf_gguf_find_tensor(const nf_gguf *g, const char *name);

#endif
