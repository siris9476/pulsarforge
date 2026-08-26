/* nf_tokenizer.h — M1: byte-level (GPT-2-style) BPE tokenizer for Qwen3.
 *
 * Loads vocabulary and merges from the GGUF metadata (tokenizer.ggml.tokens /
 * tokenizer.ggml.merges) and replicates the HuggingFace tokenizer pipeline:
 * pre-tokenization (qwen2 regex) -> byte-encoding -> BPE merge -> id.
 */
#ifndef NF_TOKENIZER_H
#define NF_TOKENIZER_H

#include <stdint.h>

typedef struct nf_tokenizer nf_tokenizer;

nf_tokenizer *nf_tokenizer_load(const char *gguf_path);
void nf_tokenizer_free(nf_tokenizer *t);

uint64_t nf_tokenizer_vocab_size(const nf_tokenizer *t);

/* Tokenizes plain text (no special token added, the same way the Qwen3
 * HF tokenizer handles a bare string). Writes up to out_cap ids
 * into out; returns the total number of tokens, or -1 on error. */
int64_t nf_tokenize(nf_tokenizer *t, const char *text,
                    int32_t *out, int64_t out_cap);

/* Decodes an id into its raw text (real bytes, not byte-encoded).
 * Returns the bytes written into buf (no trailing NUL), or -1 if id is invalid. */
int nf_token_decode(const nf_tokenizer *t, int32_t id, char *buf, int cap);

#endif
