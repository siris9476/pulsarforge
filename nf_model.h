/* nf_model.h — M2: fp32 forward pass for Qwen3.
 *
 * Loads weights from the GGUF (dequantized to fp32 once at load time:
 * didactic simplicity before memory efficiency — keeping them quantized
 * and dequantizing inside the dot product is M4's job) and runs the
 * full transformer forward pass for a token sequence.
 */
#ifndef NF_MODEL_H
#define NF_MODEL_H

#include <stdint.h>
#include <stddef.h>   /* size_t (POSIX port) */

typedef struct nf_model nf_model;

nf_model *nf_model_load(const char *gguf_path);
void nf_model_free(nf_model *m);

uint64_t nf_model_vocab(const nf_model *m);
int nf_model_n_embd(const nf_model *m);
int nf_model_is_dsk(const nf_model *m);
/* Which row-based kernel (dsk_eval_batched_rows vs
 * qwen_eval_batched_rows) the scheduler must dispatch for this model —
 * see nf_model_kind_of in nf_model.c for the detail. */
typedef enum { NF_MODEL_PLAIN, NF_MODEL_DSK, NF_MODEL_QWEN_ROWS } nf_model_kind;
nf_model_kind nf_model_kind_of(const nf_model *m);
void nf_debug_vecdot(nf_model *m);

/* Runs the forward pass over the whole sequence from scratch (no cache)
 * and returns the logits of the last position (n_vocab floats, owned by
 * the model, valid until the next call). NULL on error.
 * Correctness reference (M2): redoing everything from scratch by
 * construction can't have cache-indexing bugs. Used by M3 as an
 * internal oracle to validate nf_session_eval. */
const float *nf_model_forward(nf_model *m, const int32_t *tokens, int n_tokens);

/* RAW hidden state (no normalization — a retrieval choice
 * left to the caller) of the LAST token in the sequence after layers
 * 0..layer inclusive (0-indexed). Stops the forward pass right after
 * `layer`, saving the cost of the later layers. `out` is caller-owned
 * (n_embd floats). Returns -1 if layer is out of range [0, n_layers) —
 * no default: only layer 15 on the 0.6B model has been validated. */
int nf_model_embed(nf_model *m, const int32_t *tokens, int n_tokens,
                   int layer, float *out);

/* M3: a session holds the persistent KV cache (K/V per layer, up to
 * max_ctx positions) across multiple calls — this is what makes
 * generation O(1) per token instead of O(n) like redoing the full
 * forward pass every time. */
typedef struct nf_session nf_session;

nf_session *nf_session_create(nf_model *m, int max_ctx);
void nf_session_free(nf_session *s);
int nf_session_n_past(const nf_session *s);
/* M8 Phase 3b: the current logits (n_vocab floats, owned by the
 * session) — used by the server scheduler to re-read them after a
 * grouped tick (dsk_eval_batched_sessions) without exposing the whole
 * struct, which stays opaque outside nf_model.c. */
const float *nf_session_logits(const nf_session *s);
/* M8 Phase 3b (continuous batching): decode ONE token per session, for
 * n_sess INDEPENDENT sessions together in a single pass over the
 * weights — shares expert weight reads across sessions when the router
 * makes them converge on the same expert in the same tick. Each session
 * receives its own new token (next_tok[t]) and its own updated logits
 * (readable with nf_session_logits); each session's n_past is already
 * incremented on return. 0 = ok, -1 = context full for some session or
 * malloc failed (no session was touched in that case). Valid only for
 * deepseek2 models (is_dsk); validated in isolation with
 * `nf debugcrosssess` (Phase 3a): bit-exact against itself
 * called one session at a time, PPL -0.36% vs single decode (benign
 * delta, already in the same order of magnitude as other accepted
 * optimizations). */
int dsk_eval_batched_sessions(nf_session **sess, int n_sess,
                              const int32_t *next_tok);
/* MIXED ubatch: one row = one (session, position) pair, so
 * a new request's prefill chunk travels in the SAME round as other
 * sessions' decodes instead of blocking them for its whole duration.
 * sess[r]/tok[r]/pos[r] describe the row; want_lg[r]=1 if its logits are
 * needed (a prefill's intermediate rows don't: the head is only
 * computed for marked rows). Rows of the same session must be passed in
 * increasing pos order. */
int dsk_eval_batched_rows(nf_session **sess, const int32_t *tok,
                          const int *pos, const int *want_lg, int n_rows,
                          float *out_lg);
/* Same contract as dsk_eval_batched_rows, but for Qwen3
 * (dense or qwen3moe) — standard GQA attention instead of MLA. See
 * nf_model_kind_of for the dispatch. */
int qwen_eval_batched_rows(nf_session **sess, const int32_t *tok,
                          const int *pos, const int *want_lg, int n_rows,
                          float *out_lg);
/* PPL validation of the row-based path: evaluates n_new tokens in
 * chunks, MIXED with `decoy`'s decode (if non-NULL) — so PPL measures
 * the real path, mixed batches included, not a clean chunking that
 * doesn't exist in production. Writes the logits of all n_new
 * positions. */
int nf_ppl_eval_rows(nf_session *s, nf_session *decoy,
                     const int32_t *tokens, int n_new, int chunk,
                     float *all_logits);
/* KV cache RAM for a session at max_ctx (for declaring a budget) */
size_t nf_session_kv_bytes(const nf_model *m, int max_ctx);
/* Context shifting (K-shift RoPE): discards n_discard positions right
 * after the first n_keep, re-rotating the remaining Ks to compensate
 * for the shift (V just translates). Returns the n_discard actually
 * applied (0 if there's nothing to discard — never a silent partial
 * effect). */
int nf_session_shift(nf_session *s, int n_keep, int n_discard);

/* Extends the session with n_new new tokens (from the current position
 * onward) and returns the logits of the last of these (n_vocab floats,
 * owned by the session, valid until the next call). NULL on error
 * (e.g. context exhausted). */
const float *nf_session_eval(nf_session *s, const int32_t *tokens, int n_new);

/* Like nf_session_eval, but writes into all_logits the logits of EVERY
 * position in the batch (n_new x n_vocab) — for speculative decoding
 * (verifying k proposed tokens in a single pass over the weights). */
const float *nf_session_eval_all(nf_session *s, const int32_t *tokens,
                                 int n_new, float *all_logits);
/* Rewinds the session to n_past positions (rollback after a speculative
 * rejection: the K/V rows beyond that point remain and will be
 * overwritten by the next eval). */
void nf_session_rewind(nf_session *s, int n_past);

/* ---- draft-verify MTP glm-dsa ------------------------------
 * LAYER-MAJOR verify window: n_new in {2..4} positions processed layer
 * by layer with ONE fetch per layer of the UNION of the positions'
 * experts and per-expert decoding amortized over all the activations
 * that chose it (see glm_eval_verify in nf_model.c). all_logits
 * (n_new x n_vocab) is mandatory: the caller verifies every position.
 * glm-dsa models only. NULL on error. */
const float *nf_glm_eval_verify(nf_session *s, const int32_t *tokens,
                                int n_new, float *all_logits);
/* Phase B workbench sweep: ONE token for each of n_streams,
 * layer-major order — requires NF_GLM_MLA_ABSORB=1, streaming off
 * (Phase E), consistent n_past per stream. logits_out: n_streams x
 * n_vocab. 0 = ok. See the comment on nf_glm_eval_sweep in nf_model.c. */
int nf_glm_eval_sweep(nf_session **ss, int n_streams, const int32_t *tok,
                      float *logits_out);
/* ---- batched prefill glm-dsa -------------------------------
 * Generalized sibling of nf_glm_eval_verify for the ordinary PREFILL of
 * a long prompt: same layer-major + expert-union scheme, but without
 * the n_new in {2..4} ceiling (arbitrary NV, driven by a chunk chosen by
 * the caller — see NF_GLM_PREFILL_CHUNK in nf_model.c) and with
 * all_logits OPTIONAL: NULL computes only the logits of the last
 * position (same contract as nf_session_eval for the same n_new),
 * non-NULL computes every position (same contract as
 * nf_glm_eval_verify). Used internally by nf_session_eval/_eval_all
 * when NF_GLM_PREFILL_BATCH=1 and n_new>1 (dispatched in sequential
 * chunks); also exposed for the cmd_debugglmprefillbatch debug command.
 * glm-dsa models only. NULL on error. */
const float *nf_glm_eval_prefill_batched(nf_session *s, const int32_t *tokens,
                                         int n_new, float *all_logits);
/* Registers (buf!=NULL) or clears (NULL) the buffer where glm-dsa eval
 * paths expose the hidden state post-output_norm: with all_logits at
 * t*n_embd for every position in the batch, without only the last
 * position at offset 0. The buffer is caller-owned (size it for the
 * largest batch). Default NULL: no new behavior for existing callers. */
void nf_session_set_hidden_out(nf_session *s, float *buf);
/* Draft from the MTP head (nextn) for live generation: S pairs (h post-
 * output_norm at position t, real token at t+1), draft logits for the
 * position after the last pair in out_logits (n_vocab floats — ONLY the
 * last position, unlike the debug path). -1 if the model doesn't have
 * MTP loaded (needs NF_GLM_MTP=1). */
int nf_mtp_draft_glm(nf_model *m, const float *h_states,
                     const int32_t *next_toks, int S, float *out_logits);
/* Layer-skip drafter (--layerskip-draft): APPROXIMATE
 * forward pass of ONE new position (tok, at the session's current
 * s->n_past) with a calibrated set of MoE layers skipped entirely —
 * see the comment on glm_eval_draft_skip in nf_model.c for the KV
 * isolation argument. Does NOT touch the session (s->n_past/K/V/Gkv/
 * Ixk/logits/hidden_out remain bit-for-bit what they were before the
 * call) — never advanced, never written. out_logits (n_vocab floats,
 * caller-owned) receives the draft's logits for the position after tok.
 * Returns 0/-1 (never silent). glm-dsa only. */
int nf_glm_draft_skip(nf_session *s, int32_t tok, float *out_logits);
int nf_model_is_glm(const nf_model *m);
int nf_model_has_mtp(const nf_model *m);

/* Prefix graft: copies KV rows [0, n) from src to dst and
 * positions dst at n_past = n — bit-exact (same bytes as prefilling
 * those positions). Same model and same max_ctx required.
 * 0 = ok, -1 = incompatible sessions. */
int nf_session_copy_prefix(nf_session *dst, const nf_session *src, int n);

/* DEBUG (M4 validation): dequantizes row `row` of blk.<layer>.attn_q
 * into out (n_embd floats). Not part of the stable API — only used to
 * compare our dequantizer against an external reference. */
int nf_debug_dequant_wq_row(nf_model *m, int layer, uint64_t row, float *out);
/* Same as above, but for wv (Q6_K in Qwen3-0.6B-Q4_K_M) — out of n_embd floats. */
int nf_debug_dequant_wv_row(nf_model *m, int layer, uint64_t row, float *out);

/* DEBUG (MoE validation): isolates layer 0's router+experts on a single
 * token, without going through attention/other layers — reuses moe_ffn()
 * exactly. Writes xb (post-ffn_norm input) and out (weighted result) to
 * file, so an independent Python reference (real weights dequantized
 * with gguf.quants) can start from the SAME xb and compare the output,
 * isolating the test to just the NEW part of the code (the rest —
 * attention, RMSNorm, RoPE — is already validated on the dense path,
 * unchanged here). Returns -1 if the model isn't MoE. */
int nf_debug_moe(nf_model *m, uint64_t token, const char *xb_path,
                 const char *out_path);

/* DEBUG (profiling): prints the time accumulators (attention/router/
 * experts) to stderr if NF_DEBUG_TIMING is set, otherwise does nothing.
 * Must be called at the end of a generate/bench run. */
void nf_debug_timing_report(void);
void nf_debug_expert_hotness_report(nf_model *m);

/* DEBUG (glm-dsa MoE routing characterization, measurement only): prints
 * to NF_GLM_ROUTE_PROFILE (path, or "-" for stderr) the mass
 * concentration of mw[0..K) (top1/top2/top3) and the midx[0..K) overlap
 * between adjacent positions, per MoE layer — no-op if the variable
 * isn't set. Must be called at the end of a generate/bench/serve run,
 * same call site as nf_debug_timing_report(). */
void nf_route_profile_report(void);

/* DEBUG (MTP, "amortization via MTP"): offline
 * accuracy report for the MTP head (nextn) against a REAL trace written
 * by NF_GLM_MTP_TRACE during a generation (nf generate/serve). Compares
 * the MTP head's draft (top-1/top-5 token + overlap of the internally
 * chosen expert set) against what the full model ACTUALLY generated/
 * routed two positions later. -1 if the model has no MTP or the trace
 * is inconsistent/too short. */
int nf_debug_mtp_report(nf_model *m, const char *trace_path);

/* DEBUG (MTP, validation against the HYBRID HF oracle — see the comment
 * on the function in nf_model.c): compares nf_mtp_predict_glm against
 * the REAL logits computed by an actual GlmMoeDsaDecoderLayer/
 * GlmMoeDsaRMSNorm (transformers doesn't implement MTP natively, so the
 * only "ours" part is the enorm/hnorm/eh_proj wiring) on
 * tools/glm_tiny_mtp.gguf. -1 on error. */
int nf_debug_mtp_oracle(nf_model *m, const char *trace_path, const char *ref_path);

/* DEBUG (measurement, not an implementation): isolates the cost of the
 * horizontal reduction (hsum) of ONE superblock relative to the rest of
 * the full dot product, to estimate the theoretical ceiling of "register
 * blocking" (SIMD polish backlog item) not yet implemented, before
 * writing it. */
void nf_debug_simd_bench(nf_model *m);

/* DEBUG (wide kernel, gate 0): correctness of the v2
 * repack + wide-register kernel against the flat path on a group of 8
 * real rows (nf debugpack), and A/B streaming at the kernel level on a
 * whole tensor (nf debugwide). Tensor selection via
 * NF_DEBUG_PACK_LAYER / NF_DEBUG_PACK_TENSOR / NF_DEBUG_PACK_GROUP. */
/* on-disk expert cache: builds <gguf>.expw (all experts repacked wide,
 * mmapped from load time — zero warmup on MoE). The model must be
 * loaded with NF_NO_EXPW=1 before calling this. */
int nf_expw_build(nf_model *m, const char *gguf_path);
/* sidecar of the repacked DENSE weights: builds <gguf>.dpk (all the wide-
 * kernel buffers for dense/attention tensors, mmapped from load time —
 * the repack, ~46s of I/O on GLM-5.2, never has to be redone). The model
 * must be loaded with NF_NO_DPK=1 before calling this (the builder
 * rewrites the same file the load would mmap). */
int nf_dpk_build(nf_model *m, const char *gguf_path);
/* byte-for-byte identity check between the mmapped sidecar and the
 * repack recomputed in memory. Requires a model loaded WITH the
 * sidecar. */
int nf_dpk_verify(nf_model *m);
int nf_convert_iq4xs_model(nf_model *m, const char *manifest_path,
                           const char *blob_path);

void nf_debug_pack(nf_model *m);
void nf_debug_wide_bench(nf_model *m);
void nf_debug_gemm_bench(nf_model *m);
void nf_debug_dsk_down_bench(nf_model *m);
void nf_debug_q6k(nf_model *m);
void nf_debug_q2k(nf_model *m);
void nf_debug_iq4xs(nf_model *m);
/* Generic "N rows of a real tensor, dequantized" dump for the 5 types
 * Q2_K/Q3_K/IQ2_XXS/IQ3_XXS/IQ1_S (real GLM-5.2) — UNLIKE
 * the nf_debug_* above this does NOT require a loaded nf_model: it opens
 * the GGUF on its own and reads rows with a direct fread (no mmap, no
 * model/architecture semantics). Writes <out_prefix>_q.bin (raw bytes)
 * and <out_prefix>_x.f32 (our dequant) for comparison against the
 * gguf.quants.dequantize Python oracle. Returns 0 if ok. */
int nf_debug_dequant5(const char *gguf_path, const char *tensor_name,
                      uint64_t row0, int nrows, const char *out_prefix);
/* Comparison of fused AVX2 vecdot vs scalar dequant+dotf for IQ1_S/
 * IQ2_XXS/IQ3_XXS on nrows REAL tensor rows (same standalone scheme as
 * nf_debug_dequant5: no model load). Returns 0 if the max relative error
 * per row stays below threshold (1e-4). */
int nf_debug_iqdot(const char *gguf_path, const char *tensor_name, int nrows);
void nf_debug_scale1_bench(nf_model *m);
void nf_debug_nogather_bench(nf_model *m);
void nf_debug_headbatch_bench(nf_model *m);
void nf_debug_tailwide_bench(nf_model *m);
void nf_debug_downq6k_bench(nf_model *m);
void nf_debug_prefetch_bench(nf_model *m);
void nf_debug_crosssess_bench(nf_model *m);
void nf_debug_crosssess_qwen_bench(nf_model *m);
void nf_debug_batchscale_bench(nf_model *m);
void nf_debug_batchscale_qwen_bench(nf_model *m);
void nf_debug_moecost_qwen_bench(nf_model *m);
void nf_debug_ompcost_bench(void);
#ifdef _WIN32
void nf_debug_expwio_bench(nf_model *m);
/* Explicit glm-dsa expert streaming: byte-identity of
 * n_pairs deterministic (layer, expert) pairs, streaming vs mmap
 * (expert_slice) — requires NF_GLM_STREAM_GB>0. 0 = all identical. */
int nf_debug_glm_stream_check(nf_model *m, int n_pairs);
#endif
/* Bench of the IQ1_S expert kernel in two regimes (HOT: one matrix
 * reused, the ALU ceiling; COLD: pool beyond L3, decode's real regime)
 * — point 0 of the "CPU kernel" lead, reopened by the
 * profile: the gap between the two regimes tells whether it's worth
 * optimizing the kernel or how the bytes reach it. Synthetic data, no
 * model. */
void nf_debug_kernel_bench(int iters);
void nf_debug_twinscan(nf_model *m, int layer);
void nf_debug_sweepbench(nf_model *m, int N);
int nf_debug_forgekern(const char *path);
void nf_debug_mixedbatch_bench(nf_model *m);
void nf_debug_mixedbatch_qwen_bench(nf_model *m);
/* decode-attention bench on synthetic cache (Q8 KV investigation):
 * thread x {f16, q8} sweep of the GQA loop at np=ctx, no prefill */
void nf_debug_attn_bench(nf_model *m, int ctx, int iters);
/* context-shifting bench: shift vs direct recompute of the equivalent
 * sequence. real_toks/n_real_toks: real, already-tokenized text
 * (recommended); NULL = random out-of-distribution ids (fast bench, but
 * amplifies drift across layers). */
void nf_debug_shift_bench(nf_model *m, const int32_t *real_toks,
                         int n_real_toks, int n_keep, int n_discard,
                         int n_tail);

/* long-memory bench (Phase 1, RAM-only): Check 1 (bit-exact below
 * threshold, short seq_below) + Check 2 (needle-in-haystack, seq_far
 * above threshold with a distant fact, seq_near control with a nearby
 * fact) — see the comment on the function in nf_model.c. */
void nf_debug_longmem_bench(nf_model *m,
                            const int32_t *seq_below, int n_below,
                            const int32_t *seq_far, int n_far,
                            const int32_t *seq_near, int n_near,
                            int block, int sink, int window, int topk);

/* Phase 2a (disk-backing): ALWAYS bit-exact (not just below threshold)
 * between pure NF_LONGMEM and NF_LONGMEM+NF_LONGMEM_DISK on the SAME
 * sequence, one token at a time — see the comment on the function in
 * nf_model.c. */
void nf_debug_longmemdisk_bench(nf_model *m, const int32_t *seq, int n,
                                int block, int sink, int window, int topk);

/* Phase 2a: recall (not bit-exactness, shown to be unreachable in
 * general) — DISK against the OFF-oracle on a distant fact, control
 * with the same nearby fact. See the comment on the function in
 * nf_model.c. */
void nf_debug_longmemdisk_recall_bench(nf_model *m,
                                       const int32_t *seq_far, int n_far,
                                       const int32_t *seq_near, int n_near,
                                       int block, int sink, int window, int topk);

/* M6: the KV cache as a first-class citizen of disk (the DwarfStar
 * lesson). Saves the session's state (the n_past positions already
 * computed, K/V for every layer) together with the tokens that produced
 * it, so a long conversation can be resumed without redoing the prefill
 * from scratch. Machine-native binary format (not portable across
 * architectures — a deliberate choice, it's a disposable local cache,
 * not an exchange format). */
int nf_session_save(const nf_session *s, const int32_t *tokens, const char *path);

/* Loads a saved session: validates that n_layers/kv_dim match the
 * current model, creates a session with max_ctx capacity (must be
 * >= the saved n_past), restores K/V and n_past, writes the tokens into
 * tokens_out (up to tokens_cap) and their count into *n_tokens_out.
 * NULL on error (incompatible version, corrupt file, model mismatch). */
nf_session *nf_session_load(nf_model *m, const char *path, int max_ctx,
                            int32_t *tokens_out, int tokens_cap,
                            int *n_tokens_out);

#endif
