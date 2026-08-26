/* nf_tokenizer.c — M1: byte-level (GPT-2-style) BPE tokenizer for Qwen3.
 *
 * The pipeline replicates Qwen3's HuggingFace tokenizer:
 *
 *   1. PRE-TOKENIZATION: the text is split into "pieces" by a regex
 *      (the "qwen2" pattern). This is the step that decides, for
 *      example, that a space stays attached to the word that follows
 *      (" hello" is a single piece) and that every digit is its own
 *      piece.
 *   2. BYTE-ENCODING: every byte of the piece is mapped to a printable
 *      codepoint (GPT-2's bytes_to_unicode map). This is needed
 *      because the BPE vocabulary is stored in this alphabet: the
 *      space, for instance, appears in the vocabulary as 'Ġ' (U+0120).
 *   3. BPE MERGE: the piece starts as a sequence of one-codepoint
 *      symbols; the adjacent pair with the lowest merge rank gets
 *      fused, repeatedly, until no pair is in the merge dictionary.
 *      The final symbols are looked up in the vocabulary.
 *
 * Declared limit: the Unicode classifier (letter/number) covers Latin,
 * Greek, Cyrillic, kana, and CJK — enough for Latin-script languages.
 * If golden tests on other scripts fail, this is the first place to
 * extend.
 */
#include "nf_tokenizer.h"
#include "nf_gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

/* ---- UTF-8 -------------------------------------------------------------- */

static int utf8_decode(const unsigned char *s, uint32_t *cp) {
    if (s[0] < 0x80) { *cp = s[0]; return 1; }
    if ((s[0] & 0xE0) == 0xC0 && (s[1] & 0xC0) == 0x80) {
        *cp = ((uint32_t)(s[0] & 0x1F) << 6) | (s[1] & 0x3F);
        return 2;
    }
    if ((s[0] & 0xF0) == 0xE0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
        *cp = ((uint32_t)(s[0] & 0x0F) << 12) |
              ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        return 3;
    }
    if ((s[0] & 0xF8) == 0xF0 && (s[1] & 0xC0) == 0x80 &&
        (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) {
        *cp = ((uint32_t)(s[0] & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12) |
              ((uint32_t)(s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        return 4;
    }
    *cp = 0xFFFD;
    return 1;
}

static int utf8_encode(uint32_t cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
}

/* ---- codepoint classification for the pre-tokenization regex ----------- */

static int is_letter(uint32_t c) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return 1;
    if (c == 0xAA || c == 0xB5 || c == 0xBA) return 1;          /* ª µ º */
    if (c >= 0xC0 && c <= 0xFF && c != 0xD7 && c != 0xF7) return 1;
    if (c >= 0x100 && c <= 0x24F) return 1;                      /* Latin Extended */
    if (c >= 0x370 && c <= 0x3FF && c != 0x374 && c != 0x375 &&
        c != 0x37E && c != 0x384 && c != 0x385 && c != 0x387) return 1;
    if (c >= 0x400 && c <= 0x4FF) return 1;                      /* Cyrillic */
    if (c >= 0x3040 && c <= 0x30FF && c != 0x3097 && c != 0x3098) return 1;
    if (c >= 0x4E00 && c <= 0x9FFF) return 1;                    /* CJK */
    return 0;
}

static int is_number(uint32_t c) {
    if (c >= '0' && c <= '9') return 1;
    if (c == 0xB2 || c == 0xB3 || c == 0xB9) return 1;           /* ² ³ ¹ */
    if (c >= 0xBC && c <= 0xBE) return 1;                        /* ¼ ½ ¾ */
    if (c >= 0x660 && c <= 0x669) return 1;                      /* Arabic */
    if (c >= 0xFF10 && c <= 0xFF19) return 1;                    /* fullwidth */
    return 0;
}

static int is_ws(uint32_t c) {
    return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 ||
           c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

static unsigned char ascii_lower(unsigned char c) {
    return c >= 'A' && c <= 'Z' ? (unsigned char)(c + 32) : c;
}

/* ---- string -> i32 hash map (open addressing, FNV-1a) ------------------- */

typedef struct {
    const char **keys; /* NOT owned by the map */
    int32_t *vals;
    size_t cap;        /* power of 2 */
} strmap;

static uint64_t fnv1a(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211ULL; }
    return h;
}

static int strmap_init(strmap *m, size_t n_items) {
    m->cap = 64;
    while (m->cap < n_items * 2) m->cap <<= 1;
    m->keys = calloc(m->cap, sizeof(char *));
    m->vals = malloc(m->cap * sizeof(int32_t));
    return m->keys && m->vals ? 0 : -1;
}

static void strmap_put(strmap *m, const char *k, int32_t v) {
    size_t i = fnv1a(k) & (m->cap - 1);
    while (m->keys[i]) {
        if (strcmp(m->keys[i], k) == 0) { m->vals[i] = v; return; }
        i = (i + 1) & (m->cap - 1);
    }
    m->keys[i] = k;
    m->vals[i] = v;
}

static int32_t strmap_get(const strmap *m, const char *k) {
    size_t i = fnv1a(k) & (m->cap - 1);
    while (m->keys[i]) {
        if (strcmp(m->keys[i], k) == 0) return m->vals[i];
        i = (i + 1) & (m->cap - 1);
    }
    return -1;
}

static void strmap_free(strmap *m) {
    free(m->keys);
    free(m->vals);
}

/* ---- tokenizer structure -------------------------------------------------- */

struct nf_tokenizer {
    char   **vocab;       /* id -> byte-encoded string (owned) */
    uint64_t n_vocab;
    char   **merges;      /* owned; ' ' separator replaced by \x01 */
    uint64_t n_merges;
    strmap   vocab_map;   /* byte-encoded string -> id */
    strmap   merge_map;   /* "left\x01right" -> rank */
    uint32_t byte_to_cp[256];
    int16_t  cp_to_byte[512]; /* -1 = unmapped */

    /* M6: special tokens (<|im_start|>, <|im_end|>, ...) — in the GGUF
     * these are the entries with token_type != 1 (NORMAL). They don't
     * go through BPE: they must be recognized by EXACT literal match
     * in the raw text, before regular pre-tokenization, or BPE would
     * split them into pieces (seen literally: '<|im_start|>' became
     * 6+ tokens instead of one). Sorted by decreasing length so the
     * first match is always the longest (no prefix ambiguity, e.g.
     * '<|im_' shared by several markers). */
    char   **specials;     /* owned, point at NON-byte-encoded copies */
    int32_t *special_ids;
    int      n_specials;

    /* deepseek2: pre-tokenization pattern and automatic BOS.
     * pre: 0 = "qwen2" (historical default), 1 = "deepseek-llm",
     * 2 = "olmo", 3 = "glm4" (identical to llama3, used by
     * glm-dsa/GLM-5.2). add_bos: the vocabulary wants BOS at the head
     * of the sequence (deepseek yes, qwen no) — nf_tokenize emits it,
     * ONCE. */
    int      pre;
    int      add_bos;
    int32_t  bos_id;
};

/* GPT-2's bytes_to_unicode map: "printable" bytes stay themselves, the
 * rest (spaces, control chars, ...) get moved starting at U+0100, in
 * order. This way every byte has a visible form and BPE can work on
 * clean strings. */
static void build_byte_tables(nf_tokenizer *t) {
    for (int i = 0; i < 512; i++) t->cp_to_byte[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        const int direct = (b >= 0x21 && b <= 0x7E) ||
                           (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        const uint32_t cp = direct ? (uint32_t)b : (uint32_t)(256 + n++);
        t->byte_to_cp[b] = cp;
        t->cp_to_byte[cp] = (int16_t)b;
    }
}

/* ---- pre-tokenization ("qwen2" pattern) ---------------------------------
 *
 * The original pattern (PCRE):
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)          English contractions
 *   |[^\r\n\p{L}\p{N}]?\p{L}+             (one non-letter prefix)? + letters
 *   |\p{N}                                ONE digit at a time
 *   | ?[^\s\p{L}\p{N}]+[\r\n]*            (space)? + punctuation + newlines
 *   |\s*[\r\n]+                           whitespace ending in a newline
 *   |\s+(?!\S)                            whitespace run except the last
 *   |\s+                                  leftover whitespace
 *
 * Rewritten here as a hand-written scanner: at every position, the
 * alternatives are tried in order, the way the regex engine would.
 * Returns the byte length of the next piece. */
static size_t pretok_next(const unsigned char *s, size_t n) {
    uint32_t c0;
    const int l0 = utf8_decode(s, &c0);

    /* 1) contractions: 's 't 're 've 'm 'll 'd (case-insensitive) */
    if (c0 == '\'' && n >= 2) {
        const unsigned char a = ascii_lower(s[1]);
        const unsigned char b = n >= 3 ? ascii_lower(s[2]) : 0;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
            (a == 'l' && b == 'l')) return 3;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    }

    /* 2) [^\r\n\p{L}\p{N}]?\p{L}+ — the alternative that keeps a space
     *    (or another single punctuation mark) attached to the word. */
    {
        size_t i = 0;
        uint32_t c = c0;
        int lc = l0;
        if (c != '\r' && c != '\n' && !is_letter(c) && !is_number(c)) {
            i = (size_t)lc;
            lc = i < n ? utf8_decode(s + i, &c) : 0;
        }
        if (lc && is_letter(c)) {
            while (i < n) {
                uint32_t d;
                const int ld = utf8_decode(s + i, &d);
                if (!is_letter(d)) break;
                i += (size_t)ld;
            }
            return i;
        }
    }

    /* 3) a single digit: long numbers become one token per digit */
    if (is_number(c0)) return (size_t)l0;

    /* 4) ' '? [^\s\p{L}\p{N}]+ [\r\n]* */
    {
        size_t i = c0 == ' ' ? 1 : 0;
        const size_t start = i;
        while (i < n) {
            uint32_t d;
            const int ld = utf8_decode(s + i, &d);
            if (is_ws(d) || is_letter(d) || is_number(d)) break;
            i += (size_t)ld;
        }
        if (i > start) {
            while (i < n && (s[i] == '\r' || s[i] == '\n')) i++;
            return i;
        }
    }

    /* 5-7) whitespace */
    if (is_ws(c0)) {
        size_t i = 0, prev = 0, last_nl_end = 0;
        while (i < n) {
            uint32_t d;
            const int ld = utf8_decode(s + i, &d);
            if (!is_ws(d)) break;
            prev = i;
            i += (size_t)ld;
            if (d == '\r' || d == '\n') last_nl_end = i;
        }
        /* \s*[\r\n]+: everything up to the last newline in the run */
        if (last_nl_end) return last_nl_end;
        /* \s+(?!\S): if the run touches the end of the text, take it
         * all; otherwise leave the last whitespace char to the next
         * piece (which will absorb it via alternative 2 or 4). */
        if (i == n) return i;
        if (i > (size_t)l0) return prev;
        /* \s+: a single whitespace char followed by something the
         * earlier alternatives didn't want (e.g. a space before a
         * digit). */
        return i;
    }

    return (size_t)l0; /* unknown codepoint: a piece on its own */
}

/* ---- pre-tokenization ("deepseek-llm" pattern) --------------------------
 *
 * llama.cpp applies SIX regexes in SEQUENCE (unicode_regex_split), not
 * as alternatives at the same position:
 *   1. [CR LF]                   every newline is its own piece
 *   2. \s?[non-CJK letters]+     optional space + run of letters
 *   3. \s?[punctuation]+         ASCII minus digits + fullwidth
 *   4. \s+$                      whitespace trailing the fragment
 *   5. [U+0800-9FA5 | Hangul]+   a run of "CJK and relatives"
 *   6. \p{N}+                    a run of digits (not one at a time!)
 *
 * The sequential semantics matter for runs of spaces: rule 2/3 (via
 * its \s?) only takes the LAST space of the run; the ones before it
 * stay in a fragment that ends right there — rule 4 turns them into
 * their own piece. Emulated here with lookahead. Declared limit (same
 * caveat as the classifier at the top of the file): scripts outside
 * the verification corpus (Georgian, Cherokee, ...) may diverge — the
 * verification gate measures on Italian/English/real samples. */

static int dsk_is_r5(uint32_t c) {          /* rule 5 */
    return (c >= 0x800 && c <= 0x9FA5) || (c >= 0xAC00 && c <= 0xD7BF);
}

static int dsk_is_letter(uint32_t c) {
    return is_letter(c) && !dsk_is_r5(c);
}

static int dsk_is_punct(uint32_t c) {       /* rule 3: no digits */
    return (c >= 0x21 && c <= 0x2F) || (c >= 0x3A && c <= 0x7E) ||
           (c >= 0xFF01 && c <= 0xFF0F) || (c >= 0xFF1A && c <= 0xFF5E) ||
           (c >= 0x2018 && c <= 0x201F) || (c >= 0x3000 && c <= 0x3002);
}

static size_t pretok_next_dsk(const unsigned char *s, size_t n) {
    uint32_t c0;
    const int l0 = utf8_decode(s, &c0);

    if (c0 == '\r' || c0 == '\n') return 1;                    /* rule 1 */

    if (is_ws(c0)) {
        /* a whitespace run (never across CR/LF: rule 1 already split
         * there — the run stops before it) */
        size_t i = 0, prev = 0;
        uint32_t d = c0;
        int ld = l0;
        while (ld && is_ws(d) && d != '\r' && d != '\n') {
            prev = i;
            i += (size_t)ld;
            ld = i < n ? utf8_decode(s + i, &d) : 0;
        }
        /* what follows the run decides who gets the last space */
        if (ld && d != '\r' && d != '\n' &&
            (dsk_is_letter(d) || dsk_is_punct(d))) {
            if (i == (size_t)l0) {
                /* a single space: rule 2/3 with \s? — space + run */
                const int letters = dsk_is_letter(d);
                size_t j = i;
                while (j < n) {
                    uint32_t e;
                    const int le = utf8_decode(s + j, &e);
                    if (letters ? !dsk_is_letter(e) : !dsk_is_punct(e)) break;
                    j += (size_t)le;
                }
                return j;
            }
            /* multiple spaces: the spaces minus the last one are their
             * own piece (the fragment ends where rule 2/3 took " run") */
            return prev;
        }
        /* end of text, CR/LF, digits, or CJK: the whole run is one
         * piece (rule 4 if trailing, an advanced fragment otherwise) */
        return i;
    }

    if (dsk_is_letter(c0)) {                                   /* rule 2 */
        size_t i = 0;
        while (i < n) {
            uint32_t d;
            const int ld = utf8_decode(s + i, &d);
            if (!dsk_is_letter(d)) break;
            i += (size_t)ld;
        }
        return i;
    }

    if (is_number(c0)) {                                       /* rule 6 */
        size_t i = 0;
        while (i < n) {
            uint32_t d;
            const int ld = utf8_decode(s + i, &d);
            if (!is_number(d)) break;
            i += (size_t)ld;
        }
        return i;
    }

    if (dsk_is_punct(c0)) {                                    /* rule 3 */
        size_t i = 0;
        while (i < n) {
            uint32_t d;
            const int ld = utf8_decode(s + i, &d);
            if (!dsk_is_punct(d)) break;
            i += (size_t)ld;
        }
        return i;
    }

    if (dsk_is_r5(c0)) {                                       /* rule 5 */
        size_t i = 0;
        while (i < n) {
            uint32_t d;
            const int ld = utf8_decode(s + i, &d);
            if (!dsk_is_r5(d)) break;
            i += (size_t)ld;
        }
        return i;
    }

    return (size_t)l0; /* codepoint outside every rule: a piece on its own */
}

/* ---- pre-tokenization ("olmo" pattern, classic GPT-2) --------------------
 *
 * The original pattern (PCRE), shared in llama.cpp by gpt2/olmo/mpt/
 * jais/trillion/granite-docling (llama-vocab.cpp, same case statement):
 *   's|'t|'re|'ve|'m|'ll|'d        English contractions, CASE-SENSITIVE
 *                                  (unlike qwen2: no (?i:...))
 *   | ?\p{L}+                      (space)? + a run of letters
 *   | ?\p{N}+                      (space)? + a run of DIGITS — one
 *                                  piece for the whole number, not one
 *                                  digit at a time like pretok_next
 *                                  (qwen2)
 *   | ?[^\s\p{L}\p{N}]+            (space)? + a run of punctuation
 *   |\s+(?!\S)                     whitespace: all of it if trailing
 *                                  the text, otherwise all but the
 *                                  last char (which ends up absorbed
 *                                  by the " ?" of the next piece) — no
 *                                  leftover \s+ alternative at the end
 *                                  (unlike qwen2): every isolated space
 *                                  is by construction followed by a
 *                                  letter/digit/punctuation (never by
 *                                  more whitespace, a run is maximal),
 *                                  so one of the three alternatives
 *                                  above always absorbs it. */
static size_t pretok_next_olmo(const unsigned char *s, size_t n) {
    uint32_t c0;
    const int l0 = utf8_decode(s, &c0);

    /* 1) case-SENSITIVE contractions: 's 't 're 've 'm 'll 'd */
    if (c0 == '\'' && n >= 2) {
        const unsigned char a = s[1];
        const unsigned char b = n >= 3 ? s[2] : 0;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
            (a == 'l' && b == 'l')) return 3;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    }

    /* 2) ' '?\p{L}+ */
    {
        size_t i = c0 == ' ' ? (size_t)l0 : 0;
        uint32_t c = c0; int lc = l0;
        if (i) lc = i < n ? utf8_decode(s + i, &c) : 0;
        if (lc && is_letter(c)) {
            while (i < n) {
                uint32_t d;
                const int ld = utf8_decode(s + i, &d);
                if (!is_letter(d)) break;
                i += (size_t)ld;
            }
            return i;
        }
    }

    /* 3) ' '?\p{N}+ — digits in a RUN (unlike qwen2, which takes one at a time) */
    {
        size_t i = c0 == ' ' ? (size_t)l0 : 0;
        uint32_t c = c0; int lc = l0;
        if (i) lc = i < n ? utf8_decode(s + i, &c) : 0;
        if (lc && is_number(c)) {
            while (i < n) {
                uint32_t d;
                const int ld = utf8_decode(s + i, &d);
                if (!is_number(d)) break;
                i += (size_t)ld;
            }
            return i;
        }
    }

    /* 4) ' '?[^\s\p{L}\p{N}]+ */
    {
        size_t i = c0 == ' ' ? (size_t)l0 : 0;
        const size_t start = i;
        while (i < n) {
            uint32_t d;
            const int ld = utf8_decode(s + i, &d);
            if (is_ws(d) || is_letter(d) || is_number(d)) break;
            i += (size_t)ld;
        }
        if (i > start) return i;
    }

    /* 5) \s+(?!\S) */
    if (is_ws(c0)) {
        size_t i = 0, prev = 0;
        while (i < n) {
            uint32_t d;
            const int ld = utf8_decode(s + i, &d);
            if (!is_ws(d)) break;
            prev = i;
            i += (size_t)ld;
        }
        if (i == n) return i;               /* run to end of text: take it all */
        if (i > (size_t)l0) return prev;    /* multi-char run: all but the last */
        return i;                            /* single, not followed by whitespace */
    }

    return (size_t)l0; /* codepoint outside every rule: a piece on its own */
}

static int is_newline(uint32_t c) { return c == 0x0A || c == 0x0D; }

/* ---- pre-tokenization ("glm4" pattern, IDENTICAL to llama3) --------------
 *
 * tokenizer.ggml.pre == "glm4" (used by GLM-5.2/glm-dsa): llama-vocab.cpp
 * tags chatglm4/chatglm-bpe as LLAMA_VOCAB_PRE_TYPE_CHATGLM4, which
 * shares THE SAME regex as llama3 (the only difference: special_bos_id
 * is NULL instead of set, irrelevant for pre-tokenization alone).
 * Pattern (PCRE):
 *   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
 *                                  contractions, case-INSENSITIVE
 *                                  (unlike pretok_next_olmo/dsk, which
 *                                  are case-sensitive)
 *   |[^\r\n\p{L}\p{N}]?\p{L}+      ANY non-letter/digit/CR/LF character
 *                                  (optional, not just a space) + a
 *                                  run of letters
 *   |\p{N}{1,3}                   digits in groups of AT MOST 3 (not
 *                                  an unbounded run like olmo, not a
 *                                  single digit like qwen2)
 *   | ?[^\s\p{L}\p{N}]+[\r\n]*     (literal space)? + a run of
 *                                  punctuation + trailing \r\n absorbed
 *   |\s*[\r\n]+                   whitespace (even zero) that CONTAINS
 *                                  at least one newline
 *   |\s+(?!\S)                    whitespace trailing the text (or not
 *                                  followed by non-whitespace)
 *   |\s+                          leftover whitespace (catch-all) */
static size_t pretok_next_glm4(const unsigned char *s, size_t n) {
    uint32_t c0;
    const int l0 = utf8_decode(s, &c0);

    /* 1) case-INSENSITIVE contractions: 's 't 're 've 'm 'll 'd */
    if (c0 == '\'' && n >= 2) {
        const unsigned char a = ascii_lower(s[1]);
        const unsigned char b = n >= 3 ? ascii_lower(s[2]) : 0;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
            (a == 'l' && b == 'l')) return 3;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    }

    /* 2) [^\r\n\p{L}\p{N}]?\p{L}+ — ANY single character (not a
     * letter/digit/newline) optional, then at least one letter */
    {
        const int prefix_ok = !is_newline(c0) && !is_letter(c0) && !is_number(c0);
        size_t i = prefix_ok ? (size_t)l0 : 0;
        uint32_t c = c0; int lc = l0;
        if (i) lc = i < n ? utf8_decode(s + i, &c) : 0;
        if (lc && is_letter(c)) {
            while (i < n) {
                uint32_t d;
                const int ld = utf8_decode(s + i, &d);
                if (!is_letter(d)) break;
                i += (size_t)ld;
            }
            return i;
        }
    }

    /* 3) \p{N}{1,3} — digits, AT MOST 3 per piece */
    if (is_number(c0)) {
        size_t i = 0; int cnt = 0;
        while (i < n && cnt < 3) {
            uint32_t d;
            const int ld = utf8_decode(s + i, &d);
            if (!is_number(d)) break;
            i += (size_t)ld;
            cnt++;
        }
        return i;
    }

    /* 4) ' '?[^\s\p{L}\p{N}]+[\r\n]* — ONLY a literal space as the
     * optional prefix (not any character, unlike rule 2) */
    {
        size_t i = c0 == ' ' ? (size_t)l0 : 0;
        const size_t start = i;
        while (i < n) {
            uint32_t d;
            const int ld = utf8_decode(s + i, &d);
            if (is_ws(d) || is_letter(d) || is_number(d)) break;
            i += (size_t)ld;
        }
        if (i > start) {
            while (i < n) {
                uint32_t d;
                const int ld = utf8_decode(s + i, &d);
                if (!is_newline(d)) break;
                i += (size_t)ld;
            }
            return i;
        }
    }

    /* 5) \s*[\r\n]+ — whitespace (even none) that contains a newline:
     * the whole whitespace run is scanned and the rule is accepted
     * only if the run includes at least one \r/\n (otherwise fall
     * through to rules 6/7 below, same treatment as pretok_next_olmo). */
    if (is_ws(c0)) {
        size_t i = 0; int saw_nl = 0;
        while (i < n) {
            uint32_t d;
            const int ld = utf8_decode(s + i, &d);
            if (!is_ws(d)) break;
            if (is_newline(d)) saw_nl = 1;
            i += (size_t)ld;
        }
        if (saw_nl) return i;

        /* 6) \s+(?!\S) / 7) \s+ — same scheme as pretok_next_olmo */
        size_t prev = 0; size_t j = 0;
        while (j < n) {
            uint32_t d;
            const int ld = utf8_decode(s + j, &d);
            if (!is_ws(d)) break;
            prev = j;
            j += (size_t)ld;
        }
        if (j == n) return j;               /* run to end of text: take it all */
        if (j > (size_t)l0) return prev;    /* multi-char run: all but the last */
        return j;                            /* single, not followed by whitespace */
    }

    return (size_t)l0; /* codepoint outside every rule: a piece on its own */
}

/* ---- BPE ------------------------------------------------------------------ */

/* Encodes a raw piece: byte-encoding + merges + vocabulary lookup.
 * Writes into out (if non-NULL, up to cap); returns the tokens produced. */
static int64_t bpe_encode(nf_tokenizer *t, const unsigned char *raw,
                          size_t rawlen, int32_t *out, int64_t cap,
                          int64_t already) {
    /* byte-encoding: every byte -> codepoint -> UTF-8 (at most 2 bytes: cp<0x144) */
    const size_t enc_cap = rawlen * 2 + 1;
    char stack_enc[2048];
    char *enc = enc_cap <= sizeof(stack_enc) ? stack_enc : malloc(enc_cap);
    /* start offset of every symbol + final sentinel */
    size_t stack_off[1025];
    size_t *off = rawlen + 1 <= 1025 ? stack_off
                                     : malloc((rawlen + 1) * sizeof(size_t));
    char stack_key[2050];
    char *key = enc_cap + 2 <= sizeof(stack_key) ? stack_key
                                                 : malloc(enc_cap + 2);
    int64_t produced = -1;
    if (!enc || !off || !key) goto done;

    size_t e = 0, n_sym = 0;
    for (size_t i = 0; i < rawlen; i++) {
        off[n_sym++] = e;
        e += (size_t)utf8_encode(t->byte_to_cp[raw[i]], enc + e);
    }
    off[n_sym] = e;

    /* merge loop: always fuse the adjacent pair with the lowest rank.
     * Quadratic over the piece, but pre-tokenization pieces are short:
     * simplicity is worth more than speed here. */
    for (;;) {
        int32_t best_rank = INT32_MAX;
        size_t best = 0;
        for (size_t i = 0; i + 1 < n_sym; i++) {
            const size_t la = off[i + 1] - off[i];
            const size_t lb = off[i + 2] - off[i + 1];
            memcpy(key, enc + off[i], la);
            key[la] = '\x01';
            memcpy(key + la + 1, enc + off[i + 1], lb);
            key[la + 1 + lb] = 0;
            const int32_t r = strmap_get(&t->merge_map, key);
            if (r >= 0 && r < best_rank) { best_rank = r; best = i; }
        }
        if (best_rank == INT32_MAX) break;
        /* fusing = removing the boundary between best and best+1 */
        memmove(off + best + 1, off + best + 2,
                (n_sym - best - 1) * sizeof(size_t));
        n_sym--;
    }

    /* look up the final symbols in the vocabulary */
    produced = 0;
    for (size_t i = 0; i < n_sym; i++) {
        const size_t l = off[i + 1] - off[i];
        memcpy(key, enc + off[i], l);
        key[l] = 0;
        const int32_t id = strmap_get(&t->vocab_map, key);
        if (id < 0) {
            /* shouldn't happen with a complete byte-level vocab */
            fprintf(stderr, "nf: symbol '%s' not in vocabulary\n", key);
            produced = -1;
            goto done;
        }
        if (out && already + produced < cap) out[already + produced] = id;
        produced++;
    }

done:
    if (enc != stack_enc) free(enc);
    if (off != stack_off) free(off);
    if (key != stack_key) free(key);
    return produced;
}

/* ---- API ------------------------------------------------------------------ */

nf_tokenizer *nf_tokenizer_load(const char *gguf_path) {
    char *model = nf_gguf_read_kv_str(gguf_path, "tokenizer.ggml.model");
    if (!model || strcmp(model, "gpt2") != 0) {
        fprintf(stderr, "nf: tokenizer '%s' not supported (BPE gpt2 only)\n",
                model ? model : "?");
        free(model);
        return NULL;
    }
    free(model);

    nf_tokenizer *t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    build_byte_tables(t);

    /* Pre-tokenization pattern from the gguf. Unknown = error (a
     * wrong pattern silently produces wrong ids). */
    {
        char *pre = nf_gguf_read_kv_str(gguf_path, "tokenizer.ggml.pre");
        if (!pre || strcmp(pre, "qwen2") == 0) {
            t->pre = 0;
        } else if (strcmp(pre, "deepseek-llm") == 0) {
            t->pre = 1;
        } else if (strcmp(pre, "olmo") == 0) {
            t->pre = 2;
        } else if (strcmp(pre, "glm4") == 0) {
            t->pre = 3;
        } else {
            fprintf(stderr, "nf: pre-tokenizer '%s' not supported "
                    "(qwen2, deepseek-llm, olmo, glm4)\n", pre);
            free(pre);
            free(t);
            return NULL;
        }
        free(pre);
        uint64_t b = 0;
        if (nf_gguf_read_kv_u64(gguf_path, "tokenizer.ggml.add_bos_token",
                                &b) == 0 && b) {
            uint64_t bid = 0;
            if (nf_gguf_read_kv_u64(gguf_path, "tokenizer.ggml.bos_token_id",
                                    &bid) == 0) {
                t->add_bos = 1;
                t->bos_id = (int32_t)bid;
            }
        }
    }

    t->vocab = nf_gguf_read_str_array(gguf_path, "tokenizer.ggml.tokens",
                                      &t->n_vocab);
    t->merges = nf_gguf_read_str_array(gguf_path, "tokenizer.ggml.merges",
                                       &t->n_merges);
    if (!t->vocab || !t->merges ||
        strmap_init(&t->vocab_map, (size_t)t->n_vocab) ||
        strmap_init(&t->merge_map, (size_t)t->n_merges)) {
        fprintf(stderr, "nf: vocabulary or merges missing from GGUF\n");
        nf_tokenizer_free(t);
        return NULL;
    }

    for (uint64_t i = 0; i < t->n_vocab; i++)
        strmap_put(&t->vocab_map, t->vocab[i], (int32_t)i);

    /* merges are "left right": the only raw space is the separator
     * (the actual space in the byte-encoded vocabulary is 'Ġ'), so we
     * replace it with \x01 and use the string itself as the key. */
    for (uint64_t i = 0; i < t->n_merges; i++) {
        char *sep = strchr(t->merges[i], ' ');
        if (!sep) continue;
        *sep = '\x01';
        strmap_put(&t->merge_map, t->merges[i], (int32_t)i);
    }

    /* special tokens: vocabulary entries with token_type != 1
     * (NORMAL). If the field is missing (an old/minimal GGUF) we
     * simply recognize none — a harmless degradation, not a fatal
     * error. */
    uint64_t n_types = 0;
    int32_t *types = nf_gguf_read_i32_array(gguf_path, "tokenizer.ggml.token_type",
                                            &n_types);
    if (types && n_types == t->n_vocab) {
        int n = 0;
        for (uint64_t i = 0; i < t->n_vocab; i++) if (types[i] != 1) n++;
        t->specials = malloc((size_t)n * sizeof(char *));
        t->special_ids = malloc((size_t)n * sizeof(int32_t));
        if (t->specials && t->special_ids) {
            int k = 0;
            for (uint64_t i = 0; i < t->n_vocab; i++) {
                if (types[i] == 1) continue;
                t->specials[k] = t->vocab[i];       /* NOT owned here */
                t->special_ids[k] = (int32_t)i;
                k++;
            }
            t->n_specials = n;
            /* sort by decreasing length (selection sort: n is small,
             * a few dozen entries — clarity over speed here) */
            for (int a = 0; a < t->n_specials; a++) {
                int best = a;
                for (int b = a + 1; b < t->n_specials; b++)
                    if (strlen(t->specials[b]) > strlen(t->specials[best])) best = b;
                if (best != a) {
                    char *ts = t->specials[a]; t->specials[a] = t->specials[best]; t->specials[best] = ts;
                    int32_t ti = t->special_ids[a]; t->special_ids[a] = t->special_ids[best]; t->special_ids[best] = ti;
                }
            }
        }
    }
    free(types);
    return t;
}

void nf_tokenizer_free(nf_tokenizer *t) {
    if (!t) return;
    nf_str_array_free(t->vocab, t->n_vocab);
    nf_str_array_free(t->merges, t->n_merges);
    strmap_free(&t->vocab_map);
    strmap_free(&t->merge_map);
    free(t->specials);      /* the strings pointed to belong to t->vocab, already freed above */
    free(t->special_ids);
    free(t);
}

uint64_t nf_tokenizer_vocab_size(const nf_tokenizer *t) { return t->n_vocab; }

/* Looks for the longest special token that literally matches at
 * s[0..n). t->specials is sorted by decreasing length, so the first
 * match found is automatically the longest (no ambiguity between
 * markers sharing a common prefix, e.g. '<|im_'). Returns the index
 * into t->specials, or -1 if none match. */
static int match_special(const nf_tokenizer *t, const unsigned char *s, size_t n) {
    for (int i = 0; i < t->n_specials; i++) {
        const size_t sl = strlen(t->specials[i]);
        if (sl <= n && memcmp(s, t->specials[i], sl) == 0) return i;
    }
    return -1;
}

int64_t nf_tokenize(nf_tokenizer *t, const char *text,
                    int32_t *out, int64_t out_cap) {
    const unsigned char *s = (const unsigned char *)text;
    const size_t len = strlen(text);
    size_t pos = 0;
    int64_t n_out = 0;
    /* Automatic BOS (deepseek) — once, at the head of the call.
     * NOTE: chat flows that tokenize in fragments would get
     * it per fragment; for generate/perplexity/tokenize (one call =
     * one sequence) this is exactly what llama does. */
    if (t->add_bos) {
        if (out && n_out < out_cap) out[n_out] = t->bos_id;
        n_out++;
    }
    while (pos < len) {
        const int si = match_special(t, s + pos, len - pos);
        if (si >= 0) {
            if (out && n_out < out_cap) out[n_out] = t->special_ids[si];
            n_out++;
            pos += strlen(t->specials[si]);
            continue;
        }
        const size_t plen = t->pre == 1 ? pretok_next_dsk(s + pos, len - pos)
                          : t->pre == 2 ? pretok_next_olmo(s + pos, len - pos)
                          : t->pre == 3 ? pretok_next_glm4(s + pos, len - pos)
                          : pretok_next(s + pos, len - pos);
        const int64_t k = bpe_encode(t, s + pos, plen, out, out_cap, n_out);
        if (k < 0 || plen == 0) return -1;
        n_out += k;
        pos += plen;
    }
    return n_out;
}

int nf_token_decode(const nf_tokenizer *t, int32_t id, char *buf, int cap) {
    if (id < 0 || (uint64_t)id >= t->n_vocab) return -1;
    const unsigned char *s = (const unsigned char *)t->vocab[id];
    int w = 0;
    while (*s && w < cap) {
        uint32_t cp;
        const int l = utf8_decode(s, &cp);
        if (cp < 512 && t->cp_to_byte[cp] >= 0) {
            buf[w++] = (char)t->cp_to_byte[cp];
        } else {
            /* special tokens (e.g. <|im_start|>) or an unmapped cp:
             * copy the UTF-8 bytes as-is */
            for (int j = 0; j < l && w < cap; j++) buf[w++] = (char)s[j];
        }
        s += l;
    }
    return w;
}
