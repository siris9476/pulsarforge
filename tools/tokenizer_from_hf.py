# tokenizer_from_hf.py — builds models/glm52-tokenizer.gguf (metadata
# only, ~10MB) DIRECTLY from HuggingFace's tokenizer.json: no 200GB
# GGUF needed as an intermediary. Bootstrap for anyone starting from
# scratch (the Getting Started guide uses it as the first step).
#
#   python tools/tokenizer_from_hf.py [tokenizer.json] [out.gguf]
#
# Without arguments: downloads from zai-org/GLM-5.2 and writes
# models/glm52-tokenizer.gguf. Recommended check afterward:
#   ./nf.exe tokenize models/glm52-tokenizer.gguf "test text"
import io, json, os, struct, sys, urllib.request

HF_URL = ("https://huggingface.co/zai-org/GLM-5.2/resolve/main/"
          "tokenizer.json")

def w_str(f, s):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b))); f.write(b)

def kv_str(f, key, val):
    w_str(f, key); f.write(struct.pack("<I", 8)); w_str(f, val)

def kv_u32(f, key, val):
    w_str(f, key); f.write(struct.pack("<I", 4)); f.write(struct.pack("<I", val))

def kv_bool(f, key, val):
    w_str(f, key); f.write(struct.pack("<I", 7)); f.write(struct.pack("<B", 1 if val else 0))

def kv_arr_str(f, key, vals):
    w_str(f, key); f.write(struct.pack("<I", 9))
    f.write(struct.pack("<IQ", 8, len(vals)))
    for v in vals: w_str(f, v)

def kv_arr_i32(f, key, vals):
    w_str(f, key); f.write(struct.pack("<I", 9))
    f.write(struct.pack("<IQ", 5, len(vals)))
    for v in vals: f.write(struct.pack("<i", v))

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else None
    out = sys.argv[2] if len(sys.argv) > 2 else "models/glm52-tokenizer.gguf"
    if not src:
        src = "tokenizer.hf.json"
        if not os.path.exists(src):
            print(f"downloading {HF_URL} ...")
            urllib.request.urlretrieve(HF_URL, src)
    tk = json.load(io.open(src, encoding="utf-8"))
    model = tk["model"]
    assert model["type"] == "BPE", model["type"]
    vocab = model["vocab"]                       # str -> id
    merges = model["merges"]                     # ["a b", ...] or [[a,b],...]
    if merges and isinstance(merges[0], list):
        merges = [" ".join(m) for m in merges]
    n = max(vocab.values()) + 1
    toks = [""] * n
    for s, i in vocab.items(): toks[i] = s
    ttype = [1] * n                              # 1 = normal
    for at in tk.get("added_tokens", []):
        i = at["id"]
        if i >= n:
            toks.extend([""] * (i + 1 - n)); ttype.extend([1] * (i + 1 - n)); n = i + 1
        toks[i] = at["content"]
        ttype[i] = 3 if at.get("special") else 1   # 3 = control
    # pad to the MODEL's vocab size (154880 for GLM-5.2): ids beyond
    # the tokenizer.json exist in the output head and must decode
    # harmlessly, not error out the decode.
    PAD_TO = 154880
    while n < PAD_TO:
        toks.append(f"<|unused_{n}|>"); ttype.append(3); n += 1
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    f = io.open(out, "wb")
    f.write(struct.pack("<4sIQQ", b"GGUF", 3, 0, 6))
    kv_str(f, "general.architecture", "glm-dsa")
    kv_str(f, "tokenizer.ggml.model", "gpt2")
    kv_str(f, "tokenizer.ggml.pre", "glm4")
    kv_arr_str(f, "tokenizer.ggml.tokens", toks)
    kv_arr_str(f, "tokenizer.ggml.merges", merges)
    kv_arr_i32(f, "tokenizer.ggml.token_type", ttype)
    f.close()
    print(f"written {out}: {n} tokens, {len(merges)} merges "
          f"({os.path.getsize(out)/1e6:.1f}MB)")

if __name__ == "__main__":
    main()
