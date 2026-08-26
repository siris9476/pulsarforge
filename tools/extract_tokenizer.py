# Extracts just the metadata (the full KV block) from the large GGUF
# into a twin GGUF with NO tensors: the BPE tokenizer lives entirely in
# the KV block. The KV block is copied VERBATIM (no reserialization:
# zero risk of altering the token strings).
import io, struct

SRC = 'models/glm52-merged.gguf'
DST = 'models/glm52-tokenizer.gguf'

f = io.open(SRC, 'rb')
hdr = f.read(4 + 4 + 8 + 8)
magic, ver, n_tensors, n_kv = struct.unpack('<4sIQQ', hdr)
assert magic == b'GGUF' and ver == 3, (magic, ver)
print(f'GGUF v{ver}: {n_tensors} tensors, {n_kv} KV')

# KV walker to find the end of the block (GGUF v3 types)
SIMPLE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

def skip_str(f):
    (n,) = struct.unpack('<Q', f.read(8))
    f.seek(n, 1)

def skip_val(f, t):
    if t in SIMPLE:
        f.seek(SIMPLE[t], 1)
    elif t == 8:
        skip_str(f)
    elif t == 9:
        et, n = struct.unpack('<IQ', f.read(12))
        if et in SIMPLE:
            f.seek(SIMPLE[et] * n, 1)
        elif et == 8:
            for _ in range(n):
                skip_str(f)
        else:
            raise SystemExit(f'unhandled array of arrays (et={et})')
    else:
        raise SystemExit(f'unknown KV type {t}')

kv_start = f.tell()
names = []
for _ in range(n_kv):
    (n,) = struct.unpack('<Q', f.read(8))
    names.append(f.read(n).decode('utf-8', 'replace'))
    (t,) = struct.unpack('<I', f.read(4))
    skip_val(f, t)
kv_end = f.tell()
print(f'KV block: {kv_end - kv_start} bytes '
      f'({sum(1 for x in names if x.startswith("tokenizer"))} tokenizer keys)')

f.seek(kv_start)
kv_blob = f.read(kv_end - kv_start)
f.close()

out = io.open(DST, 'wb')
out.write(struct.pack('<4sIQQ', b'GGUF', 3, 0, n_kv))   # 0 tensors
out.write(kv_blob)
out.close()
print(f'written {DST}')
