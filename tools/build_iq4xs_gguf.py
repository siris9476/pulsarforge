"""Assembles a real GGUF with the Q4_K tensors requantized to IQ4_XS.

Hybrid architecture ("whole-model converter to IQ4_XS" plan): the C
side has already quantized the weights (nf convertiq4xs, validated
against the Python oracle) and written a manifest + a raw byte blob.
Here we assemble the actual GGUF CONTAINER using GGUFReader (to read
metadata and the tensors NOT converted, from the source file) and
GGUFWriter (to write the output) — both from the official `gguf`
library, never reinvented: same principle as always, C for the math,
Python for the format/bookkeeping.

Usage:
    python tools/build_iq4xs_gguf.py <source.gguf> <manifest.txt> <blob.bin> <output.gguf>

The manifest is text: one line per CONVERTED tensor,
"name<TAB>offset<TAB>n_bytes" (offset/n_bytes into the blob). Source
tensors that don't appear in the manifest are copied VERBATIM (same
type, same bytes) — the conversion rule is per-TYPE (Q4_K -> IQ4_XS),
not per-name, so there's no need to know which tensors were converted:
just read the manifest.
"""
import sys
import numpy as np
from gguf.gguf_reader import GGUFReader
from gguf.gguf_writer import GGUFWriter
from gguf.constants import GGMLQuantizationType, GGUFValueType

def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    src_path, manifest_path, blob_path, out_path = sys.argv[1:5]

    manifest = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, offset, nbytes = line.split("\t")
            manifest[name] = (int(offset), int(nbytes))
    print(f"manifest: {len(manifest)} tensors to convert to IQ4_XS")

    blob = np.fromfile(blob_path, dtype=np.uint8)

    reader = GGUFReader(src_path)

    arch = reader.get_field("general.architecture")
    arch_name = arch.contents() if arch is not None else "unknown"
    writer = GGUFWriter(out_path, arch_name)

    n_kv = 0
    for key, field in reader.fields.items():
        if key == "general.architecture":
            continue   # already written by GGUFWriter's constructor
        val = field.contents()
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        writer.add_key_value(key, val, vtype, sub_type=sub_type)
        n_kv += 1
    print(f"KV metadata copied: {n_kv}")

    n_converted = 0
    n_copied = 0
    for t in reader.tensors:
        if t.name in manifest:
            # t.shape is in native GGUF order (fastest axis = columns,
            # FIRST element) — the "byte" shape add_tensor expects is
            # in numpy order (fastest axis = LAST) with the LAST axis
            # in BYTES, not elements (the same trick the reader uses
            # for quantized tensors: gguf_reader.py calls
            # quant_shape_to_byte_shape before reshape — must be
            # replicated identically here, otherwise add_tensor_info
            # tries to reconvert an ELEMENT count as if it were bytes,
            # raising a ValueError like "bytes per row (1024) is not a
            # multiple of Q6_K...").
            numpy_shape_elems = tuple(int(x) for x in reversed(t.shape))
            cols_elems = numpy_shape_elems[-1]
            nsb = cols_elems // 256
            bytes_per_row = nsb * 136
            byte_shape = (*numpy_shape_elems[:-1], bytes_per_row)
            offset, nbytes = manifest[t.name]
            raw = blob[offset:offset + nbytes]
            assert raw.size == nbytes, (
                f"{t.name}: expected {nbytes} bytes from the blob, read {raw.size}")
            raw = raw.reshape(byte_shape)
            writer.add_tensor(t.name, raw, raw_dtype=GGMLQuantizationType.IQ4_XS)
            n_converted += 1
        else:
            # t.data is already shaped correctly by the reader (byte
            # shape for quantized types, natural shape for F32/F16) —
            # pass it through as-is, no shape conversion here.
            writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
            n_copied += 1
    print(f"tensors: {n_converted} converted to IQ4_XS, {n_copied} copied verbatim")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"written: {out_path}")

if __name__ == "__main__":
    main()
