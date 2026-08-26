#!/bin/sh
# Native Linux validation (live USB / real Linux machine).
# Usage: sh validate_native.sh    (from the repo root, where this file lives)
# Requires: gcc (sudo apt install build-essential). Results in
# validate_native_results.txt (persists on the NTFS disk).
cd "$(dirname "$0")" || exit 1
OUT=validate_native_results.txt
{
echo "=== native validation $(uname -srm) — $(date) ==="
echo "libc: $(ldd --version 2>&1 | head -1)"

command -v gcc >/dev/null || { echo "MISSING gcc: sudo apt install build-essential"; exit 1; }

echo "--- [1/3] build ---"
sh build_posix.sh /tmp/nf || { echo "BUILD FAILED"; exit 1; }

echo "--- [2/3] tiny gate (byte-identical to the cross-platform reference) ---"
IDS=$(cat tools/prompt_tiny_ids.txt)
NF_GLM_STREAM_GB=0.01 NF_GLM_PREFETCH=1 NF_GLM_OVERLAP=1 NF_GLM_QPRIO=1 \
  /tmp/nf forgelogits tools/glm_tiny_sparse.forge "$IDS" /tmp/native.bin 2>&1 | tail -1
if cmp tools/glm_tiny_forgelogits_ref.bin /tmp/native.bin; then
    echo "TINY GATE: BYTE-IDENTICAL (third libc validated)"
else
    echo "TINY GATE: FAILED — investigate before proceeding"
    exit 1
fi

echo "--- [3/3] real model (if present) ---"
M=../glm52.forgezh
FSTYPE=$(stat -f -c %T . 2>/dev/null)
if [ "$FSTYPE" = "v9fs" ] || [ "$FSTYPE" = "9p" ]; then
    echo "(repo on 9p/WSL: the 744B at ~5MB/s would take hours — skipping."
    echo " This script should be run from NATIVE Linux with the X9 mounted via ntfs3)"
elif [ -f "$M" ]; then
    echo "chat on the 744B, timed (expected: 'Paris.'):"
    T0=$(date +%s)
    printf "What is the capital of France? Answer with a single word.\n" | \
      /tmp/nf chat "$M" --max-tokens 16 2>/tmp/native_chat.err
    T1=$(date +%s)
    echo "total time: $((T1-T0))s"
    tail -3 /tmp/native_chat.err
else
    echo "($M not found: skipping — mount the X9 and rerun from the repo on the disk)"
fi
echo "=== validation finished ==="
} 2>&1 | tee "$OUT"
