# Point 0 of the multi-stream sweep: from REAL gate traces, simulates N
# streams interleaved PER LAYER against N separate runs, with the
# existing LRU (776 slots), and reports the byte ratio (= the maximum
# multiplier on the I/O wall, which dominates the wall clock).
import struct, sys, collections

def load(path):
    pos_list, cur, last_layer = [], collections.defaultdict(list), -1
    with open(path, 'rb') as f:
        while True:
            r = f.read(12)
            if len(r) < 12: break
            l, e, w = struct.unpack('<IIf', r)
            if l < last_layer:          # new position
                pos_list.append(cur); cur = collections.defaultdict(list)
            cur[l].append(e); last_layer = l
    if cur: pos_list.append(cur)
    return pos_list                     # [pos] -> {layer: [experts]}

A, B = load('tools/gate_A.bin'), load('tools/gate_B.bin')
print(f'traces: A={len(A)} positions, B={len(B)} positions')
NSLOT = 776

def run_bytes(streams, interleave):
    # streams: list of pos_list; step t = one position per stream
    cache, tick, fetches, accesses = {}, 0, 0, 0
    lru = collections.OrderedDict()
    T = min(len(s) for s in streams)
    layers = sorted({l for s in streams for p in s for l in p})
    def touch(key):
        nonlocal fetches
        if key in lru: lru.move_to_end(key)
        else:
            if len(lru) >= NSLOT: lru.popitem(last=False)
            lru[key] = 1; fetches += 1
    for t in range(T):
        if interleave:
            for l in layers:
                for s in streams:
                    for e in s[t].get(l, []): touch((l, e))
        else:
            for s in streams:
                for l in layers:
                    for e in s[t].get(l, []): touch((l, e))
    return fetches

for N in (2, 4, 8):
    streams = [(A if i % 2 == 0 else B) for i in range(N)]
    # different offsets for duplicated streams (honest partial decorrelation)
    streams = [s[i // 2:] for i, s in enumerate(streams)]
    sep = run_bytes(streams, interleave=False)
    inter = run_bytes(streams, interleave=True)
    print(f'N={N}: separate fetches={sep}  interleaved={inter}  '
          f'ratio={sep/inter:.2f}x  (relative bytes/token: {inter/sep:.2f})')
