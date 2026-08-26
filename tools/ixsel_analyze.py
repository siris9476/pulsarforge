# Point 0 analysis for LazyLLM x DSA: from the NF_GLM_IXSEL_TRACE dump
# (record: int32 layer, pos, keep, selb[keep]) measures, per layer, how
# many prompt positions are NEVER selected by any query — candidates
# for early exit from prefill (the LazyLLM signal).
import struct, sys, collections

path = sys.argv[1] if len(sys.argv) > 1 else 'tools/ixsel_trace.bin'
sel_by_layer = collections.defaultdict(set)   # layer -> union of chosen indices
queries = collections.defaultdict(int)        # layer -> n queries seen
max_pos = 0
with open(path, 'rb') as f:
    while True:
        h = f.read(12)
        if len(h) < 12:
            break
        layer, pos, keep = struct.unpack('<iii', h)
        idx = struct.unpack(f'<{keep}i', f.read(4 * keep))
        sel_by_layer[layer].update(idx)
        queries[layer] += 1
        max_pos = max(max_pos, pos)

n_ctx = max_pos + 1
print(f'max context seen: {n_ctx} positions')
print('layer  query  |union|  dead  frac_dead')
bands = collections.defaultdict(list)
for l in sorted(sel_by_layer):
    u = len(sel_by_layer[l])
    dead = n_ctx - u
    frac = dead / n_ctx
    print(f'{l:5d}  {queries[l]:5d}  {u:7d}  {dead:5d}  {frac:.2f}')
    bands[l // 20].append(frac)
print('\nlayer band       mean dead frac (NEVER-selected positions)')
for b in sorted(bands):
    fr = bands[b]
    print(f'  {b*20:3d}-{b*20+19:3d}        {sum(fr)/len(fr):.2f}')
