# Point 0 "two-phase fetch": is the router a good oracle of the actual
# contribution? CSV: layer,pos,rank,mw,norm_fg,norm_out.
# The effective contribution to the output is mw*norm_out (the weight
# multiplies the down-output). POLICY comparison at equal budget:
# discard 25% of the experts (per call), choosing them (a) by minimum
# mw — what dynamic-k can do BEFORE reading the bytes; (b) by minimum
# mw*norm_fg — what could be known in-flight after gate+up. If (b)
# loses much less mass than (a), in-flight cancellation carries real
# information.
import csv, sys, collections, math

path = sys.argv[1] if len(sys.argv) > 1 else 'tools/contrib_trace.csv'
calls = collections.defaultdict(list)   # (layer,pos) -> [(mw, nfg, nout)]
for row in csv.reader(open(path)):
    l, p, k = int(row[0]), int(row[1]), int(row[2])
    mw, nfg, nout = float(row[3]), float(row[4]), float(row[5])
    calls[(l, p)].append((mw, nfg, nout))

import statistics as st
pearson_in = []
loss_router, loss_flight = [], []
for key, ex in calls.items():
    if len(ex) < 3:
        continue
    contrib = [mw * nout for mw, nfg, nout in ex]
    tot = sum(contrib)
    if tot <= 0:
        continue
    # rank correlation within the call (simple proxy: pearson on ranks)
    mws = [e[0] for e in ex]
    def ranks(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0]*len(v)
        for ri, i in enumerate(s): r[i] = ri
        return r
    rm, rc = ranks(mws), ranks(contrib)
    mr, mc = st.mean(rm), st.mean(rc)
    num = sum((a-mr)*(b-mc) for a, b in zip(rm, rc))
    den = math.sqrt(sum((a-mr)**2 for a in rm) * sum((b-mc)**2 for b in rc))
    if den > 0:
        pearson_in.append(num/den)
    ndrop = max(1, len(ex)//4)
    by_mw = sorted(range(len(ex)), key=lambda i: mws[i])[:ndrop]
    by_fl = sorted(range(len(ex)), key=lambda i: ex[i][0]*ex[i][1])[:ndrop]
    loss_router.append(sum(contrib[i] for i in by_mw)/tot)
    loss_flight.append(sum(contrib[i] for i in by_fl)/tot)

print(f'calls analyzed: {len(loss_router)}')
print(f'rank correlation mw vs contribution (intra-call mean): {st.mean(pearson_in):.3f}')
print(f'mass lost discarding 25% by ROUTER WEIGHT:    mean {st.mean(loss_router)*100:.2f}%  p90 {sorted(loss_router)[int(len(loss_router)*0.9)]*100:.2f}%')
print(f'mass lost discarding 25% by IN-FLIGHT ENERGY: mean {st.mean(loss_flight)*100:.2f}%  p90 {sorted(loss_flight)[int(len(loss_flight)*0.9)]*100:.2f}%')
