"""Q1: where do excursions start, and what happened before. Q4: zoom-in. Uses traces/*.npz."""
import sys; sys.path.insert(0, '/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad/work/diag')
from common import *
E = experts()
# expert envelope of signed offset per 5 m bin (start segment handled by unwrap: negative = before start line)
BIN = 5.0
def binned(rows):
    u = unwrap(rows[:, ALONG]); return u
env_lo = {}; env_hi = {}; exp_dir = {}
for r in E:
    u = unwrap(r[:, ALONG]); b = np.floor(u / BIN).astype(int)
    for k in np.unique(b):
        m = b == k
        env_lo[k] = min(env_lo.get(k, 99), r[m, OFF].min()); env_hi[k] = max(env_hi.get(k, -99), r[m, OFF].max())
        exp_dir.setdefault(k, []).append(DIRS[r[m, CLS].astype(int)].mean())
res = results()
print('expert envelope |offset| max over lap:', max(max(abs(v) for v in env_lo.values()), max(abs(v) for v in env_hi.values())))
rows_out = []
hdr = 'name status lap prog | exc# | onset_along corner side(+L/-R of centre) | pre2s: L% R% (expert L% R%) head_err_deg margin_med cls0% | peak_off back?'
print(hdr)
for n in names():
    r, meta, lg = load(n)
    u = unwrap(r[:, ALONG]); b = np.floor(u / BIN).astype(int)
    lo = np.array([env_lo.get(k, -1) for k in b]); hi = np.array([env_hi.get(k, 1) for k in b])
    out = (r[:, OFF] > hi + 1.0) | (r[:, OFF] < lo - 1.0)          # outside expert envelope by > 1 unit
    offroad = np.abs(r[:, OFF]) > HALF
    # excursions = runs where |offset| > HALF (centre of car off the road); onset = last frame before the run that was inside the envelope(+1)
    i = 0; T = len(r); excs = []
    while i < T:
        if offroad[i]:
            j = i
            while j + 1 < T and (offroad[j + 1] or (np.abs(r[j + 1, OFF]) > 4.0)): j += 1
            k = i
            while k > 0 and out[k]: k -= 1
            excs.append((k, i, j)); i = j + 1
        else: i += 1
    # also 'large deviations' that did not leave the road: |off| in (5, HALF)
    st = meta.get('status'); lap = meta.get('lap_time')
    base = f"{n:32s} {st[:4]:4s} {lap if lap else float('nan'):6.2f} {meta.get('progress', 0):5.1f}"
    if not excs:
        print(base, '| 0 | max|off|', f"{np.abs(r[:, OFF]).max():.2f} at {r[np.abs(r[:, OFF]).argmax(), ALONG]:.0f}"); 
    for (k, i, j) in excs:
        a0 = r[k, ALONG]; pre = slice(max(0, k - 100), k + 1)
        d = DIRS[r[pre, CLS].astype(int)]
        eb = b[pre]; ed = np.array([np.mean(exp_dir.get(q, [0])) for q in eb])
        # expert left/right share at same positions
        eL = np.mean([np.mean([min(x, 0) for x in exp_dir.get(q, [0])]) for q in eb]) * -1
        eR = np.mean([np.mean([max(x, 0) for x in exp_dir.get(q, [0])]) for q in eb])
        side = np.sign(r[i, OFF])
        ended = 'END(abort)' if j >= T - 2 else f'back@{r[j, ALONG]:.0f} after {(j - i) / 50:.1f}s'
        print(base, f"| {len(excs)} | onset {a0:6.1f} f{k:4d} {corner_of(a0):26s} exit_side {'L' if side > 0 else 'R'} offroad@{r[i, ALONG]:.0f} | pre2s L {100 * (d < 0).mean():3.0f} R {100 * (d > 0).mean():3.0f} (exp L {100 * eL:3.0f} R {100 * eR:3.0f}) head {np.degrees(r[k, HEAD]):+5.1f} off {r[k, OFF]:+.1f} margin_med {np.median(r[pre, MARGIN]):.2f} cls0 {100 * (r[pre, CLS] == 0).mean():3.0f}% | peak {np.abs(r[i:j + 1, OFF]).max():5.1f} {ended}")
        rows_out.append(dict(name=n, onset_along=float(a0), onset_frame=int(k), corner=corner_of(a0), side='L' if side > 0 else 'R', offroad_along=float(r[i, ALONG]),
                             preL=float((d < 0).mean()), preR=float((d > 0).mean()), expL=float(eL), expR=float(eR), ended=ended, n_exc=len(excs), first=(k, i, j) == excs[0]))
json.dump(rows_out, open(W / 'excursions.json', 'w'), indent=1)
json.dump(dict(lo={int(k): float(v) for k, v in env_lo.items()}, hi={int(k): float(v) for k, v in env_hi.items()}), open(W / 'envelope.json', 'w'))
