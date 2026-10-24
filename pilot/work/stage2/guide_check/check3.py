"""Second independent pass: key timelines from the SAVED ACTIONS, start numbers, segment table, human laps, H5 key states."""
import json
from pathlib import Path

import numpy as np

SP = Path('/home/sukyun02/workspace/aiad/pilot')
OUT = SP / 'work/stage2/out'
HERE = SP / 'work/stage2/guide_check'
GUIDE = SP / 'work/stage2/guide'

CORNERS = [('C1', 45.5, 77.0, 1), ('C2', 248.5, 273.0, 1), ('C3', 381.5, 399.0, 1), ('C4', 518.0, 549.5, 1),
           ('C5', 609.0, 633.5, 1), ('C6', 728.0, 738.5, 1), ('C7', 749.0, 766.5, -1), ('C8', 879.0, 913.5, 1)]
SECTION = {'C1': (30, 95), 'C2': (236, 295), 'C3': (340, 460), 'C4': (500, 580), 'C5': (595, 658),
           'C6': (715, 747), 'C7': (747, 785), 'C8': (862, 933)}
SEGMENTS = [('start', -1, 30), ('C1', 30, 95), ('s12', 95, 190), ('K1', 190, 236), ('C2', 236, 295), ('s23', 295, 340),
            ('C3', 340, 460), ('s34', 460, 500), ('C4', 500, 580), ('s45', 580, 595), ('C5', 595, 658), ('K2', 658, 715),
            ('C6', 715, 747), ('C7', 747, 785), ('s78', 785, 862), ('C8', 862, 933), ('finish', 933, 960)]


def runs(mask):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, len(mask) - start))
    return out


def unwrap_along(along):
    """make along monotone-ish across the start/finish wrap: frames in the second half with along < 100 get +959"""
    a = along.copy()
    n = len(a)
    idx = np.arange(n)
    a[(idx > n // 2) & (a < 200)] += 959.0
    a[(idx < n // 2) & (a > 800)] -= 959.0
    return a


def apex_frame(along, inside, a0, a1):
    mask = (along >= a0 - 15) & (along < a1 + 15)
    idx = np.nonzero(mask)[0]
    return int(idx[np.argmax(inside[idx])])


def fmt(v):
    v = np.asarray(v, dtype=float)
    return f'{np.median(v):.2f} [{v.min():.2f}-{v.max():.2f}]'


# ---------------------------------------------------------------- set A from saved actions + traces
A = []
for i in range(1, 9):
    rows = np.load(OUT / f'A_lap{i}.npz')['rows'].astype(float)
    folder = SP / f'data_s2/scriptA/lap{i}'
    acts = np.array([np.load(p) for p in sorted(folder.glob('action_*.npy'))])
    assert len(acts) == len(rows)
    A.append(dict(along=unwrap_along(rows[:, 0]), off=rows[:, 1], steer_key=acts[:, 0], speed=rows[:, 3], state=rows[:, 4]))
E0rows = np.load(OUT / 'E0.npz')['rows'].astype(float)
E0 = dict(along=unwrap_along(E0rows[:, 0]), off=E0rows[:, 1], steer_key=np.where(E0rows[:, 2] == 5, -1.0, np.where(E0rows[:, 2] == 6, 1.0, 0.0)),
          speed=E0rows[:, 3], state=E0rows[:, 4])

print('===== 1. key timelines relative to the apex, 0.2 s bins (10 frames), from action_*.npy of data_s2/scriptA')
for name in ['C1', 'C4', 'C8', 'C6', 'C7']:
    a0, a1, side = [(c[1], c[2], c[3]) for c in CORNERS if c[0] == name][0]
    print(f'--- {name}: inside key = {"LEFT" if side > 0 else "RIGHT"}')
    table_in, table_out, table_along, table_off = [], [], [], []
    for lap in A:
        inside = side * lap['off']
        ap = apex_frame(lap['along'], inside, a0, a1)
        key_in = (lap['steer_key'] == -side)
        key_out = (lap['steer_key'] == side)
        rin, rout, ral, rof = [], [], [], []
        for b in range(-9, 9):
            lo, hi = ap + b * 10, ap + (b + 1) * 10
            lo2, hi2 = max(lo, 0), min(hi, len(inside))
            if hi2 <= lo2:
                rin.append(np.nan); rout.append(np.nan); ral.append(np.nan); rof.append(np.nan)
                continue
            rin.append(key_in[lo2:hi2].mean()); rout.append(key_out[lo2:hi2].mean())
            ral.append(lap['along'][lo2:hi2].mean()); rof.append(inside[lo2:hi2].mean())
        table_in.append(rin); table_out.append(rout); table_along.append(ral); table_off.append(rof)
    tin, tout, tal, tof = map(np.array, (table_in, table_out, table_along, table_off))
    for k, b in enumerate(range(-9, 9)):
        print(f'  {b*0.2:+.1f}..{(b+1)*0.2:+.1f}s  inside med {np.nanmedian(tin[:,k]):.2f} [{np.nanmin(tin[:,k]):.2f}-{np.nanmax(tin[:,k]):.2f}] mean {np.nanmean(tin[:,k]):.2f}'
              f' | outside med {np.nanmedian(tout[:,k]):.2f} [{np.nanmin(tout[:,k]):.2f}-{np.nanmax(tout[:,k]):.2f}] mean {np.nanmean(tout[:,k]):.2f}'
              f' | along med {np.nanmedian(tal[:,k]):.1f} | inside offset med {np.nanmedian(tof[:,k]):+.2f}')

print('===== 2. first keys per A lap')
for name in ['C1', 'C3', 'C4', 'C8']:
    a0, a1, side = [(c[1], c[2], c[3]) for c in CORNERS if c[0] == name][0]
    for gap in (10, 15, 25):
        f_first, al_first, rel = [], [], []
        for lap in A:
            inside = side * lap['off']
            ap = apex_frame(lap['along'], inside, a0, a1)
            t = ap
            while t > 0 and inside[t - 1] >= 1.0:
                t -= 1
            key_in = np.nonzero(lap['steer_key'] == -side)[0]
            key_in = key_in[key_in <= t]
            f = key_in[-1]
            for g in key_in[::-1]:
                if f - g > gap:
                    break
                f = g
            f_first.append(f / 50); al_first.append(lap['along'][f]); rel.append((f - ap) / 50)
        print(f'{name} approach cluster start (gap>{gap} frames breaks): t {fmt(f_first)} along {fmt(al_first)} rel apex {fmt(rel)}')
print('first steering key of any kind / first LEFT key, per A lap (frame, along):')
for i, lap in enumerate(A, 1):
    anyk = int(np.argmax(lap['steer_key'] != 0)); leftk = int(np.argmax(lap['steer_key'] < 0))
    print(f'  lap{i}: any key frame {anyk} along {lap["along"][anyk]:.1f} | first LEFT frame {leftk} t {leftk/50:.2f} along {lap["along"][leftk]:.1f}')

print('===== 3. start')
l1 = A[0]
print('A lap1 speed at frames 5,10,15,20,25,30,50:', [round(float(l1['speed'][f]), 2) for f in (5, 10, 15, 20, 25, 30, 50)])
print('A lap1 along at frames 25,50,75,100:', [round(float(l1['along'][f]), 1) for f in (25, 50, 75, 100)])
print('A frames to along>=30:', [int(np.argmax(lap['along'] >= 30)) for lap in A], 'E0:', int(np.argmax(E0['along'] >= 30)))

# ---------------------------------------------------------------- human laps from my own replay
H = []
for i in range(1, 8):
    r = np.load(HERE / f'replay/H{i}.npz')['rows']
    js = json.loads((HERE / f'replay/H{i}.json').read_text())
    H.append(dict(along=unwrap_along(r[:, 0]), off=r[:, 1], steer_key=r[:, 2], gas=r[:, 3], speed=r[:, 5], state=r[:, 6], js=js))
print('human frames to along>=30:', [int(np.argmax(h['along'] >= 30)) for h in H], 'hold', [h['js']['start_hold_frames'] for h in H],
      'lap', [h['js']['lap_time'] for h in H])

print('===== 4. segment table (frames/50)')


def seg_times(lap):
    al = lap['along']
    return [((al >= lo) & (al < hi)).sum() / 50 for _, lo, hi in SEGMENTS]


segA = np.array([seg_times(l) for l in A]); segE = np.array(seg_times(E0)); segH = np.array([seg_times(h) for h in H])
for k, (nm, lo, hi) in enumerate(SEGMENTS):
    print(f'  {nm:7s} {lo:4d}-{hi:4d}  E0 {segE[k]:.2f}  A med {np.median(segA[:,k]):.2f} [{segA[:,k].min():.2f}-{segA[:,k].max():.2f}] gain {segE[k]-np.median(segA[:,k]):+.2f}'
          f'  human med {np.median(segH[:,k]):.2f} [{segH[:,k].min():.2f}-{segH[:,k].max():.2f}] hum-E0 {np.median(segH[:,k])-segE[k]:+.2f}')
print(f'  sum of segment medians: E0 {segE.sum():.2f}  A {np.median(segA,axis=0).sum():.2f}  human {np.median(segH,axis=0).sum():.2f}')
print(f'  per-lap totals: A {segA.sum(1).round(2).tolist()} median {np.median(segA.sum(1)):.2f} mean {segA.sum(1).mean():.3f} | human {segH.sum(1).round(2).tolist()} median {np.median(segH.sum(1)):.2f}')
print('  human minus hold: ', [round(h['js']['lap_time'] - h['js']['start_hold_frames'] / 50, 2) for h in H])

print('===== 5. human laps per corner (inside-key presses that start inside the section)')
for name, a0, a1, side in CORNERS:
    s0, s1 = SECTION[name]
    n_in, longest, maxin, n_out = [], [], [], []
    for h in H:
        al = h['along']
        sm = (al >= s0) & (al < s1)
        pr = [(s, l) for s, l in runs(h['steer_key'] == -side) if sm[s]]
        po = [(s, l) for s, l in runs(h['steer_key'] == side) if sm[s]]
        n_in.append(len(pr)); n_out.append(len(po)); longest.append(max([l for _, l in pr], default=0))
        m = (al >= a0 - 15) & (al < a1 + 15)
        maxin.append(float((side * h['off'][m]).max()))
    print(f'  {name}: inside presses med {np.median(n_in):.1f} [{min(n_in)}-{max(n_in)}] longest med {np.median(longest):.1f} [{min(longest)}-{max(longest)}]'
          f' | outside presses med {np.median(n_out):.1f} [{min(n_out)}-{max(n_out)}] | max inside {[round(v, 2) for v in maxin]}')

print('===== 6. whole-lap key press statistics')
for label, laps in (('A', A), ('human', H)):
    counts, lens_all = [], []
    for lap in laps:
        pr = runs(lap['steer_key'] < 0) + runs(lap['steer_key'] > 0)
        counts.append(len(pr)); lens_all += [l for _, l in pr]
    lens_all = np.array(lens_all)
    print(f'  {label}: presses per lap {counts} median {np.median(counts):.1f} mean {np.mean(counts):.1f} | length median {np.median(lens_all):.0f}'
          f' p10 {np.percentile(lens_all,10):.0f} p90 {np.percentile(lens_all,90):.0f} max {lens_all.max()} | share 1f {100*(lens_all==1).mean():.0f}% 2f {100*(lens_all==2).mean():.0f}% <=2f {100*(lens_all<=2).mean():.0f}%')

print('===== 7. K2 / K1: largest RIGHT offset of set A')
for nm, lo, hi in (('K1', 190, 236), ('K2', 650, 712)):
    vals, als = [], []
    for lap in A:
        m = np.nonzero((lap['along'] >= lo) & (lap['along'] < hi))[0]
        k = m[np.argmin(lap['off'][m])]
        vals.append(-lap['off'][k]); als.append(lap['along'][k])
    print(f'  {nm}: right offset {fmt(vals)} at along {fmt(als)}')
print('===== 8. C6/C7 steering magnitude of set A (|controller steering|) in 715-785, and per section any direction')
for lo, hi in ((715, 747), (747, 785), (715, 785)):
    v = [float(np.abs(lap['state'][(lap['along'] >= lo) & (lap['along'] < hi)]).max()) for lap in A]
    e = float(np.abs(E0['state'][(E0['along'] >= lo) & (E0['along'] < hi)]).max())
    print(f'  {lo}-{hi}: A max |steer| {fmt(v)}  E0 {e:.2f}')
print('  longest press of any key in 715-785 per A lap:', [max(l for s, l in runs(lap['steer_key'] != 0) if 715 <= lap['along'][s] < 785) for lap in A],
      ' (same key):', [max(l for s, l in (runs(lap['steer_key'] < 0) + runs(lap['steer_key'] > 0)) if 715 <= lap['along'][s] < 785) for lap in A])

print('===== 9. set H5 (guide/hold/H5_lap*.npz): key-state run lengths and literal sequences')
for i in range(1, 9):
    r = np.load(GUIDE / f'hold/H5_lap{i}.npz')['rows'].astype(float)
    js = json.loads((GUIDE / f'hold/H5_lap{i}.json').read_text())
    cls = r[:, 2].astype(int)
    change = np.nonzero(np.diff(cls) != 0)[0] + 1
    bounds = np.r_[0, change, len(cls)]
    lens = np.diff(bounds)
    inner = lens[1:-1]
    pr = runs(cls == 5) + runs(cls == 6)
    print(f'  H5 lap{i}: lap_time {js.get("lap_time")} progress {js.get("progress")} center {js.get("center")} max_abs_offset {js.get("max_abs_offset")} frames {len(r)}'
          f' | key states {len(lens)} shortest inner state {inner.min()} (count <5: {(inner<5).sum()}) | presses {len(pr)} | params E {js.get("E")} W {js.get("W")} k0 {js.get("k0")} look {js.get("lookahead")} gain {js.get("gain")}')
r = np.load(GUIDE / 'hold/H5_lap1.npz')['rows'].astype(float)
al = unwrap_along(r[:, 0]); cls = r[:, 2].astype(int); off = r[:, 1]
for name in ['C1', 'C4', 'C8']:
    a0, a1, side = [(c[1], c[2], c[3]) for c in CORNERS if c[0] == name][0]
    ap = apex_frame(al, side * off, a0, a1)
    lo, hi = ap - 90, min(ap + 90, len(cls))
    seq, f = [], lo
    while f < hi:
        g = f
        while g < hi and cls[g] == cls[f]:
            g += 1
        seq.append(f"{ {1: '-', 5: 'L', 6: 'R', 0: '0'}[cls[f]] }{g - f}" + ('*' if f <= ap < g else ''))
        f = g
    print(f'  H5 lap1 {name}: apex frame {ap} t {ap/50:.2f} along {al[ap]:.1f} inside {side*off[ap]:.2f} | states from apex-1.8s to apex+1.8s (* = contains apex): ' + ' '.join(seq))
print('  hold scan files:')
for p in sorted((GUIDE / 'hold').glob('*.json')):
    js = json.loads(p.read_text())
    print(f'    {p.name}: lap_time {js.get("lap_time")} progress {js.get("progress")} center {js.get("center")} max_abs_offset {js.get("max_abs_offset")} status {js.get("status")} E {js.get("E")} min_hold {js.get("min_hold")}')
