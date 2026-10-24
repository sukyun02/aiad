"""Read-only check of the two extra models' traces (numpy only)."""
import numpy as np

R = '/home/sukyun02/workspace/aiad/pilot/work/report'
BENDS = [(45, 77), (248, 273), (381, 399), (518, 550), (609, 634), (728, 767), (879, 914)]


def runs_of(flag):
    out, start = [], None
    for i, f in enumerate(np.r_[flag, False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - start))
            start = None
    return out


def where(a):
    for n, (lo, hi) in enumerate(BENDS, 1):
        if lo <= a <= hi:
            return f'in bend{n}({lo}-{hi})'
    prev = [b for b in BENDS if b[1] < a]
    nxt = [b for b in BENDS if b[0] > a]
    s = ''
    if prev:
        s += f'{a - prev[-1][1]:.1f} after bend end {prev[-1][1]}'
    if nxt:
        s += f', {nxt[0][0] - a:.1f} before bend start {nxt[0][0]}'
    return s


ref = np.load(f'{R}/final_model/trace.npz')['rows']
for name in ('final_model', 'm_20261006_232759', 'm_20261006_233500'):
    z = np.load(f'{R}/{name}/trace.npz')
    rows = z['rows']
    x, y, off, cls, speed, along, err = rows.T
    off = off > 0.5
    cls = cls.astype(int)
    n = len(rows)
    print('=====', name, 'frames', n, 't_end', n / 50, 'off', int(off.sum()), 'class counts', np.bincount(cls, minlength=9).tolist())
    print('  first gas frame', int(np.argmax(np.isin(cls, (1, 5, 6)))))
    for s, ln in runs_of(off):
        e = s + ln - 1
        print(f'  run start {s:5d} len {ln:4d} ({ln / 50:.2f}s) along {along[s]:7.1f} [{where(along[s])}] -> {along[e]:7.1f} [{where(along[e])}]'
              f' max err {err[s:s + ln].max():6.2f}; ends at last frame: {e == n - 1}')
    print(f'  last frame: along {along[-1]:.1f} err {err[-1]:.2f} speed {speed[-1]:.2f} class {cls[-1]} off {bool(off[-1])}')
    print('  last 12 frames speed', np.round(speed[-12:], 2).tolist())
    print('  classes in the last off-road run', np.bincount(cls[runs_of(off)[-1][0]:], minlength=9).tolist())
    s, ln = runs_of(off)[-1]
    k = slice(s, s + ln)
    print(f'  last run: speed min {speed[k].min():.2f} max {speed[k].max():.2f} mean {speed[k].mean():.2f}; err at start {err[s]:.2f}, at end {err[s + ln - 1]:.2f}, max {err[k].max():.2f}')
    # err trajectory inside the last run every 25 frames
    print('  err every 25 frames in last run', np.round(err[k][::25], 1).tolist())
    print('  along every 25 frames in last run', np.round(along[k][::25], 1).tolist())

# compare model 232759 with the final model: frame at which they first differ in position by more than 1
for name in ('m_20261006_232759', 'm_20261006_233500'):
    r = np.load(f'{R}/{name}/trace.npz')['rows']
    m = min(len(r), len(ref))
    d = np.hypot(r[:m, 0] - ref[:m, 0], r[:m, 1] - ref[:m, 1])
    first = int(np.argmax(d > 1.0)) if (d > 1.0).any() else -1
    print(name, 'first frame >1 apart from the final model', first, 'along there', ref[first, 5] if first >= 0 else None, 'max distance', float(d.max()))
