"""Read-only check of final_model/trace.npz and stage1_model/trace.npz (numpy only)."""
import json
import numpy as np

R = '/home/sukyun02/workspace/aiad/pilot/work/report'
BENDS = [(45, 77), (248, 273), (381, 399), (518, 550), (609, 634), (728, 767), (879, 914)]
LENGTH = 959.0


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
            return f'in bend{n}'
    best = min(range(len(BENDS)), key=lambda n: min(abs(a - BENDS[n][0]), abs(a - BENDS[n][1])))
    lo, hi = BENDS[best]
    side = 'before' if a < lo else 'after'
    return f'{side} bend{best + 1} by {min(abs(a - lo), abs(a - hi)):.1f}'


for name in ('final_model', 'stage1_model'):
    z = np.load(f'{R}/{name}/trace.npz')
    rows = z['rows']
    print('=====', name, 'keys', list(z.keys()), 'rows', rows.shape, 'half_width', float(z['half_width']),
          'centre pts', z['centre'].shape, 'road', z['road'].shape)
    x, y, off, cls, speed, along, err = rows.T
    off = off > 0.5
    cls = cls.astype(int)
    centre = z['centre']
    seg = np.roll(centre, -1, axis=0) - centre
    print('centre line length', float(np.linalg.norm(seg, axis=1).sum()))
    print('frames', len(rows), 'lap', len(rows) / 50, 'off', int(off.sum()), f'{100 * off.mean():.2f}%')
    print('class counts', np.bincount(cls, minlength=9).tolist())
    gas = np.isin(cls, (1, 5, 6))
    print('first gas frame index', int(np.argmax(gas)), '-> waiting frames', int(np.argmax(gas)), 'sec', np.argmax(gas) / 50)
    print('speed at frames 0,10,42,43,44,60,100,150:', [round(float(speed[i]), 2) for i in (0, 10, 42, 43, 44, 60, 100, 150)])
    moving = np.argmax(speed > 0.5)
    print('first frame with speed>0.5', int(moving))
    for thr in (20.0, 22.0, 22.4):
        print('  first frame speed >=', thr, int(np.argmax(speed >= thr)))
    print('mean speed after 100', float(speed[100:].mean()), 'min', float(speed[100:].min()), 'max', float(speed.max()))
    d = np.linalg.norm(np.diff(rows[:, :2], axis=0), axis=1)
    print('path length (sum of steps)', float(d.sum()))
    print('max err', float(err.max()), 'mean err', float(err.mean()))
    rr = runs_of(off)
    print('off-road runs:', len(rr))
    for s, n in rr:
        a0, a1 = along[s], along[s + n - 1]
        print(f'  start frame {s:5d} len {n:4d} ({n / 50:.2f}s) t={s / 50:.2f}-{(s + n) / 50:.2f}s along {a0:7.1f}->{a1:7.1f}'
              f' [{where(a0)} .. {where(a1)}] max err {err[s:s + n].max():.2f} margin to 5s: {5 - n / 50:.2f}s ({250 - n} frames)')
    # which bends are crossed off-road at all / share of each bend's along range covered off-road
    for n, (lo, hi) in enumerate(BENDS, 1):
        m = (along >= lo) & (along <= hi)
        print(f'  bend{n} along {lo}-{hi}: frames {int(m.sum())}, off-road frames {int((m & off).sum())}, max err {err[m].max() if m.any() else float("nan"):.2f}')
    # where along the track do off-road frames sit relative to straights
    in_bend = np.zeros(len(along), bool)
    for lo, hi in BENDS:
        in_bend |= (along >= lo) & (along <= hi)
    print('off-road frames inside a bend range', int((off & in_bend).sum()), 'outside', int((off & ~in_bend).sum()))
    # on which side of the centre line: sign via cross product with nearest segment direction
    idx = np.array([np.argmin(np.linalg.norm(centre - p, axis=1)) for p in rows[:, :2]])
    direction = seg[idx] / np.linalg.norm(seg[idx], axis=1, keepdims=True)
    rel = rows[:, :2] - centre[idx]
    cross = direction[:, 0] * rel[:, 1] - direction[:, 1] * rel[:, 0]   # >0 = left of driving direction
    for s, n in rr:
        print(f'   run at frame {s}: mean signed offset {cross[s:s + n].mean():+.2f} (+ = left of driving direction)')
    # total turning of centre line -> direction of the lap
    ang = np.arctan2(seg[:, 1], seg[:, 0])
    dang = (np.diff(np.r_[ang, ang[0]]) + np.pi) % (2 * np.pi) - np.pi
    print('total signed turning of centre line (deg):', float(np.degrees(dang.sum())), ' left-turn deg', float(np.degrees(dang[dang > 0].sum())),
          'right-turn deg', float(np.degrees(dang[dang < 0].sum())))
    # the last frame
    print('last frame along', float(along[-1]), 'off', bool(off[-1]), ' first frame along', float(along[0]))
    # recompute progress and centre score with the rule of metrics.py from the trace (position before step, off flag after step)
    hw = float(z['half_width'])
    print('---- time on each stretch (frames) final vs for comparison printed per model')
    edges = [0, 45, 77, 248, 273, 381, 399, 518, 550, 609, 634, 728, 767, 879, 914, 959.1]
    names = ['S0', 'B1', 'S1', 'B2', 'S2', 'B3', 'S3', 'B4', 'S4', 'B5', 'S5', 'B6-7', 'S6', 'B8', 'S7']
    cnt = {}
    un = np.unwrap(along * 2 * np.pi / LENGTH) * LENGTH / (2 * np.pi)
    for nm, lo, hi in zip(names, edges[:-1], edges[1:]):
        m = (along >= lo) & (along < hi)
        cnt[nm] = int(m.sum())
    print(json.dumps(cnt))
    np.save(f'{R}/check_b/{name}_stretch.npy', np.array(list(cnt.values())))

a = np.load(f'{R}/check_b/final_model_stretch.npy')
b = np.load(f'{R}/check_b/stage1_model_stretch.npy')
names = ['S0', 'B1', 'S1', 'B2', 'S2', 'B3', 'S3', 'B4', 'S4', 'B5', 'S5', 'B6-7', 'S6', 'B8', 'S7']
print('stretch: final frames, stage1 frames, diff (stage1-final), sec')
for nm, fa, fb in zip(names, a, b):
    print(f'  {nm:5s} {fa:5d} {fb:5d} {fb - fa:+5d} {(fb - fa) / 50:+.2f}')
print('  total', a.sum(), b.sum(), (b.sum() - a.sum()) / 50)
