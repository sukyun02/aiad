"""Second read-only pass over the traces (numpy only): metric recomputation, bend directions, off-road depth."""
import numpy as np

R = '/home/sukyun02/workspace/aiad/pilot/work/report'
BENDS = [(45, 77), (248, 273), (381, 399), (518, 550), (609, 634), (728, 767), (879, 914)]
C8 = [('C1', 45.5, 77.0), ('C2', 248.5, 273.0), ('C3', 381.5, 399.0), ('C4', 518.0, 549.5), ('C5', 609.0, 633.5),
      ('C6', 728.0, 738.5), ('C7', 749.0, 766.5), ('C8', 879.0, 913.5)]

for name in ('final_model', 'stage1_model'):
    z = np.load(f'{R}/{name}/trace.npz')
    rows, centre, hw = z['rows'], z['centre'], float(z['half_width'])
    x, y, off, cls, speed, along, err = rows.T
    off = off > 0.5
    n = len(rows)
    L = 959.0
    # metrics.py replay: update k uses the position AFTER step k (= row k+1 position) and the flag stored in row k
    prev = along[0]
    distance = furthest = scored = centre_int = 0.0
    was_off = False
    for k in range(n - 1):
        a, e = along[k + 1], err[k + 1]
        on = not off[k]
        delta = (a - prev + L / 2) % L - L / 2
        prev = a
        distance += delta
        reached = min(L, max(furthest, distance))
        gained = reached - furthest
        if on and not was_off:
            scored += gained
            centre_int += gained * (.2 + .8 * (1 - min(e / hw, 1.)) ** 4)
        furthest = reached
        was_off = not on
    # last update (position after the final step is not in the trace): assume the remaining distance is gained on road at the last error
    gained = L - furthest
    if not off[n - 1] and not was_off:
        scored += gained
        centre_int += gained * (.2 + .8 * (1 - min(err[n - 1] / hw, 1.)) ** 4)
    print('=====', name)
    print(f'recomputed progress {100 * scored / L:.2f}  centre {100 * centre_int / L:.2f}   (unscored share of the centre line {100 - 100 * scored / L:.2f}%)')
    print(f'mean centre weight on scored distance {centre_int / scored:.3f}')
    # on-road frames: distribution of distance from the centre line
    on = ~off
    print(f'on-road frames {int(on.sum())}: mean err {err[on].mean():.2f}, median {np.median(err[on]):.2f}, share with err > half of half-width {100 * (err[on] > hw / 2).mean():.1f}%')
    print(f'off-road frames {int(off.sum())}: err min {err[off].min() if off.any() else float("nan"):.2f}, frames with err < half-width {int((err[off] < hw).sum()) if off.any() else 0}')
    # bend directions from the centre line
    seg = np.roll(centre, -1, axis=0) - centre
    lens = np.linalg.norm(seg, axis=1)
    starts = np.r_[0., np.cumsum(lens)[:-1]]
    ang = np.arctan2(seg[:, 1], seg[:, 0])
    dang = (np.diff(np.r_[ang, ang[0]]) + np.pi) % (2 * np.pi) - np.pi      # turn at the end of segment i
    pos = starts + lens
    if name == 'final_model':
        for lo, hi in BENDS:
            m = (pos >= lo) & (pos <= hi)
            print(f'  bend {lo}-{hi}: turning {np.degrees(dang[m].sum()):+7.1f} deg (+ = left)')
        for nm, lo, hi in C8:
            m = (pos >= lo) & (pos <= hi)
            print(f'  {nm} {lo}-{hi}: turning {np.degrees(dang[m].sum()):+7.1f} deg')
        m = (pos >= 767) & (pos <= 879)
        print(f'  straight 767-879: turning {np.degrees(dang[m].sum()):+7.1f} deg')
        m = (pos >= 914) | (pos <= 45)
        print(f'  914-959-45: turning {np.degrees(dang[m].sum()):+7.1f} deg')
        # frames after the car is back on the road in the long run until the finish
        runs, s = [], None
        for i, f in enumerate(np.r_[off, False]):
            if f and s is None:
                s = i
            elif not f and s is not None:
                runs.append((s, i - s)); s = None
        s, ln = max(runs, key=lambda r: r[1])
        print(f'  longest run: frames {s}..{s + ln - 1}, back on road at frame {s + ln} (t={(s + ln) / 50:.2f}s), finish at frame {n} (t={n / 50:.2f}s): {n - s - ln} frames = {(n - s - ln) / 50:.2f}s before the line')
        print(f'  along at leave {along[s]:.1f} ({879 - along[s]:.1f} before the bend start 879), along at return {along[s + ln]:.1f} ({959 - along[s + ln]:.1f} before the finish line)')
        print(f'  straight-line chord of the long run {np.hypot(x[s + ln - 1] - x[s], y[s + ln - 1] - y[s]):.1f} vs centre-line distance {along[s + ln - 1] - along[s]:.1f}')
        for s2, l2 in runs:
            chord = np.hypot(np.diff(x[s2:s2 + l2]), np.diff(y[s2:s2 + l2])).sum()
            print(f'   run @{s2} len {l2}: driven {chord:.1f}, centre-line distance covered {along[s2 + l2 - 1] - along[s2]:.1f}, saved {along[s2 + l2 - 1] - along[s2] - chord:.1f}; err max {err[s2:s2 + l2].max():.2f} (beyond edge {err[s2:s2 + l2].max() - hw:.2f})')
        print('  frames per class while off-road', np.bincount(cls[off].astype(int), minlength=9).tolist())
    d = np.hypot(np.diff(x), np.diff(y)).sum()
    wait = int(np.argmax(np.isin(cls.astype(int), (1, 5, 6))))
    print(f'path {d:.2f}, wait {wait} frames, frames {n}; (frames - wait) = {n - wait} -> {(n - wait) / 50:.2f}s moving')

f = np.load(f'{R}/final_model/trace.npz')['rows']
s = np.load(f'{R}/stage1_model/trace.npz')['rows']
pf = np.hypot(np.diff(f[:, 0]), np.diff(f[:, 1])).sum()
ps = np.hypot(np.diff(s[:, 0]), np.diff(s[:, 1])).sum()
print(f'path difference {ps - pf:.2f} -> at 22.49: {(ps - pf) / 22.49:.2f}s ; wait difference {(43 - 26) / 50:.2f}s ; lap difference {(len(s) - len(f)) / 50:.2f}s ; moving-time difference {((len(s) - 26) - (len(f) - 43)) / 50:.2f}s')
