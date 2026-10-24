"""Regions of the lap where the driven line is more than 1.0 away from the centre, per reference lap (numpy only)."""
from pathlib import Path
import numpy as np

OUT = Path('/home/sukyun02/workspace/aiad/pilot/work/stage2/out')


def regions(along, off, thr=1.0):
    out, start = [], None
    for i in range(len(off)):
        on = abs(off[i]) >= thr
        if on and start is None:
            start = i
        if (not on or i == len(off) - 1) and start is not None:
            seg = off[start:i]
            k = start + int(np.argmax(np.abs(seg)))
            out.append((start, i, k))
            start = None
    return out


for name in ['A_lap1', 'A_lap2', 'A_lap3', 'A_lap4', 'A_lap5', 'A_lap6', 'A_lap7', 'A_lap8', 'E3_lap1', 'E0']:
    rows = np.load(OUT / f'{name}.npz')['rows']
    along, off = rows[:, 0], rows[:, 1]
    print('==', name, 'frames', len(rows))
    for a, b, k in regions(along, off):
        print(f'   f{a:5d}-{b:5d} along {along[a]:6.1f}-{along[b]:6.1f}  peak {off[k]:+.2f} at along {along[k]:6.1f} f{k} t={k/50:.2f}')

data = np.load(OUT / 'A_lap1.npz')
s, kappa, target = data['s'], data['kappa'], data['offset']
print('kappa nonzero vertices (heading change in deg) along the centre line:')
idx = np.nonzero(np.abs(kappa) > 1e-6)[0]
print(' count', len(idx))
# group heading change per 10 units
for a in range(0, 960, 10):
    m = (s >= a) & (s < a + 10)
    turn = np.degrees(kappa[m].sum() * 0.5)
    print(f'  {a:4d}-{a+10:4d}: turn {turn:+6.1f} deg   target offset A1 {target[m].mean():+.2f}')
