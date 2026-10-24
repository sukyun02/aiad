"""Quick look at the reference traces (numpy only). usage: explore.py"""
import json
from pathlib import Path
import numpy as np

OUT = Path('/home/sukyun02/workspace/aiad/pilot/work/stage2/out')
CORNERS = [('C1', 45.5, 77.0, 1), ('C2', 248.5, 273.0, 1), ('C3', 381.5, 399.0, 1), ('C4', 518.0, 549.5, 1),
           ('C5', 609.0, 633.5, 1), ('C6', 728.0, 738.5, 1), ('C7', 749.0, 766.5, -1), ('C8', 879.0, 913.5, 1)]


def rle(values):
    out, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            out.append((int(values[start]), start, i - start))
            start = i
    return out


for name in ['A_lap1', 'E0', 'E3_lap1']:
    data = np.load(OUT / f'{name}.npz')
    rows = data['rows']
    print('==', name, rows.shape, json.loads((OUT / f'{name}.json').read_text())['lap_time'],
          'keys', list(data.keys()), 'offset', data['offset'].shape, 's', data['s'][:3], data['s'][-1])
    print(' first rows', np.round(rows[:6], 3).tolist())
    print(' last rows', np.round(rows[-3:], 3).tolist())
    speed = rows[:, 3]
    for v in (5, 10, 15, 20, 22, 22.4):
        f = int(np.argmax(speed >= v))
        print(f'  speed>={v}: frame {f} t={f/50:.2f} along={rows[f,0]:.1f}')
    along, off, cls = rows[:, 0], rows[:, 1], rows[:, 2].astype(int)
    for cname, start, end, side in CORNERS:
        mask = (along >= start - 40) & (along < end + 40)
        frames = np.nonzero(mask)[0]
        runs = rle(cls[frames])
        text = ' '.join(f"{ {1: '-', 5: 'L', 6: 'R', 0: '0'}[c]}{n}" for c, _, n in runs)
        print(f' {cname} frames {frames[0]}..{frames[-1]} along {along[frames[0]]:.1f}..{along[frames[-1]]:.1f}')
        print('   keys:', text)
        prof = [(round(float(along[f]), 1), round(float(off[f]), 2)) for f in frames[::10]]
        print('   offs:', prof)
