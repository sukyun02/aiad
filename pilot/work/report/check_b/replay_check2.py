"""Same as replay_check.py but aligned at the first gas frame (the car stands still before it, so the physical
trajectory after the first gas frame depends only on the action sequence from there on)."""
import os
import numpy as np

R = '/home/sukyun02/workspace/aiad/pilot/work/report'
A = '/home/sukyun02/workspace/aiad/assignment1/data'


def to_class(a):
    steer, gas, brake = float(a[0]), float(a[1]), float(a[2])
    left, right, g, b = steer < -0.1, steer > 0.1, gas > 0.1, brake > 0.1
    if left and g: return 5
    if right and g: return 6
    if left and b: return 7
    if right and b: return 8
    if left: return 3
    if right: return 4
    if g: return 1
    if b: return 2
    return 0


laps = {}
for group in ('stage2', 'stage2_over41', 'collected'):
    for lap in sorted(os.listdir(f'{A}/{group}')):
        d = f'{A}/{group}/{lap}'
        if not os.path.isdir(d):
            continue
        names = sorted(n for n in os.listdir(d) if n.startswith('action_'))
        laps[lap] = (group, np.array([to_class(np.load(f'{d}/{n}', allow_pickle=False)) for n in names]))
np.savez(f'{R}/check_b/lap_classes.npz', **{k: v[1] for k, v in laps.items()})


def gas0(c):
    return int(np.argmax(np.isin(c, (1, 5, 6))))


def best(c, pool, top=4):
    c = c[gas0(c):]
    out = []
    for lap, (group, lc) in pool.items():
        l2 = lc[gas0(lc):]
        m = min(len(c), len(l2))
        diff = np.flatnonzero(c[:m] != l2[:m])
        out.append((int(diff[0]) if len(diff) else m, lap, group, len(lc), gas0(lc)))
    return sorted(out, reverse=True)[:top]


train = {k: v for k, v in laps.items() if v[0] != 'collected'}
for name in ('final_model', 'm_20261006_232759', 'm_20261006_233500', 'stage1_model'):
    c = np.load(f'{R}/{name}/trace.npz')['rows'][:, 3].astype(int)
    pool = laps if name == 'stage1_model' else train
    print('=====', name, 'frames', len(c), 'first gas', gas0(c), '(pool:', len(pool), 'laps)')
    for prefix, lap, group, n, hold in best(c, pool):
        print(f'  lap {lap} ({group}, {n} frames, hold {hold}): identical for the first {prefix} frames after the first gas frame')

# pairwise between training laps themselves: how long do two human laps stay identical after the first gas frame?
keys = list(train)
pre = []
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = train[keys[i]][1], train[keys[j]][1]
        a, b = a[gas0(a):], b[gas0(b):]
        m = min(len(a), len(b))
        diff = np.flatnonzero(a[:m] != b[:m])
        pre.append(int(diff[0]) if len(diff) else m)
pre = np.array(pre)
print('pairs of training laps: identical frames after first gas: min', pre.min(), 'median', np.median(pre), 'max', pre.max())
# first steering frame (after first gas) per training lap and per model
print('first steering frame after first gas, training laps:', sorted(int(np.argmax(np.isin(v[1][gas0(v[1]):], (5, 6)))) for v in train.values()))
for name in ('final_model', 'm_20261006_232759', 'm_20261006_233500', 'stage1_model'):
    c = np.load(f'{R}/{name}/trace.npz')['rows'][:, 3].astype(int)
    c = c[gas0(c):]
    sw = np.flatnonzero(np.diff(c) != 0) + 1
    print(name, 'first steering frame after first gas', int(np.argmax(np.isin(c, (5, 6)))), 'class switches in first 607 frames', int((sw < 607).sum()), 'switch frames', sw[:30].tolist())
