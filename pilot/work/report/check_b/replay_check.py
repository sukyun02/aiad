"""Does the final model replay one of its training laps? Compare the driven action-class sequence with the recorded
key sequence of every training lap, frame by frame from frame 0 (numpy only, reads action_*.npy read-only)."""
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
for group in ('stage2', 'stage2_over41'):
    for lap in sorted(os.listdir(f'{A}/{group}')):
        d = f'{A}/{group}/{lap}'
        names = sorted(n for n in os.listdir(d) if n.startswith('action_'))
        laps[lap] = (group, np.array([to_class(np.load(f'{d}/{n}', allow_pickle=False)) for n in names]))
print('laps', len(laps), 'frames', sum(len(v[1]) for v in laps.values()))
allc = np.concatenate([v[1] for v in laps.values()])
print('class counts of the 21 laps', np.bincount(allc, minlength=9).tolist())
print('first gas frame per lap', {k[9:15]: int(np.argmax(np.isin(v[1], (1, 5, 6)))) for k, v in laps.items()})

for name in ('final_model', 'm_20261006_232759', 'm_20261006_233500'):
    rows = np.load(f'{R}/{name}/trace.npz')['rows']
    c = rows[:, 3].astype(int)
    print('=====', name, 'frames', len(c), 'first gas', int(np.argmax(np.isin(c, (1, 5, 6)))))
    out = []
    for lap, (group, lc) in laps.items():
        m = min(len(c), len(lc))
        diff = np.flatnonzero(c[:m] != lc[:m])
        prefix = int(diff[0]) if len(diff) else m
        out.append((prefix, lap, group, len(lc), float((c[:m] == lc[:m]).mean()), len(diff)))
    for prefix, lap, group, n, agree, nd in sorted(out, reverse=True)[:6]:
        print(f'  lap {lap} ({group}, {n} frames = {n / 50:.2f}s): identical prefix {prefix} frames, same-index agreement {agree:.4f}, mismatches {nd}')

# detail for the best matching lap of the final model: where do the mismatches sit
rows = np.load(f'{R}/final_model/trace.npz')['rows']
c = rows[:, 3].astype(int)
best = max(laps.items(), key=lambda kv: (np.flatnonzero(c[:min(len(c), len(kv[1][1]))] != kv[1][1][:min(len(c), len(kv[1][1]))])[:1].tolist() or [10 ** 9])[0])
lap, (group, lc) = best
m = min(len(c), len(lc))
diff = np.flatnonzero(c[:m] != lc[:m])
print('best lap', lap, 'len', len(lc), 'model len', len(c), 'mismatch frames (first 40)', diff[:40].tolist())
print('along at first mismatches', np.round(rows[diff[:10], 5], 1).tolist())
# steering frames: model vs that lap
print('model steering frames', int(np.isin(c, (5, 6)).sum()), 'lap steering frames', int(np.isin(lc, (5, 6)).sum()))
