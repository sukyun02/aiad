"""check_a: recount action classes of the training laps and inspect trace.npz (numpy only, read-only)."""
import json
import os
import sys

import numpy as np

A = '/home/sukyun02/workspace/aiad/assignment1'
R = '/home/sukyun02/workspace/aiad/pilot/work/report'
OUT = os.path.join(R, 'check_a', 'check_data.json')


def cls(a):
    steer, gas, brake = float(a[0]), float(a[1]), float(a[2])
    left, right, g, b = steer < -0.1, steer > 0.1, gas > 0.1, brake > 0.1
    if left and g:
        return 5
    if right and g:
        return 6
    if left and b:
        return 7
    if right and b:
        return 8
    if left:
        return 3
    if right:
        return 4
    if g:
        return 1
    if b:
        return 2
    return 0


def lap_stats(folder):
    names = sorted(n for n in os.listdir(folder) if n.startswith('action_'))
    idx = [int(n[len('action_'):-4]) for n in names]
    assert idx == list(range(len(idx))), (folder, 'indices not contiguous')
    acts = np.stack([np.load(os.path.join(folder, n), allow_pickle=False) for n in names])
    c = np.array([cls(a) for a in acts])
    counts = np.bincount(c, minlength=9).tolist()
    zeros = np.flatnonzero(c == 0)
    lead = 0
    while lead < len(c) and c[lead] == 0:
        lead += 1
    zeros_all_leading = bool(len(zeros) == lead)
    # first frame where gas pressed
    gas_idx = np.flatnonzero(acts[:, 1] > 0.1)
    first_gas = int(gas_idx[0]) if len(gas_idx) else None
    # steering runs
    steer = np.where(acts[:, 0] < -0.1, -1, np.where(acts[:, 0] > 0.1, 1, 0))
    runs = []
    start = 0
    for i in range(1, len(steer) + 1):
        if i == len(steer) or steer[i] != steer[start]:
            if steer[start] != 0:
                runs.append((int(steer[start]), i - start))
            start = i
    obs0 = np.load(os.path.join(folder, 'observation_00000.npy'), allow_pickle=False)
    n_obs = sum(1 for n in os.listdir(folder) if n.startswith('observation_'))
    return dict(frames=len(c), n_obs=n_obs, counts=counts, lead_zero=lead, zeros=int(len(zeros)),
                zeros_all_leading=zeros_all_leading, first_gas=first_gas,
                zero_positions_after_lead=[int(z) for z in zeros if z >= lead][:20],
                uniq_steer=sorted(set(np.round(acts[:, 0], 3).tolist())),
                uniq_gas=sorted(set(np.round(acts[:, 1], 3).tolist())),
                uniq_brake=sorted(set(np.round(acts[:, 2], 3).tolist())),
                brake_frames=int((acts[:, 2] > 0.1).sum()),
                nogas_after_first=int((acts[first_gas:, 1] <= 0.1).sum()) if first_gas is not None else None,
                left_runs=[r[1] for r in runs if r[0] < 0], right_runs=[r[1] for r in runs if r[0] > 0],
                obs_shape=list(obs0.shape), obs_dtype=str(obs0.dtype), act_dtype=str(acts.dtype))


res = {}
for group in ['stage2', 'stage2_over41', 'collected']:
    base = os.path.join(A, 'data', group)
    for lap in sorted(os.listdir(base)):
        res[f'{group}/{lap}'] = lap_stats(os.path.join(base, lap))


def agg(keys, label):
    tot = np.zeros(9, int)
    frames = 0
    lead = 0
    allz = True
    lruns, rruns = [], []
    for k in keys:
        s = res[k]
        tot += np.array(s['counts'])
        frames += s['frames']
        lead += s['lead_zero']
        allz &= s['zeros_all_leading']
        lruns += s['left_runs']
        rruns += s['right_runs']
    share = (100 * tot / frames).round(3).tolist()
    print(f'== {label}: laps {len(keys)} frames {frames}')
    print('   counts', tot.tolist())
    print('   share %', share)
    print('   class0 total', int(tot[0]), 'sum of leading zeros', lead, 'all class-0 frames are leading:', allz)
    print('   left(3+5+7)', int(tot[3] + tot[5] + tot[7]), 'right(4+6+8)', int(tot[4] + tot[6] + tot[8]),
          'ratio', round(float(tot[3] + tot[5] + tot[7]) / max(1, float(tot[4] + tot[6] + tot[8])), 3))
    print('   lead_zero min/max', min(res[k]['lead_zero'] for k in keys), max(res[k]['lead_zero'] for k in keys))
    print('   brake frames', sum(res[k]['brake_frames'] for k in keys), 'no-gas after first gas',
          sum(res[k]['nogas_after_first'] for k in keys))
    for name, runs in (('left', lruns), ('right', rruns)):
        r = np.array(runs)
        print(f'   {name} steering presses: n={len(r)} median={np.median(r)} mean={r.mean():.2f} '
              f'p90={np.percentile(r, 90)} max={r.max()} share<=5fr={np.mean(r <= 5):.3f} share<=10fr={np.mean(r <= 10):.3f}')
    print('   steering presses per lap', round((len(lruns) + len(rruns)) / len(keys), 1))
    print('   uniq steer', sorted({v for k in keys for v in res[k]['uniq_steer']}),
          'gas', sorted({v for k in keys for v in res[k]['uniq_gas']}),
          'brake', sorted({v for k in keys for v in res[k]['uniq_brake']}))
    print('   obs shapes', sorted({tuple(res[k]['obs_shape']) for k in keys}), sorted({res[k]['obs_dtype'] for k in keys}),
          'n_obs==frames', all(res[k]['n_obs'] == res[k]['frames'] for k in keys))
    return dict(counts=tot.tolist(), share=share, frames=frames)


k21 = [k for k in res if k.startswith('stage2')]
k10 = [k for k in res if k.startswith('stage2/')]
k11 = [k for k in res if k.startswith('stage2_over41/')]
k7 = [k for k in res if k.startswith('collected/')]
summary = dict(all21=agg(k21, '21 laps (stage2 + stage2_over41)'), stage2_now=agg(k10, 'stage2 as it is now'),
               over41=agg(k11, 'stage2_over41'), provided7=agg(k7, 'provided 7 laps'))
for k in k21:
    s = res[k]
    print(k, s['frames'], 'lead0', s['lead_zero'], 'first_gas', s['first_gas'], 'zeros', s['zeros'], s['counts'])

# ---- trace.npz of the final model and of the stage1 model
for name in ['final_model', 'stage1_model']:
    z = np.load(os.path.join(R, name, 'trace.npz'), allow_pickle=True)
    print(f'== {name}/trace.npz keys', {k: (z[k].shape, str(z[k].dtype)) for k in z.files})
    for k in z.files:
        v = z[k]
        if v.dtype == object:
            try:
                print('  ', k, 'object; len', len(v), 'first shape', np.asarray(v[0]).shape)
            except Exception as e:  # noqa
                print('  ', k, 'object', e)
    cand = [k for k in z.files if 'cent' in k.lower() or 'line' in k.lower() or 'track' in k.lower()]
    print('   centre-line candidates', cand)
    for k in cand:
        p = np.asarray(z[k], float)
        if p.ndim == 2 and p.shape[1] >= 2:
            p = p[:, -2:] if p.shape[1] == 4 else p[:, :2]
            seg = np.roll(p, -1, axis=0) - p
            length = float(np.linalg.norm(seg, axis=1).sum())
            area = 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
            head = np.arctan2(seg[:, 1], seg[:, 0])
            dh = (np.roll(head, -1) - head + np.pi) % (2 * np.pi) - np.pi
            print(f'   {k}: points {len(p)} closed length {length:.2f} signed area {area:.1f} (>0 = counter-clockwise) '
                  f'total turning {np.degrees(dh.sum()):.1f} deg, left-turn sum {np.degrees(dh[dh > 0].sum()):.1f}, '
                  f'right-turn sum {np.degrees(dh[dh < 0].sum()):.1f}; 959/22.5 = {959 / 22.5:.3f}, len/22.5 = {length / 22.5:.3f}')
    rows = [k for k in z.files if z[k].ndim == 2 and z[k].shape[1] == 7]
    for k in rows:
        t = np.asarray(z[k], float)
        off = t[:, 2] > 0.5
        print(f'   {k}: frames {len(t)} off_road {int(off.sum())} ({100 * off.mean():.2f}%)')
        # consecutive off-road stretches
        st = []
        i = 0
        while i < len(off):
            if off[i]:
                j = i
                while j < len(off) and off[j]:
                    j += 1
                st.append((i, j - i))
                i = j
            else:
                i += 1
        print('   off-road stretches (start frame, length):', st)
        c = t[:, 3].astype(int)
        print('   class counts', np.bincount(c, minlength=9).tolist(), 'first non-zero class frame', int(np.flatnonzero(c != 0)[0]))
        sp = t[:, 4]
        print(f'   speed max {sp.max():.3f} mean after 100 {sp[100:].mean():.3f} min after 100 {sp[100:].min():.3f}; '
              f'frame where speed first >= 22.0: {int(np.flatnonzero(sp >= 22.0)[0])}')
        print(f'   |offset| max {np.abs(t[:, 6]).max():.2f}; along range {t[:, 5].min():.1f}..{t[:, 5].max():.1f}')
        d = np.linalg.norm(np.diff(t[:, :2], axis=0), axis=1).sum()
        print(f'   path length {d:.1f}')
        # offset when off_road toggles -> estimate of half width
        on_max = np.abs(t[~off, 6]).max() if (~off).any() else None
        off_min = np.abs(t[off, 6]).min() if off.any() else None
        print('   max |offset| while on road', on_max, ' min |offset| while off road', off_min)

json.dump(dict(laps=res, summary=summary), open(OUT, 'w'), indent=1)
print('written', OUT)
