"""Fact-check, part 1: numbers that can be read from files that already exist (no simulation, no torch).

Reads  $SP/work/stage2/out/*.json|npz  and writes  factcheck/out/existing.json  + prints tables.
"""
import json
import sys
from pathlib import Path

import numpy as np

SP = Path('/home/sukyun02/workspace/aiad/pilot')
OUT = SP / 'work/stage2/out'
HERE = Path(__file__).resolve().parent
(HERE / 'out').mkdir(exist_ok=True)

CORNERS = [('C1', 45.5, 77.0, 1), ('C2', 248.5, 273.0, 1), ('C3', 381.5, 399.0, 1), ('C4', 518.0, 549.5, 1),
           ('C5', 609.0, 633.5, 1), ('C6', 728.0, 738.5, 1), ('C7', 749.0, 766.5, -1), ('C8', 879.0, 913.5, 1)]
LENGTH = 959.0
result = {}


def load(name):
    return json.loads((OUT / f'{name}.json').read_text())


def rows_of(name):
    return np.load(OUT / f'{name}.npz')['rows']


# ---------------------------------------------------------------- provided human laps
provided = json.loads((OUT / 'laps_provided.json').read_text())
print('== provided human laps (laps_provided.json)')
for r in provided:
    print(f"  {r['lap']}  lap {r['lap_time']:.2f}  prog {r['progress']:.2f}  centre {r['center']:.2f}  hold {r['start_hold_frames']}  "
          f"coast {r['coast_frames']}  brake {r['brake_frames']}  maxoff {r['max_abs_offset']}")
result['provided'] = [dict(lap=r['lap'], lap_time=r['lap_time'], hold=r['start_hold_frames'], center=r['center'],
                           coast=r['coast_frames'], brake=r['brake_frames']) for r in provided]

# ---------------------------------------------------------------- scripted sets
print('\n== scripted sets (json)')
sets = {}
for prefix in ('A', 'E2', 'E3', 'E5', 'A16x', 'C'):
    items = []
    for path in sorted(OUT.glob(f'{prefix}_lap*.json'), key=lambda p: int(p.stem.split('lap')[1])):
        d = json.loads(path.read_text())
        items.append(d)
        print(f"  {path.stem:10s} E {d['E']:<4} W {d['W']:<4} k0 {d['k0']:<5} hold {d['hold']:<2} {d['status']:8s} lap {d['lap_time']}  prog {d['progress']:6.2f} "
              f"centre {d['center']:6.2f}  path {d['path_len']}  maxoff {d['max_abs_offset']}  minspeed {d['min_speed_after100']}  meanspeed {d['mean_speed']}")
    sets[prefix] = items
    fin = [d for d in items if d['status'] == 'finished']
    summary = dict(n=len(items), finished=len(fin),
                   lap_min=min(d['lap_time'] for d in fin), lap_max=max(d['lap_time'] for d in fin),
                   centre_min=min(d['center'] for d in fin), centre_max=max(d['center'] for d in fin),
                   centre_mean=round(float(np.mean([d['center'] for d in fin])), 2),
                   progress_min=min(d['progress'] for d in fin),
                   minspeed_min=min(d['min_speed_after100'] for d in fin),
                   maxoff_max=max(d['max_abs_offset'] for d in fin))
    result[f'set_{prefix}'] = summary
    print('   ->', summary)

e0 = load('E0')
print('\n== E0 (centre-line lap):', {k: e0[k] for k in ('lap_time', 'progress', 'center', 'path_len', 'min_speed_after100', 'frames')})
result['E0'] = {k: e0[k] for k in ('lap_time', 'progress', 'center', 'path_len', 'min_speed_after100', 'frames')}

# ---------------------------------------------------------------- E sweep s1
print('\n== s1 sweep (E x W x k0)')
sweep = []
for path in sorted(OUT.glob('s1_*.json')):
    d = json.loads(path.read_text())
    rows = np.load(path.with_suffix('.npz'))['rows']
    inside = {}
    for name, start, end, side in CORNERS:
        mask = (rows[:, 0] >= start - 15) & (rows[:, 0] < end + 15)
        inside[name] = round(float((side * rows[mask, 1]).max()), 2)
    sweep.append(dict(name=path.stem, E=d['E'], W=d['W'], k0=d['k0'], status=d['status'], lap_time=d['lap_time'], progress=d['progress'],
                      center=d['center'], maxoff=d['max_abs_offset'], minspeed=d['min_speed_after100'], inside=inside))
    print(f"  {path.stem:22s} {d['status']:8s} lap {d['lap_time']}  prog {d['progress']:6.2f}  centre {d['center']:6.2f}  maxoff {d['max_abs_offset']:5.2f} "
          f"minspeed {d['min_speed_after100']}  inside " + ' '.join(f"{inside[n]:5.2f}" for n in inside))
result['s1_sweep'] = sweep


# ---------------------------------------------------------------- claim 10 / 11: sector times from traces
def crossing_times(rows, marks):
    """Time (s) at which the unwrapped along-track distance first reaches each mark (linear interpolation).
    rows[i, 0] is the position BEFORE step i, i.e. at simulated time i/50."""
    along = rows[:, 0].astype(float)
    unwrapped = np.r_[along[0], along[0] + np.cumsum((np.diff(along) + LENGTH / 2) % LENGTH - LENGTH / 2)]
    if along[0] > LENGTH / 2:          # the start position may project to just before the line (along ~ 959)
        unwrapped -= LENGTH
    t = np.arange(len(along)) / 50.0
    out = []
    for m in marks:
        idx = int(np.argmax(unwrapped >= m)) if (unwrapped >= m).any() else None
        if idx is None:
            out.append(float('nan'))
        elif idx == 0:
            out.append(0.0)
        else:
            a, b = unwrapped[idx - 1], unwrapped[idx]
            out.append(float(t[idx - 1] + (m - a) / (b - a) * 0.02))
    return out


# partition 1: corner windows with +-15 margin (C6 and C7 merged because their windows overlap), rest = straights
windows = [('C1', 30.5, 92.0), ('C2', 233.5, 288.0), ('C3', 366.5, 414.0), ('C4', 503.0, 564.5), ('C5', 594.0, 648.5),
           ('C6+C7', 713.0, 781.5), ('C8', 864.0, 928.5)]
# partition 2: the whole lap cut at the midpoints between neighbouring corners (start line = midpoint C8/C1)
mids = [0.0]
for (n1, s1, e1, _), (n2, s2, e2, _) in zip(CORNERS[:-1], CORNERS[1:]):
    mids.append((e1 + s2) / 2)
mids.append(LENGTH)


def sector_times(name):
    rows = rows_of(name)
    lap = load(name)['lap_time']
    marks = sorted({v for _, a, b in windows for v in (a, b)})
    tc = dict(zip(marks, crossing_times(rows, marks)))
    win = {n: tc[b] - tc[a] for n, a, b in windows}
    win['straights'] = lap - sum(win.values())
    tm = crossing_times(rows, mids[:-1]) + [lap]
    mid = {CORNERS[i][0]: tm[i + 1] - tm[i] for i in range(8)}
    return win, mid, lap


win0, mid0, lap0 = sector_times('E0')
print('\n== claim 10: time per section, A laps minus E0 (seconds; negative = faster than the centre-line lap)')
gains_win, gains_mid = [], []
for i in range(1, 9):
    win, mid, lap = sector_times(f'A_lap{i}')
    gw = {k: win0[k] - win[k] for k in win}
    gm = {k: mid0[k] - mid[k] for k in mid}
    gains_win.append(gw)
    gains_mid.append(gm)
    print(f"  A_lap{i} lap {lap:.2f} gain {lap0 - lap:.2f} | +-15 windows: " + ' '.join(f"{k} {v:+.3f}" for k, v in gw.items()))
    print(f"           midpoint partition: " + ' '.join(f"{k} {v:+.3f}" for k, v in gm.items()))
mean_win = {k: float(np.mean([g[k] for g in gains_win])) for k in gains_win[0]}
mean_mid = {k: float(np.mean([g[k] for g in gains_mid])) for k in gains_mid[0]}
big_win = mean_win['C1'] + mean_win['C4'] + mean_win['C8']
big_mid = mean_mid['C1'] + mean_mid['C4'] + mean_mid['C8']
tot_mid = sum(mean_mid.values())
tot_win = sum(mean_win.values())
print('  mean over 8 A laps, +-15 windows :', {k: round(v, 3) for k, v in mean_win.items()}, '| C1+C4+C8', round(big_win, 3),
      '| other corners', round(tot_win - big_win - mean_win['straights'], 3), '| straights', round(mean_win['straights'], 3), '| total', round(tot_win, 3))
print('  mean over 8 A laps, midpoint part.:', {k: round(v, 3) for k, v in mean_mid.items()}, '| C1+C4+C8', round(big_mid, 3),
      '| other five', round(tot_mid - big_mid, 3), '| total', round(tot_mid, 3))
result['claim10'] = dict(E0_lap=lap0, mean_gain_windows={k: round(v, 3) for k, v in mean_win.items()},
                         mean_gain_midpoint={k: round(v, 3) for k, v in mean_mid.items()},
                         big3_windows=round(big_win, 3), other_corners_windows=round(tot_win - big_win - mean_win['straights'], 3),
                         straights_windows=round(mean_win['straights'], 3), total=round(tot_win, 3),
                         big3_midpoint=round(big_mid, 3), other5_midpoint=round(tot_mid - big_mid, 3),
                         per_lap_big3_midpoint=[round(g['C1'] + g['C4'] + g['C8'], 3) for g in gains_mid],
                         per_lap_total=[round(sum(g.values()), 3) for g in gains_mid])

print('\n== claim 11: time since start (s) at corner entry / exit (along-track marks of laps.py CORNERS, no margin)')
marks = [v for _, a, b, _ in CORNERS for v in (a, b)]
table = {}
for name in ['E0'] + [f'A_lap{i}' for i in range(1, 9)]:
    table[name] = crossing_times(rows_of(name), marks)
arr = np.array([table[f'A_lap{i}'] for i in range(1, 9)])
claim11 = {}
for k, (cn, a, b, _) in enumerate(CORNERS):
    lo_in, hi_in = arr[:, 2 * k].min(), arr[:, 2 * k].max()
    lo_out, hi_out = arr[:, 2 * k + 1].min(), arr[:, 2 * k + 1].max()
    claim11[cn] = dict(A_entry=[round(float(lo_in), 2), round(float(hi_in), 2)], A_exit=[round(float(lo_out), 2), round(float(hi_out), 2)],
                       E0_entry=round(table['E0'][2 * k], 2), E0_exit=round(table['E0'][2 * k + 1], 2))
    print(f"  {cn} ({a}-{b}): A laps entry {lo_in:.2f}-{hi_in:.2f}  exit {lo_out:.2f}-{hi_out:.2f} | E0 entry {table['E0'][2 * k]:.2f} exit {table['E0'][2 * k + 1]:.2f}")
result['claim11'] = claim11

# when does the steering actually start / end around each corner in the A laps (first/last steering-class frame inside the +-25 window)
print('  first / last frame with a steering key towards the bend inside [start-25, end+10] (A laps, seconds):')
steer_times = {}
for cn, a, b, side in CORNERS:
    firsts, lasts = [], []
    for i in range(1, 9):
        rows = rows_of(f'A_lap{i}')
        cls_in = 5 if side > 0 else 6
        mask = (rows[:, 0] >= a - 25) & (rows[:, 0] < b + 10) & (rows[:, 2] == cls_in) & (np.arange(len(rows)) > 20)
        idx = np.nonzero(mask)[0]
        firsts.append(idx[0] / 50)
        lasts.append(idx[-1] / 50)
    steer_times[cn] = [round(min(firsts), 2), round(max(firsts), 2), round(min(lasts), 2), round(max(lasts), 2)]
    print(f"    {cn}: first {min(firsts):.2f}-{max(firsts):.2f}  last {min(lasts):.2f}-{max(lasts):.2f}")
result['claim11_steer'] = steer_times

# ---------------------------------------------------------------- claim 13: speed through the bends
print('\n== claim 13: speed (trace column 3)')
claim13 = {}
for prefix in ('A', 'E3', 'E5', 'A16x'):
    per_corner = {cn: [] for cn, *_ in CORNERS}
    overall, reach = [], []
    for path in sorted(OUT.glob(f'{prefix}_lap*.npz')):
        rows = np.load(path)['rows']
        speed = rows[:, 3]
        first = int(np.argmax(speed >= 22.2))
        reach.append(first)
        overall.append(float(speed[first:].min()))
        for cn, a, b, _ in CORNERS:
            mask = (rows[:, 0] >= a - 15) & (rows[:, 0] < b + 15) & (np.arange(len(rows)) >= first)
            per_corner[cn].append(float(speed[mask].min()))
    claim13[prefix] = dict(frames_to_22p2=[min(reach), max(reach)], min_speed_after_reaching=round(min(overall), 3),
                           per_corner_min={k: round(min(v), 3) for k, v in per_corner.items()})
    print(f"  {prefix}: frames until speed>=22.2: {min(reach)}-{max(reach)} | min speed afterwards {min(overall):.3f} | per-corner min "
          + ' '.join(f"{k} {min(v):.2f}" for k, v in per_corner.items()))
rows = rows_of('E0')
first = int(np.argmax(rows[:, 3] >= 22.2))
claim13['E0'] = dict(frames_to_22p2=first, min_after=round(float(rows[first:, 3].min()), 3), max=round(float(rows[:, 3].max()), 3),
                     speed_at_frames={str(f): round(float(rows[f, 3]), 2) for f in (10, 25, 50, 75, 100, 150)},
                     along_at_frames={str(f): round(float(rows[f, 0]), 1) for f in (10, 25, 50, 75, 100, 150)})
print('  E0:', claim13['E0'])
result['claim13'] = claim13

(HERE / 'out' / 'existing.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))
print('\nwritten', HERE / 'out' / 'existing.json')
