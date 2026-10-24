"""Pooled key statistics, frames per corner section, and where constrained runs leave the road (numpy only).
Reads out/keystats.json, out/keystats_H8.json, traces/*.npz, ../out/E0.npz and every sweep result/trace in out/*/.
Writes out/analysis.json and prints the tables.
"""
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
CORNERS = [('C1', 45.5, 77.0, 1), ('C2', 248.5, 273.0, 1), ('C3', 381.5, 399.0, 1), ('C4', 518.0, 549.5, 1),
           ('C5', 609.0, 633.5, 1), ('C6', 728.0, 738.5, 1), ('C7', 749.0, 766.5, -1), ('C8', 879.0, 913.5, 1)]
MARGIN = 15.0
HALF = 6.67
result = {}

# ---- 1. pooled press / gap histograms -------------------------------------------------------------------
stats = json.loads((OUT / 'keystats.json').read_text()) + json.loads((OUT / 'keystats_H8.json').read_text())
groups = {'human': [s for s in stats if s['name'].startswith('H') and not s['name'].startswith('H8')],
          'scriptA': [s for s in stats if s['name'].startswith('A_')], 'scriptC': [s for s in stats if s['name'].startswith('C_')],
          'H8': [s for s in stats if s['name'].startswith('H8')]}
bins = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 10), (11, 15), (16, 999)]
print('pooled press lengths (frames): share of presses per bin')
print(f"{'set':8s} {'laps':>4s} {'presses':>7s} " + ' '.join(f"{(str(a) if a == b else f'{a}-{b}' if b < 999 else f'{a}+'):>6s}" for a, b in bins) + '   min/p25/med/p75/max')
result['pooled'] = {}
for name, laps in groups.items():
    lengths = np.array([l for s in laps for l in s['press_lengths']])
    gaps = np.array([g for s in laps for g in s['gaps']])
    hist = [float(((lengths >= a) & (lengths <= b)).mean()) for a, b in bins]
    ghist = [float(((gaps >= a) & (gaps <= b)).mean()) for a, b in [(0, 0)] + bins]
    q = lambda v: [float(v.min()), float(np.percentile(v, 25)), float(np.median(v)), float(np.percentile(v, 75)), float(v.max())]
    rng = lambda key: [min(s[key] for s in laps), max(s[key] for s in laps)]
    result['pooled'][name] = dict(laps=len(laps), presses=int(len(lengths)), press_hist=dict(zip([f'{a}-{b}' for a, b in bins], [round(h, 3) for h in hist])),
                                  press_q=q(lengths), gap_q=q(gaps), gap_hist=dict(zip(['0'] + [f'{a}-{b}' for a, b in bins], [round(h, 3) for h in ghist])),
                                  presses_per_lap=rng('steer_presses'), key_changes=rng('key_changes'), steer_share=rng('steer_share'),
                                  start_hold=rng('start_hold'), coast=rng('coast_frames'), brake=rng('brake_frames'), lap_time=rng('lap_time'),
                                  frames=rng('frames'), press_le2_share=round(float((lengths <= 2).mean()), 3), press_le3_share=round(float((lengths <= 3).mean()), 3),
                                  gap_le2_share=round(float((gaps <= 2).mean()), 3), gap_le3_share=round(float((gaps <= 3).mean()), 3),
                                  left_frames=rng('steer_frames_left'), right_frames=rng('steer_frames_right'))
    print(f"{name:8s} {len(laps):4d} {len(lengths):7d} " + ' '.join(f'{100 * h:6.1f}' for h in hist) + '   ' + '/'.join(f'{x:g}' for x in q(lengths)))
print('pooled gaps between presses (frames): share per bin (first column = direct left<->right switch)')
for name in groups:
    p = result['pooled'][name]
    print(f"{name:8s} " + ' '.join(f'{100 * h:6.1f}' for h in p['gap_hist'].values()) + '   ' + '/'.join(f'{x:g}' for x in p['gap_q']))

# ---- per corner: medians over the laps of each set ------------------------------------------------------
print('\nper corner, median over laps (range): inside-key presses | inside frames | longest inside press | counter frames | max inside offset')
result['corner_keys'] = {}
for name, laps in groups.items():
    result['corner_keys'][name] = {}
    line = []
    for c, *_ in CORNERS:
        vals = {k: [s['corners'][c][k] for s in laps] for k in ('inside_presses', 'inside_frames', 'inside_longest', 'counter_presses', 'counter_frames', 'inside_max_offset', 'outside_max_offset')}
        result['corner_keys'][name][c] = {k: dict(median=float(np.median(v)), min=float(min(v)), max=float(max(v))) for k, v in vals.items()}
        m = result['corner_keys'][name][c]
        line.append(f"{c}: {m['inside_presses']['median']:g}({m['inside_presses']['min']:g}-{m['inside_presses']['max']:g}) | {m['inside_frames']['median']:g}({m['inside_frames']['min']:g}-{m['inside_frames']['max']:g}) | "
                    f"{m['inside_longest']['median']:g} | {m['counter_frames']['median']:g} | {m['inside_max_offset']['median']:+.1f}")
    print(name)
    for l in line:
        print('   ' + l)

# ---- 2. frames per corner section -----------------------------------------------------------------------
def section_frames(along):
    out = {}
    used = np.zeros(len(along), dtype=bool)
    for c, start, end, _ in CORNERS:
        mask = (along >= start - MARGIN) & (along < end + MARGIN)
        out[c] = int(mask.sum())
        used |= mask
    out['rest'] = int((~used).sum())
    out['lap_frames'] = int(len(along))      # C6 and C7 sections overlap, so the sections do not add up to this
    return out

sec = {}
e0 = np.load('/home/sukyun02/workspace/aiad/pilot/work/stage2/out/E0.npz')['rows']
sec['E0_centre_line'] = [section_frames(e0[:, 0])]
for label, prefix in (('human', 'H?'), ('scriptA', 'A_lap*'), ('H8', 'H8_lap*')):
    sec[label] = []
    for f in sorted(glob.glob(str(HERE / 'traces' / f'{prefix}.npz'))):
        rows = np.load(f)['rows']
        hold = int(np.argmax(rows[:, 3] > 0.1))
        d = section_frames(rows[hold:, 0])        # the start hold is not counted
        d['hold'] = hold
        sec[label].append(d)
print('\nframes per corner section (corner +-15 units), start hold excluded; mean over laps; (difference to the centre-line lap E0)')
names = [c for c, *_ in CORNERS] + ['rest', 'lap_frames']
base = sec['E0_centre_line'][0]
result['section_frames'] = {}
print(f"{'set':16s} " + ' '.join(f'{n:>13s}' for n in names))
for label, laps in sec.items():
    mean = {n: float(np.mean([l[n] for l in laps])) for n in names}
    result['section_frames'][label] = dict(mean={n: round(v, 1) for n, v in mean.items()}, diff_to_E0={n: round(mean[n] - base[n], 1) for n in names},
                                           laps=len(laps))
    print(f"{label:16s} " + ' '.join(f"{mean[n]:6.1f}({mean[n] - base[n]:+5.1f})" for n in names))

# ---- 3. where constrained runs leave the road ------------------------------------------------------------
def corner_of(along):
    for c, start, end, side in CORNERS:
        if start - MARGIN <= along < end + MARGIN:
            return c, side
    return None, 0

episodes = []
runs = []
for f in sorted(glob.glob(str(OUT / '*' / '*.json'))):
    p = Path(f)
    if p.parent.name not in ('pre', 'tune', 'grid', 'follow', 'elimit', 'h8') or p.name.startswith(('H8_', 'keystats')):
        continue
    r = json.loads(p.read_text())
    if 'min_press' not in r:
        continue
    r['file'] = f'{p.parent.name}/{p.stem}'
    runs.append(r)
    if r['clean']:
        continue
    rows = np.load(p.with_suffix('.npz'))['rows']
    out = np.abs(rows[:, 1]) > HALF
    idx = np.where(out)[0]
    if len(idx) == 0:
        episodes.append(dict(file=r['file'], corner=corner_of(r['first_offroad_along'] or -1)[0], side='edge', along=r['first_offroad_along'], frames=r['offroad_frames'], peak=r['max_abs_offset']))
        continue
    start = idx[0]
    for a, b in zip(idx, list(idx[1:]) + [None]):
        if b is None or b != a + 1:
            seg = rows[start:a + 1]
            c, side = corner_of(float(seg[0, 0]))
            peak = float(seg[np.argmax(np.abs(seg[:, 1])), 1])
            where = 'straight' if c is None else ('inside' if side * peak > 0 else 'outside')
            episodes.append(dict(file=r['file'], corner=c or 'straight', side=where, along=round(float(seg[0, 0]), 1), frames=int(len(seg)), peak=round(peak, 2)))
            start = b
result['offroad_episodes'] = episodes
count = {}
for e in episodes:
    key = f"{e['corner']}:{e['side']}"
    count[key] = count.get(key, 0) + 1
first = {}
for r in runs:
    if not r['clean'] and r['first_offroad_along'] is not None:
        c, _ = corner_of(r['first_offroad_along'])
        first[c or 'straight'] = first.get(c or 'straight', 0) + 1
result['offroad_episode_counts'] = dict(sorted(count.items()))
result['first_offroad_corner_counts'] = dict(sorted(first.items()))
result['runs_total'] = len(runs)
result['runs_unclean'] = sum(not r['clean'] for r in runs)
print(f"\nconstrained runs: {len(runs)}, unclean: {result['runs_unclean']}")
print('off-road episodes (|offset| > 6.67) by corner section and side:', result['offroad_episode_counts'])
print('corner section of the FIRST off-road frame per unclean run:', result['first_offroad_corner_counts'])

# per corner: tracking error of the clean human-like runs (min-press >= 4, any react), relative to the target line
print('\nclean runs with min-press >= 4: per corner mean of [max inside offset - E, max outside offset], and rms tracking error in the section')
track = {c: [] for c, *_ in CORNERS}
wide = {c: [] for c, *_ in CORNERS}
over = {c: [] for c, *_ in CORNERS}
used = 0
for r in runs:
    if not r['clean'] or r['min_press'] < 4:
        continue
    p = OUT / (r['file'] + '.npz')
    rows = np.load(p)['rows']
    used += 1
    for c, start, end, side in CORNERS:
        mask = (rows[:, 0] >= start - MARGIN) & (rows[:, 0] < end + MARGIN)
        err = rows[mask, 1] - rows[mask, 5]
        track[c].append(float(np.sqrt(np.mean(err ** 2))))
        wide[c].append(float((-side * err).max()))          # furthest OUTSIDE of the target line
        over[c].append(float((side * err).max()))           # furthest INSIDE of the target line
result['corner_tracking_clean_minpress_ge4'] = dict(runs=used, per_corner={c: dict(rms=round(float(np.mean(track[c])), 2), outside_of_target_mean=round(float(np.mean(wide[c])), 2),
                                                    outside_of_target_max=round(float(np.max(wide[c])), 2), inside_of_target_mean=round(float(np.mean(over[c])), 2),
                                                    inside_of_target_max=round(float(np.max(over[c])), 2)) for c in track})
print(f'runs used: {used}')
for c, v in result['corner_tracking_clean_minpress_ge4']['per_corner'].items():
    print(f'   {c}: {v}')
(OUT / 'analysis.json').write_text(json.dumps(result, indent=1))
