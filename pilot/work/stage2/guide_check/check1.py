"""Independent recomputation of the corner guide numbers (no torch, no simulator).
Reads traces out/A_lap*.npz, E0.npz, E3_lap*.npz, guide/hold/H5_lap*.npz and the saved actions of data_s2/scriptA.
"""
import json
import sys
from pathlib import Path

import numpy as np

SP = Path('/home/sukyun02/workspace/aiad/pilot')
OUT = SP / 'work/stage2/out'
HERE = SP / 'work/stage2/guide_check'

CORNERS = [('C1', 45.5, 77.0, 1), ('C2', 248.5, 273.0, 1), ('C3', 381.5, 399.0, 1), ('C4', 518.0, 549.5, 1),
           ('C5', 609.0, 633.5, 1), ('C6', 728.0, 738.5, 1), ('C7', 749.0, 766.5, -1), ('C8', 879.0, 913.5, 1)]
SECTION = {'C1': (30, 95), 'C2': (236, 295), 'C3': (340, 460), 'C4': (500, 580), 'C5': (595, 658),
           'C6': (715, 747), 'C7': (747, 785), 'C8': (862, 933)}
KERB_START = {'C1': 52.5, 'C2': 255.5, 'C4': 525.0, 'C5': 612.5, 'C8': 889.0}


def load(path):
    rows = np.load(path)['rows'].astype(np.float64)
    return rows


def runs(mask):
    """list of (start, length) of True runs"""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, len(mask) - start))
    return out


def corner_stats(rows, name, a0, a1, side):
    along, off, cls, speed, steer = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4]
    n = len(rows)
    frames = np.arange(n)
    inside = side * off
    # guard against wrap of along at the start/finish (frames near the end may have along ~0)
    mask = (along >= a0 - 15) & (along < a1 + 15)
    if name == 'C1':
        mask &= frames < n // 2
    if name == 'C8':
        mask &= frames > n // 2
    idx = frames[mask]
    apex = int(idx[np.argmax(inside[idx])])
    # turn-in: start of the unbroken run inside>=1 containing the apex
    f = apex
    while f > 0 and inside[f - 1] >= 1.0:
        f -= 1
    turn = f
    f = apex
    while f < n - 1 and inside[f] >= 1.0:
        f += 1
    back = f
    s0, s1 = SECTION[name]
    smask = (along >= s0) & (along < s1)
    if name == 'C1':
        smask &= frames < n // 2
    if name == 'C8':
        smask &= frames > n // 2
    sidx = frames[smask]
    sec_frames = len(sidx)
    inkey = 5 if side > 0 else 6      # class 5 = steer -1 = LEFT, class 6 = steer +1 = RIGHT
    outkey = 6 if side > 0 else 5
    c = cls[sidx]
    share_in = float((c == inkey).mean())
    share_out = float((c == outkey).mean())
    share_none = float((c == 1).mean())
    # longest inside-key press that lies (starts) in the section
    press = runs(cls == inkey)
    press_in = [(s, l) for s, l in press if smask[s]]
    longest = max(press_in, key=lambda t: t[1]) if press_in else (None, 0)
    # steering magnitude towards the inside in the section (controller.steering negative = left)
    st = -side * steer[sidx]
    peak = float(st.max())
    over15 = int((st >= 0.15).sum())
    full = bool((np.abs(steer[sidx]) >= 0.45 - 1e-6).any())
    # first inside key before the apex: walk back from turn-in to find start of approach (first press in section)
    first_press = press_in[0][0] if press_in else None
    return dict(apex_f=apex, apex_along=float(along[apex]), apex_inside=float(inside[apex]), t_apex=apex / 50,
                turn_f=turn, turn_along=float(along[turn]), t_turn=turn / 50,
                back_f=back, back_along=float(along[back]), t_back=back / 50,
                sec_time=sec_frames / 50, share_in=share_in, share_out=share_out, share_none=share_none,
                longest_len=longest[1], longest_start_f=longest[0],
                longest_start_along=float(along[longest[0]]) if longest[0] is not None else None,
                longest_start_rel_apex=(longest[0] - apex) / 50 if longest[0] is not None else None,
                peak_steer=peak, frames_ge015=over15, full_lock=full,
                n_in_press=len(press_in), n_out_press=len([1 for s, l in runs(cls == outkey) if smask[s]]),
                in_total_frames=int((c == inkey).sum()),
                first_press_f=first_press, first_press_along=float(along[first_press]) if first_press is not None else None)


def summarize(label, files, keys=None):
    per = {name: [] for name, *_ in CORNERS}
    for path in files:
        rows = load(path)
        for name, a0, a1, side in CORNERS:
            per[name].append(corner_stats(rows, name, a0, a1, side))
    print(f'===== set {label}: {len(files)} laps')
    result = {}
    for name, *_ in CORNERS:
        result[name] = {}
        line = [name]
        for key in ['apex_along', 't_apex', 'apex_inside', 'turn_along', 't_turn', 'back_along', 't_back', 'sec_time',
                    'share_in', 'share_out', 'share_none', 'longest_len', 'longest_start_along', 'longest_start_rel_apex',
                    'peak_steer', 'frames_ge015', 'n_in_press', 'n_out_press', 'in_total_frames', 'first_press_along']:
            vals = np.array([d[key] for d in per[name] if d[key] is not None], dtype=float)
            result[name][key] = dict(med=float(np.median(vals)), min=float(vals.min()), max=float(vals.max()),
                                     all=[round(float(v), 3) for v in vals])
        result[name]['full_lock'] = [d['full_lock'] for d in per[name]]
        r = result[name]
        print(f"{name}: apex along {r['apex_along']['med']:.1f} [{r['apex_along']['min']:.1f}-{r['apex_along']['max']:.1f}]"
              f" t {r['t_apex']['med']:.2f} | inside med {r['apex_inside']['med']:.2f} min {r['apex_inside']['min']:.2f} max {r['apex_inside']['max']:.2f}"
              f" | turn-in along {r['turn_along']['med']:.1f} t {r['t_turn']['med']:.2f} [{r['t_turn']['min']:.2f}-{r['t_turn']['max']:.2f}]"
              f" | back along {r['back_along']['med']:.1f} t {r['t_back']['med']:.2f}"
              f" | section {r['sec_time']['med']:.2f} [{r['sec_time']['min']:.2f}-{r['sec_time']['max']:.2f}]")
        print(f"    keys: inside {100*r['share_in']['med']:.0f}% [{100*r['share_in']['min']:.0f}-{100*r['share_in']['max']:.0f}]"
              f" outside {100*r['share_out']['med']:.0f}% [{100*r['share_out']['min']:.0f}-{100*r['share_out']['max']:.0f}]"
              f" none {100*r['share_none']['med']:.0f}% [{100*r['share_none']['min']:.0f}-{100*r['share_none']['max']:.0f}]"
              f" | longest inside press {r['longest_len']['med']:.1f} [{r['longest_len']['min']:.0f}-{r['longest_len']['max']:.0f}] frames,"
              f" start along {r['longest_start_along']['med']:.1f}, {r['longest_start_rel_apex']['med']:+.2f}s rel apex"
              f" | peak steer {r['peak_steer']['med']:.2f} [{r['peak_steer']['min']:.2f}-{r['peak_steer']['max']:.2f}]"
              f" >=0.15 for {r['frames_ge015']['med']/50:.2f}s | full lock laps {sum(r['full_lock'])}/{len(r['full_lock'])}"
              f" | presses in {r['n_in_press']['med']:.1f} [{r['n_in_press']['min']:.0f}-{r['n_in_press']['max']:.0f}] ({r['in_total_frames']['med']/50:.2f}s)"
              f" out {r['n_out_press']['med']:.1f} [{r['n_out_press']['min']:.0f}-{r['n_out_press']['max']:.0f}]"
              f" | first inside press in section along {r['first_press_along']['med']:.1f} [{r['first_press_along']['min']:.1f}-{r['first_press_along']['max']:.1f}]")
    return result, per


all_out = {}
A_files = [OUT / f'A_lap{i}.npz' for i in range(1, 9)]
all_out['A'], perA = summarize('A', A_files)
print('--- set A lap 1 only')
for name, *_ in CORNERS:
    d = perA[name][0]
    print(f"{name}: turn {d['t_turn']:.2f} (along {d['turn_along']:.1f}) apex {d['t_apex']:.2f} (along {d['apex_along']:.1f}, inside {d['apex_inside']:.2f}) back {d['t_back']:.2f} (along {d['back_along']:.1f})")
all_out['E0'], perE0 = summarize('E0', [OUT / 'E0.npz'])
all_out['E3'], _ = summarize('E3', [OUT / f'E3_lap{i}.npz' for i in range(1, 9)])
H5 = [SP / f'work/stage2/guide/hold/H5_lap{i}.npz' for i in range(1, 9)]
if all(p.exists() for p in H5):
    all_out['H5'], perH = summarize('H5', H5)

# lap totals
print('===== lap frames')
for path in A_files + [OUT / 'E0.npz']:
    rows = load(path)
    js = json.loads(path.with_suffix('.json').read_text())
    print(path.name, 'frames', len(rows), 'lap_time', js['lap_time'], 'progress', js['progress'], 'center', js['center'],
          'max_abs_offset', js['max_abs_offset'], 'counts', js['action_counts'])

# cross-check trace classes against saved actions (scriptA) and key statistics from the action files
print('===== saved actions of data_s2/scriptA vs traces')
for i in range(1, 9):
    folder = SP / f'data_s2/scriptA/lap{i}'
    paths = sorted(folder.glob('action_*.npy'))
    acts = np.array([np.load(p) for p in paths])
    rows = load(OUT / f'A_lap{i}.npz')
    steer = acts[:, 0]
    cls_from_act = np.where(steer < -0.5, 5, np.where(steer > 0.5, 6, 1))
    same = bool(len(acts) == len(rows) and np.array_equal(cls_from_act, rows[:, 2].astype(int)))
    gas0 = int((acts[:, 1] <= 0.1).sum())
    brake = int((acts[:, 2] > 0.1).sum())
    press = runs(steer != 0)
    # presses = runs of the same nonzero key
    allruns = runs(steer < -0.5) + runs(steer > 0.5)
    lens = np.array([l for _, l in allruns])
    print(f'lap{i}: actions {len(acts)} trace {len(rows)} classes identical {same} | no-gas frames {gas0} brake frames {brake}'
          f' | key presses {len(allruns)} share<=2 frames {100*(lens<=2).mean():.0f}% longest {lens.max()}'
          f' | unique steer {np.unique(steer).tolist()} unique gas {np.unique(acts[:,1]).tolist()}')

(HERE / 'check1.json').write_text(json.dumps(all_out, indent=1))
