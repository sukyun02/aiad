"""Corner numbers for the driver's guide, from the saved traces (numpy only, no simulation).

Inputs : $SP/work/stage2/out/A_lap1..8.npz (reference, E about 4), E3_lap1..8.npz (milder), E0.npz (centre line),
         guide/human/*.npz (the 7 provided keyboard laps, written by replay_trace.py), guide/centre_lap.npz.
Outputs: guide/corner_numbers.json, guide/frames_A1.json, guide/frames_centre.json, and a text report on stdout.
Definitions (frame f = state before action f; time = f / 50 s since the first frame of the lap):
  inside offset s = side * signed lateral offset (side +1 for a left bend, -1 for a right bend)
  apex      = frame of max s with along-track in [corner start - 15, corner end + 15]   (as laps.py)
  turn-in   = first frame of the unbroken run with s >= 1.0 that contains the apex
  back      = first frame after the apex with s < 1.0
  section   = fixed along-track window per corner (SECTIONS), chosen to contain every A lap's turn-in..back run
"""
import json
from pathlib import Path
import numpy as np

SP = Path('/home/sukyun02/workspace/aiad/pilot')
OUT = SP / 'work/stage2/out'
GUIDE = SP / 'work/stage2/guide'
LENGTH = 959.0
CORNERS = [('C1', 45.5, 77.0, 1, 76), ('C2', 248.5, 273.0, 1, 42), ('C3', 381.5, 399.0, 1, 19), ('C4', 518.0, 549.5, 1, 79),
           ('C5', 609.0, 633.5, 1, 32), ('C6', 728.0, 738.5, 1, 14), ('C7', 749.0, 766.5, -1, 42), ('C8', 879.0, 913.5, 1, 97)]
MARGIN = 15.0
# tiling of the lap: (name, start, end). Corner sections carry the corner's name.
SEGMENTS = [('start straight', -1, 30), ('C1', 30, 95), ('straight 1-2', 95, 190), ('right kink K1', 190, 236),
            ('C2', 236, 295), ('straight 2-3', 295, 340), ('C3', 340, 460), ('straight 3-4', 460, 500), ('C4', 500, 580),
            ('straight 4-5', 580, 595), ('C5', 595, 658), ('right kink K2', 658, 715), ('C6', 715, 747), ('C7', 747, 785),
            ('straight 7-8', 785, 862), ('C8', 862, 933), ('finish straight', 933, 959.001)]
SECTION = {name: (a, b) for name, a, b in SEGMENTS}
THRESHOLD = 1.0
DEG_PER_IMPULSE = 22.5 / 3.24 / 50 * 180 / 3.141592653589793


def load_scripted(path):
    rows = np.load(path)['rows'].astype(float)
    cls = rows[:, 2].astype(int)
    key = np.where(cls == 5, -1, np.where(cls == 6, 1, 0))
    return dict(name=path.stem, along=fix_along(rows[:, 0]), off=rows[:, 1], key=key, speed=rows[:, 3], state=rows[:, 4],
                gas=(cls != 0).astype(float), frames=len(rows))


def load_replayed(path):
    rows = np.load(path)['rows']
    return dict(name=path.stem, along=fix_along(rows[:, 0]), off=rows[:, 1], key=np.round(rows[:, 2]).astype(int),
                speed=rows[:, 3], state=rows[:, 4], gas=rows[:, 8], brake=rows[:, 9], frames=len(rows))


def fix_along(along):
    along = along.copy()
    head = np.arange(len(along)) < 100
    along[head & (along > LENGTH / 2)] -= LENGTH          # the car starts a hair before the line: 959.0 -> 0.0
    return along


def runs(mask):
    """(start, length) of every run of True."""
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


def merged_runs(key, wanted, gap=2):
    """Runs of `wanted`, joined when separated by at most `gap` frames in which no opposite key is pressed."""
    base = runs(key == wanted)
    out = []
    for start, length in base:
        if out and start - (out[-1][0] + out[-1][1]) <= gap and not (key[out[-1][0] + out[-1][1]:start] == -wanted).any():
            out[-1] = (out[-1][0], start + length - out[-1][0])
        else:
            out.append((start, length))
    return out


def corner_frames(lap, corner):
    name, start, end, side, _ = corner
    along, s = lap['along'], side * lap['off']
    window = np.nonzero((along >= start - MARGIN) & (along < end + MARGIN))[0]
    apex = int(window[np.argmax(s[window])])
    result = dict(apex_frame=apex, apex_along=float(along[apex]), apex_inside=float(s[apex]))
    if s[apex] < THRESHOLD:
        return dict(result, turn_frame=None, back_frame=None)
    a = apex
    while a > 0 and s[a - 1] >= THRESHOLD:
        a -= 1
    b = apex
    while b < len(s) - 1 and s[b] >= THRESHOLD:
        b += 1
    return dict(result, turn_frame=a, turn_along=float(along[a]), back_frame=b, back_along=float(along[b]))


def phases(lap, corner, cf):
    """Driver-facing phases around one bend of one lap (frames; None when the line never leaves the centre by 1.0)."""
    name, start, end, side, _ = corner
    if cf['turn_frame'] is None:
        return None
    key, s = lap['key'], side * lap['off']
    inside = key == -side
    steer = side * -lap['state']                                  # steering towards the inside of the bend
    # first inside key of the approach: walk back from turn-in while inside presses are less than 10 frames apart
    first, gap, f = cf['turn_frame'], 0, cf['turn_frame']
    while f > 0 and gap < 10:
        f -= 1
        if inside[f]:
            first, gap = f, 0
        else:
            gap += 1
    drift = cf['turn_frame']
    while drift > 0 and s[drift - 1] >= 0.3:
        drift -= 1
    lo, hi = max(0, cf['apex_frame'] - 60), min(lap['frames'], cf['apex_frame'] + 60)
    press = max(runs(inside[lo:hi]), key=lambda r: r[1], default=(0, 0))
    hard = np.nonzero(steer[lo:hi] >= 0.15)[0]
    # steering "impulse" = sum of the steering state over frames. Kinematic estimate of the heading change it produces:
    # degrees = impulse * (speed 22.5 / wheelbase 3.24) / 50 fps * 180/pi = impulse * 7.96 (checked against the bend angles).
    a0, a1, a2, a3 = first, max(first, cf['apex_frame'] - 15), cf['apex_frame'] + 25, min(lap['frames'], cf['back_frame'] + 50)
    imp = lambda x, y: float(steer[x:y].sum())
    extra = dict(approach_impulse=imp(a0, a1), bend_impulse=imp(a1, a2), exit_impulse=imp(a2, a3),
                 approach_deg=imp(a0, a1) * DEG_PER_IMPULSE, bend_deg=imp(a1, a2) * DEG_PER_IMPULSE, exit_deg=imp(a2, a3) * DEG_PER_IMPULSE,
                 approach_frames=a1 - a0, bend_single_press_frames=single_hold_equivalent(max(imp(a1, a2), 0.0)),
                 approach_single_press_frames=single_hold_equivalent(max(imp(a0, a1), 0.0)),
                 approach_inside_key_frames=int(inside[a0:a1].sum()), approach_outside_key_frames=int((key[a0:a1] == side).sum()),
                 bend_inside_key_frames=int(inside[a1:a2].sum()), exit_inside_key_frames=int(inside[a2:a3].sum()),
                 exit_outside_key_frames=int((key[a2:a3] == side).sum()))
    return dict(extra, first_key_frame=first, first_key_along=float(lap['along'][first]), first_key_rel_apex=(first - cf['apex_frame']) / 50,
                drift_frame=drift, drift_along=float(lap['along'][drift]),
                press_frames=int(press[1]), press_start_rel_apex=(lo + press[0] - cf['apex_frame']) / 50,
                press_start_along=float(lap['along'][lo + press[0]]),
                hard_frames=int(len(hard)), hard_start_rel_apex=None if not len(hard) else (lo + hard[0] - cf['apex_frame']) / 50,
                hard_end_rel_apex=None if not len(hard) else (lo + hard[-1] + 1 - cf['apex_frame']) / 50,
                peak_steer=float(steer[lo:hi].max()), peak_steer_rel_apex=(lo + int(np.argmax(steer[lo:hi])) - cf['apex_frame']) / 50,
                frames_turn_to_apex=cf['apex_frame'] - cf['turn_frame'], frames_apex_to_back=cf['back_frame'] - cf['apex_frame'])


def key_runs_text(lap, cf, side, half=90):
    """Run-length text of the steering keys around the apex, e.g. '-12 L5 -7 L14': 12 frames none, LEFT 5 frames, ..."""
    lo, hi = max(0, cf['apex_frame'] - half), min(lap['frames'], cf['apex_frame'] + half)
    key = lap['key'][lo:hi]
    out, start = [], 0
    for i in range(1, len(key) + 1):
        if i == len(key) or key[i] != key[start]:
            mark = ' <apex> ' if lo + start <= cf['apex_frame'] < lo + i else ''
            out.append(f"{mark}{ {-1: 'L', 1: 'R', 0: '-'}[int(key[start])]}{i - start}")
            start = i
    return ' '.join(out)


def phase_stats(laps, corner, cfs):
    rows = [phases(l, corner, cf) for l, cf in zip(laps, cfs)]
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    return {k: stat([r[k] for r in rows], 1 if 'along' in k else 2) for k in rows[0]}


def segment_frames(lap, a, b):
    return np.nonzero((lap['along'] >= a) & (lap['along'] < b))[0]


def key_stats(lap, frames, side):
    key, state = lap['key'][frames], lap['state'][frames]
    inside_key = -side                                       # left bend -> key -1
    inside_runs = runs(key == inside_key)
    outside_runs = runs(key == -inside_key)
    merged = merged_runs(key, inside_key)
    longest = max(inside_runs, key=lambda r: r[1]) if inside_runs else (0, 0)
    longest_merged = max(merged, key=lambda r: r[1]) if merged else (0, 0)
    inside_state = np.clip(-side * -state, 0, None) if False else np.clip(side * -state, 0, None)   # steering towards the inside
    return dict(frames=len(frames), inside_share=float((key == inside_key).mean()), outside_share=float((key == -inside_key).mean()),
                none_share=float((key == 0).mean()), inside_frames=int((key == inside_key).sum()),
                outside_frames=int((key == -inside_key).sum()), inside_presses=len(inside_runs), outside_presses=len(outside_runs),
                longest_inside=int(longest[1]), longest_inside_start=int(frames[longest[0]]) if inside_runs else None,
                longest_inside_merged=int(longest_merged[1]),
                longest_inside_merged_start=int(frames[longest_merged[0]]) if merged else None,
                longest_outside=int(max((r[1] for r in outside_runs), default=0)),
                inside_impulse=float(inside_state.sum()), peak_inside_steer=float(inside_state.max()),
                coast_frames=int((lap['gas'][frames] <= 0.1).sum()))


def stat(values, digits=2):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return dict(median=round(float(np.median(values)), digits), min=round(float(np.min(values)), digits),
                max=round(float(np.max(values)), digits), n=len(values))


def fmt(s, unit=''):
    if s is None:
        return 'n/a'
    return f"{s['median']}{unit} [{s['min']}..{s['max']}]"


def single_hold_equivalent(impulse):
    """Frames of ONE continuous key press (then release) that give the same summed steering as `impulse`
    (steering ramps 0.03 per frame up to 0.45 and back: n frames -> 0.03 n^2 for n <= 15, then +0.45 per frame)."""
    if impulse <= 6.75:
        return float(np.sqrt(impulse / 0.03))
    return 15 + (impulse - 6.75) / 0.45


def timeline(laps, frames_by_lap, side, half=90, step=10):
    """Key use in 0.2 s bins relative to each lap's own apex frame: median share of inside / outside key and mean steering."""
    out = []
    for lo in range(-half, half, step):
        ins, outs, steer, offs, alongs = [], [], [], [], []
        for lap, cf in zip(laps, frames_by_lap):
            a, b = cf['apex_frame'] + lo, cf['apex_frame'] + lo + step
            if a < 0 or b > lap['frames']:
                continue
            key = lap['key'][a:b]
            ins.append(float((key == -side).mean()))
            outs.append(float((key == side).mean()))
            steer.append(float((side * -lap['state'][a:b]).mean()))
            offs.append(float((side * lap['off'][a:b]).mean()))
            alongs.append(float(lap['along'][a:b].mean()))
        out.append(dict(t_rel=(lo / 50, (lo + step) / 50), inside=round(float(np.median(ins)), 2),
                        outside=round(float(np.median(outs)), 2), steer=round(float(np.median(steer)), 3),
                        offset=round(float(np.median(offs)), 2), along=round(float(np.median(alongs)), 1)))
    return out


# kerbs.json is written by trackmap.py (which itself needs corner_numbers.json): on a first run it is missing, the kerb
# distances are then skipped; run analyze.py again after trackmap.py.
KERBS = json.loads((GUIDE / 'kerbs.json').read_text())['kerbs'] if (GUIDE / 'kerbs.json').exists() else []


def main():
    A = [load_scripted(OUT / f'A_lap{i}.npz') for i in range(1, 9)]
    E3 = [load_scripted(OUT / f'E3_lap{i}.npz') for i in range(1, 9)]
    E0 = load_scripted(OUT / 'E0.npz')
    centre = load_scripted(GUIDE / 'centre_lap.npz')
    human = [load_replayed(p) for p in sorted((GUIDE / 'human').glob('*.npz'))]
    # set H5: the 8 parameter combinations of set A driven by driver_hold.py --min-hold 5 (every key state >= 0.10 s)
    H5 = [load_scripted(GUIDE / 'hold' / f'H5_lap{i}.npz') for i in range(1, 9)]
    h5_results = [json.loads((GUIDE / 'hold' / f'H5_lap{i}.json').read_text()) for i in range(1, 9)]
    hold_scan = {n: json.loads((GUIDE / 'hold' / f'N{n}.json').read_text()) for n in (0, 3, 4, 5, 6, 8)}
    print('set H5 (min key state 5 frames):', [(round(r['lap_time'], 2), r['progress'], r['center'], r['max_abs_offset']) for r in h5_results])
    print('min-hold scan on the lap-1 line (E 4, W 28, k0 0.03):', {n: (round(r['lap_time'], 2), r['progress'], r['max_abs_offset']) for n, r in hold_scan.items()})
    depth_scan = {name: json.loads((GUIDE / 'hold' / f'{name}.json').read_text()) for name in ('E0_N5', 'E3_N4', 'E3_N5', 'E5_N5')}
    print('depth scan with human-rate keys (name = E<depth>_N<min-hold>; W 28, k0 0.03 except E0):',
          {n: (round(r['lap_time'], 2), r['progress'], r['center'], r['max_abs_offset']) for n, r in depth_scan.items()})
    lap_time = lambda name: json.loads((OUT / f'{name}.json').read_text())['lap_time']
    print('laps: A', [lap_time(l['name']) for l in A], ' E3', [lap_time(l['name']) for l in E3], ' E0', lap_time('E0'),
          ' own centre lap frames', centre['frames'], ' identical to E0 trace:',
          bool(centre['frames'] == E0['frames'] and np.allclose(centre['off'], E0['off']) and (centre['key'] == E0['key']).all()))
    print('human laps (frames/50):', [round(l['frames'] / 50, 2) for l in human])
    report = dict(definitions=__doc__, corners={}, segments=[], start={}, sets=dict(
        A=[dict(lap=l['name'], lap_time=lap_time(l['name']), frames=l['frames']) for l in A],
        E3=[dict(lap=l['name'], lap_time=lap_time(l['name']), frames=l['frames']) for l in E3],
        E0=dict(lap_time=lap_time('E0'), frames=E0['frames']),
        human=[dict(lap=l['name'], frames=l['frames'], lap_time=l['frames'] / 50) for l in human],
        H5=[dict(lap=f'H5_lap{i + 1}', lap_time=round(r['lap_time'], 2), progress=r['progress'], center=r['center'],
                 max_abs_offset=r['max_abs_offset'], E=r['E'], W=r['W'], k0=r['k0'], lookahead=r['lookahead'], gain=r['gain'])
            for i, r in enumerate(h5_results)],
        depth_scan_human_rate={n: dict(E=r['E'], min_hold=r['min_hold'], lap_time=round(r['lap_time'], 2), progress=r['progress'],
                                       center=r['center'], max_abs_offset=r['max_abs_offset']) for n, r in depth_scan.items()},
        min_hold_scan={str(n): dict(lap_time=round(r['lap_time'], 2), progress=r['progress'], center=r['center'],
                                    max_abs_offset=r['max_abs_offset'], track_err_rms=r['track_err_rms']) for n, r in hold_scan.items()}))

    # ---- segments: where the time goes ---------------------------------------------------------------
    print('\n== time per segment (s = frames / 50); gain = E0 - set median')
    print(f"{'segment':16s} {'along':>11s} {'E0':>6s} {'A med':>6s} {'A min':>6s} {'A max':>6s} {'gain A':>7s} {'E3 med':>7s} {'gain E3':>8s} {'human med':>10s} {'hum-E0':>7s}")
    total = dict(E0=0.0, A=0.0, E3=0.0, human=0.0)
    for name, a, b in SEGMENTS:
        t0 = len(segment_frames(E0, a, b)) / 50
        tA = [len(segment_frames(l, a, b)) / 50 for l in A]
        t3 = [len(segment_frames(l, a, b)) / 50 for l in E3]
        tH = [len(segment_frames(l, a, b)) / 50 for l in human]
        t5 = [len(segment_frames(l, a, b)) / 50 for l in H5]
        row = dict(segment=name, along=[a, round(b, 0)], E0=t0, A=stat(tA), E3=stat(t3), human=stat(tH), H5=stat(t5),
                   gain_H5=round(t0 - float(np.median(t5)), 2),
                   gain_A=round(t0 - float(np.median(tA)), 2), gain_E3=round(t0 - float(np.median(t3)), 2),
                   A_lap1=tA[0], gain_A_lap1=round(t0 - tA[0], 2))
        report['segments'].append(row)
        total['E0'] += t0; total['A'] += float(np.median(tA)); total['E3'] += float(np.median(t3)); total['human'] += float(np.median(tH))
        print(f"{name:16s} {a:5.0f}-{b:5.0f} {t0:6.2f} {np.median(tA):6.2f} {min(tA):6.2f} {max(tA):6.2f} {row['gain_A']:7.2f} "
              f"{np.median(t3):7.2f} {row['gain_E3']:8.2f} {np.median(tH):10.2f} {np.median(tH) - t0:7.2f}   H5 med {np.median(t5):5.2f} gain {row['gain_H5']:5.2f}")
    print(f"{'sum':16s} {'':11s} {total['E0']:6.2f} {total['A']:6.2f} {'':6s} {'':6s} {total['E0'] - total['A']:7.2f} {total['E3']:7.2f} "
          f"{total['E0'] - total['E3']:8.2f} {total['human']:10.2f} {total['human'] - total['E0']:7.2f}")
    report['segment_sum'] = {k: round(v, 2) for k, v in total.items()}

    # ---- start -----------------------------------------------------------------------------------------
    lap1 = A[0]
    first22 = int(np.argmax(lap1['speed'] >= 22.0))
    first_key = int(np.argmax(lap1['key'] != 0))
    holds = [int(np.argmax(l['gas'] > 0.1)) for l in human]
    human_first22 = [int(np.argmax(l['speed'] >= 22.0)) for l in human]
    report['start'] = dict(A_lap1_frames_to_speed22=first22, A_lap1_along_at_speed22=round(float(lap1['along'][first22]), 1),
                           A_lap1_speed_by_frame={str(f): round(float(lap1['speed'][f]), 2) for f in (0, 5, 10, 15, 20, 25, 30, 50)},
                           A_lap1_along_by_frame={str(f): round(float(lap1['along'][f]), 1) for f in (0, 25, 50, 75, 100)},
                           A_lap1_first_steer_key_frame=first_key, A_lap1_first_steer_along=round(float(lap1['along'][first_key]), 1),
                           all_A_laps_identical_first_60_frames=bool(all((l['key'][:60] == lap1['key'][:60]).all() for l in A)),
                           human_start_hold_frames=holds, human_frames_to_speed22=human_first22,
                           human_frames_at_along30=[int(np.argmax(l['along'] >= 30)) for l in human],
                           A_frames_at_along30=[int(np.argmax(l['along'] >= 30)) for l in A],
                           E0_frames_at_along30=int(np.argmax(E0['along'] >= 30)))
    print('\n== start'); print(json.dumps(report['start'], indent=1))

    # ---- corners ---------------------------------------------------------------------------------------
    frames_A1, frames_centre = {}, {}
    for corner in CORNERS:
        name, start, end, side, angle = corner
        a, b = SECTION[name]
        cfA = [corner_frames(l, corner) for l in A]
        cf3 = [corner_frames(l, corner) for l in E3]
        cf0 = corner_frames(E0, corner)
        cfH = [corner_frames(l, corner) for l in human]
        # sanity: the fixed section contains every A lap's turn-in..back run
        for l, cf in zip(A, cfA):
            assert cf['turn_frame'] is not None and a <= cf['turn_along'] and cf['back_along'] <= b, (name, l['name'], cf)
        ksA = [key_stats(l, segment_frames(l, a, b), side) for l in A]
        ks3 = [key_stats(l, segment_frames(l, a, b), side) for l in E3]
        ks0 = key_stats(E0, segment_frames(E0, a, b), side)
        ksH = [key_stats(l, segment_frames(l, a, b), side) for l in human]
        g = lambda cfs, k: [cf.get(k) for cf in cfs]
        t = lambda cfs, k: [None if cf.get(k) is None else cf[k] / 50 for cf in cfs]
        entry = dict(
            direction='left' if side > 0 else 'right', angle_deg=angle, bend_along=[start, end], section_along=[a, b],
            A=dict(turn_in_along=stat(g(cfA, 'turn_along'), 1), turn_in_t=stat(t(cfA, 'turn_frame')),
                   apex_along=stat(g(cfA, 'apex_along'), 1), apex_t=stat(t(cfA, 'apex_frame')),
                   apex_inside=stat(g(cfA, 'apex_inside')), back_along=stat(g(cfA, 'back_along'), 1),
                   back_t=stat(t(cfA, 'back_frame')),
                   section_time=stat([k['frames'] / 50 for k in ksA]),
                   keys={k: stat([ks[k] for ks in ksA], 3 if 'share' in k or 'steer' in k else 2) for k in ksA[0] if 'start' not in k},
                   longest_inside_start_t=stat([ks['longest_inside_start'] / 50 for ks in ksA]),
                   longest_inside_start_along=stat([float(l['along'][ks['longest_inside_start']]) for l, ks in zip(A, ksA)], 1),
                   longest_merged_start_t=stat([ks['longest_inside_merged_start'] / 50 for ks in ksA]),
                   longest_merged_start_along=stat([float(l['along'][ks['longest_inside_merged_start']]) for l, ks in zip(A, ksA)], 1),
                   longest_merged_rel_apex_s=stat([(ks['longest_inside_merged_start'] - cf['apex_frame']) / 50 for ks, cf in zip(ksA, cfA)]),
                   single_hold_equivalent_frames=stat([single_hold_equivalent(ks['inside_impulse']) for ks in ksA], 1),
                   per_lap=[dict(lap=l['name'], **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in cf.items()})
                            for l, cf in zip(A, cfA)]),
            E3=dict(turn_in_along=stat(g(cf3, 'turn_along'), 1), turn_in_t=stat(t(cf3, 'turn_frame')),
                    apex_along=stat(g(cf3, 'apex_along'), 1), apex_t=stat(t(cf3, 'apex_frame')),
                    apex_inside=stat(g(cf3, 'apex_inside')), back_along=stat(g(cf3, 'back_along'), 1),
                    back_t=stat(t(cf3, 'back_frame')), section_time=stat([k['frames'] / 50 for k in ks3]),
                    keys={k: stat([ks[k] for ks in ks3], 3 if 'share' in k or 'steer' in k else 2) for k in ks3[0] if 'start' not in k}),
            E0=dict(apex_along=round(cf0['apex_along'], 1), apex_t=cf0['apex_frame'] / 50, apex_inside=round(cf0['apex_inside'], 2),
                    section_time=ks0['frames'] / 50, keys={k: (round(v, 3) if isinstance(v, float) else v) for k, v in ks0.items()},
                    single_hold_equivalent_frames=round(single_hold_equivalent(ks0['inside_impulse']), 1)),
            human=dict(apex_inside=stat(g(cfH, 'apex_inside')), section_time=stat([k['frames'] / 50 for k in ksH]),
                       keys={k: stat([ks[k] for ks in ksH], 3 if 'share' in k or 'steer' in k else 2) for k in ksH[0] if 'start' not in k},
                       per_lap_apex_inside=[round(cf['apex_inside'], 2) for cf in cfH]),
            timeline_A=timeline(A, cfA, side), timeline_E0=timeline([E0], [cf0], side),
        )
        cf5 = [corner_frames(l, corner) for l in H5]
        ks5 = [key_stats(l, segment_frames(l, a, b), side) for l in H5]
        entry['H5'] = dict(turn_in_along=stat(g(cf5, 'turn_along'), 1), turn_in_t=stat(t(cf5, 'turn_frame')),
                           apex_along=stat(g(cf5, 'apex_along'), 1), apex_t=stat(t(cf5, 'apex_frame')),
                           apex_inside=stat(g(cf5, 'apex_inside')), back_along=stat(g(cf5, 'back_along'), 1),
                           back_t=stat(t(cf5, 'back_frame')), section_time=stat([k['frames'] / 50 for k in ks5]),
                           keys={k: stat([ks[k] for ks in ks5], 3 if 'share' in k or 'steer' in k else 2) for k in ks5[0] if 'start' not in k},
                           phases=phase_stats(H5, corner, cf5),
                           lap1_key_runs=key_runs_text(H5[0], cf5[0], side))
        entry['A']['phases'] = phase_stats(A, corner, cfA)
        entry['E3']['phases'] = phase_stats(E3, corner, cf3)
        entry['E0']['phases'] = phase_stats([E0], corner, [cf0])
        entry['human']['phases'] = phase_stats(human, corner, cfH)
        kerb = [k for k in KERBS if k['start'] < end + MARGIN and k['end'] > start - MARGIN]
        entry['kerb'] = kerb[0] if kerb else None
        if kerb:
            entry['A']['turn_in_before_kerb_start'] = stat([kerb[0]['start'] - cf['turn_along'] for cf in cfA], 1)
            entry['A']['first_key_before_kerb_start'] = stat([kerb[0]['start'] - phases(l, corner, cf)['first_key_along'] for l, cf in zip(A, cfA)], 1)
            entry['A']['apex_after_kerb_start'] = stat([cf['apex_along'] - kerb[0]['start'] for cf in cfA], 1)
        entry['gain_vs_E0_s'] = round(entry['E0']['section_time'] - entry['A']['section_time']['median'], 2)
        entry['gain_E3_vs_E0_s'] = round(entry['E0']['section_time'] - entry['E3']['section_time']['median'], 2)
        report['corners'][name] = entry

        c1 = cfA[0]
        frames_A1[f'{name}_1_turnin'] = c1['turn_frame']
        frames_A1[f'{name}_2_apex'] = c1['apex_frame']
        frames_A1[f'{name}_3_exit'] = c1['back_frame']
        for tag, key in (('1_turnin', 'turn_along'), ('2_apex', 'apex_along'), ('3_exit', 'back_along')):
            target = c1[key]
            cand = np.nonzero(np.arange(centre['frames']) > 10)[0]
            f = int(cand[np.argmin(np.abs(centre['along'][cand] - target))])
            frames_centre[f'{name}_{tag}_centreline'] = f

        A_, k = entry['A'], entry['A']['keys']
        print(f"\n== {name} {entry['direction']} {angle} deg, bend {start}-{end}, section {a}-{b}")
        print(f"  A  turn-in along {fmt(A_['turn_in_along'])} t {fmt(A_['turn_in_t'], 's')} | apex along {fmt(A_['apex_along'])} t {fmt(A_['apex_t'], 's')} "
              f"inside {fmt(A_['apex_inside'])} | back along {fmt(A_['back_along'])} t {fmt(A_['back_t'], 's')}")
        E = entry['E3']
        print(f"  E3 turn-in along {fmt(E['turn_in_along'])} t {fmt(E['turn_in_t'], 's')} | apex along {fmt(E['apex_along'])} t {fmt(E['apex_t'], 's')} "
              f"inside {fmt(E['apex_inside'])} | back along {fmt(E['back_along'])} t {fmt(E['back_t'], 's')}")
        print(f"  E0 max inside {entry['E0']['apex_inside']} at along {entry['E0']['apex_along']} t {entry['E0']['apex_t']}s ; human laps max inside {entry['human']['per_lap_apex_inside']}")
        print(f"  section time: E0 {entry['E0']['section_time']:.2f}  A {fmt(A_['section_time'], 's')} (gain {entry['gain_vs_E0_s']})  "
              f"E3 {fmt(E['section_time'], 's')} (gain {entry['gain_E3_vs_E0_s']})  human {fmt(entry['human']['section_time'], 's')}")
        for label, kk in (('A ', k), ('E3', E['keys']), ('hu', entry['human']['keys'])):
            print(f"  keys {label}: inside {fmt(kk['inside_share'])} outside {fmt(kk['outside_share'])} none {fmt(kk['none_share'])} | inside frames {fmt(kk['inside_frames'])} "
                  f"in {fmt(kk['inside_presses'])} presses | longest inside {fmt(kk['longest_inside'])} f, merged(gap<=2) {fmt(kk['longest_inside_merged'])} f | "
                  f"longest outside {fmt(kk['longest_outside'])} f | impulse {fmt(kk['inside_impulse'])} peak steer {fmt(kk['peak_inside_steer'])} coast {fmt(kk['coast_frames'])}")
        k0 = entry['E0']['keys']
        print(f"  keys E0: inside {k0['inside_share']} outside {k0['outside_share']} none {k0['none_share']} | inside frames {k0['inside_frames']} in {k0['inside_presses']} presses | "
              f"longest inside {k0['longest_inside']} f, merged {k0['longest_inside_merged']} f | longest outside {k0['longest_outside']} | impulse {k0['inside_impulse']:.2f} "
              f"peak steer {k0['peak_inside_steer']:.2f} | single-hold equivalent {entry['E0']['single_hold_equivalent_frames']} f")
        print(f"  A longest merged inside press starts: t {fmt(A_['longest_merged_start_t'], 's')} along {fmt(A_['longest_merged_start_along'])} "
              f"= {fmt(A_['longest_merged_rel_apex_s'], 's')} relative to the apex ; single-hold equivalent of all inside steering {fmt(A_['single_hold_equivalent_frames'])} f")
        ph = A_['phases']
        print(f"  A phases: first approach key {fmt(ph['first_key_rel_apex'], 's')} rel apex at along {fmt(ph['first_key_along'])} | offset passes 0.3 at along {fmt(ph['drift_along'])} | "
              f"turn-in->apex {fmt(ph['frames_turn_to_apex'])} f, apex->back {fmt(ph['frames_apex_to_back'])} f")
        print(f"            longest inside press within 1.2 s of the apex: {fmt(ph['press_frames'])} f starting {fmt(ph['press_start_rel_apex'], 's')} rel apex at along {fmt(ph['press_start_along'])} | "
              f"steer >= 0.15: {fmt(ph['hard_frames'])} f from {fmt(ph['hard_start_rel_apex'], 's')} to {fmt(ph['hard_end_rel_apex'], 's')} | peak steer {fmt(ph['peak_steer'])} at {fmt(ph['peak_steer_rel_apex'], 's')}")
        print(f"            approach (first key .. apex-0.3s, {fmt(ph['approach_frames'])} f): inside key {fmt(ph['approach_inside_key_frames'])} f, outside key {fmt(ph['approach_outside_key_frames'])} f, "
              f"net impulse {fmt(ph['approach_impulse'])} = {fmt(ph['approach_deg'])} deg = one press of {fmt(ph['approach_single_press_frames'])} f")
        print(f"            bend (apex-0.3s .. apex+0.5s): inside key {fmt(ph['bend_inside_key_frames'])} f, net impulse {fmt(ph['bend_impulse'])} = {fmt(ph['bend_deg'])} deg = one press of {fmt(ph['bend_single_press_frames'])} f")
        print(f"            exit (apex+0.5s .. back+1.0s): inside key {fmt(ph['exit_inside_key_frames'])} f, outside key {fmt(ph['exit_outside_key_frames'])} f, net impulse {fmt(ph['exit_impulse'])} = {fmt(ph['exit_deg'])} deg")
        h5, k5, p5 = entry['H5'], entry['H5']['keys'], entry['H5']['phases']
        print(f"  H5 (human-rate keys, every key state >= 5 f): apex along {fmt(h5['apex_along'])} t {fmt(h5['apex_t'], 's')} inside {fmt(h5['apex_inside'])} | section {fmt(h5['section_time'], 's')}")
        print(f"     keys: inside {fmt(k5['inside_share'])} outside {fmt(k5['outside_share'])} none {fmt(k5['none_share'])} | inside {fmt(k5['inside_frames'])} f in {fmt(k5['inside_presses'])} presses, "
              f"outside {fmt(k5['outside_frames'])} f in {fmt(k5['outside_presses'])} presses | longest inside {fmt(k5['longest_inside'])} f, longest outside {fmt(k5['longest_outside'])} f | peak steer {fmt(k5['peak_inside_steer'])}")
        if p5:
            print(f"     longest inside press within 1.2 s of the apex: {fmt(p5['press_frames'])} f starting {fmt(p5['press_start_rel_apex'], 's')} rel apex at along {fmt(p5['press_start_along'])} | "
                  f"first approach key {fmt(p5['first_key_rel_apex'], 's')} rel apex at along {fmt(p5['first_key_along'])}")
        print(f"     H5 lap 1 keys from 1.8 s before to 1.8 s after its apex (L/R = key, - = none, number = frames): {h5['lap1_key_runs']}")
        p0, pH = entry['E0']['phases'], entry['human']['phases']
        if p0:
            print(f"  E0 phases: longest press {fmt(p0['press_frames'])} f at {fmt(p0['press_start_rel_apex'], 's')} | steer >= 0.15: {fmt(p0['hard_frames'])} f from {fmt(p0['hard_start_rel_apex'], 's')} to {fmt(p0['hard_end_rel_apex'], 's')} | peak {fmt(p0['peak_steer'])}")
        if pH:
            print(f"  human phases (laps that go 1.0 inside: {pH['press_frames']['n']}): longest press {fmt(pH['press_frames'])} f | steer >= 0.15: {fmt(pH['hard_frames'])} f | peak {fmt(pH['peak_steer'])}")
        print(f"  kerb: {entry['kerb']} | A turn-in is {fmt(A_.get('turn_in_before_kerb_start'))} before the kerb start, first key {fmt(A_.get('first_key_before_kerb_start'))} before, apex {fmt(A_.get('apex_after_kerb_start'))} after it")
        print('  timeline relative to the apex (0.2 s bins): A [inside share, outside share, mean steer, inside offset, along]   |   E0 [inside, outside, steer, offset]')
        for ra, r0 in zip(entry['timeline_A'], entry['timeline_E0']):
            print(f"    {ra['t_rel'][0]:+.1f}..{ra['t_rel'][1]:+.1f}s  A in {ra['inside']:.2f} out {ra['outside']:.2f} steer {ra['steer']:+.3f} off {ra['offset']:+.2f} along {ra['along']:6.1f}"
                  f"   |   E0 in {r0['inside']:.2f} out {r0['outside']:.2f} steer {r0['steer']:+.3f} off {r0['offset']:+.2f}")

    # extra frames for the screenshots of lap 1
    frames_A1['S0_start_waiting'] = 0
    frames_A1['S1_start_025'] = 25
    frames_A1['S2_start_050'] = 50
    frames_A1['S3_straight_mid'] = int(np.argmax(lap1['along'] >= 142.5))
    frames_A1['S4_straight_before_C8'] = int(np.argmax(lap1['along'] >= 823.5))
    frames_A1['K1_right_kink'] = int(np.argmax(lap1['along'] >= 207.0))
    frames_A1['K2_right_kink'] = int(np.argmax(lap1['along'] >= 672.0))
    frames_centre['S3_straight_mid_centreline'] = int(np.argmax(centre['along'] >= 142.5))
    (GUIDE / 'frames_A1.json').write_text(json.dumps(frames_A1, indent=1))
    (GUIDE / 'frames_centre.json').write_text(json.dumps(frames_centre, indent=1))
    (GUIDE / 'corner_numbers.json').write_text(json.dumps(report, indent=1))
    print('\nframes A lap1:', frames_A1)
    print('frames centre lap:', frames_centre)

    # ---- how the provided human laps press keys (run lengths over the whole lap) -------------------------
    print('\n== key press lengths (frames) over the whole lap')
    for label, laps in (('human', human), ('A', A), ('E0', [E0]), ('H5', H5)):
        lengths = np.array([n for l in laps for kv in (-1, 1) for _, n in runs(l['key'] == kv)])
        q = np.percentile(lengths, [10, 25, 50, 75, 90])
        print(f"  {label:5s}: presses per lap {len(lengths) / len(laps):.0f}, length percentiles 10/25/50/75/90 = {q.round(1).tolist()}, max {lengths.max()}, "
              f"steer share {np.mean([np.mean(l['key'] != 0) for l in laps]):.3f}, coast frames after start/lap "
              f"{np.mean([(l['gas'][int(np.argmax(l['gas'] > 0.1)):] <= 0.1).sum() for l in laps]):.1f}")
    report['press_lengths'] = {}
    for label, laps in (('human', human), ('A', A), ('E0', [E0]), ('H5', H5)):
        lengths = np.array([n for l in laps for kv in (-1, 1) for _, n in runs(l['key'] == kv)])
        report['press_lengths'][label] = dict(presses_per_lap=round(len(lengths) / len(laps), 1),
                                              percentiles_10_25_50_75_90=np.percentile(lengths, [10, 25, 50, 75, 90]).round(1).tolist(),
                                              max=int(lengths.max()))
    (GUIDE / 'corner_numbers.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
