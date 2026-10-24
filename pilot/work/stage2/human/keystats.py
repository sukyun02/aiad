"""Key statistics of recorded laps from the traces written by replay_trace.py (numpy only).

usage: keystats.py TRACE_DIR OUT.json NAME [NAME ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

CORNERS = [('C1', 45.5, 77.0, 1), ('C2', 248.5, 273.0, 1), ('C3', 381.5, 399.0, 1), ('C4', 518.0, 549.5, 1),
           ('C5', 609.0, 633.5, 1), ('C6', 728.0, 738.5, 1), ('C7', 749.0, 766.5, -1), ('C8', 879.0, 913.5, 1)]
MARGIN = 15.0


def runs(values):
    """[(value, start, length)] of maximal constant runs."""
    out, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            out.append((values[start], start, i - start))
            start = i
    return out


def q(values):
    if len(values) == 0:
        return None
    v = np.asarray(values, dtype=float)
    return dict(n=len(v), min=float(v.min()), p25=float(np.percentile(v, 25)), median=float(np.median(v)),
                p75=float(np.percentile(v, 75)), max=float(v.max()), mean=round(float(v.mean()), 2))


def lap_stats(name, rows, info):
    along, signed = rows[:, 0], rows[:, 1]
    steer = np.sign(np.where(np.abs(rows[:, 2]) > 0.1, rows[:, 2], 0)).astype(int)     # -1 left, +1 right
    gas = (rows[:, 3] > 0.1).astype(int)
    brake = (rows[:, 4] > 0.1).astype(int)
    n = len(rows)
    hold = int(np.argmax(gas)) if gas.any() else n
    steer_runs = runs(list(steer))
    presses = [(v, s, l) for v, s, l in steer_runs if v != 0]
    left = [l for v, s, l in presses if v < 0]
    right = [l for v, s, l in presses if v > 0]
    # gaps: no-steer runs strictly between two presses; a direct left<->right switch is a gap of 0
    gaps, reversals = [], 0
    for a, b in zip(presses[:-1], presses[1:]):
        gap = b[1] - (a[1] + a[2])
        gaps.append(gap)
        reversals += int(gap == 0)
    state = steer * 100 + gas * 10 + brake
    changes = int((np.diff(state) != 0).sum())
    steer_changes = int((np.diff(steer) != 0).sum())
    lengths = [l for v, s, l in presses]
    corners = {}
    for cname, start, end, side in CORNERS:
        mask = (along >= start - MARGIN) & (along < end + MARGIN)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        lo, hi = idx[0], idx[-1] + 1
        inside_key = -side                       # left bend -> left key (-1)
        ins = [(max(s, lo), min(s + l, hi)) for v, s, l in presses if v == inside_key and s < hi and s + l > lo]
        out = [(max(s, lo), min(s + l, hi)) for v, s, l in presses if v == -inside_key and s < hi and s + l > lo]
        corners[cname] = dict(frames=int(hi - lo), inside_presses=len(ins), inside_frames=int(sum(b - a for a, b in ins)),
                              inside_longest=int(max([b - a for a, b in ins], default=0)),
                              counter_presses=len(out), counter_frames=int(sum(b - a for a, b in out)),
                              inside_max_offset=round(float((side * signed[lo:hi]).max()), 2),
                              outside_max_offset=round(float((-side * signed[lo:hi]).max()), 2))
    return dict(name=name, frames=n, lap_time=info.get('lap_time'), progress=round(info.get('progress', 0), 2),
                center=round(info.get('center', 0), 2), start_hold=hold,
                steer_presses=len(presses), left_presses=len(left), right_presses=len(right),
                press_len=q(lengths), press_len_left=q(left), press_len_right=q(right),
                press_le2=int(sum(l <= 2 for l in lengths)), press_le3=int(sum(l <= 3 for l in lengths)),
                press_le5=int(sum(l <= 5 for l in lengths)),
                gap_len=q(gaps), gap_le2=int(sum(g <= 2 for g in gaps)), gap_le3=int(sum(g <= 3 for g in gaps)),
                direct_reversals=reversals,
                steer_share=round(float((steer != 0).mean()), 3), steer_frames_left=int((steer < 0).sum()),
                steer_frames_right=int((steer > 0).sum()),
                coast_frames=int(((gas == 0) & (brake == 0))[hold:].sum()), brake_frames=int(brake.sum()),
                key_changes=changes, steer_changes=steer_changes,
                max_abs_offset=round(float(np.abs(signed).max()), 2),
                press_lengths=lengths, gaps=gaps, corners=corners)


if __name__ == '__main__':
    trace_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    results = []
    for name in sys.argv[3:]:
        data = np.load(trace_dir / f'{name}.npz')
        results.append(lap_stats(name, data['rows'], json.loads(str(data['info']))))
    out.write_text(json.dumps(results, indent=1))
    hdr = (f"{'lap':8s} {'time':>6s} {'frm':>5s} {'hold':>4s} {'press':>5s} {'L':>3s} {'R':>3s} | {'len min/p25/med/p75/max':>24s} | "
           f"{'L med':>5s} {'R med':>5s} | {'<=2':>3s} {'<=3':>3s} {'<=5':>3s} | {'gap min/p25/med/p75/max':>24s} {'g<=2':>4s} {'rev':>3s} | "
           f"{'steer%':>6s} {'coast':>5s} {'brake':>5s} {'chg':>4s} {'maxoff':>6s} {'ctr':>5s}")
    print(hdr)
    for r in results:
        p, g = r['press_len'], r['gap_len']
        fmt = lambda d: f"{d['min']:.0f}/{d['p25']:.1f}/{d['median']:.1f}/{d['p75']:.1f}/{d['max']:.0f}" if d else '-'
        med = lambda d: f"{d['median']:.1f}" if d else '-'
        print(f"{r['name']:8s} {r['lap_time']:6.2f} {r['frames']:5d} {r['start_hold']:4d} {r['steer_presses']:5d} {r['left_presses']:3d} {r['right_presses']:3d} | "
              f"{fmt(p):>24s} | {med(r['press_len_left']):>5s} {med(r['press_len_right']):>5s} | {r['press_le2']:3d} {r['press_le3']:3d} {r['press_le5']:3d} | "
              f"{fmt(g):>24s} {r['gap_le2']:4d} {r['direct_reversals']:3d} | {100 * r['steer_share']:6.1f} {r['coast_frames']:5d} {r['brake_frames']:5d} "
              f"{r['key_changes']:4d} {r['max_abs_offset']:6.2f} {r['center']:5.1f}")
    print()
    print('per corner (section = corner +-15): inside-key presses / inside frames / longest / counter presses / counter frames / max inside offset / max outside offset')
    for r in results:
        print(f"{r['name']:8s} " + '  '.join(
            f"{c}:{d['inside_presses']}/{d['inside_frames']}/{d['inside_longest']}/{d['counter_presses']}/{d['counter_frames']}/{d['inside_max_offset']:+.1f}/{d['outside_max_offset']:+.1f}"
            for c, d in r['corners'].items()))
