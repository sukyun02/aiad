"""Tables from the fact-check simulations in out/ (no simulation here)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
CORNERS = [('C1', 45.5, 77.0, 1), ('C2', 248.5, 273.0, 1), ('C3', 381.5, 399.0, 1), ('C4', 518.0, 549.5, 1),
           ('C5', 609.0, 633.5, 1), ('C6', 728.0, 738.5, 1), ('C7', 749.0, 766.5, -1), ('C8', 879.0, 913.5, 1)]
summary = {}


def load(name):
    return json.loads((OUT / f'{name}.json').read_text())


def rows(name):
    return np.load(OUT / f'{name}.npz')['rows']


pattern = sys.argv[1] if len(sys.argv) > 1 else ''

# ------------------------------------------------------------------ claim 6
print('== claim 6: releasing the gas key (driver_fc.py)')
print(f"{'run':22s} {'status':8s} {'lap':>6s} {'d_lap':>6s} {'coast':>5s} {'minspd':>6s} {'prog':>6s} {'centre':>6s}  classes[0..8]")
c6 = {}
for base, names in (('c6_A_base', ['c6_A_base', 'c6_A_once10_s150', 'c6_A_once25_s150', 'c6_A_once50_s150', 'c6_A_once10_s450',
                                   'c6_A_once25_s450', 'c6_A_once50_s450', 'c6_A_entry5', 'c6_A_entry10', 'c6_A_entry25',
                                   'c6_A_steer', 'c6_A_corners'] + [p.stem for p in sorted(OUT.glob('c6x_A_*.json'))]),
                    ('c6_E0_base', ['c6_E0_base', 'c6_E0_once10_s150', 'c6_E0_steer'])):
    b = load(base)
    for name in names:
        d = load(name)
        delta = d['lap_time'] - b['lap_time'] if d['lap_time'] else float('nan')
        c6[name] = dict(status=d['status'], lap_time=d['lap_time'], delta=None if delta != delta else round(delta, 2), coast_frames=d['coast_total'],
                        min_speed_after25=d['min_speed_after25'], progress=d['progress'], center=d['center'], classes=d['action_counts'])
        lap_text = f"{d['lap_time']:6.2f}" if d['lap_time'] else '     -'
        print(f"{name:22s} {d['status']:8s} {lap_text} {delta:+6.2f} {d['coast_total']:5d} {d['min_speed_after25']:6.2f} {d['progress']:6.2f} {d['center']:6.2f}  {d['action_counts']}")
summary['claim6'] = c6
# speed profile around a single 10-frame lift
r = rows('c6_A_once10_s150')
gas = r[:, 6]
first = int(np.argmax((gas == 0) & (np.arange(len(r)) > 30)))
prof = {str(k): round(float(r[first + k, 3]), 2) for k in (-1, 0, 5, 10, 15, 20, 30, 40, 60)}
print('  speed around a 10-frame lift (frame offset from the first gas-off frame -> speed):', prof)
back = int(np.argmax(r[first + 10:, 3] >= 22.2))
print('  frames after the lift ends until speed >= 22.2 again:', back)
summary['claim6_lift_profile'] = dict(speed=prof, frames_to_recover_22p2=back)

# ------------------------------------------------------------------ claim 8
print('\n== claim 8: start hold (original driver.py)')
c8 = {}
for prefix in ('c8_E0', 'c8_A'):
    b = load(f'{prefix}_hold0')
    for path in sorted(OUT.glob(f'{prefix}_hold*.json'), key=lambda p: int(p.stem.split('hold')[1])):
        d = json.loads(path.read_text())
        delta = d['lap_time'] - b['lap_time']
        c8[path.stem] = dict(hold=d['hold'], lap_time=d['lap_time'], delta=round(delta, 2), expected=round(d['hold'] / 50, 2),
                             frames=d['frames'], center=d['center'], path_len=d['path_len'])
        print(f"  {path.stem:16s} hold {d['hold']:2d} lap {d['lap_time']:.2f}  delta {delta:+.2f}  hold/50 {d['hold'] / 50:.2f}  frames {d['frames']}  centre {d['center']}  path {d['path_len']}")
summary['claim8'] = c8

# ------------------------------------------------------------------ claim 9
print('\n== claim 9: constant offset to the left all around (original driver.py --line)')
c9 = {}
xs, ys = [], []
for path in sorted(OUT.glob('c9_const*.json'), key=lambda p: float(p.stem.split('const')[1])):
    d = json.loads(path.read_text())
    r = np.load(path.with_suffix('.npz'))['rows']
    off = r[:, 1]
    mean_signed = float(off[100:].mean())
    formula = 100 * (0.2 + 0.8 * (1 - min(float(path.stem.split('const')[1]) / (40 / 6), 1)) ** 4)
    c9[path.stem] = dict(target=float(path.stem.split('const')[1]), status=d['status'], lap_time=d['lap_time'], progress=d['progress'],
                         center=d['center'], path_len=d['path_len'], mean_signed_offset_after100=round(mean_signed, 2),
                         min_offset_after100=round(float(off[100:].min()), 2), max_offset=round(float(off.max()), 2),
                         formula_centre_for_exact_constant=round(formula, 2))
    xs.append(mean_signed)
    ys.append(d['center'])
    print(f"  {path.stem:14s} {d['status']:8s} lap {d['lap_time']:.2f}  prog {d['progress']:.2f}  centre {d['center']:.2f}  (formula for an exact constant: {formula:.2f})  "
          f"path {d['path_len']}  actual offset after frame 100: mean {mean_signed:.2f} min {off[100:].min():.2f} max {off.max():.2f}")
targets = [v['target'] for v in c9.values()]
centres = [v['center'] for v in c9.values()]
order = np.argsort(centres)
t30 = float(np.interp(30.0, np.array(centres)[order], np.array(targets)[order]))
m30 = float(np.interp(30.0, np.array(centres)[order], np.array(xs)[order]))
analytic = (40 / 6) * (1 - 0.125 ** 0.25)
print(f"  centre = 30 exactly: analytic constant offset {analytic:.3f}; interpolated from the simulated laps: target {t30:.2f} (actual mean offset {m30:.2f})")
summary['claim9'] = dict(runs=c9, analytic_offset_for_30=round(analytic, 3), simulated_target_for_30=round(t30, 2), simulated_mean_offset_for_30=round(m30, 2))

# ------------------------------------------------------------------ claim 12
print('\n== claim 12: inside limit (driver_fc.py, trace column 7 = car centre on road before the step)')
c12 = {}
all_on, all_off = [], []
for path in sorted(OUT.glob('c12_*.json')) + [OUT / 's3_source.json']:
    d = json.loads(path.read_text())
    r = np.load(path.with_suffix('.npz'))['rows']
    on = r[:, 7] > 0.5
    absoff = np.abs(r[:, 1])
    inside = {}
    where = []
    for name, a, b, side in CORNERS:
        mask = (r[:, 0] >= a - 15) & (r[:, 0] < b + 15)
        inside[name] = round(float((side * r[mask, 1]).max()), 2)
        n_off = int((mask & ~on).sum())
        if n_off:
            where.append(f"{name}:{n_off}")
    n_off_else = int((~on).sum()) - sum(int(w.split(':')[1]) for w in where)
    max_on = float(absoff[on].max())
    min_off = float(absoff[~on].min()) if (~on).any() else None
    all_on.append(max_on)
    if min_off is not None:
        all_off.append(min_off)
    c12[path.stem] = dict(status=d['status'], lap_time=d['lap_time'], progress=d['progress'], center=d['center'],
                          offroad_frames=int((~on).sum()), offroad_where=where, max_abs_offset_while_on_road=round(max_on, 3),
                          min_abs_offset_while_off_road=None if min_off is None else round(min_off, 3), inside_max=inside)
    print(f"  {path.stem:22s} {d['status']:8s} lap {d['lap_time']}  prog {d['progress']:6.2f}  centre {d['center']:5.2f}  off-road frames {int((~on).sum()):3d} {where} other {n_off_else} | "
          f"max |offset| on-road {max_on:.3f}  min |offset| off-road {min_off if min_off is None else round(min_off, 3)} | inside max " + ' '.join(f"{inside[n]:.2f}" for n in inside))
print(f"  over all these laps: largest |offset| of an on-road frame {max(all_on):.3f}; smallest |offset| of an off-road frame {min(all_off):.3f}")
summary['claim12'] = dict(runs=c12, largest_onroad_offset=round(max(all_on), 3), smallest_offroad_offset=round(min(all_off), 3))

(OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
print('\nwritten', OUT / 'summary.json')
