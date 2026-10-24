"""Table of driver_h.py results sorted by (min_press, react, lookahead, E).  usage: table2.py GLOB [GLOB ...]"""
import glob
import json
import sys
from pathlib import Path
rows = []
for pattern in sys.argv[1:]:
    for f in glob.glob(pattern):
        r = json.loads(Path(f).read_text())
        r['file'] = Path(f).stem
        rows.append(r)
rows.sort(key=lambda r: (r['decide'], r['react_steer'], r['min_press'], r['press_jitter'], r['react'], r['lookahead'], r['gain'], r['E'], r['noise_seed']))
print(f"{'file':26s} {'mp':>2s} {'j':>1s} {'rD':>2s} {'E':>4s} {'la':>4s} {'g':>4s} {'status':8s} {'lap':>6s} {'prog':>6s} {'ctr':>5s} {'maxoff':>6s} {'offrd':>5s} {'1st':>6s} {'chg':>4s} {'prs':>4s} {'press min/p25/med/p75/max':>24s} {'gap min/p25/med/p75/max':>24s} {'st%':>5s} {'rms':>5s}  widest corner [inside, outside]")
for r in rows:
    lap = f"{r['lap_time']:.2f}" if r['lap_time'] is not None else '-'
    fmt = lambda v: '/'.join('-' if x is None else f'{x:g}' for x in v)
    worst = max(r['corner_inside_outside_max'].items(), key=lambda kv: max(kv[1]))
    print(f"{r['file'][:26]:26s} {r['min_press']:2d} {r['press_jitter']:1d} {r['react']:2d} {r['E']:4.1f} {r['lookahead']:4.1f} {r['gain']:4.2f} "
          f"{('clean' if r['clean'] else r['status']):8s} {lap:>6s} {r['progress']:6.2f} {r['center']:5.1f} {r['max_abs_offset']:6.2f} {r['offroad_frames']:5d} "
          f"{str(r['first_offroad_along'] or '-'):>6s} {r['key_changes']:4d} {r['steer_presses']:4d} {fmt(r['press_len']):>24s} {fmt(r['gap_len']):>24s} {100 * r['steer_frac']:5.1f} {r['track_err_rms']:5.2f}  {worst[0]} {worst[1]}")
