"""Print a table of driver_h.py result jsons.  usage: table.py FILE.json [FILE.json ...] [--json OUT]"""
import json
import sys
from pathlib import Path
args = sys.argv[1:]
out = None
if '--json' in args:
    i = args.index('--json')
    out = args[i + 1]
    args = args[:i] + args[i + 2:]
rows = []
print(f"{'file':34s} {'dec':6s} {'mp':>2s} {'rD':>2s} {'rs':3s} {'E':>4s} {'la':>4s} {'g':>4s} {'status':8s} {'lap':>6s} {'prog':>6s} {'ctr':>5s} {'maxoff':>6s} {'offrd':>5s} {'1st':>6s} {'chg':>4s} {'prs':>4s} {'press min/p25/med/p75/max':>24s} {'gap min/p25/med/p75/max':>24s} {'st%':>5s} {'rms':>5s}")
for f in args:
    r = json.loads(Path(f).read_text())
    r['file'] = Path(f).stem
    rows.append(r)
    lap = f"{r['lap_time']:.2f}" if r['lap_time'] is not None else '-'
    fmt = lambda v: '/'.join('-' if x is None else f'{x:g}' for x in v)
    print(f"{r['file'][:34]:34s} {r['decide']:6s} {r['min_press']:2d} {r['react']:2d} {r['react_steer'][:3]:3s} {r['E']:4.1f} {r['lookahead']:4.1f} {r['gain']:4.2f} "
          f"{('clean' if r['clean'] else r['status']):8s} {lap:>6s} {r['progress']:6.2f} {r['center']:5.1f} {r['max_abs_offset']:6.2f} {r['offroad_frames']:5d} "
          f"{str(r['first_offroad_along'] or '-'):>6s} {r['key_changes']:4d} {r['steer_presses']:4d} {fmt(r['press_len']):>24s} {fmt(r['gap_len']):>24s} {100 * r['steer_frac']:5.1f} {r['track_err_rms']:5.2f}")
if out:
    Path(out).write_text(json.dumps(rows, indent=1))
