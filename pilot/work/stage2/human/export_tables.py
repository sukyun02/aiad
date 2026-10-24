"""Write out/sweep_table.json (all 118 sweep laps) and out/stats_table.json (per-lap key statistics) and print them compactly."""
import glob
import json
from pathlib import Path
OUT = Path('/home/sukyun02/workspace/aiad/pilot/work/stage2/human/out')
sweep = []
order = {'grid': 0, 'follow': 1, 'elimit': 2, 'tune': 3, 'pre': 4}
for f in glob.glob(str(OUT / '*' / '*.json')):
    p = Path(f)
    if p.parent.name not in order:
        continue
    r = json.loads(p.read_text())
    status = 'clean' if r['clean'] else ('unclean' if r['status'] == 'finished' else r['status'])
    other = f"{p.parent.name}/{p.stem}; {r['decide']}; steer={r['react_steer']}; la={r['lookahead']:g}; gain={r['gain']:g}; offroad={r['offroad_frames']}"
    if r['first_offroad_along'] is not None:
        other += f"@{r['first_offroad_along']:g}"
    other += f"; presses={r['steer_presses']}; rms={r['track_err_rms']}"
    sweep.append((order[p.parent.name], r['decide'], r['react_steer'], r['min_press'], r['react'], r['lookahead'], r['gain'], r['E'],
                  dict(min_press=r['min_press'], react=r['react'], E=r['E'], status=status, lap_time=r['lap_time'], progress=r['progress'],
                       centre=r['center'], max_offset=r['max_abs_offset'], key_changes=r['key_changes'], other=other)))
sweep.sort(key=lambda t: t[:8])
sweep = [t[8] for t in sweep]
(OUT / 'sweep_table.json').write_text(json.dumps(sweep, indent=1))
stats = json.loads((OUT / 'keystats.json').read_text()) + json.loads((OUT / 'keystats_H8.json').read_text())
table = []
for s in stats:
    name = s['name']
    group = 'human' if name[0] == 'H' and not name.startswith('H8') else 'scriptA' if name.startswith('A_') else 'scriptC' if name.startswith('C_') else 'H8'
    p = s['press_len']
    table.append(dict(set=group, lap=name, frames=s['frames'], start_hold=s['start_hold'], steer_presses=s['steer_presses'], press_len_min=p['min'], press_len_p25=p['p25'],
                      press_len_median=p['median'], press_len_p75=p['p75'], press_len_max=p['max'], steer_share=s['steer_share'], coast_frames=s['coast_frames'],
                      brake_frames=s['brake_frames'], key_changes=s['key_changes']))
(OUT / 'stats_table.json').write_text(json.dumps(table, indent=1))
print(len(sweep), len(table))
print('SWEEP')
for r in sweep:
    print(json.dumps(r, separators=(',', ':')))
print('STATS')
for r in table:
    print(json.dumps(r, separators=(',', ':')))
