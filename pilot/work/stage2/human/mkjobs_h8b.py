"""Job list: six more jittered candidates for H8 (seeds 9-14)."""
from pathlib import Path
RAW = '/home/sukyun02/workspace/aiad/pilot/data_s2/H8_raw'
common = '--W 28 --k0 0.03 --react 6 --react-steer current --decide commit'
cands = {'b9': (4.0, 15, 1.05, 5, 2, 9), 'b10': (4.5, 16, 1.0, 5, 2, 10), 'b11': (4.0, 16, 1.05, 5, 2, 11),
         'b12': (4.5, 15, 1.05, 5, 2, 12), 'b13': (4.0, 15, 1.0, 5, 2, 13), 'b14': (4.5, 16, 1.0, 5, 2, 14)}
lines = []
for name, (e, la, g, n, j, seed) in cands.items():
    lines.append(f'driver_h.py --E {e} --lookahead {la} --gain {g} --min-press {n} --press-jitter {j} --noise-seed {seed} {common} '
                 f'--save-dir {RAW}/{name} --out out/h8/{name}.json --trace out/h8/{name}.npz')
Path('jobs_h8b.txt').write_text('\n'.join(lines) + '\n')
print(len(lines))
