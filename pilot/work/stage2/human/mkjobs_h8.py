"""Job list: candidates for the human-like dataset H8 (saved into data_s2/H8_raw/<name>)."""
from pathlib import Path
RAW = '/home/sukyun02/workspace/aiad/pilot/data_s2/H8_raw'
common = '--W 28 --k0 0.03 --react 6 --react-steer current --decide commit'
# name: (E, lookahead, gain, min_press, jitter, seed)
cands = {
    'b1': (4.0, 15, 1.0, 5, 2, 1), 'b2': (3.5, 15, 1.0, 5, 2, 2), 'b3': (4.5, 15, 1.0, 5, 2, 3), 'b4': (4.0, 14, 1.0, 5, 2, 4),
    'b5': (4.0, 16, 1.0, 5, 2, 5), 'b6': (4.5, 16, 1.05, 5, 2, 6), 'b7': (3.5, 14, 0.95, 5, 2, 7), 'b8': (4.5, 15, 0.95, 5, 2, 8),
    'a1': (4.0, 15, 1.0, 6, 0, 0), 'a2': (3.5, 15, 1.0, 6, 0, 0), 'a3': (4.5, 15, 1.0, 6, 0, 0), 'a4': (4.0, 14, 1.0, 6, 0, 0),
    'a5': (4.0, 16, 1.0, 6, 0, 0), 'a6': (4.5, 15, 0.95, 6, 0, 0),
}
lines = []
for name, (e, la, g, n, j, seed) in cands.items():
    lines.append(f'driver_h.py --E {e} --lookahead {la} --gain {g} --min-press {n} --press-jitter {j} --noise-seed {seed} {common} '
                 f'--save-dir {RAW}/{name} --out out/h8/{name}.json --trace out/h8/{name}.npz')
Path('jobs_h8.txt').write_text('\n'.join(lines) + '\n')
print(len(lines))
