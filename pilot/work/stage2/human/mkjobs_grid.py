"""Job list: main sweep min-press x react x E with the default line parameters (W 28, k0 0.03, lookahead 9, gain 1.0)."""
from pathlib import Path
lines = []
for n in (2, 4, 6, 8, 10):
    for d in (0, 3, 6):
        for e in (2, 3, 4, 5):
            tag = f'g_m{n}_r{d}_E{e}'
            lines.append(f'driver_h.py --E {e} --W 28 --k0 0.03 --lookahead 9 --gain 1.0 --min-press {n} --react {d} --react-steer current --decide commit --out out/grid/{tag}.json --trace out/grid/{tag}.npz')
Path('jobs_grid.txt').write_text('\n'.join(lines) + '\n')
print(len(lines))
