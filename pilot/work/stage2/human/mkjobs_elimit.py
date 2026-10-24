"""Job list: where does the line stop being clean? (larger E for the chosen human-like setting and two neighbours)"""
from pathlib import Path
cases = [(6, 6, 5.5, 15), (6, 6, 6.0, 15), (6, 6, 6.5, 15), (6, 6, 7.0, 15), (8, 6, 6.0, 15), (4, 3, 6.0, 9)]
lines = []
for n, d, e, la in cases:
    tag = f'x_m{n}_r{d}_E{e}_la{la}'
    lines.append(f'driver_h.py --E {e} --W 28 --k0 0.03 --lookahead {la} --gain 1.0 --min-press {n} --react {d} --react-steer current --decide commit --out out/elimit/{tag}.json --trace out/elimit/{tag}.npz')
Path('jobs_elimit.txt').write_text('\n'.join(lines) + '\n')
print(len(lines))
