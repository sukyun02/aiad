"""Job list: lookahead/gain tuning for the delayed / long-press cases (commit rule, react-steer current, E=4)."""
from pathlib import Path
cases = [(6, 6, 12, 1.0), (6, 6, 15, 1.0), (6, 6, 12, 0.7), (4, 6, 12, 1.0), (6, 3, 9, 1.0), (6, 3, 11, 1.0), (10, 0, 9, 1.0), (10, 0, 12, 1.0)]
lines = []
for n, d, la, g in cases:
    tag = f'tune_m{n}_r{d}_la{la}_g{g}'
    lines.append(f'driver_h.py --E 4 --W 28 --k0 0.03 --lookahead {la} --gain {g} --min-press {n} --react {d} --react-steer current --decide commit --out out/tune/{tag}.json --trace out/tune/{tag}.npz')
Path('jobs_tune.txt').write_text('\n'.join(lines) + '\n')
print(len(lines))
