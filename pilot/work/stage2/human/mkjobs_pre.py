"""Job list for the preliminary comparison of decision rules (12 laps, E=4)."""
from pathlib import Path
base = '--E 4 --W 28 --k0 0.03 --lookahead 9 --gain 1.0'
cases = [('greedy', 4, 0, 'delayed'), ('greedy', 6, 0, 'delayed'), ('greedy', 4, 3, 'delayed'), ('greedy', 6, 6, 'delayed'),
         ('commit', 4, 0, 'delayed'), ('commit', 6, 0, 'delayed'), ('commit', 4, 3, 'delayed'), ('commit', 4, 3, 'current'),
         ('commit', 6, 6, 'delayed'), ('commit', 6, 6, 'current'), ('greedy', 4, 3, 'current'), ('greedy', 6, 6, 'current')]
lines = []
for decide, n, d, rs in cases:
    tag = f'pre_{decide}_m{n}_r{d}_{rs}'
    lines.append(f'driver_h.py {base} --min-press {n} --react {d} --react-steer {rs} --decide {decide} --out out/pre/{tag}.json --trace out/pre/{tag}.npz')
Path('jobs_pre.txt').write_text('\n'.join(lines) + '\n')
print(len(lines))
