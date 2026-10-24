"""Job list: follow-up with the lookahead tuned per constraint level (commit rule, react-steer current, gain 1.0)."""
from pathlib import Path
cases = []
for n in (4, 6, 8, 10):                       # reaction delay 6: lookahead 15
    for e in (2, 3, 4, 5):
        cases.append((n, 6, e, 15))
for d in (0, 3):                              # min-press 10: lookahead 12
    for e in (2, 3, 4, 5):
        cases.append((10, d, e, 12))
for e in (2, 3, 5):                           # min-press 6, react 3: lookahead 11 (E4 is in out/tune)
    cases.append((6, 3, e, 11))
for e in (2, 3, 4, 5):                        # min-press 8, react 3: lookahead 12
    cases.append((8, 3, e, 12))
cases.append((2, 6, 4, 15))
lines = []
for n, d, e, la in cases:
    tag = f'f_m{n}_r{d}_E{e}_la{la}'
    lines.append(f'driver_h.py --E {e} --W 28 --k0 0.03 --lookahead {la} --gain 1.0 --min-press {n} --react {d} --react-steer current --decide commit --out out/follow/{tag}.json --trace out/follow/{tag}.npz')
Path('jobs_follow.txt').write_text('\n'.join(lines) + '\n')
print(len(lines))
