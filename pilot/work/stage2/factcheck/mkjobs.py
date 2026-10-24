"""Write the job list (one line of arguments per scripted lap) and the constant-offset line files."""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)
(HERE / 'lines').mkdir(exist_ok=True)
ORIG = '/home/sukyun02/workspace/aiad/pilot/work/stage2/driver.py'
FC = str(HERE / 'driver_fc.py')
A = '--E 4 --W 28 --k0 0.03 --lookahead 9 --gain 1.0'          # = scriptA lap1 (41.72 s)

count = len(np.load('/home/sukyun02/workspace/aiad/pilot/work/stage2/out/E0.npz')['offset'])
jobs = []


def add(script, name, args):
    jobs.append(f'{script} {args} --out {OUT}/{name}.json --trace {OUT}/{name}.npz')


# claim 6: releasing the gas key
add(FC, 'c6_A_base', A)
for at in (150, 450):
    for n in (10, 25, 50):
        add(FC, f'c6_A_once{n}_s{at}', f'{A} --coast-mode once --coast-at {at} --coast-frames {n}')
for n in (5, 10, 25):
    add(FC, f'c6_A_entry{n}', f'{A} --coast-mode entry --coast-frames {n}')
add(FC, 'c6_A_steer', f'{A} --coast-mode steer')
add(FC, 'c6_A_corners', f'{A} --coast-mode corners')
add(FC, 'c6_E0_base', '--E 0')
add(FC, 'c6_E0_once10_s150', '--E 0 --coast-mode once --coast-at 150 --coast-frames 10')
add(FC, 'c6_E0_steer', '--E 0 --coast-mode steer')

# claim 8: start hold (ORIGINAL driver.py)
for hold in (0, 12, 20, 32):
    add(ORIG, f'c8_E0_hold{hold}', f'--E 0 --hold {hold}')
for hold in (0, 20):
    add(ORIG, f'c8_A_hold{hold}', f'{A} --hold {hold}')

# claim 9: constant offset to the left all around (ORIGINAL driver.py with --line)
for value in (2.0, 2.5, 2.7, 3.0, 3.5):
    line = HERE / 'lines' / f'const_{value}.npz'
    np.savez(line, offset=np.full(count, value))
    add(ORIG, f'c9_const{value}', f'--line {line}')

# claim 12: how far inside can the car centre go (variant, on-road flag per frame)
for E in (6.0, 6.25, 6.5, 6.75, 7.0, 7.5):
    add(FC, f'c12_E{E}_W28_k0.03', f'--E {E} --W 28 --k0 0.03 --lookahead 9 --gain 1.0')
for E in (6.0, 6.5, 7.0):
    add(FC, f'c12_E{E}_W21_k0.02', f'--E {E} --W 21 --k0 0.02 --lookahead 9 --gain 1.0')
for value in (5.5, 6.0, 6.3, 6.6):
    line = HERE / 'lines' / f'const_{value}.npz'
    np.savez(line, offset=np.full(count, value))
    add(FC, f'c12_const{value}', f'--line {line}')

(HERE / 'jobs.txt').write_text('\n'.join(jobs) + '\n')
print(len(jobs), 'jobs; line length', count)
