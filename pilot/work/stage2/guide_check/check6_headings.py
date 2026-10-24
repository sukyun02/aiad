"""Node-by-node heading of the centre line around C6/C7 and the other bends (official env, headless, no torch)."""
import os
import sys

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
sys.path.insert(0, '/home/sukyun02/workspace/aiad/assignment1/src')
import numpy as np
from gui import make_driving_env
from sdc_wrapper import SDC_Wrapper
from track import TRACK_SEED

env = SDC_Wrapper(make_driving_env(display=False, collect=True))
env.reset(seed=TRACK_SEED)
m = env.unwrapped.metrics
vec = m.vectors
head = np.degrees(np.unwrap(np.arctan2(vec[:, 1], vec[:, 0])))
turn = np.diff(np.r_[head, head[0] + 360.0])      # turn at the END node of segment j (= start of segment j+1)
starts = m.starts
print('segment j: start along, length, heading, turn at its end node (+ = left)')
for lo, hi in ((690, 800),):
    for j in range(len(starts)):
        if lo <= starts[j] <= hi:
            print(f'  j {j:3d} along {starts[j]:6.1f} len {m.lengths[j]:.2f} heading {head[j]:+8.2f} turn at along {starts[j]+m.lengths[j]:6.1f}: {turn[j]:+6.2f}')


def total(lo, hi):
    """sum of the turns at nodes whose along lies in [lo, hi)"""
    node_along = starts + m.lengths
    sel = (node_along >= lo) & (node_along < hi)
    return float(turn[sel].sum()), float(np.abs(turn[sel]).max()) if sel.any() else 0.0


for name, lo, hi in (('C1', 45.5, 77.0), ('C2', 248.5, 273.0), ('C3', 381.5, 399.0), ('C4', 518.0, 549.5), ('C5', 609.0, 633.5),
                     ('C6', 728.0, 738.5), ('C7', 749.0, 766.5), ('C8', 879.0, 913.5), ('C6 wide', 715, 747), ('C7 wide', 747, 785),
                     ('C6+C7', 715, 785), ('K2', 655, 712)):
    s, mx = total(lo, hi)
    s2, _ = total(lo - 0.01, hi + 0.01)
    print(f'{name}: sum of node turns in [{lo}, {hi}) = {s:+.1f} deg (closed interval {s2:+.1f}), largest single node turn {mx:.1f}')
# all runs of same-sign turning above 1 deg per node
print('runs of consecutive nodes turning the same way by more than 1.5 deg each:')
node_along = starts + m.lengths
run = []
for j in range(len(turn)):
    if abs(turn[j]) > 1.5 and (not run or np.sign(turn[j]) == np.sign(turn[run[-1]])) and (not run or j == run[-1] + 1):
        run.append(j)
    else:
        if run:
            print(f'  along {node_along[run[0]]:6.1f}-{node_along[run[-1]]:6.1f}: {turn[run].sum():+6.1f} deg over {len(run)} nodes')
        run = [j] if abs(turn[j]) > 1.5 else []
if run:
    print(f'  along {node_along[run[0]]:6.1f}-{node_along[run[-1]]:6.1f}: {turn[run].sum():+6.1f} deg over {len(run)} nodes')
env.close()
