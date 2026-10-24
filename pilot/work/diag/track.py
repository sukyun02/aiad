import os, sys, json
from pathlib import Path
SP = Path('/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad')
sys.path.insert(0, str(SP / 'kit')); import kit
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy'); os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
sys.path.insert(0, str(kit.SRC))
import numpy as np
from gui import make_driving_env
from sdc_wrapper import SDC_Wrapper
from track import TRACK_SEED
env = SDC_Wrapper(make_driving_env(display=False, unlimited=True))
env.reset(seed=TRACK_SEED)
m = env.unwrapped.metrics
pts = m.points; vec = m.vectors
ang = np.arctan2(vec[:, 1], vec[:, 0])
d = np.diff(np.r_[ang, ang[0]]); d = (d + np.pi) % (2 * np.pi) - np.pi     # + = left turn (ccw)
np.savez(str(SP / 'work/diag/track.npz'), points=pts, starts=m.starts, lengths=m.lengths, dtheta=d, length=m.length, half_width=m.half_width,
         car0=np.array(env.unwrapped.car.hull.position))
print('n', len(pts), 'length', m.length, 'half', m.half_width, 'total turn deg', np.degrees(d.sum()))
# corners: contiguous runs of |curvature| > thr smoothed
s = m.starts; k = d / m.lengths   # rad per unit
ks = np.convolve(np.r_[k[-3:], k, k[:3]], np.ones(7) / 7, mode='valid')
on = np.abs(ks) > 0.015
i = 0; n = len(k); out = []
while i < n:
    if on[i]:
        j = i
        while j + 1 < n and on[j + 1] and np.sign(ks[j + 1]) == np.sign(ks[i]): j += 1
        turn = np.degrees(d[i:j + 1].sum())
        if abs(turn) > 8: out.append((round(float(s[i]), 1), round(float(s[j] + m.lengths[j]), 1), 'L' if turn > 0 else 'R', round(float(turn), 1), round(float(np.abs(ks[i:j+1]).max()), 4)))
        i = j + 1
    else: i += 1
for c in out: print(c)
json.dump(out, open(SP / 'work/diag/corners.json', 'w'))
env.close()
