"""Offset profiles for 'partial' laps: the inside line of driver.py with some bends driven on the centre line (no torch).

usage: mkline.py OUT.npz --E 4 --W 28 --k0 0.03 --off C2,C67      (units: C1 C2 C3 C4 C5 C67 C8)
The npz holds 'offset' (for driver.py --line), plus 'off_windows' = along-track windows [a, b] that were flattened.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import driver

UNITS = {'C1': (45.5, 77.0), 'C2': (248.5, 273.0), 'C3': (381.5, 399.0), 'C4': (518.0, 549.5), 'C5': (609.0, 633.5),
         'C67': (728.0, 766.5), 'C8': (879.0, 913.5)}
PAD = 30.0      # the inside line leaves the centre well before the bend and returns well after it
TAPER = 10.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--E', type=float, default=4.0)
    parser.add_argument('--W', type=float, default=28.0)
    parser.add_argument('--k0', type=float, default=0.03)
    parser.add_argument('--off', default='')
    parser.add_argument('--step', type=float, default=0.5)
    args = parser.parse_args()
    from gui import make_driving_env
    from sdc_wrapper import SDC_Wrapper
    from track import TRACK_SEED
    env = SDC_Wrapper(make_driving_env(display=False, collect=True))
    try:
        env.reset(seed=TRACK_SEED)
        points = env.unwrapped.metrics.points.copy()
    finally:
        env.close()
    centre, s, tangent, normal, kappa, total = driver.dense_centre(points, args.step)
    offset = driver.offset_profile(kappa, args.step, args.E, args.W, args.k0)
    windows = []
    for name in [v for v in args.off.split(',') if v]:
        a, b = UNITS[name][0] - PAD, UNITS[name][1] + PAD
        windows.append((a, b))
        # 1 outside the window, 0 inside, cosine taper of TAPER units just inside both ends
        rise = np.clip((s - a) / TAPER, 0, 1)
        fall = np.clip((b - s) / TAPER, 0, 1)
        inside = 0.5 - 0.5 * np.cos(np.pi * np.minimum(rise, fall))
        offset = offset * (1 - inside)
    np.savez(args.out, offset=offset, off_windows=np.array(windows).reshape(-1, 2), s=s)
    print(args.out, 'off', args.off or '-', 'max', round(float(np.abs(offset).max()), 2))


if __name__ == '__main__':
    main()
