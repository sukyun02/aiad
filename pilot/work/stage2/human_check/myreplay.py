"""Checker's own replay of saved action files through the official env/controller/metrics (no torch, observations not loaded).

usage: myreplay.py OUT.json NAME=FOLDER [NAME=FOLDER ...]
Per lap: official lap time / progress / centre, off-road frame count and episodes, per-corner numbers.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
sys.path.insert(0, '/home/sukyun02/workspace/aiad/assignment1/src')

import numpy as np

CORNERS = [('C1', 45.5, 77.0, 1), ('C2', 248.5, 273.0, 1), ('C3', 381.5, 399.0, 1), ('C4', 518.0, 549.5, 1),
           ('C5', 609.0, 633.5, 1), ('C6', 728.0, 738.5, 1), ('C7', 749.0, 766.5, -1), ('C8', 879.0, 913.5, 1)]
MARGIN = 15.0


def one(name, folder):
    from gui import make_driving_env
    from sdc_wrapper import SDC_Wrapper
    from track import TRACK_SEED
    from driving import DrivingController, DEFAULT_DRIVING

    files = sorted(Path(folder).glob('action_*.npy'))
    acts = [np.load(f) for f in files]
    env = SDC_Wrapper(make_driving_env(display=False, collect=True))
    ctl = DrivingController(DEFAULT_DRIVING, env.metadata['render_fps'])
    along, off, onroad, speed = [], [], [], []
    info = {}
    try:
        env.reset(seed=TRACK_SEED)
        ctl.reset()
        m = env.unwrapped.metrics
        car = env.unwrapped.car
        pts, vec, ln, st = m.points, m.vectors, m.lengths, m.starts
        used = 0
        for a in acts:
            p = np.array(car.hull.position)
            v = float(np.linalg.norm(car.hull.linearVelocity))
            rel = p - pts
            fr = np.clip((rel * vec).sum(1) / ln ** 2, 0, 1)
            err = np.linalg.norm(rel - fr[:, None] * vec, axis=1)
            j = int(err.argmin())
            cross = vec[j, 0] * rel[j, 1] - vec[j, 1] * rel[j, 0]
            along.append(float(st[j] + fr[j] * ln[j]))
            off.append(float(np.sign(cross) * err[j]))
            speed.append(v)
            _, _, done, trunc, info = env.step(ctl.apply(a, v))
            onroad.append(not info['lane_departure'])
            used += 1
            if done or trunc:
                break
    finally:
        env.close()
    along, off, onroad, speed = np.array(along), np.array(off), np.array(onroad), np.array(speed)
    acts = np.array(acts)[:used]
    steer = np.where(acts[:, 0] < -0.1, -1, np.where(acts[:, 0] > 0.1, 1, 0))
    gas = acts[:, 1] > 0.1
    hold = int(np.argmax(gas)) if gas.any() else used
    # off-road episodes
    eps, i = [], 0
    while i < used:
        if not onroad[i]:
            k = i
            while k < used and not onroad[k]:
                k += 1
            eps.append(dict(start_along=round(float(along[i]), 1), frames=int(k - i), peak_offset=round(float(off[i:k][np.abs(off[i:k]).argmax()]), 2)))
            i = k
        else:
            i += 1
    corners = {}
    for c, s, e, side in CORNERS:
        mask = (along >= s - MARGIN) & (along < e + MARGIN)
        mask[:hold] = False
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        seg = steer[lo:hi]
        inside_key = -side
        # runs of the inside key that touch the section (clipped to it)
        longest, n_runs, run = 0, 0, 0
        for x in list(seg) + [0]:
            if x == inside_key:
                run += 1
            else:
                if run:
                    n_runs += 1
                    longest = max(longest, run)
                run = 0
        corners[c] = dict(frames=int(mask.sum()), inside_presses=n_runs, inside_frames=int((seg == inside_key).sum()), inside_longest=int(longest),
                          counter_frames=int((seg == -inside_key).sum()), inside_max_offset=round(float((side * off[lo:hi]).max()), 2),
                          outside_max_offset=round(float((-side * off[lo:hi]).max()), 2))
    return dict(name=name, folder=str(folder), frames_saved=len(files), frames_used=used, hold=hold, finished=bool(info.get('lap_finished')),
                lap_time=info.get('lap_time'), progress=round(info.get('progress', 0.0), 2), center=round(info.get('center', 0.0), 2),
                offroad_frames=int((~onroad).sum()), offroad_episodes=eps, max_abs_offset=round(float(np.abs(off).max()), 2),
                mean_abs_offset=round(float(np.abs(off[hold:]).mean()), 2),
                min_speed_after_hold_plus_150=round(float(speed[hold + 150:].min()), 2) if used > hold + 150 else None,
                driving_frames=used - hold, corners=corners)


if __name__ == '__main__':
    out = Path(sys.argv[1])
    res = [one(*item.split('=', 1)) for item in sys.argv[2:]]
    out.write_text(json.dumps(res, indent=1))
    for r in res:
        print(json.dumps({k: v for k, v in r.items() if k != 'corners'}), flush=True)
