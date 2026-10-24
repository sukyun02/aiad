"""Own replay of saved laps through the official env (headless, no torch): per-frame trace + result.
usage: check2_replay.py OUT_DIR NAME=FOLDER [NAME=FOLDER ...]
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


def one(name, folder, out_dir):
    from gui import make_driving_env
    from sdc_wrapper import SDC_Wrapper
    from track import TRACK_SEED
    from driving import DrivingController, DEFAULT_DRIVING

    folder = Path(folder)
    obs_paths = sorted(folder.glob('observation_*.npy'))
    acts = np.array([np.load(folder / p.name.replace('observation_', 'action_', 1)) for p in obs_paths])
    env = SDC_Wrapper(make_driving_env(display=False, collect=True))
    ctrl = DrivingController(DEFAULT_DRIVING, env.metadata['render_fps'])
    rows = []
    mismatch = 0
    info = {}
    try:
        obs, _ = env.reset(seed=TRACK_SEED)
        ctrl.reset()
        un = env.unwrapped
        m = un.metrics
        pts, vec, ln, st = m.points, m.vectors, m.lengths, m.starts
        for k, (p, a) in enumerate(zip(obs_paths, acts)):
            if not np.array_equal(obs, np.load(p)):
                mismatch += 1
            pos = np.array(un.car.hull.position, dtype=float)
            speed = float(np.linalg.norm(un.car.hull.linearVelocity))
            rel = pos - pts
            fr = np.clip((rel * vec).sum(1) / ln ** 2, 0, 1)
            perp = rel - fr[:, None] * vec
            err = np.linalg.norm(perp, axis=1)
            j = int(np.argmin(err))
            along = st[j] + fr[j] * ln[j]
            cross = vec[j, 0] * rel[j, 1] - vec[j, 1] * rel[j, 0]
            signed = np.sign(cross) * err[j]
            on_road = any(fx.TestPoint(un.car.hull.position) for tile in un.road for fx in tile.fixtures)
            rows.append((along, signed, a[0], a[1], a[2], speed, float(ctrl.steering), float(on_road), pos[0], pos[1]))
            obs, _, done, trunc, info = env.step(ctrl.apply(a, speed))
            if done or trunc:
                break
        steps_done = len(rows)
    finally:
        env.close()
    rows = np.array(rows)
    gas = acts[:, 1]
    hold = int(np.argmax(gas > 0.1)) if (gas > 0.1).any() else len(gas)
    res = dict(name=name, folder=str(folder), saved_frames=len(obs_paths), steps_replayed=steps_done, obs_mismatch=mismatch,
               lap_finished=bool(info.get('lap_finished')), lap_time=info.get('lap_time'), progress=round(float(info.get('progress', 0)), 3),
               center=round(float(info.get('center', 0)), 3), start_hold_frames=hold,
               off_road_frames=int((rows[:, 7] < 0.5).sum()), max_abs_offset=round(float(np.abs(rows[:, 1]).max()), 3),
               no_gas_frames_after_hold=int((gas[hold:] <= 0.1).sum()), brake_frames=int((acts[:, 2] > 0.1).sum()))
    np.savez_compressed(Path(out_dir) / f'{name}.npz', rows=rows)
    (Path(out_dir) / f'{name}.json').write_text(json.dumps(res) + '\n')
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in sys.argv[2:]:
        name, folder = spec.split('=', 1)
        one(name, folder, out_dir)
