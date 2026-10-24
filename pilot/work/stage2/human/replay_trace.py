"""Replay saved laps (action_XXXXX.npy) on the fixed track through the official env/controller/metrics and write a
per-frame trace (no torch, observations are not loaded).

usage: replay_trace.py OUT_DIR NAME=FOLDER [NAME=FOLDER ...]
writes OUT_DIR/NAME.npz with 'rows' = per frame [along_track, signed lateral offset (+ = left), steer, gas, brake,
speed, steering state before the frame, on_road after the step] and 'info' (json string of the final metrics).
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
REPO = Path('/home/sukyun02/workspace/aiad/assignment1')
sys.path.insert(0, str(REPO / 'src'))

import numpy as np


def replay(folder):
    from gui import make_driving_env
    from sdc_wrapper import SDC_Wrapper
    from track import TRACK_SEED
    from driving import DrivingController, DEFAULT_DRIVING

    folder = Path(folder)
    paths = sorted(folder.glob('action_*.npy'))
    if not paths:
        raise SystemExit(f'no action_*.npy in {folder}')
    actions = [np.load(p) for p in paths]
    env = SDC_Wrapper(make_driving_env(display=False, collect=True))
    controller = DrivingController(DEFAULT_DRIVING, env.metadata['render_fps'])
    rows, info = [], {}
    try:
        env.reset(seed=TRACK_SEED)
        controller.reset()
        metrics = env.unwrapped.metrics
        car = env.unwrapped.car
        for action in actions:
            position = np.array(car.hull.position)
            speed = float(np.linalg.norm(car.hull.linearVelocity))
            relative = position - metrics.points
            fraction = np.clip(np.sum(relative * metrics.vectors, axis=1) / metrics.lengths ** 2, 0, 1)
            errors = np.linalg.norm(relative - fraction[:, None] * metrics.vectors, axis=1)
            j = int(np.argmin(errors))
            v = metrics.vectors[j]
            along = float(metrics.starts[j] + fraction[j] * metrics.lengths[j])
            signed = float(np.sign(v[0] * relative[j][1] - v[1] * relative[j][0]) * errors[j])
            state = float(controller.steering)
            _, _, done, truncated, info = env.step(controller.apply(action, speed))
            rows.append((along, signed, action[0], action[1], action[2], speed, state, 0.0 if info['lane_departure'] else 1.0))
            if done or truncated:
                break
    finally:
        env.close()
    return np.array(rows, dtype=np.float32), info


if __name__ == '__main__':
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for item in sys.argv[2:]:
        name, folder = item.split('=', 1)
        rows, info = replay(folder)
        np.savez_compressed(out / f'{name}.npz', rows=rows, info=json.dumps(info))
        print(name, len(rows), json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}), flush=True)
