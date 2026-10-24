"""Replay saved laps headless (official env/controller/metrics, no torch, no window) and write a per-frame trace.

usage: replay_trace.py OUT_DIR LAP_DIR [LAP_DIR ...]
For each lap: OUT_DIR/<lap name>.npz with rows = [along_track, signed lateral offset (+ = left), steer key (-1/0/+1),
speed, steering state before the frame's action, x, y, hull angle, gas, brake]  and OUT_DIR/<lap name>.json (result).
Start through the kit:  k job -- python replay_trace.py ...
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


def replay(folder, out_dir):
    from gui import make_driving_env
    from sdc_wrapper import SDC_Wrapper
    from track import TRACK_SEED
    from driving import DrivingController, DEFAULT_DRIVING

    folder = Path(folder)
    obs_paths = sorted(folder.glob('observation_*.npy'))
    if not obs_paths:
        raise SystemExit(f'no observation_*.npy in {folder}')
    actions = [np.load(p.with_name(p.name.replace('observation_', 'action_', 1))) for p in obs_paths]
    env = SDC_Wrapper(make_driving_env(display=False, collect=True))
    controller = DrivingController(DEFAULT_DRIVING, env.metadata['render_fps'])
    rows, mismatch, info = [], 0, {}
    try:
        observation, _ = env.reset(seed=TRACK_SEED)
        controller.reset()
        metrics, car = env.unwrapped.metrics, env.unwrapped.car
        for obs_path, action in zip(obs_paths, actions):
            mismatch += int(not np.array_equal(observation, np.load(obs_path)))
            position = np.array(car.hull.position)
            speed = float(np.linalg.norm(car.hull.linearVelocity))
            relative = position - metrics.points
            fraction = np.clip(np.sum(relative * metrics.vectors, axis=1) / metrics.lengths ** 2, 0, 1)
            errors = np.linalg.norm(relative - fraction[:, None] * metrics.vectors, axis=1)
            j = int(np.argmin(errors))
            v = metrics.vectors[j]
            along = float(metrics.starts[j] + fraction[j] * metrics.lengths[j])
            signed = float(np.sign(v[0] * relative[j][1] - v[1] * relative[j][0]) * errors[j])
            rows.append((along, signed, float(action[0]), speed, float(controller.steering), position[0], position[1],
                         float(car.hull.angle), float(action[1]), float(action[2])))
            observation, _, done, truncated, info = env.step(controller.apply(action, speed))
            if done or truncated:
                break
    finally:
        env.close()
    rows = np.array(rows, dtype=np.float64)
    gas = rows[:, 8]
    result = dict(lap=str(folder), frames=len(rows), finished=bool(info.get('lap_finished', False)),
                  lap_time=info.get('lap_time'), progress=round(info.get('progress', 0.0), 2),
                  center=round(info.get('center', 0.0), 2), obs_mismatch_frames=mismatch,
                  start_hold_frames=int(np.argmax(gas > 0.1)) if (gas > 0.1).any() else len(gas),
                  max_abs_offset=round(float(np.abs(rows[:, 1]).max()), 2))
    np.savez_compressed(Path(out_dir) / f'{folder.name}.npz', rows=rows)
    (Path(out_dir) / f'{folder.name}.json').write_text(json.dumps(result, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for lap in sys.argv[2:]:
        replay(lap, out)
