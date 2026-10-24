"""Replay expert laps with the same extended row format as trace.py (signed offset, speed, steering state, heading)."""
import json, os, sys
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy'); os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
REPO = Path('/home/sukyun02/workspace/aiad/assignment1'); sys.path.insert(0, str(REPO / 'src'))
SP = Path('/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad')
import numpy as np
from gui import make_driving_env
from sdc_wrapper import SDC_Wrapper
from track import TRACK_SEED
from driving import DrivingController, DEFAULT_DRIVING
data = {k: v for k, v in np.load(SP / 'cache/data.npz').items()}
for lap in sorted(set(data['lap'].tolist())):
    sel = np.where(data['lap'] == lap)[0]; sel = sel[np.argsort(data['frame'][sel])]
    env = SDC_Wrapper(make_driving_env(display=False, collect=True))
    controller = DrivingController(DEFAULT_DRIVING, env.metadata['render_fps'])
    obs, _ = env.reset(seed=TRACK_SEED); controller.reset()
    m = env.unwrapped.metrics
    rows = []; mism = 0
    for i in sel:
        mism += int(not np.array_equal(obs, data['obs'][i]))
        car = env.unwrapped.car
        speed = float(np.linalg.norm(car.hull.linearVelocity)); pos = np.array(car.hull.position)
        rel = pos - m.points
        fr = np.clip(np.sum(rel * m.vectors, axis=1) / m.lengths**2, 0, 1)
        perp = rel - fr[:, None] * m.vectors
        errs = np.linalg.norm(perp, axis=1); j = int(np.argmin(errs))
        along = float(m.starts[j] + fr[j] * m.lengths[j]); v = m.vectors[j]
        sign = np.sign(v[0] * rel[j][1] - v[1] * rel[j][0])
        head = float(car.hull.angle) + np.pi / 2 - np.arctan2(v[1], v[0]); head = (head + np.pi) % (2 * np.pi) - np.pi
        st = float(controller.steering)
        obs, _, done, trunc, info = env.step(controller.apply(data['act'][i], speed))
        rows.append((along, sign * errs[j], int(data['labels'][i]), np.nan, speed, st, head, float(info.get('lane_departure', False)), 0))
    env.close()
    np.savez_compressed(SP / f'work/diag/traces/expert_lap{lap}.npz', rows=np.array(rows, dtype=np.float32))
    print(lap, len(rows), 'mismatch', mism, info.get('lap_time'))
