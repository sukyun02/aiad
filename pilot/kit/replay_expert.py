"""Replay the recorded expert actions on the fixed track to recover their trajectories and scores."""
import json, os, sys
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy'); os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
REPO = Path('/home/sukyun02/workspace/aiad/assignment1'); sys.path.insert(0, str(REPO / 'src'))
SP = Path(__file__).resolve().parents[1]
import numpy as np
from gui import make_driving_env
from sdc_wrapper import SDC_Wrapper
from track import TRACK_SEED
from driving import DrivingController, DEFAULT_DRIVING
data = {k: v for k, v in np.load(SP / 'cache/data.npz').items()}
out = SP / 'expert'; out.mkdir(exist_ok=True)
summary = []
for lap in sorted(set(data['lap'].tolist())):
    sel = np.where(data['lap'] == lap)[0]
    sel = sel[np.argsort(data['frame'][sel])]
    env = SDC_Wrapper(make_driving_env(display=False, collect=True))
    controller = DrivingController(DEFAULT_DRIVING, env.metadata['render_fps'])
    obs, _ = env.reset(seed=TRACK_SEED); controller.reset()
    metrics = env.unwrapped.metrics
    rows, mismatch, path, prev = [], 0, 0.0, None
    for k, i in enumerate(sel):
        mismatch += int(not np.array_equal(obs, data['obs'][i]))
        pos = np.array(env.unwrapped.car.hull.position)
        path += 0.0 if prev is None else float(np.linalg.norm(pos - prev)); prev = pos
        along, err = metrics.project(pos)
        rows.append((round(along, 2), round(err, 3), int(data['labels'][i])))
        speed = np.linalg.norm(env.unwrapped.car.hull.linearVelocity)
        obs, _, done, trunc, info = env.step(controller.apply(data['act'][i], speed))
    env.close()
    offs = np.array([r[1] for r in rows])
    s = dict(lap=int(lap), frames=len(sel), obs_mismatch_frames=mismatch, lap_time=info.get('lap_time'), finished=info.get('lap_finished'),
             progress=round(info['progress'], 2), center=round(info['center'], 2), path_len=round(path, 1),
             mean_offset=round(float(offs.mean()), 3), max_offset=round(float(offs.max()), 2),
             steer_frac=round(float(np.mean(data['labels'][sel] >= 3)), 4),
             switches=int(np.sum(np.diff(data['labels'][sel]) != 0)))
    summary.append(s); print(json.dumps(s))
    (out / f'lap{lap}.json').write_text(json.dumps(rows))
(out / 'summary.json').write_text(json.dumps(summary, indent=1))
