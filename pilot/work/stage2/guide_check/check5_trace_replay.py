"""Re-drive a lap from the ACTION CLASS column of a trace npz through the official env (headless, no torch) and compare
with what the trace / its json say. Independent of driver_hold.py: only the key sequence is taken from the trace.
usage: check5_trace_replay.py OUT.jsonl NAME=TRACE.npz [...]
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


def one(name, path):
    from gui import make_driving_env
    from sdc_wrapper import SDC_Wrapper
    from track import TRACK_SEED
    from driving import DrivingController, DEFAULT_DRIVING

    rows = np.load(path)['rows'].astype(float)
    cls = rows[:, 2].astype(int)
    steer = np.where(cls == 5, -1.0, np.where(cls == 6, 1.0, 0.0))
    gas = np.where(cls == 0, 0.0, 0.5)
    env = SDC_Wrapper(make_driving_env(display=False, collect=True))
    ctrl = DrivingController(DEFAULT_DRIVING, env.metadata['render_fps'])
    info = {}
    max_dev_along = max_dev_off = 0.0
    off_road = 0
    max_abs = 0.0
    steps = 0
    try:
        env.reset(seed=TRACK_SEED)
        ctrl.reset()
        un = env.unwrapped
        m = un.metrics
        for k in range(len(rows)):
            pos = np.array(un.car.hull.position, dtype=float)
            speed = float(np.linalg.norm(un.car.hull.linearVelocity))
            along, err = m.project(pos)
            d = abs(along - rows[k, 0])
            d = min(d, abs(d - m.length))
            max_dev_along = max(max_dev_along, d)
            max_dev_off = max(max_dev_off, abs(err - abs(rows[k, 1])))
            max_abs = max(max_abs, err)
            if not any(fx.TestPoint(un.car.hull.position) for tile in un.road for fx in tile.fixtures):
                off_road += 1
            a = np.array([steer[k], gas[k], 0.0], dtype=np.float32)
            _, _, done, trunc, info = env.step(ctrl.apply(a, speed))
            steps += 1
            if done or trunc:
                break
    finally:
        env.close()
    js_path = Path(path).with_suffix('.json')
    js = json.loads(js_path.read_text()) if js_path.exists() else {}
    res = dict(name=name, trace_frames=len(rows), steps=steps, lap_finished=bool(info.get('lap_finished')), lap_time=info.get('lap_time'),
               progress=round(float(info.get('progress', 0)), 3), center=round(float(info.get('center', 0)), 3), off_road_frames=off_road,
               max_abs_offset=round(max_abs, 3), max_dev_along_vs_trace=round(max_dev_along, 4), max_dev_offset_vs_trace=round(max_dev_off, 4),
               json_lap_time=js.get('lap_time'), json_progress=js.get('progress'), json_center=js.get('center'), json_max_abs_offset=js.get('max_abs_offset'))
    print(json.dumps(res), flush=True)
    return res


if __name__ == '__main__':
    out = Path(sys.argv[1])
    results = [one(*spec.split('=', 1)) for spec in sys.argv[2:]]
    out.write_text('\n'.join(json.dumps(r) for r in results) + '\n')
