"""Replay the 7 provided human keyboard laps (official env/controller/metrics, no torch) and report per lap:
start hold, time at which each corner starts/ends, min speed after the launch, min speed per corner, inside offsets,
action-class counts.  Writes out/human.json.
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
HERE = Path(__file__).resolve().parent

import numpy as np

CORNERS = [('C1', 45.5, 77.0, 1), ('C2', 248.5, 273.0, 1), ('C3', 381.5, 399.0, 1), ('C4', 518.0, 549.5, 1),
           ('C5', 609.0, 633.5, 1), ('C6', 728.0, 738.5, 1), ('C7', 749.0, 766.5, -1), ('C8', 879.0, 913.5, 1)]

from gui import make_driving_env
from sdc_wrapper import SDC_Wrapper
from track import TRACK_SEED
from driving import DrivingController, DEFAULT_DRIVING


def classify(a):
    left, right, gas, brake = a[0] < -0.1, a[0] > 0.1, a[1] > 0.1, a[2] > 0.1
    if left and gas:
        return 5
    if right and gas:
        return 6
    if left and brake:
        return 7
    if right and brake:
        return 8
    if left:
        return 3
    if right:
        return 4
    if gas:
        return 1
    if brake:
        return 2
    return 0


results = []
env = SDC_Wrapper(make_driving_env(display=False, collect=True))
controller = DrivingController(DEFAULT_DRIVING, env.metadata['render_fps'])
try:
    for folder in sorted((REPO / 'data/collected').glob('20260923_*')):
        actions = [np.load(p) for p in sorted(folder.glob('action_*.npy'))]
        observation, _ = env.reset(seed=TRACK_SEED)
        controller.reset()
        metrics = env.unwrapped.metrics
        car = env.unwrapped.car
        along, signed, speeds = [], [], []
        info = {}
        for action in actions:
            position = np.array(car.hull.position)
            speed = float(np.linalg.norm(car.hull.linearVelocity))
            relative = position - metrics.points
            fraction = np.clip(np.sum(relative * metrics.vectors, axis=1) / metrics.lengths ** 2, 0, 1)
            errors = np.linalg.norm(relative - fraction[:, None] * metrics.vectors, axis=1)
            j = int(np.argmin(errors))
            v = metrics.vectors[j]
            along.append(float(metrics.starts[j] + fraction[j] * metrics.lengths[j]))
            signed.append(float(np.sign(v[0] * relative[j][1] - v[1] * relative[j][0]) * errors[j]))
            speeds.append(speed)
            observation, _, done, truncated, info = env.step(controller.apply(action, speed))
            if done or truncated:
                break
        along, signed, speeds = np.array(along), np.array(signed), np.array(speeds)
        acts = np.array(actions)
        gas = acts[:, 1]
        hold = int(np.argmax(gas > 0.1))
        classes = np.bincount([classify(a) for a in acts], minlength=9).tolist()
        length = metrics.length
        unwrapped = np.r_[along[0], along[0] + np.cumsum((np.diff(along) + length / 2) % length - length / 2)]
        if along[0] > length / 2:
            unwrapped -= length
        launch = hold + int(np.argmax(speeds[hold:] >= 22.2))
        corner = {}
        for name, a, b, side in CORNERS:
            t_in = float(np.argmax(unwrapped >= a)) / 50
            t_out = float(np.argmax(unwrapped >= b)) / 50
            mask = (along >= a - 15) & (along < b + 15) & (np.arange(len(along)) >= launch)
            corner[name] = dict(t_in=round(t_in, 2), t_out=round(t_out, 2), min_speed=round(float(speeds[mask].min()), 2),
                                inside_max=round(float((side * signed[mask]).max()), 2))
        results.append(dict(lap=folder.name, lap_time=info.get('lap_time'), progress=round(info['progress'], 2), center=round(info['center'], 2),
                            frames=len(actions), hold=hold, frames_hold_to_22p2=launch - hold,
                            min_speed_after_launch=round(float(speeds[launch:].min()), 2), max_speed=round(float(speeds.max()), 2),
                            classes=classes, corners=corner))
        r = results[-1]
        print(f"{r['lap']} lap {r['lap_time']:.2f} hold {hold} launch+{launch - hold} minspeed {r['min_speed_after_launch']} classes {classes}")
        print('   corner entry-exit s: ' + ' '.join(f"{n} {c['t_in']:.2f}-{c['t_out']:.2f}" for n, c in corner.items()))
        print('   corner min speed  : ' + ' '.join(f"{n} {c['min_speed']:.2f}" for n, c in corner.items()))
finally:
    env.close()
(HERE / 'out' / 'human.json').write_text(json.dumps(results, ensure_ascii=False, indent=1))
print('written', HERE / 'out' / 'human.json')
