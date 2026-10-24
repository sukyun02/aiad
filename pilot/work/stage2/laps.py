"""Inspect recorded laps by replaying their actions on the fixed track (no torch).

usage: laps.py SESSION_DIR [SESSION_DIR ...] [--json out.json]
A session dir holds observation_XXXXX.npy / action_XXXXX.npy (what scripts/collect.py saves).
For each lap it prints lap time, progress, centre score, path length, start hold, how far inside each big
corner was driven, and whether the replay reproduced the recorded frames (it must: the simulator is deterministic).
Start it through the kit when several run at once:  k job -- python laps.py ...
"""
import argparse
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

# The eight bends of the fixed track: (name, start, end along the centre line, +1 = left / -1 = right).
# Each is widened by 15 units on both sides, as in the stage-2 analysis.
CORNERS = [('C1', 45.5, 77.0, 1), ('C2', 248.5, 273.0, 1), ('C3', 381.5, 399.0, 1), ('C4', 518.0, 549.5, 1),
           ('C5', 609.0, 633.5, 1), ('C6', 728.0, 738.5, 1), ('C7', 749.0, 766.5, -1), ('C8', 879.0, 913.5, 1)]
MARGIN = 15.0


def replay(folder):
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
    try:
        observation, _ = env.reset(seed=TRACK_SEED)
        controller.reset()
        metrics = env.unwrapped.metrics
        car = env.unwrapped.car
        mismatch, path, previous = 0, 0.0, None
        along, signed = [], []
        info = {}
        for obs_path, action in zip(obs_paths, actions):
            mismatch += int(not np.array_equal(observation, np.load(obs_path)))
            position = np.array(car.hull.position)
            speed = float(np.linalg.norm(car.hull.linearVelocity))
            path += 0.0 if previous is None else float(np.linalg.norm(position - previous))
            previous = position
            relative = position - metrics.points
            fraction = np.clip(np.sum(relative * metrics.vectors, axis=1) / metrics.lengths ** 2, 0, 1)
            errors = np.linalg.norm(relative - fraction[:, None] * metrics.vectors, axis=1)
            j = int(np.argmin(errors))
            v = metrics.vectors[j]
            along.append(float(metrics.starts[j] + fraction[j] * metrics.lengths[j]))
            signed.append(float(np.sign(v[0] * relative[j][1] - v[1] * relative[j][0]) * errors[j]))   # + = left
            observation, _, done, truncated, info = env.step(controller.apply(action, speed))
            if done or truncated:
                break
    finally:
        env.close()
    along, signed = np.array(along), np.array(signed)
    acts = np.array(actions)
    steer, gas, brake = acts[:, 0], acts[:, 1], acts[:, 2]
    hold = int(np.argmax(gas > 0.1)) if (gas > 0.1).any() else len(gas)
    inside = {}
    for name, start, end, side in CORNERS:
        mask = (along >= start - MARGIN) & (along < end + MARGIN)
        inside[name] = round(float((side * signed[mask]).max()), 2) if mask.any() else None
    return dict(lap=folder.name, frames=len(obs_paths), finished=bool(info.get('lap_finished', False)),
                lap_time=info.get('lap_time'), progress=round(info.get('progress', 0.0), 2),
                center=round(info.get('center', 0.0), 2), path_len=round(path, 1), start_hold_frames=hold,
                max_abs_offset=round(float(np.abs(signed).max()), 2), obs_mismatch_frames=mismatch,
                steer_frac=round(float((np.abs(steer) > 0.1).mean()), 3), brake_frames=int((brake > 0.1).sum()),
                coast_frames=int(((gas <= 0.1) & (brake <= 0.1)).sum()) - hold, inside_max=inside)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folders', nargs='+')
    parser.add_argument('--json')
    args = parser.parse_args()
    results = [replay(folder) for folder in args.folders]
    names = [name for name, *_ in CORNERS]
    print(f"{'lap':24s} {'lap_time':>8s} {'prog':>6s} {'centre':>6s} {'path':>6s} {'hold':>4s} {'maxoff':>6s} {'replay':>6s} | inside max at "
          + ' '.join(f'{n:>5s}' for n in names))
    for r in results:
        lap = f"{r['lap_time']:.2f}" if r['lap_time'] is not None else '-'
        replay_ok = 'ok' if r['obs_mismatch_frames'] == 0 else f"{r['obs_mismatch_frames']}!"
        print(f"{r['lap'][:24]:24s} {lap:>8s} {r['progress']:6.2f} {r['center']:6.2f} {r['path_len']:6.1f} {r['start_hold_frames']:4d} "
              f"{r['max_abs_offset']:6.2f} {replay_ok:>6s} | " + ' '.join(f"{r['inside_max'][n]:5.2f}" for n in names))
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=1))
