"""Replay a saved lap in the official COLLECTION WINDOW (render_mode human, headless SDL) and save screenshots.

usage: shots.py LAP_DIR --frames frames.json --out-dir DIR [--trace t.npz] [--result r.json] [--stop-after N]
  frames.json: {"file_stem": frame_index, ...}; frame f = the picture the driver sees while choosing action f
  (after f simulation steps; frame 0 = the waiting screen before Enter).
What is patched (display only, the simulation is untouched):
  - SDL_VIDEODRIVER=dummy (no real window);
  - gui.fit_window -> (1000, 900): the size of the real window on the student's laptop, non-compact layout;
  - FramePacer.tick -> no sleep (the real pacer only waits to hold 50 fps);
  - pygame.time.get_ticks -> 0 so the blinking record light is always in its bright phase.
The replay is checked against the saved observations frame by frame (the simulator is deterministic).
Start through the kit:  k job -- python shots.py ...
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
REPO = Path('/home/sukyun02/workspace/aiad/assignment1')
sys.path.insert(0, str(REPO / 'src'))

import numpy as np
import pygame

import frame_pacing
import gui

WINDOW = (1000, 900)
gui.fit_window = lambda desktop: WINDOW
frame_pacing.FramePacer.tick = lambda self, fps: 0.0
pygame.time.get_ticks = lambda: 0

# geometry of the collection window (derived from gui._render): canvas 1000 x (720 + 330), scaled to the window height
CANVAS = (1000, 720 + 330)
SCALE = min(WINDOW[0] / CANVAS[0], WINDOW[1] / CANVAS[1])
LEFT = (WINDOW[0] - int(CANVAS[0] * SCALE)) // 2
CAR_PX = (LEFT + 500 * SCALE, 600 * SCALE)             # hull centre: (WINDOW_W/2, WINDOW_H - WINDOW_H/4) on the road surface
PX_PER_UNIT = 2.7 * 6.0 * SCALE                        # ZOOM * SCALE of CarRacing, after the first second


def classify(rgb):
    r, g, b = (int(v) for v in rgb)
    if g > r + 50 and g > b + 50:
        return 'grass'
    if r > 215 and g > 215 and b > 215:
        return 'white'
    if r > 170 and g < 90 and b < 90:
        return 'red'
    if abs(r - g) < 14 and abs(g - b) < 14 and 85 < r < 130:
        return 'road'
    return 'other'


def edge_scan(screen, direction):
    """Walk sideways from the car centre along the screen row of the car; return (pixels to the first non-road pixel
    that stays non-road for 3 pixels, what it is). direction -1 = to the left on screen, +1 = to the right."""
    y = int(round(CAR_PX[1]))
    x0 = CAR_PX[0]
    for d in range(26, 400):                              # 26 px: just outside the car's own wheels (half width 19 px)
        x = int(round(x0 + direction * d))
        if not 0 <= x < WINDOW[0]:
            return None, 'window_edge'
        kinds = [classify(screen.get_at((int(round(x0 + direction * (d + k))), y))[:3]) for k in range(3)
                 if 0 <= int(round(x0 + direction * (d + k))) < WINDOW[0]]
        if kinds and all(kind != 'road' for kind in kinds):
            return d, kinds[-1]
    return None, 'none'


def run(args):
    from sdc_wrapper import SDC_Wrapper
    from track import TRACK_SEED
    from driving import DrivingController, DEFAULT_DRIVING

    folder = Path(args.lap)
    obs_paths = sorted(folder.glob('observation_*.npy'))
    if not obs_paths:
        raise SystemExit(f'no observation_*.npy in {folder}')
    actions = [np.load(p.with_name(p.name.replace('observation_', 'action_', 1))) for p in obs_paths]
    wanted = json.loads(Path(args.frames).read_text()) if args.frames else {}
    by_frame = {}
    for stem, frame in wanted.items():
        by_frame.setdefault(int(frame), []).append(stem)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = SDC_Wrapper(gui.make_driving_env(display=True, collect=True))
    controller = DrivingController(DEFAULT_DRIVING, env.metadata['render_fps'])
    shots, rows = {}, []
    info = {}
    mismatch = 0
    try:
        observation, _ = env.reset(seed=TRACK_SEED)
        controller.reset()
        base = env.unwrapped
        metrics, car = base.metrics, base.car
        assert base.screen.get_size() == WINDOW, base.screen.get_size()

        def capture(frame):
            for stem in by_frame.get(frame, []):
                path = out_dir / f'{stem}.png'
                pygame.image.save(base.screen, str(path))
                position = np.array(car.hull.position)
                along, error = metrics.project(position)
                left = edge_scan(base.screen, -1)
                right = edge_scan(base.screen, +1)
                shots[stem] = dict(file=str(path), frame=frame, t=round(frame / 50, 2), along=round(along, 2),
                                   offset_abs=round(error, 2), status=base.gui_status,
                                   px_to_left_edge=left[0], left_edge_kind=left[1],
                                   px_to_right_edge=right[0], right_edge_kind=right[1],
                                   units_to_left_edge=None if left[0] is None else round(left[0] / PX_PER_UNIT, 2),
                                   units_to_right_edge=None if right[0] is None else round(right[0] / PX_PER_UNIT, 2))

        capture(0)                                            # waiting screen (status 'ready'), as before pressing Enter
        limit = len(actions) if args.stop_after is None else min(len(actions), args.stop_after)
        for frame in range(limit):
            mismatch += int(not np.array_equal(observation, np.load(obs_paths[frame])))
            position = np.array(car.hull.position)
            speed = float(np.linalg.norm(car.hull.linearVelocity))
            relative = position - metrics.points
            fraction = np.clip(np.sum(relative * metrics.vectors, axis=1) / metrics.lengths ** 2, 0, 1)
            errors = np.linalg.norm(relative - fraction[:, None] * metrics.vectors, axis=1)
            j = int(np.argmin(errors))
            v = metrics.vectors[j]
            along = float(metrics.starts[j] + fraction[j] * metrics.lengths[j])
            signed = float(np.sign(v[0] * relative[j][1] - v[1] * relative[j][0]) * errors[j])       # + = left
            action = actions[frame]
            rows.append((along, signed, float(action[0]), speed, float(controller.steering), position[0], position[1],
                         float(car.hull.angle), float(action[1]), float(action[2])))
            base.gui_status = 'recording'                     # what collect.py sets while a lap is being recorded
            base.gui_frames = frame + 1
            observation, _, done, truncated, info = env.step(controller.apply(action, speed))
            capture(frame + 1)
            if done or truncated:
                break
        if 'final' in wanted or args.final:
            path = out_dir / f'{args.final or "final"}.png'
            pygame.image.save(base.screen, str(path))
            shots[path.stem] = dict(file=str(path), frame=len(rows), status=base.gui_status)
        points = np.array(metrics.points)
        track = np.array(base.track)
    finally:
        env.close()
    result = dict(lap=str(folder), frames_saved=len(obs_paths), frames_replayed=len(rows),
                  finished=bool(info.get('lap_finished', False)), lap_time=info.get('lap_time'),
                  progress=round(info.get('progress', 0.0), 2), center=round(info.get('center', 0.0), 2),
                  obs_mismatch_frames=mismatch, window=list(WINDOW), px_per_unit=round(PX_PER_UNIT, 3),
                  car_px=[round(v, 1) for v in CAR_PX], shots=shots)
    if args.trace:
        np.savez_compressed(args.trace, rows=np.array(rows, dtype=np.float64), points=points, track=track)
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.result:
        Path(args.result).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('lap')
    parser.add_argument('--frames')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--trace')
    parser.add_argument('--result')
    parser.add_argument('--final', help='also save the last screen under this file stem')
    parser.add_argument('--stop-after', type=int)
    run(parser.parse_args())
