"""Scripted driver on the fixed track, for stage-2 feasibility analysis (no torch).

It follows a target line = centre line + lateral offset profile with keyboard-like commands
(steer -1 / 0 / +1, gas always on) through the official environment, controller and metrics.

usage: driver.py --E 4 [--W 21] [--k0 0.02] [--lookahead 9] [--gain 1.0] [--hold 0] [--out result.json] [--trace t.npz]
Start it through the kit:  k job -- python driver.py ...
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

WHEELBASE = 3.24          # CarRacing wheel positions (+1.60 / -1.64 after SIZE scaling)
STEER_STEP = 0.03         # controller: steering_rate 1.5 / 50 fps
STEER_LIMIT = 0.45


def dense_centre(points, step=0.5):
    """Closed centre line resampled every `step`: positions, arc length, unit tangent, left normal, curvature."""
    closed = np.vstack([points, points[:1]])
    seg = np.diff(closed, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.r_[0.0, np.cumsum(seg_len)]
    total = cum[-1]
    s = np.arange(0.0, total, step)
    x = np.interp(s, cum, closed[:, 0])
    y = np.interp(s, cum, closed[:, 1])
    pos = np.c_[x, y]
    index = np.clip(np.searchsorted(cum, s, side='right') - 1, 0, len(seg) - 1)
    tangent = seg[index] / seg_len[index][:, None]
    heading = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    # curvature of the polyline is concentrated at the vertices; spread it with a short window
    dh = np.diff(np.r_[heading, heading[0] + 2 * np.pi * np.round((heading[-1] - heading[0]) / (2 * np.pi))])
    kappa = dh / step
    return pos, s, tangent, np.c_[-tangent[:, 1], tangent[:, 0]], kappa, total


def smooth_closed(values, window):
    """Moving average on a closed loop (window in samples), applied twice for a smoother shape."""
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    for _ in range(2):
        padded = np.r_[values[-window:], values, values[:window]]
        values = np.convolve(padded, kernel, mode='same')[window:-window]
    return values


def offset_profile(kappa, step, E, W, k0, shape='clip'):
    """Lateral offset (positive = left of the driving direction): towards the inside of each bend, up to E."""
    k = smooth_closed(kappa, max(1, int(round(W / step))))
    if shape == 'sign':                       # full offset wherever the smoothed curvature exceeds k0
        g = np.where(np.abs(k) > k0, np.sign(k), 0.0)
        g = smooth_closed(g, max(1, int(round(W / step))))
    else:                                     # proportional, saturating at k0
        g = np.clip(k / k0, -1.0, 1.0)
    return E * g


def run(args):
    from gui import make_driving_env
    from sdc_wrapper import SDC_Wrapper
    from track import TRACK_SEED
    from driving import DrivingController, DEFAULT_DRIVING

    env = SDC_Wrapper(make_driving_env(display=False, collect=True))
    controller = DrivingController(DEFAULT_DRIVING, env.metadata['render_fps'])
    try:
        observation, _ = env.reset(seed=TRACK_SEED)
        controller.reset()
        saved_obs, saved_act = [], []
        metrics = env.unwrapped.metrics
        centre, s, tangent, normal, kappa, total = dense_centre(metrics.points, args.step)
        if args.line:
            offset = np.load(args.line)['offset']
            assert len(offset) == len(s), (len(offset), len(s))
        else:
            offset = offset_profile(kappa, args.step, args.E, args.W, args.k0, args.shape)
        target = centre + offset[:, None] * normal
        count = len(target)
        look = max(1, int(round(args.lookahead / args.step)))
        car = env.unwrapped.car
        nearest = 0
        rows, counts = [], [0] * 9
        rng = np.random.default_rng(args.noise_seed)
        burst_left, burst_steer = 0, 0.0
        path, previous = 0.0, None
        info = {}
        status = 'running'
        while True:
            position = np.array(car.hull.position)
            speed = float(np.linalg.norm(car.hull.linearVelocity))
            path += 0.0 if previous is None else float(np.linalg.norm(position - previous))
            previous = position
            # nearest point of the target line, searched forward from the last one
            window = (nearest + np.arange(-10, 80)) % count
            nearest = int(window[np.argmin(np.linalg.norm(target[window] - position, axis=1))])
            goal = target[(nearest + look) % count]
            forward = np.array(car.hull.GetWorldVector((0.0, 1.0)))
            to_goal = goal - position
            distance = float(np.linalg.norm(to_goal)) + 1e-9
            alpha = np.arctan2(forward[0] * to_goal[1] - forward[1] * to_goal[0], float(forward @ to_goal))
            curvature = 2.0 * np.sin(alpha) / distance                     # pure pursuit, positive = left
            wanted = float(np.clip(-args.gain * np.arctan(WHEELBASE * curvature), -STEER_LIMIT, STEER_LIMIT))
            state = float(controller.steering)                             # negative = left
            frame = len(rows)
            if frame < args.hold:
                command = (0.0, 0.0, 0.0)
            else:
                # the three keyboard choices and where each would move the rate-limited steering state
                options = {-1.0: max(state - STEER_STEP, -STEER_LIMIT), 1.0: min(state + STEER_STEP, STEER_LIMIT),
                           0.0: state - np.clip(state, -STEER_STEP, STEER_STEP)}
                steer = min(options, key=lambda key: (abs(options[key] - wanted), abs(key)))
                # human-like slips: now and then a short burst of a random key instead of the intended one
                if burst_left == 0 and args.tap_noise > 0 and rng.random() < args.tap_noise:
                    burst_left, burst_steer = int(rng.integers(3, 8)), float(rng.choice([-1.0, 0.0, 1.0]))
                if burst_left > 0:
                    burst_left -= 1
                    steer = burst_steer
                command = (steer, 0.5, 0.0)
            cls = 0 if frame < args.hold else {(-1.0): 5, 1.0: 6, 0.0: 1}[command[0]]
            counts[cls] += 1
            along, error = metrics.project(position)
            j = int(np.argmin(np.linalg.norm(centre - position, axis=1)))
            sign = np.sign(tangent[j, 0] * (position - centre[j])[1] - tangent[j, 1] * (position - centre[j])[0])
            rows.append((along, sign * error, cls, speed, state, offset[j]))
            action = np.array(command, dtype=np.float32)
            if args.save_dir:
                saved_obs.append(observation.copy())
                saved_act.append(action)
            observation, _, done, truncated, info = env.step(controller.apply(action, speed))
            if truncated or (done and not info.get('lap_finished', False)):
                status = 'aborted'
                break
            if info.get('lap_finished', False):
                status = 'finished'
                break
            if info['elapsed'] >= 150:
                status = 'timeout'
                break
    finally:
        env.close()
    rows = np.array(rows, dtype=np.float32)
    off = rows[:, 1]
    result = dict(tap_noise=args.tap_noise, noise_seed=args.noise_seed, E=args.E, W=args.W, k0=args.k0, shape=args.shape, lookahead=args.lookahead, gain=args.gain, hold=args.hold,
                  line=args.line, status=status, lap_time=info.get('lap_time'), progress=round(info['progress'], 2),
                  center=round(info['center'], 2), frames=len(rows), path_len=round(path, 1),
                  max_abs_offset=round(float(np.abs(off).max()), 2), mean_abs_offset=round(float(np.abs(off).mean()), 2),
                  track_err_rms=round(float(np.sqrt(np.mean((off - rows[:, 5]) ** 2))), 2),
                  mean_speed=round(float(rows[100:, 3].mean()), 2), min_speed_after100=round(float(rows[100:, 3].min()), 2),
                  steer_frac=round(float((rows[:, 2] >= 5).mean()), 3), action_counts=counts,
                  target_len=round(float(np.linalg.norm(np.diff(np.vstack([target, target[:1]]), axis=0), axis=1).sum()), 1))
    if args.save_dir and status == 'finished':      # same layout as src/demonstrations.save_demonstrations
        folder = Path(args.save_dir)
        folder.mkdir(parents=True, exist_ok=False)
        for index, (obs, act) in enumerate(zip(saved_obs, saved_act)):
            np.save(folder / f'observation_{index:05d}.npy', obs)
            np.save(folder / f'action_{index:05d}.npy', act)
    if args.trace:
        np.savez_compressed(args.trace, rows=rows, offset=offset, kappa=kappa, s=s)
    text = json.dumps(result, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--E', type=float, default=0.0, help='maximum inside offset in track units (road half-width 6.67)')
    parser.add_argument('--W', type=float, default=21.0, help='smoothing window of the curvature in track units')
    parser.add_argument('--k0', type=float, default=0.02, help='curvature at which the full offset is used')
    parser.add_argument('--shape', default='clip', choices=['clip', 'sign'])
    parser.add_argument('--line', help='npz with an explicit offset array (overrides the profile)')
    parser.add_argument('--lookahead', type=float, default=9.0)
    parser.add_argument('--gain', type=float, default=1.0)
    parser.add_argument('--hold', type=int, default=0, help='frames of doing nothing at the start')
    parser.add_argument('--step', type=float, default=0.5)
    parser.add_argument('--tap-noise', type=float, default=0.0, help='probability per frame of a 3-7 frame random-key burst')
    parser.add_argument('--noise-seed', type=int, default=0)
    parser.add_argument('--out')
    parser.add_argument('--save-dir', help='write the lap as observation_/action_ .npy files (only if it finishes)')
    parser.add_argument('--trace')
    run(parser.parse_args())
