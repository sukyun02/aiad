"""Scripted driver with HUMAN keyboard constraints (variant of ../driver.py; no torch).

Same target line, same pure pursuit, same official environment / controller / metrics as driver.py.
Added constraints on the steer key (left / right / none):

  --min-press N      once the steer key state changes it stays for at least N frames (N=1: no constraint).
  --press-jitter J   each new key state must be held N + randint(0..J) frames (seed --noise-seed); J=0: off.
  --react D          the key decision at frame t uses the car state of frame t-D (position, heading and, by default,
                     also the steering state).
  --react-steer current   variant: position/heading are D frames old, but the steering state is the present one
                     (a driver knows which key he is holding).
  --decide greedy    (default) the original one-frame rule picks the key, the hold constraint is applied afterwards.
  --decide commit    a key change is judged by the MEAN steering state over the frames it commits to (N), staying on
                     the current key by the next frame only. For N=1 this is the original rule.

With --min-press 1 --react 0 (defaults) the computation is the original one, frame for frame.

usage: driver_h.py --E 4 --W 28 --k0 0.03 --min-press 5 --react 3 [--decide commit] [--out r.json] [--trace t.npz] [--save-dir DIR]
Start it through the kit:  k job -- python driver_h.py ...
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

CORNERS = [('C1', 45.5, 77.0, 1), ('C2', 248.5, 273.0, 1), ('C3', 381.5, 399.0, 1), ('C4', 518.0, 549.5, 1),
           ('C5', 609.0, 633.5, 1), ('C6', 728.0, 738.5, 1), ('C7', 749.0, 766.5, -1), ('C8', 879.0, 913.5, 1)]
MARGIN = 15.0


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


def one_step(state, key):
    """Steering state after one frame of holding `key` (same expressions as the original driver)."""
    if key < 0:
        return max(state - STEER_STEP, -STEER_LIMIT)
    if key > 0:
        return min(state + STEER_STEP, STEER_LIMIT)
    return state - np.clip(state, -STEER_STEP, STEER_STEP)


def mean_state(state, key, frames):
    """Mean steering state over the next `frames` frames of holding `key`."""
    total = 0.0
    for _ in range(frames):
        state = one_step(state, key)
        total += state
    return total / frames


def runs(values):
    out, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            out.append((values[start], start, i - start))
            start = i
    return out


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
        hold_rng = np.random.default_rng(args.noise_seed + 7919)
        burst_left, burst_steer = 0, 0.0
        path, previous = 0.0, None
        info = {}
        status = 'running'
        history = []                      # per frame: (position, forward vector, steering state)
        current_key, held, need = 0.0, 10 ** 9, 1      # steer key state, frames it has been held, frames it must be held
        keys, offroad_frames, first_offroad = [], 0, None
        while True:
            position = np.array(car.hull.position)
            speed = float(np.linalg.norm(car.hull.linearVelocity))
            path += 0.0 if previous is None else float(np.linalg.norm(position - previous))
            previous = position
            forward = np.array(car.hull.GetWorldVector((0.0, 1.0)))
            state = float(controller.steering)                             # negative = left
            frame = len(rows)
            history.append((position, forward, state))
            # what the driver reacts to: the car as it was `react` frames ago
            seen_position, seen_forward, seen_state = history[max(0, frame - args.react)]
            if args.react_steer == 'current':
                seen_state = state
            # nearest point of the target line, searched forward from the last one
            window = (nearest + np.arange(-10, 80)) % count
            nearest = int(window[np.argmin(np.linalg.norm(target[window] - seen_position, axis=1))])
            goal = target[(nearest + look) % count]
            to_goal = goal - seen_position
            distance = float(np.linalg.norm(to_goal)) + 1e-9
            alpha = np.arctan2(seen_forward[0] * to_goal[1] - seen_forward[1] * to_goal[0], float(seen_forward @ to_goal))
            curvature = 2.0 * np.sin(alpha) / distance                     # pure pursuit, positive = left
            wanted = float(np.clip(-args.gain * np.arctan(WHEELBASE * curvature), -STEER_LIMIT, STEER_LIMIT))
            if frame < args.hold:
                command = (0.0, 0.0, 0.0)
            else:
                if args.decide == 'commit' and args.min_press > 1:
                    # a key change commits to `min_press` frames: judge it by the mean steering state over them
                    options = {key: (one_step(seen_state, key) if key == current_key else mean_state(seen_state, key, args.min_press))
                               for key in (-1.0, 1.0, 0.0)}
                else:
                    # the three keyboard choices and where each would move the rate-limited steering state
                    options = {-1.0: max(seen_state - STEER_STEP, -STEER_LIMIT), 1.0: min(seen_state + STEER_STEP, STEER_LIMIT),
                               0.0: seen_state - np.clip(seen_state, -STEER_STEP, STEER_STEP)}
                steer = min(options, key=lambda key: (abs(options[key] - wanted), abs(key)))
                # human-like slips: now and then a short burst of a random key instead of the intended one
                if burst_left == 0 and args.tap_noise > 0 and rng.random() < args.tap_noise:
                    burst_left, burst_steer = int(rng.integers(3, 8)), float(rng.choice([-1.0, 0.0, 1.0]))
                if burst_left > 0:
                    burst_left -= 1
                    steer = burst_steer
                # human constraint: a key state lasts at least `need` frames
                if steer != current_key and held < need:
                    steer = current_key
                if steer != current_key:
                    current_key, held = steer, 0
                    need = args.min_press + (int(hold_rng.integers(0, args.press_jitter + 1)) if args.press_jitter > 0 else 0)
                held += 1
                command = (steer, 0.5, 0.0)
            keys.append(command[0])
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
            if info.get('lane_departure', False):
                offroad_frames += 1
                if first_offroad is None:
                    first_offroad = round(float(along), 1)
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
    keys = np.array(keys, dtype=np.float32)
    key_runs = runs(list(keys[args.hold:]))
    presses = [length for value, _, length in key_runs if value != 0]
    press_starts = [(value, start, length) for value, start, length in key_runs if value != 0]
    gaps = [b[1] - (a[1] + a[2]) for a, b in zip(press_starts[:-1], press_starts[1:])]
    pct = lambda values, p: (round(float(np.percentile(values, p)), 1) if len(values) else None)
    corners = {}
    for name, start, end, side in CORNERS:
        mask = (rows[:, 0] >= start - MARGIN) & (rows[:, 0] < end + MARGIN)
        if mask.any():
            corners[name] = [round(float((side * off[mask]).max()), 2), round(float((-side * off[mask]).max()), 2)]
    result = dict(tap_noise=args.tap_noise, noise_seed=args.noise_seed, E=args.E, W=args.W, k0=args.k0, shape=args.shape, lookahead=args.lookahead, gain=args.gain, hold=args.hold,
                  line=args.line, status=status, lap_time=info.get('lap_time'), progress=round(info['progress'], 2),
                  center=round(info['center'], 2), frames=len(rows), path_len=round(path, 1),
                  max_abs_offset=round(float(np.abs(off).max()), 2), mean_abs_offset=round(float(np.abs(off).mean()), 2),
                  track_err_rms=round(float(np.sqrt(np.mean((off - rows[:, 5]) ** 2))), 2),
                  mean_speed=round(float(rows[100:, 3].mean()), 2), min_speed_after100=round(float(rows[100:, 3].min()), 2),
                  steer_frac=round(float((rows[:, 2] >= 5).mean()), 3), action_counts=counts,
                  target_len=round(float(np.linalg.norm(np.diff(np.vstack([target, target[:1]]), axis=0), axis=1).sum()), 1),
                  # additions of driver_h.py
                  min_press=args.min_press, press_jitter=args.press_jitter, react=args.react, react_steer=args.react_steer,
                  decide=args.decide, offroad_frames=offroad_frames, first_offroad_along=first_offroad,
                  clean=bool(status == 'finished' and info['progress'] > 99.995 and offroad_frames == 0),
                  key_changes=int((np.diff(rows[:, 2]) != 0).sum()), steer_presses=len(presses),
                  press_len=[min(presses, default=None), pct(presses, 25), pct(presses, 50), pct(presses, 75), max(presses, default=None)],
                  gap_len=[min(gaps, default=None), pct(gaps, 25), pct(gaps, 50), pct(gaps, 75), max(gaps, default=None)],
                  corner_inside_outside_max=corners)
    if args.save_dir and status == 'finished':      # same layout as src/demonstrations.save_demonstrations
        folder = Path(args.save_dir)
        folder.mkdir(parents=True, exist_ok=False)
        for index, (obs, act) in enumerate(zip(saved_obs, saved_act)):
            np.save(folder / f'observation_{index:05d}.npy', obs)
            np.save(folder / f'action_{index:05d}.npy', act)
    if args.trace:
        np.savez_compressed(args.trace, rows=rows, offset=offset, kappa=kappa, s=s, keys=keys)
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
    parser.add_argument('--min-press', type=int, default=1, help='a steer key state (left/right/none) lasts at least this many frames')
    parser.add_argument('--press-jitter', type=int, default=0, help='extra 0..J frames (random per key change) on top of --min-press')
    parser.add_argument('--react', type=int, default=0, help='the decision uses the car state of this many frames ago')
    parser.add_argument('--react-steer', default='delayed', choices=['delayed', 'current'],
                        help='delayed: the steering state is also the old one; current: only position/heading are old')
    parser.add_argument('--decide', default='greedy', choices=['greedy', 'commit'])
    parser.add_argument('--out')
    parser.add_argument('--save-dir', help='write the lap as observation_/action_ .npy files (only if it finishes)')
    parser.add_argument('--trace')
    run(parser.parse_args())
