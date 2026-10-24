"""How much does ONE key tap of N frames turn the car at full speed?  (official env + controller, no torch)

For each N: drive the centre line with the original 1-frame rule up to along-track 165 (between C1 and C2),
release the key for 20 frames, hold LEFT for N frames, release for 50 frames, and measure the heading change and the
sideways drift that results. Writes out/tap_response.json.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
sys.path.insert(0, '/home/sukyun02/workspace/aiad/assignment1/src')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from driver_h import dense_centre, smooth_closed, WHEELBASE, STEER_STEP, STEER_LIMIT


def experiment(n_press, start_along=165.0):
    from gui import make_driving_env
    from sdc_wrapper import SDC_Wrapper
    from track import TRACK_SEED
    from driving import DrivingController, DEFAULT_DRIVING
    env = SDC_Wrapper(make_driving_env(display=False, collect=True))
    controller = DrivingController(DEFAULT_DRIVING, env.metadata['render_fps'])
    try:
        env.reset(seed=TRACK_SEED)
        controller.reset()
        metrics = env.unwrapped.metrics
        centre, s, tangent, normal, kappa, total = dense_centre(metrics.points, 0.5)
        car = env.unwrapped.car
        count, look, nearest = len(centre), 18, 0
        phase, timer = 'approach', 0
        log = []
        k_smooth = smooth_closed(kappa, 20)
        straight = float(np.abs(k_smooth[(s > 165) & (s < 215)]).max())
        while True:
            position = np.array(car.hull.position)
            speed = float(np.linalg.norm(car.hull.linearVelocity))
            along, error = metrics.project(position)
            state = float(controller.steering)
            if phase == 'approach':
                window = (nearest + np.arange(-10, 80)) % count
                nearest = int(window[np.argmin(np.linalg.norm(centre[window] - position, axis=1))])
                goal = centre[(nearest + look) % count]
                forward = np.array(car.hull.GetWorldVector((0.0, 1.0)))
                to_goal = goal - position
                alpha = np.arctan2(forward[0] * to_goal[1] - forward[1] * to_goal[0], float(forward @ to_goal))
                wanted = float(np.clip(-np.arctan(WHEELBASE * 2.0 * np.sin(alpha) / (np.linalg.norm(to_goal) + 1e-9)), -STEER_LIMIT, STEER_LIMIT))
                options = {-1.0: max(state - STEER_STEP, -STEER_LIMIT), 1.0: min(state + STEER_STEP, STEER_LIMIT),
                           0.0: state - np.clip(state, -STEER_STEP, STEER_STEP)}
                steer = min(options, key=lambda key: (abs(options[key] - wanted), abs(key)))
                if start_along <= along < start_along + 60:
                    phase, timer = 'settle', 0
            if phase == 'settle':
                steer = 0.0
                timer += 1
                if timer > 20:
                    phase, timer = 'press', 0
                    heading0, position0, speed0 = float(car.hull.angle), position.copy(), speed
                    direction0 = np.array(car.hull.linearVelocity) / speed
            if phase == 'press':
                steer = -1.0
                timer += 1
                if timer > n_press:
                    phase, timer = 'release', 0
            if phase == 'release':
                steer = 0.0
                timer += 1
                log.append((float(car.hull.angle), state))
                if timer > 50:
                    velocity = np.array(car.hull.linearVelocity)
                    direction1 = velocity / np.linalg.norm(velocity)
                    moved = position - position0
                    lateral = float(direction0[0] * moved[1] - direction0[1] * moved[0])     # + = left of the old direction
                    course = float(np.degrees(np.arctan2(direction0[0] * direction1[1] - direction0[1] * direction1[0], direction0 @ direction1)))
                    return dict(press_frames=n_press, heading_change_deg=round(float(np.degrees(car.hull.angle - heading0)), 2),
                                course_change_deg=round(course, 2), lateral_shift_after=round(lateral, 2),
                                frames_after_press_start=n_press + 50, peak_steering=round(0.03 * n_press if n_press < 15 else 0.45, 2),
                                speed_before=round(speed0, 2), speed_after=round(speed, 2), start_along=round(float(metrics.project(position0)[0]), 1),
                                max_abs_smoothed_curvature_165_215=round(straight, 4))
            env.step(controller.apply(np.array([steer, 0.5, 0.0], dtype=np.float32), speed))
    finally:
        env.close()


if __name__ == '__main__':
    results = []
    for n in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15):
        r = experiment(n)
        results.append(r)
        print(json.dumps(r), flush=True)
    Path(__file__).resolve().parent.joinpath('out', 'tap_response.json').write_text(json.dumps(results, indent=1))
