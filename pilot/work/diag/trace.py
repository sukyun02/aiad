"""diag lens: extended headless drive (replica of kit.drive) with signed offset, heading error, speed, steering state, logits.
usage: trace.py MODEL.pth OUT.npz [--hud zero] [--noise K --noise-seed S] [--nudge-seed S] [--device cuda] [--keep-obs]
"""
import argparse, json, os, sys
from pathlib import Path
SP = Path('/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad')
sys.path.insert(0, str(SP / 'kit'))
import kit
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy'); os.environ.setdefault('SDL_AUDIODRIVER', 'dummy'); os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
sys.path.insert(0, str(kit.SRC))
import numpy as np, torch

def drive(model_path, out, device='cuda', hud='none', noise=0, noise_seed=0, nudge_seed=None, keep_obs=False, max_seconds=150.0,
          sens='none'):
    from gui import make_driving_env
    from sdc_wrapper import SDC_Wrapper
    from track import TRACK_SEED
    from driving import DrivingController, DEFAULT_DRIVING
    model_path = Path(model_path)
    Net, _ = kit.load_network_class(model_path.with_suffix('.network.py'), 'eval_network')
    device = torch.device(device)
    ck = torch.load(model_path, map_location='cpu', weights_only=True)
    model = Net(device=device); model.load_state_dict(ck['state_dict'], strict=True); model.device = device; model.eval()
    env = SDC_Wrapper(make_driving_env(display=False, unlimited=True))
    controller = DrivingController(DEFAULT_DRIVING, env.metadata['render_fps'])
    rng = np.random.default_rng(noise_seed)
    nudges = {}
    if nudge_seed is not None:
        r2 = np.random.default_rng(1000 + nudge_seed)
        for f in r2.integers(80, 2050, size=6):
            c = int(r2.choice([5, 6]))
            for k in range(4):
                nudges[int(f) + k] = c
    rows, logits, obs_keep = [], [], []
    info = {}; status = 'running'; path = 0.0; prev = None; t = 0
    try:
        observation, _ = env.reset(seed=TRACK_SEED); controller.reset()
        m = env.unwrapped.metrics
        with torch.inference_mode():
            while True:
                seen = observation
                if hud == 'zero':
                    seen = observation.copy(); seen[kit.HUD_BLOB] = 0
                if noise:
                    seen = np.clip(seen.astype(np.int16) + rng.integers(-noise, noise + 1, size=seen.shape), 0, 255).astype(np.uint8)
                if keep_obs:
                    obs_keep.append(seen.copy())
                image = torch.as_tensor(np.ascontiguousarray(seen[None]), dtype=torch.float32, device=device)
                scores = model(image)
                cls = int(torch.argmax(scores, dim=1).item())
                forced = 0
                if t in nudges:
                    cls = nudges[t]; forced = 1
                action = model.action_classes[cls]
                top = torch.topk(scores[0], 2).values
                car = env.unwrapped.car
                speed = float(np.linalg.norm(car.hull.linearVelocity))
                pos = np.array(car.hull.position)
                path += 0.0 if prev is None else float(np.linalg.norm(pos - prev)); prev = pos
                rel = pos - m.points
                fr = np.clip(np.sum(rel * m.vectors, axis=1) / m.lengths**2, 0, 1)
                perp = rel - fr[:, None] * m.vectors
                errs = np.linalg.norm(perp, axis=1); i = int(np.argmin(errs))
                along = float(m.starts[i] + fr[i] * m.lengths[i])
                v = m.vectors[i]
                sign = np.sign(v[0] * rel[i][1] - v[1] * rel[i][0])      # + = car left of centre line (track direction)
                head = float(car.hull.angle) + np.pi / 2 - np.arctan2(v[1], v[0])
                head = (head + np.pi) % (2 * np.pi) - np.pi                # + = car pointing left of track direction
                steer_state = float(controller.steering)
                observation, _, done, truncated, info = env.step(controller.apply(action, speed))
                rows.append((along, sign * errs[i], cls, float(top[0] - top[1]), speed, steer_state, head,
                             float(info.get('lane_departure', False)), forced))
                logits.append(scores[0].float().cpu().numpy())
                t += 1
                if truncated or (done and not info.get('lap_finished', False)):
                    status = 'aborted'; break
                if info.get('lap_finished', False):
                    status = 'finished'; break
                if info['elapsed'] >= max_seconds:
                    status = 'timeout'; break
    finally:
        env.close()
    rows = np.array(rows, dtype=np.float32)
    res = dict(model=str(model_path), hud=hud, noise=noise, noise_seed=noise_seed, nudge_seed=nudge_seed, status=status,
               lap_time=info.get('lap_time'), progress=round(info['progress'], 2), center=round(info['center'], 2),
               frames=len(rows), path_len=round(path, 1), max_offset=round(float(np.abs(rows[:, 1]).max()), 2),
               steer_frac=round(float((rows[:, 2] >= 3).mean()), 4), offroad_frames=int(rows[:, 7].sum()))
    if out:
        extra = dict(obs=np.array(obs_keep)) if keep_obs else {}
        np.savez_compressed(out, rows=rows, logits=np.array(logits, dtype=np.float32), meta=json.dumps(res), **extra)
    return res

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model'); p.add_argument('out')
    p.add_argument('--device', default='cuda'); p.add_argument('--hud', default='none')
    p.add_argument('--noise', type=int, default=0); p.add_argument('--noise-seed', type=int, default=0)
    p.add_argument('--nudge-seed', type=int); p.add_argument('--keep-obs', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(1)
    with kit.Slot('eval', kit.EVAL_SLOTS):
        r = drive(a.model, a.out if a.out != '-' else None, a.device, a.hud, a.noise, a.noise_seed, a.nudge_seed, a.keep_obs)
    print(json.dumps(r))
