"""Drive one checkpoint through the official inference window (headless SDL) and keep what the report needs.

usage: drive_model.py MODEL.pth --out DIR [--every 10]
Writes DIR/trace.npz (per frame: x, y, off_road, class, speed, along, error; plus road polygons and the centre line),
DIR/result.json (official metrics of the lap), DIR/final.png (the window when the lap ends) and DIR/frames/fNNNN.png
(the window every --every frames).
Same loader and drive loop as src/evaluation.drive_until_return; only the display is patched
(SDL dummy driver, window fixed to 1000x900, frame pacer does not sleep).
Start through the kit:  k job --kind eval -- python drive_model.py ...
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')      # same as scripts/infer.py
REPO = Path('/home/sukyun02/workspace/aiad/assignment1')
sys.path.insert(0, str(REPO / 'src'))

import numpy as np
import pygame
import torch

import frame_pacing
import gui

gui.fit_window = lambda desktop: (1000, 900)
frame_pacing.FramePacer.tick = lambda self, fps: 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--out', required=True)
    parser.add_argument('--every', type=int, default=10)
    args = parser.parse_args()
    from device import select_device
    from driving import DEFAULT_DRIVING, DrivingController
    from evaluation import warm_up_model
    from model_io import load_model
    from sdc_wrapper import SDC_Wrapper
    from track import TRACK_SEED

    out = Path(args.out)
    (out / 'frames').mkdir(parents=True, exist_ok=True)
    device, _ = select_device()
    model, checkpoint = load_model(str(Path(args.model).resolve()), device, allow_equivalent_source=True)
    model.eval()
    model.device = device
    warm_up_model(model, device)
    env = SDC_Wrapper(gui.make_driving_env(display=True, unlimited=True, language='kor'))
    base = env.unwrapped
    base.gui_model = Path(args.model).name
    base.gui_score_mode = False
    controller = DrivingController(dict(DEFAULT_DRIVING), env.metadata['render_fps'])
    rows, info, failed = [], {}, False
    try:
        base.gui_status = 'driving'
        observation, _ = env.reset(seed=TRACK_SEED)
        controller.reset()
        metrics, car = base.metrics, base.car
        with torch.inference_mode():
            while True:
                position = np.array(car.hull.position)
                along, error = metrics.project(position)
                image = torch.as_tensor(np.ascontiguousarray(observation[None]), dtype=torch.float32, device=device)
                scores = model(image)
                action = model.scores_to_action(scores)
                speed = float(np.linalg.norm(car.hull.linearVelocity))
                observation, _, done, truncated, info = env.step(controller.apply(action, speed))
                rows.append((position[0], position[1], float(metrics.off_road), int(scores.argmax(1)), speed, along, error))
                failed = truncated or (done and not info.get('lap_finished', False))
                if failed or info.get('lap_finished', False):
                    base.gui_status = 'aborted' if failed else 'finished'
                    env.render()
                    pygame.image.save(base.screen, str(out / 'final.png'))
                    break
                if len(rows) % args.every == 0:
                    pygame.image.save(base.screen, str(out / 'frames' / f'f{len(rows):04d}.png'))
                if len(rows) >= 30000:
                    break
        road = np.array([np.array(poly, dtype=float) for poly, _ in base.road_poly])
        colors = np.array([np.array(color[:3], dtype=float) for _, color in base.road_poly])
        np.savez_compressed(out / 'trace.npz', rows=np.array(rows), road=road, colors=colors,
                            centre=np.array(metrics.points), half_width=metrics.half_width)
    finally:
        env.close()
    rows = np.array(rows)
    cls = rows[:, 3].astype(int)
    gas = np.isin(cls, (1, 5, 6))
    result = dict(model=str(args.model), device=str(device), checkpoint_epoch=checkpoint.get('epoch'),
                  status='aborted' if failed else 'finished' if info.get('lap_finished') else 'timeout',
                  lap_time=info.get('lap_time'), progress=round(info['progress'], 2), center=round(info['center'], 2),
                  frames=len(rows), off_road_frames=int(rows[:, 2].sum()), first_gas_frame=int(np.argmax(gas)),
                  class_counts=np.bincount(cls, minlength=9).tolist(), max_abs_offset=round(float(rows[:, 6].max()), 2),
                  mean_speed_after_100=round(float(rows[100:, 4].mean()), 2), min_speed_after_100=round(float(rows[100:, 4].min()), 2))
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
