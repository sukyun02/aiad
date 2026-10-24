"""Fact-check, claim 12: where exactly does the official off-road test (src/gui.py step(): fixture.TestPoint(car.hull.position)
over the road tiles) flip, measured as lateral distance from the centre line?  No driving, no torch.

For every 0.5 units along the centre line and for both sides, bisect the largest lateral offset d for which the point
centre + d * normal is still inside a road tile, and also express that edge point as the metric offset (metrics.project).
Writes out/geom.json.
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
sys.path.insert(0, str(HERE))

import numpy as np
from driver_fc import dense_centre

CORNERS = [('C1', 45.5, 77.0, 1), ('C2', 248.5, 273.0, 1), ('C3', 381.5, 399.0, 1), ('C4', 518.0, 549.5, 1),
           ('C5', 609.0, 633.5, 1), ('C6', 728.0, 738.5, 1), ('C7', 749.0, 766.5, -1), ('C8', 879.0, 913.5, 1)]

from gui import make_driving_env
from sdc_wrapper import SDC_Wrapper
from track import TRACK_SEED
from gymnasium.envs.box2d.car_racing import TRACK_WIDTH, BORDER

env = SDC_Wrapper(make_driving_env(display=False, collect=True))
try:
    env.reset(seed=TRACK_SEED)
    u = env.unwrapped
    metrics = u.metrics
    centre, s, tangent, normal, kappa, total = dense_centre(metrics.points, 0.5)
    n_tiles = len(u.road)
    assert n_tiles == len(metrics.points), (n_tiles, len(metrics.points))

    def on_road(point, node):
        p = (float(point[0]), float(point[1]))
        for k in range(node - 6, node + 7):
            tile = u.road[k % n_tiles]
            if any(f.TestPoint(p) for f in tile.fixtures):
                return True
        return False

    def on_road_full(point):
        p = (float(point[0]), float(point[1]))
        return any(f.TestPoint(p) for tile in u.road for f in tile.fixtures)

    edge = np.zeros((len(s), 2))        # [:, 0] left, [:, 1] right : largest on-road distance along the normal
    edge_metric = np.zeros((len(s), 2))  # the same edge point expressed as metrics.project() error
    for j in range(len(s)):
        node = int(np.searchsorted(metrics.starts, s[j], side='right') - 1)
        for col, side in enumerate((1.0, -1.0)):
            lo, hi = 3.0, 10.0
            assert on_road(centre[j] + side * lo * normal[j], node)
            for _ in range(24):
                mid = (lo + hi) / 2
                if on_road(centre[j] + side * mid * normal[j], node):
                    lo = mid
                else:
                    hi = mid
            edge[j, col] = lo
            edge_metric[j, col] = metrics.project(centre[j] + side * lo * normal[j])[1]
    # sanity: the local search agrees with the full search on a sample of points
    rng = np.random.default_rng(0)
    for j in rng.integers(0, len(s), 60):
        for side, col in ((1.0, 0), (-1.0, 1)):
            assert on_road_full(centre[j] + side * (edge[j, col] - 0.01) * normal[j])
            assert not on_road_full(centre[j] + side * (edge[j, col] + 0.01) * normal[j])

    result = dict(TRACK_WIDTH=TRACK_WIDTH, BORDER=BORDER, track_length=total, tiles=n_tiles,
                  start_position=[float(v) for v in u.car.hull.position],
                  start_along_error=[float(v) for v in metrics.project(u.car.hull.position)],
                  start_on_road=bool(on_road_full(np.array(u.car.hull.position))))
    straight = np.abs(kappa) < 1e-9
    result['whole_lap'] = dict(left_min=float(edge[:, 0].min()), left_max=float(edge[:, 0].max()),
                               right_min=float(edge[:, 1].min()), right_max=float(edge[:, 1].max()),
                               left_metric_min=float(edge_metric[:, 0].min()), right_metric_min=float(edge_metric[:, 1].min()))
    print('TRACK_WIDTH', TRACK_WIDTH, 'BORDER', BORDER, 'length', total, 'tiles', n_tiles)
    print('whole lap:', result['whole_lap'])
    per = {}
    for name, a, b, side in CORNERS:
        mask = (s >= a - 15) & (s < b + 15)
        core = (s >= a) & (s < b)
        col = 0 if side > 0 else 1
        angle = float(np.degrees(kappa[core].sum() * 0.5))
        per[name] = dict(side='L' if side > 0 else 'R', turn_deg=round(angle, 1),
                         inside_edge_along_normal_min=round(float(edge[mask, col].min()), 3),
                         inside_edge_along_normal_max=round(float(edge[mask, col].max()), 3),
                         inside_edge_metric_min=round(float(edge_metric[mask, col].min()), 3),
                         inside_edge_metric_max=round(float(edge_metric[mask, col].max()), 3),
                         outside_edge_metric_min=round(float(edge_metric[mask, 1 - col].min()), 3))
        print(name, per[name])
    result['corners'] = per

    # kerbs (red/white border polygons): drawn only, never part of self.road; which side of the bend are they on?
    kerbs = []
    for poly, color in u.road_poly:
        color = tuple(int(c) for c in color)
        if color in ((255, 255, 255), (255, 0, 0)):
            c = np.mean(np.array(poly), axis=0)
            j = int(np.argmin(np.linalg.norm(centre - c, axis=1)))
            lateral = float((c - centre[j]) @ normal[j])
            kerbs.append((float(s[j]), lateral))
    kerbs = np.array(kerbs)
    result['kerb_polygons'] = len(kerbs)
    info = {}
    for name, a, b, side in CORNERS:
        mask = (kerbs[:, 0] >= a - 15) & (kerbs[:, 0] < b + 15) if len(kerbs) else np.zeros(0, bool)
        if mask.any():
            lat = kerbs[mask, 1]
            info[name] = dict(count=int(mask.sum()), along=[float(kerbs[mask, 0].min()), float(kerbs[mask, 0].max())],
                              lateral_centre=[round(float(lat.min()), 2), round(float(lat.max()), 2)],
                              where='inside' if np.sign(lat.mean()) == side else 'outside')
        else:
            info[name] = dict(count=0)
        print('kerb', name, info[name])
    outside_any = [(round(float(a), 1), round(float(l), 2)) for a, l in kerbs
                   if not any(c[1] - 15 <= a < c[2] + 15 for c in CORNERS)]
    result['kerbs'] = info
    result['kerbs_outside_corner_windows'] = outside_any
    print('kerbs outside the corner windows:', outside_any)
    np.savez_compressed(HERE / 'out' / 'geom.npz', s=s, edge=edge, edge_metric=edge_metric, kappa=kappa)
    (HERE / 'out' / 'geom.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))
finally:
    env.close()
