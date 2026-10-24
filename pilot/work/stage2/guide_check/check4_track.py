"""Track facts straight from the official env (headless, no torch): kerbs, their real length, off-road test on a kerb,
heading change of the centre line, world positions of the lap-1 apexes."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
sys.path.insert(0, '/home/sukyun02/workspace/aiad/assignment1/src')
import numpy as np

HERE = Path('/home/sukyun02/workspace/aiad/pilot/work/stage2/guide_check')

from gui import make_driving_env
from sdc_wrapper import SDC_Wrapper
from track import TRACK_SEED
from gymnasium.envs.box2d import car_racing as cr

env = SDC_Wrapper(make_driving_env(display=False, collect=True))
env.reset(seed=TRACK_SEED)
un = env.unwrapped
m = un.metrics
track = np.asarray(un.track, dtype=float)       # alpha, beta, x, y
print('track nodes', len(track), 'length', round(m.length, 2), 'TRACK_WIDTH', cr.TRACK_WIDTH, 'BORDER', cr.BORDER,
      'ZOOM*SCALE', cr.ZOOM * cr.SCALE, 'WINDOW', cr.WINDOW_W, cr.WINDOW_H, 'FPS', cr.FPS)
print('t after reset', un.t, '-> zoom after reset', 0.1 * cr.SCALE * max(1 - un.t, 0) + cr.ZOOM * cr.SCALE * min(un.t, 1))


def on_road(p):
    return any(fx.TestPoint((float(p[0]), float(p[1]))) for tile in un.road for fx in tile.fixtures)


# kerb polygons: road_poly entries coloured white (255,255,255) or red (255,0,0)
kerbs = []
for poly, color in un.road_poly:
    c = tuple(int(v) for v in color[:3])
    if c in ((255, 255, 255), (255, 0, 0)):
        kerbs.append((np.array(poly, dtype=float), c))
print('kerb blocks', len(kerbs))
blocks = []
for poly, c in kerbs:
    # vertices: b1_l (road edge at node1), b1_r (outer at node1), b2_r (outer at node2), b2_l (road edge at node2)
    e1, o1, o2, e2 = poly
    a1, d1 = m.project(e1)
    a2, d2 = m.project(e2)
    mid_edge = (e1 + e2) / 2
    am, dm = m.project(mid_edge)
    # side: sign of cross(track direction, edge - centre)
    j = int(np.argmin(np.linalg.norm(m.points - mid_edge, axis=1)))
    v = m.vectors[j]
    rel = mid_edge - m.points[j]
    side = 'left' if (v[0] * rel[1] - v[1] * rel[0]) > 0 else 'right'
    centre_of_block = poly.mean(axis=0)
    blocks.append(dict(along1=a1, along2=a2, along_mid=am, side=side, edge_len=float(np.linalg.norm(e2 - e1)),
                       outer_len=float(np.linalg.norm(o2 - o1)), width=float(np.linalg.norm(o1 - e1)),
                       dist_edge_from_centre=(d1, d2), block_centre_on_road=on_road(centre_of_block),
                       just_inside_on_road=on_road(mid_edge + 0.15 * (mid_edge - centre_of_block) / np.linalg.norm(mid_edge - centre_of_block)),
                       color='white' if c == (255, 255, 255) else 'red'))
blocks.sort(key=lambda b: b['along_mid'])
groups, cur = [], []
for b in blocks:
    if cur and b['along_mid'] - cur[-1]['along_mid'] > 8:
        groups.append(cur); cur = []
    cur.append(b)
if cur:
    groups.append(cur)
out = []
for g in groups:
    lo = min(min(b['along1'], b['along2']) for b in g); hi = max(max(b['along1'], b['along2']) for b in g)
    edge = sum(b['edge_len'] for b in g)
    print(f"kerb: {len(g)} blocks, along {lo:.1f}-{hi:.1f} (centre-line extent {hi-lo:.1f}), side {set(b['side'] for b in g)},"
          f" real length along the road edge {edge:.1f} units = {edge*13.886:.0f} px in the 1000x900 window, width {g[0]['width']:.2f},"
          f" block centres on road: {[b['block_centre_on_road'] for b in g]}, 0.15 inside the edge on road: {[b['just_inside_on_road'] for b in g]},"
          f" first block colour {g[0]['color']}")
    out.append(dict(along_start=lo, along_end=hi, blocks=len(g), edge_len=edge))
(HERE / 'kerbs_check.json').write_text(json.dumps(out, indent=1))

# heading of the centre line vs along
pts = m.points
vec = m.vectors
head = np.unwrap(np.arctan2(vec[:, 1], vec[:, 0]))
starts = m.starts


def heading_at(a):
    j = int(np.clip(np.searchsorted(starts, a, side='right') - 1, 0, len(starts) - 1))
    return head[j]


for label, lo, hi in (('C1', 40, 85), ('K1', 190, 236), ('C2', 240, 280), ('C3 window 350-450', 350, 450), ('C3 381.5-399', 381.5, 399.0),
                      ('C3 340-460', 340, 460), ('C4', 510, 560), ('C5', 600, 640), ('K2 660-690', 660, 690), ('K2 650-712', 650, 712),
                      ('C6 728-738.5', 726, 741), ('C6 715-747', 715, 747), ('C7 749-766.5', 747.5, 768), ('C8', 870, 925)):
    d = np.degrees(heading_at(hi) - heading_at(lo))
    print(f'heading change {label}: {d:+.1f} deg (+ = left)')
# segment lengths and node spacing
print('node spacing: median', round(float(np.median(m.lengths)), 3), 'min', round(float(m.lengths.min()), 3), 'max', round(float(m.lengths.max()), 3))

# geometric half-width at the inside of the big bends: nearest on-road limit measured from the centre line polyline
# sample: for along positions in each bend, walk from the centre line towards the inside along the local normal until off-road
print('on-road limit measured along the local normal towards the inside (step 0.02):')
for name, lo, hi, side in (('C1', 52.5, 70.0, 1), ('C2', 255.5, 269.5, 1), ('C4', 525.0, 549.5, 1), ('C5', 612.5, 630.0, 1), ('C7', 749.0, 766.5, -1), ('C8', 889.0, 910.0, 1)):
    lims = []
    for a in np.arange(lo, hi, 0.5):
        j = int(np.clip(np.searchsorted(starts, a, side='right') - 1, 0, len(starts) - 1))
        t = vec[j] / m.lengths[j]
        p0 = pts[j] + t * (a - starts[j])
        n = np.array([-t[1], t[0]]) * side
        r = 0.0
        while r < 9 and on_road(p0 + n * (r + 0.02)):
            r += 0.02
        lims.append(r)
    print(f'  {name}: inside limit min {min(lims):.2f} median {np.median(lims):.2f} max {max(lims):.2f}')

# world positions of the lap-1 apexes from my own replay of scriptA/lap1
r = np.load(HERE / 'replay/A1.npz')['rows']
apex_frames = dict(C1=139, C2=579, C3=856, C4=1173, C5=1361, C6=1615, C7=1651, C8=1949)
print('lap-1 apex world positions (x, y) and offsets from my replay:')
pos = {}
for k, f in apex_frames.items():
    pos[k] = r[f, 8:10]
    print(f'  {k}: frame {f} along {r[f,0]:.1f} offset {r[f,1]:+.2f} pos ({r[f,8]:.1f}, {r[f,9]:.1f}) on_road {bool(r[f,7])}')
print('  start pos', r[0, 8:10].round(1).tolist(), 'heading of first centre segment (deg)', round(float(np.degrees(head[0])), 1))
# pairwise distances for comparison with the track map (3.0 px per unit)
names = list(pos)
for a, b in (('C1', 'C2'), ('C2', 'C3'), ('C3', 'C4'), ('C4', 'C5'), ('C5', 'C6'), ('C6', 'C7'), ('C7', 'C8'), ('C8', 'C1'), ('C1', 'C4')):
    print(f'  dist {a}-{b}: {np.linalg.norm(pos[a]-pos[b]):.1f} units = {3.0*np.linalg.norm(pos[a]-pos[b]):.0f} px on the map')
env.close()
