"""Whole-track map for the driver's guide, drawn with pygame (no matplotlib here, no simulation).

Inputs : guide/replay_A1.npz (rows with x,y of scriptA/lap1 during the window replay, the track nodes and the metric
         centre line), guide/corner_numbers.json (turn-in / apex / back frames of lap 1 and the set-A statistics).
Outputs: guide/track_map.png (about 1400 px wide) and guide/kerbs.json (where CarRacing draws red/white kerbs).
Orientation: as on the waiting screen before Enter (car pointing up at the start line).
Start through the kit:  k job -- python trackmap.py
"""
import json
import math
import os
from pathlib import Path

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import numpy as np
import pygame

GUIDE = Path('/home/sukyun02/workspace/aiad/pilot/work/stage2/guide')
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
TRACK_WIDTH = 40 / 6.0
BORDER = 8 / 6.0
TURN_RATE = 0.31
WIDTH = 1400
MARGIN = (150, 130, 150, 250)             # left, top, right, bottom
BG = (255, 255, 255)
ROAD = (205, 208, 214)
EDGE = (90, 96, 108)
CENTRE = (255, 255, 255)
LINE = (20, 90, 220)
INK = (25, 30, 40)
SOFT = (95, 105, 120)
RED = (215, 35, 45)
# label anchor offsets in pixels (dx, dy) from the apex point, tuned by eye after the first render
LABEL_SHIFT = {'C1': (-423, 10), 'C2': (-185, 95), 'C3': (41, 73), 'C4': (39, -115), 'C5': (-360, 67),
               'C6': (-253, 141), 'C7': (130, 155), 'C8': (-350, -155)}


def kerb_flags(track):
    n = len(track)
    border = [False] * n
    for i in range(n):
        good, oneside = True, 0
        for neg in range(4):
            beta1, beta2 = track[i - neg][1], track[i - neg - 1][1]
            good &= abs(beta1 - beta2) > TURN_RATE * 0.2
            oneside += np.sign(beta1 - beta2)
        border[i] = bool(good and abs(oneside) == 4)
    for i in range(n):
        for neg in range(4):
            border[i - neg] |= border[i]
    return border


def main():
    pygame.font.init()
    data = np.load(GUIDE / 'replay_A1.npz')
    rows, track, points = data['rows'], data['track'], data['points']
    numbers = json.loads((GUIDE / 'corner_numbers.json').read_text())
    n = len(track)
    beta, xy = track[:, 1], track[:, 2:4]
    normal = np.c_[np.cos(beta), np.sin(beta)]                   # points to the RIGHT of the driving direction
    left, right = xy - TRACK_WIDTH * normal, xy + TRACK_WIDTH * normal
    seg = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    starts = np.r_[0.0, np.cumsum(seg)[:-1]]

    theta = -rows[0, 7]                                           # same rotation as the game's camera at the start
    c, s = math.cos(theta), math.sin(theta)
    rotate = lambda p: np.c_[p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c]
    every = rotate(np.vstack([left, right]))
    lo, hi = every.min(axis=0), every.max(axis=0)
    scale = (WIDTH - MARGIN[0] - MARGIN[2]) / (hi[0] - lo[0])
    height = int((hi[1] - lo[1]) * scale + MARGIN[1] + MARGIN[3])

    def px(p):
        r = rotate(np.atleast_2d(np.asarray(p, dtype=float)))
        out = np.c_[MARGIN[0] + (r[:, 0] - lo[0]) * scale, MARGIN[1] + (hi[1] - r[:, 1]) * scale]
        return [tuple(v) for v in out]

    surface = pygame.Surface((WIDTH, height))
    surface.fill(BG)
    big, mid, small = (pygame.font.Font(FONT, size) for size in (26, 20, 17))

    def text(value, pos, font=mid, color=INK, anchor='topleft'):
        image = font.render(value, True, color)
        rect = image.get_rect(**{anchor: pos})
        surface.blit(image, rect)
        return rect

    # road tiles and kerbs
    border = kerb_flags(track)
    kerb_tiles = []
    for i in range(n):
        j = i - 1
        pygame.draw.polygon(surface, ROAD, px([left[i], right[i], right[j], left[j]]))
    for i in range(n):
        j = i - 1
        if border[i]:
            side = float(np.sign(track[j][1] - track[i][1]))      # +1 = right of the driving direction, -1 = left
            a1, a2 = xy[i] + side * TRACK_WIDTH * normal[i], xy[i] + side * (TRACK_WIDTH + BORDER) * normal[i]
            b1, b2 = xy[j] + side * TRACK_WIDTH * normal[j], xy[j] + side * (TRACK_WIDTH + BORDER) * normal[j]
            pygame.draw.polygon(surface, RED if i % 2 else (250, 250, 250), px([a1, a2, b2, b1]))
            pygame.draw.polygon(surface, RED, px([a1, a2, b2, b1]), 1)
            kerb_tiles.append((float(starts[j % n]), float(starts[j % n] + seg[j % n]), 'left' if side < 0 else 'right'))
    for edge in (left, right):
        pygame.draw.lines(surface, EDGE, True, px(edge), 2)
    centre_px = px(points)
    for k in range(0, len(centre_px), 2):                         # dashed centre line
        pygame.draw.line(surface, CENTRE, centre_px[k], centre_px[(k + 1) % len(centre_px)], 2)

    # merge kerb tiles into along-track intervals
    kerbs = []
    for a, b, side in sorted(kerb_tiles):
        if kerbs and kerbs[-1]['side'] == side and a - kerbs[-1]['end'] < 1e-6:
            kerbs[-1]['end'] = b
        else:
            kerbs.append(dict(start=a, end=b, side=side))
    for k in kerbs:
        k['start'], k['end'] = round(k['start'], 1), round(k['end'], 1)
    (GUIDE / 'kerbs.json').write_text(json.dumps(dict(
        note='red/white kerb strips drawn by CarRacing: 1.33 wide, OUTSIDE the road edge (not road tiles -> off-road for the '
             'car centre), on the inside of the bend; along-track intervals of the centre line',
        kerbs=kerbs), indent=1))

    # reference trajectory of scriptA/lap1
    path = px(rows[:, 5:7])
    pygame.draw.lines(surface, LINE, False, path, 3)
    for second in range(5, 45, 5):                                # time ticks
        f = second * 50
        if f < len(path):
            pygame.draw.circle(surface, BG, path[f], 6)
            pygame.draw.circle(surface, LINE, path[f], 6, 2)
            direction = np.array(path[min(f + 5, len(path) - 1)]) - np.array(path[f - 5])
            out = np.array([-direction[1], direction[0]]) / (np.linalg.norm(direction) + 1e-9)   # screen-right of travel = outside of the loop
            text(f'{second}s', tuple(np.array(path[f]) + out * 40), small, LINE, 'center')

    # start line and direction arrow
    a, b = px([left[0], right[0]])
    pygame.draw.line(surface, INK, a, b, 5)
    p0, p1 = px([xy[0] + 1.6 * TRACK_WIDTH * normal[0], xy[0] + 1.6 * TRACK_WIDTH * normal[0] + 26 * np.array([-normal[0][1], normal[0][0]])])
    pygame.draw.line(surface, INK, p0, p1, 4)
    d = np.array(p1) - np.array(p0)
    d = d / np.linalg.norm(d)
    side_v = np.array([-d[1], d[0]])
    pygame.draw.polygon(surface, INK, [tuple(np.array(p1) + d * 16), tuple(np.array(p1) + side_v * 9), tuple(np.array(p1) - side_v * 9)])
    text('START / FINISH  (t = 0 → 41.72 s)', (a[0] - 14, a[1] - 56), mid, INK, 'topright')
    text('출발선 · 주행 방향 ↑ (반시계 방향)', (a[0] - 14, a[1] - 28), small, SOFT, 'topright')

    # corner labels
    for name, entry in numbers['corners'].items():
        lap1 = entry['A']['per_lap'][0]
        apex, turn, back = lap1['apex_frame'], lap1['turn_frame'], lap1['back_frame']
        for f, color in ((turn, (20, 150, 70)), (back, (20, 150, 70))):
            pygame.draw.circle(surface, color, path[f], 6)
        pygame.draw.circle(surface, RED, path[apex], 8)
        pygame.draw.circle(surface, BG, path[apex], 3)
        shift = LABEL_SHIFT[name]
        anchor = (path[apex][0] + shift[0], path[apex][1] + shift[1])
        lines = [(f"{name}  {'좌' if entry['direction'] == 'left' else '우'} {entry['angle_deg']}°  apex {apex / 50:.2f} s", big, INK),
                 (f"in {turn / 50:.2f} s → out {back / 50:.2f} s · {lap1['apex_inside']:.1f} inside", mid, INK),
                 (f"set A {entry['A']['apex_inside']['min']:.1f}–{entry['A']['apex_inside']['max']:.1f} inside · saves {entry['gain_vs_E0_s']:.2f} s", small, SOFT)]
        y = anchor[1]
        rects = []
        for value, font, color in lines:
            rects.append(text(value, (anchor[0], y), font, color))
            y = rects[-1].bottom + 1
        box = rects[0].unionall(rects[1:])
        target = (min(max(path[apex][0], box.left), box.right), box.top if path[apex][1] < box.top else box.bottom)
        pygame.draw.line(surface, SOFT, path[apex], target, 1)

    # the two gentle right kinks that have no corner number
    for label, along in (('K1 우 13° (완만)', 186.0), ('K2 우 19° (완만)', 655.0)):
        f = 10 + int(np.argmax(rows[10:, 0] >= along))
        direction = np.array(path[f + 5]) - np.array(path[f - 5])
        out = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        text(label, tuple(np.array(path[f]) + out * 58), small, SOFT, 'center')

    # title, legend, scale bar
    text('Fixed track (seed 22597174) — reference lap scriptA/lap1: 41.72 s, centre score 67.6', (24, 16), big)
    text('centre line 959 long → 42.60 s at the 22.5 speed cap;  road half-width 6.67;  times = seconds since the first frame',
         (24, 54), small, SOFT)
    lx, ly = 40, height - 78                                      # legend: two rows of three items
    col = 440
    pygame.draw.line(surface, LINE, (lx, ly + 10), (lx + 44, ly + 10), 3)
    text('reference line (lap 1 of set A)', (lx + 54, ly - 2), small)
    pygame.draw.circle(surface, RED, (lx + col + 22, ly + 10), 8)
    pygame.draw.circle(surface, BG, (lx + col + 22, ly + 10), 3)
    text('apex = largest inside offset', (lx + col + 54, ly - 2), small)
    pygame.draw.circle(surface, (20, 150, 70), (lx + 2 * col + 22, ly + 10), 6)
    text('turn-in / back: line is 1.0 from the centre', (lx + 2 * col + 54, ly - 2), small)
    ly += 34
    pygame.draw.rect(surface, RED, (lx, ly + 3, 22, 14))
    pygame.draw.rect(surface, (250, 250, 250), (lx + 22, ly + 3, 22, 14))
    pygame.draw.rect(surface, RED, (lx, ly + 3, 44, 14), 1)
    text('red/white kerb: outside the road = off-road', (lx + 54, ly - 2), small)
    pygame.draw.rect(surface, ROAD, (lx + col, ly + 1, 44, 18))
    pygame.draw.line(surface, CENTRE, (lx + col, ly + 10), (lx + col + 18, ly + 10), 2)
    pygame.draw.line(surface, CENTRE, (lx + col + 26, ly + 10), (lx + col + 44, ly + 10), 2)
    text('road (13.3 wide) with centre line', (lx + col + 54, ly - 2), small)
    bar = 50 * scale
    bx = lx + 2 * col
    pygame.draw.line(surface, INK, (bx, ly + 10), (bx + bar, ly + 10), 3)
    for x in (bx, bx + bar):
        pygame.draw.line(surface, INK, (x, ly + 4), (x, ly + 16), 3)
    text('50 track units = 2.2 s at full speed', (bx + bar + 12, ly - 2), small)

    pygame.image.save(surface, str(GUIDE / 'track_map.png'))
    print(json.dumps(dict(size=[WIDTH, height], px_per_unit=round(scale, 3), kerbs=kerbs)))


if __name__ == '__main__':
    main()
