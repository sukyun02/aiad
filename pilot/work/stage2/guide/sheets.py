"""Contact sheets for the driver's guide from the collection-window screenshots (pygame only, no simulation).

Inputs : guide/shots/*.png, guide/replay_A1.json, guide/replay_centre.json (per-shot frame, time, offset, edge distances)
Outputs: guide/sheet_C1.png, sheet_C4.png, sheet_C8.png (turn-in / apex / exit / centre-line apex), sheet_apexes.png (all 8 apex
         close-ups), sheet_C6C7.png (the left-right combination), sheet_start.png (waiting screen, zoom-in, straight).
The yellow bar on a tile runs from the car's centre point to the inside road edge on the car's own screen row; its
length in track units is measured on the screenshot (13.886 px per unit) and printed in the caption.
Start through the kit:  k job -- python sheets.py
"""
import json
import os
from pathlib import Path

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame

GUIDE = Path('/home/sukyun02/workspace/aiad/pilot/work/stage2/guide')
SHOTS = GUIDE / 'shots'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
ROAD_VIEW = pygame.Rect(71, 0, 857, 600)          # the road picture inside the 1000x900 window (panel and bars cut off)
CAR = (499.6, 514.3)
PX_PER_UNIT = 13.886
CAR_WIDTH = 2 * (55 + 14) * 0.02                  # outer faces of the wheels (car_dynamics.py): 2.76 track units
BG = (255, 255, 255)
INK = (25, 30, 40)
SOFT = (95, 105, 120)
YELLOW = (255, 221, 0)
SIDE = {'C1': 'left', 'C2': 'left', 'C3': 'left', 'C4': 'left', 'C5': 'left', 'C6': 'left', 'C7': 'right', 'C8': 'left'}
ANGLE = {'C1': 76, 'C2': 42, 'C3': 19, 'C4': 79, 'C5': 32, 'C6': 14, 'C7': 42, 'C8': 97}


def load_meta():
    meta = {}
    for name in ('replay_A1', 'replay_centre'):
        meta.update(json.loads((GUIDE / f'{name}.json').read_text())['shots'])
    return meta


def tile(stem, meta, crop, scale, side=None):
    """Cropped, scaled screenshot with the centre-to-inside-edge bar drawn on it."""
    image = pygame.image.load(str(SHOTS / f'{stem}.png'))
    part = image.subsurface(crop).copy()
    info = meta[stem]
    if side and info.get(f'px_to_{side}_edge'):
        direction = -1 if side == 'left' else 1
        x0, y = CAR[0] - crop.x, CAR[1] - crop.y
        x1 = x0 + direction * info[f'px_to_{side}_edge']
        pygame.draw.line(part, (0, 0, 0), (x0, y), (x1, y), 5)
        pygame.draw.line(part, YELLOW, (x0, y), (x1, y), 3)
        for x in (x0, x1):
            pygame.draw.line(part, (0, 0, 0), (x, y - 9), (x, y + 9), 5)
            pygame.draw.line(part, YELLOW, (x, y - 8), (x, y + 8), 3)
    size = (round(crop.w * scale), round(crop.h * scale))
    return pygame.transform.smoothscale(part, size)


def edge_text(info, side):
    units = info.get(f'units_to_{side}_edge')
    if units is None:
        return ''
    kind = info[f'{side}_edge_kind']
    kerb = ', red/white kerb there' if kind in ('white', 'red') else ', grass there (no kerb)'
    return f'inside edge {units:.1f} from the car centre = {units / CAR_WIDTH:.1f} car widths{kerb}'


def wrap(value, font, width):
    """Split a caption into lines that fit the tile width."""
    lines, current = [], ''
    for word in value.split(' '):
        trial = word if not current else current + ' ' + word
        if font.size(trial)[0] <= width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    return lines + ([current] if current else [])


def sheet(path, title, subtitle, tiles, columns, gap=16, caption_height=84):
    """tiles: list of (surface, caption line 1, caption line 2)."""
    pygame.font.init()
    big, mid, small = (pygame.font.Font(FONT, size) for size in (26, 19, 16))
    tw, th = tiles[0][0].get_size()
    rows = (len(tiles) + columns - 1) // columns
    top = 84
    width = columns * tw + (columns + 1) * gap
    height = top + rows * (th + caption_height + gap)
    surface = pygame.Surface((width, height))
    surface.fill(BG)
    surface.blit(big.render(title, True, INK), (gap, 12))
    surface.blit(small.render(subtitle, True, SOFT), (gap, 50))
    for index, (image, line1, line2) in enumerate(tiles):
        x = gap + (index % columns) * (tw + gap)
        y = top + (index // columns) * (th + caption_height + gap)
        surface.blit(image, (x, y))
        pygame.draw.rect(surface, (60, 66, 78), (x, y, tw, th), 1)
        surface.blit(mid.render(line1, True, INK), (x + 2, y + th + 4))
        for k, part in enumerate(wrap(line2, small, tw - 6)[:2]):
            surface.blit(small.render(part, True, SOFT), (x + 2, y + th + 32 + 22 * k))
    pygame.image.save(surface, str(path))
    return dict(file=str(path), size=[width, height])


def main():
    meta = load_meta()
    numbers = json.loads((GUIDE / 'corner_numbers.json').read_text())
    made = []
    sub = ('what the collection window shows (road part of the 1000x900 window; the car is always at this spot, heading up).  '
           'Yellow bar: car centre → inside road edge.')

    for name in ('C1', 'C4', 'C8'):
        side = SIDE[name]
        entry = numbers['corners'][name]
        t = lambda stem: meta[stem]
        a, b, c, d = (f'{name}_1_turnin', f'{name}_2_apex', f'{name}_3_exit', f'{name}_2_apex_centreline')
        tiles = [
            (tile(a, meta, ROAD_VIEW, 0.75, side), f"① turn-in 진입   t = {t(a)['t']:.2f} s   (along {t(a)['along']:.1f})",
             f"line is {t(a)['offset_abs']:.1f} inside the centre and still moving in;  {edge_text(t(a), side)}"),
            (tile(b, meta, ROAD_VIEW, 0.75, side), f"② apex 정점   t = {t(b)['t']:.2f} s   (along {t(b)['along']:.1f})   {t(b)['offset_abs']:.1f} inside",
             edge_text(t(b), side)),
            (tile(c, meta, ROAD_VIEW, 0.75, side), f"③ exit 탈출   t = {t(c)['t']:.2f} s   (along {t(c)['along']:.1f})",
             f"back to {t(c)['offset_abs']:.1f} from the centre;  {edge_text(t(c), side)}"),
            (tile(d, meta, ROAD_VIEW, 0.75, side), f"④ centre-line lap 중앙선 주행, same place   t = {t(d)['t']:.2f} s   {t(d)['offset_abs']:.1f} inside",
             edge_text(t(d), side)),
        ]
        title = (f"{name}  {'left' if side == 'left' else 'right'} {ANGLE[name]}°  —  reference lap (scriptA/lap1, 41.72 s) vs centre-line lap (42.60 s);  "
                 f"this bend: {entry['A']['section_time']['median']:.2f} s vs {entry['E0']['section_time']:.2f} s")
        made.append(sheet(GUIDE / f'sheet_{name}.png', title, sub, tiles, 2))

    close = pygame.Rect(290, 180, 420, 420)
    tiles = []
    for name in SIDE:
        stem, side = f'{name}_2_apex', SIDE[name]
        info = meta[stem]
        units = info[f'units_to_{side}_edge']
        kind = 'kerb' if info[f'{side}_edge_kind'] in ('white', 'red') else 'grass'
        tiles.append((tile(stem, meta, close, 340 / 420, side),
                      f"{name} {'좌' if side == 'left' else '우'} {ANGLE[name]}°  t={info['t']:.2f}s  {info['offset_abs']:.1f} inside",
                      f"edge {units:.1f} away = {units / CAR_WIDTH:.1f} car widths ({kind})"))
    made.append(sheet(GUIDE / 'sheet_apexes.png', 'Apex of every bend — reference lap scriptA/lap1 (close-up, 30 x 30 track units)',
                      'Yellow bar: car centre → inside road edge on the car\'s screen row.  Car width (wheel to wheel) = 2.76 units;  road = 13.3 wide.',
                      tiles, 4, gap=12, caption_height=62))

    seq = [('C6_1_turnin', 'left', 'C6 ① turn-in'), ('C6_2_apex', 'left', 'C6 ② apex (left side)'), ('C7_1_turnin', 'right', 'C7 ① crossing to the right'),
           ('C7_2_apex', 'right', 'C7 ② apex (right side)'), ('C7_3_exit', 'right', 'C7 ③ exit'), ('C7_2_apex_centreline', 'right', 'centre-line lap at the C7 apex')]
    tiles = []
    for stem, side, label in seq:
        info = meta[stem]
        tiles.append((tile(stem, meta, ROAD_VIEW, 0.5, side), f"{label}   t = {info['t']:.2f} s",
                      f"{info['offset_abs']:.1f} from the centre;  {side} edge {info[f'units_to_{side}_edge']:.1f} away"))
    made.append(sheet(GUIDE / 'sheet_C6C7.png', 'C6 (left 14°) → C7 (right 42°): from the left side to the right side in 0.72 s — reference lap scriptA/lap1',
                      'No kerbs here. Yellow bar: car centre → road edge on the inside of the bend being taken.', tiles, 3, gap=14, caption_height=62))

    seq = [('S0_start_waiting', 'waiting screen before Enter: whole track, zoomed out', 'frame 0 — the car is the dot on the white start line at the right, heading up (the lap runs anticlockwise)'),
           ('S1_start_025', 't = 0.50 s: camera still zooming in, speed already 22.2', 'the zoom-in takes exactly the first 1.0 s of every lap'),
           ('S2_start_050', 't = 1.00 s: normal view from here on (along 17.8)', 'about 37 track units of road visible ahead of the car = 1.6 s'),
           ('S3_straight_mid', 't = 6.32 s: mid-straight between C1 and C2 (along 142.7)', 'car on the centre line: 6.67 to each edge = 2.4 car widths')]
    tiles = [(tile(stem, meta, ROAD_VIEW, 0.75), l1, l2) for stem, l1, l2 in seq]
    made.append(sheet(GUIDE / 'sheet_start.png', 'Start and straight — reference lap scriptA/lap1',
                      'The lap clock starts with the first recorded frame (Enter); the provided human laps wait 12-32 frames (0.24-0.64 s) before accelerating.',
                      tiles, 2))
    tiles = []
    for name in ('C2', 'C3', 'C5'):
        side = SIDE[name]
        for tag, label in (('1_turnin', '① turn-in'), ('2_apex', '② apex'), ('3_exit', '③ exit')):
            info = meta[f'{name}_{tag}']
            tiles.append((tile(f'{name}_{tag}', meta, ROAD_VIEW, 0.5, side), f"{name} {label}   t = {info['t']:.2f} s   (along {info['along']:.1f})",
                          f"{info['offset_abs']:.1f} from the centre;  {edge_text(info, side)}"))
    made.append(sheet(GUIDE / 'sheet_C2C3C5.png', 'The three medium left bends C2 (42°), C3 (19°, long sweep), C5 (32°) — reference lap scriptA/lap1',
                      'Yellow bar: car centre → inside road edge. Kerbs at C2 and C5, none at C3.', tiles, 3, gap=14, caption_height=84))
    print(json.dumps(made, indent=1))


if __name__ == '__main__':
    main()
