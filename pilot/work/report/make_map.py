"""Track map with the driven lines of the stage-1 model and the final model (system python3 + matplotlib).
usage: make_map.py OUT.png"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D

FONT = font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
BLUE, ORANGE, INK, MUTED, ROAD = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#d9dadd'
CORNERS = [('C1', 45.5, 77.0), ('C2', 248.5, 273.0), ('C3', 381.5, 399.0), ('C4', 518.0, 549.5), ('C5', 609.0, 633.5),
           ('C6', 728.0, 738.5), ('C7', 749.0, 766.5), ('C8', 879.0, 913.5)]

final = np.load('final_model/trace.npz')
stage1 = np.load('stage1_model/trace.npz')
rows, rows1 = final['rows'], stage1['rows']
xy, off, along = rows[:, :2], rows[:, 2] > 0.5, rows[:, 5]
road, colors, centre = final['road'], final['colors'], final['centre']

# off-road runs of the final model: where on the track, how long, how far from the centre line
runs, start = [], None
for i, flag in enumerate(np.r_[off, False]):
    if flag and start is None:
        start = i
    elif not flag and start is not None:
        seg = slice(start, i)
        mid = float(np.median(along[seg]))
        near = min(CORNERS, key=lambda c: min(abs(mid - c[1]), abs(mid - c[2]), 0 if c[1] <= mid <= c[2] else 1e9))
        runs.append((start, i - start, round(float(along[start]), 1), round(float(along[i - 1]), 1), round(float(rows[seg, 6].max()), 1), near[0]))
        start = None
print('off-road runs (first frame, frames, along from, along to, max distance from centre line, nearest bend):')
for run in runs:
    print('  ', run)
print('off-road frames', int(off.sum()), 'of', len(off), f'= {100 * off.mean():.1f}%')

is_road = (np.abs(colors[:, 0] - colors[:, 1]) < 20) & (colors[:, 0] < 200)          # grey tiles; kerbs are red / white
fig, ax = plt.subplots(figsize=(4.6, 2.5), dpi=300)
ax.add_collection(PolyCollection(road[is_road], facecolors=ROAD, edgecolors=ROAD, linewidths=0.3))
ax.plot(rows1[:, 0], rows1[:, 1], color=MUTED, lw=0.9, ls=(0, (4, 2)), zorder=3)
points = xy.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
seg_off = off[1:]
ax.add_collection(LineCollection(segments[~seg_off], colors=BLUE, linewidths=1.3, zorder=4, capstyle='round'))
ax.add_collection(LineCollection(segments[seg_off], colors=ORANGE, linewidths=2.4, zorder=5, capstyle='round'))
# start / finish and driving direction
p0, p1 = centre[0], centre[3]
direction = (p1 - p0) / np.linalg.norm(p1 - p0)
normal = np.array([-direction[1], direction[0]])
ax.plot(*np.c_[p0 - 9 * normal, p0 + 9 * normal], color=INK, lw=1.6, zorder=6)
a0 = p0 - 22 * normal
ax.annotate('', xy=a0 + 38 * direction, xytext=a0 - 4 * direction, arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1), zorder=6)
ax.text(*(p0 - 30 * normal - 2 * direction), '출발 · 도착', fontproperties=FONT, fontsize=8.8, color=INK, ha='left', va='center')
ax.set_aspect('equal')
pad = 10
ax.set_xlim(road[..., 0].min() - pad, road[..., 0].max() + pad + 74)
ax.set_ylim(road[..., 1].min() - pad, road[..., 1].max() + pad)
ax.axis('off')
handles = [Line2D([], [], color=MUTED, lw=0.9, ls=(0, (4, 2)), label='제공 데이터로 학습한 모델 43.28초'),
           Line2D([], [], color=BLUE, lw=1.3, label='최종 모델 40.94초: 도로 위'),
           Line2D([], [], color=ORANGE, lw=2.4, label=f'최종 모델: 도로 밖 ({100 * off.mean():.1f}% 프레임)')]
ax.legend(handles=handles, prop=font_manager.FontProperties(fname=FONT.get_file(), size=8.6), loc='center', bbox_to_anchor=(0.43, 0.57),
          frameon=False, handlelength=2.2, labelcolor=INK)
longest = max(runs, key=lambda r: r[1])
mid = xy[longest[0] + longest[1] // 2]
ax.annotate(f'{longest[1] / 50:.2f}초 연속 이탈 (한도 5초)', xy=mid, xytext=(mid[0] - 62, mid[1] + 30), fontproperties=FONT, fontsize=8.6,
            color=INK, ha='center', va='center', arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6, shrinkB=2), zorder=7)
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(sys.argv[1], dpi=300, facecolor='white')
print('saved', sys.argv[1], 'xlim', ax.get_xlim(), 'ylim', ax.get_ylim())
