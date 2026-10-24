"""Q4: behaviour during the zoom-in (first ~60 frames) and class-0 usage."""
import sys; sys.path.insert(0, '/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad/work/diag')
from common import *
def rle(c):
    out = []; i = 0
    while i < len(c):
        j = i
        while j + 1 < len(c) and c[j + 1] == c[i]: j += 1
        out.append(f'{int(c[i])}x{j - i + 1}'); i = j + 1
    return ' '.join(out)
def line(n, r):
    c = r[:, CLS].astype(int); sp = r[:, SPEED]
    mv = np.argmax(sp > 0.5) if (sp > 0.5).any() else -1
    g = np.argmax(c != 0) if (c != 0).any() else -1
    st = np.where(DIRS[c[:80]] != 0)[0]
    late0 = int((c[80:] == 0).sum())
    f20 = np.argmax(sp > 20) if (sp > 20).any() else -1
    print(f"{n:34s} first_non0 f{g:3d} moving f{mv:3d} v>20 f{f20:4d} steer_frames<80: {len(st):2d} off@f100 {r[min(100, len(r) - 1), OFF]:+.2f} off@f150 {r[min(150, len(r) - 1), OFF]:+.2f} cls0_after_f80 {late0:4d} | first80: {rle(c[:80])[:90]}")
for i, r in enumerate(experts()): line(f'expert_lap{i}', r)
for n in names(): 
    if 'hudzero' in n: continue
    line(n, load(n)[0])
