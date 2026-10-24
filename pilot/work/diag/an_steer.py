"""Q2: steering timing/frequency along the lap: experts vs cnnA vs starter. Per 40 m bin left/right command share, burst stats, corner-entry onset."""
import sys; sys.path.insert(0, '/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad/work/diag')
from common import *
E = experts()
groups = {
 'expert(7)': E,
 'cnnA_e100(4)': [load(f'cnnA_b256_s{s}_e100')[0] for s in range(4)],
 'cnnA_e50(4)': [load(f'cnnA_b256_s{s}_e50')[0] for s in range(4)],
 'cnnA_e25(4)': [load(f'cnnA_b256_s{s}_e25')[0] for s in range(4)],
 'base_b4096_fin(s0,s1)': [load(f'base_b4096_s{s}_e100')[0] for s in (0, 1)],
 'baseline.pth': [load('baseline')[0]],
}
def bursts(r):
    d = DIRS[r[:, CLS].astype(int)]; out = []; i = 0
    while i < len(d):
        if d[i] != 0:
            j = i
            while j + 1 < len(d) and d[j + 1] == d[i]: j += 1
            out.append((i, j - i + 1, d[i])); i = j + 1
        else: i += 1
    return out
print('== global steering statistics (per lap; mean over group members) ==')
print(f"{'group':24s} frames steer% L% R%  bursts  medlen p90len  maxlen  |steer_state| mean  sat%(|s|>=0.44)  mean|off| max|off| mean|head|deg  rms_lat_vel")
for g, rs in groups.items():
    v = []
    for r in rs:
        r = r[60:]                      # skip zoom-in
        d = DIRS[r[:, CLS].astype(int)]; bs = bursts(r); ln = np.array([b[1] for b in bs]) if bs else np.array([0])
        v.append([len(r) + 60, 100 * (d != 0).mean(), 100 * (d < 0).mean(), 100 * (d > 0).mean(), len(bs), np.median(ln), np.percentile(ln, 90), ln.max(),
                  np.abs(r[:, STEER]).mean(), 100 * (np.abs(r[:, STEER]) >= 0.44).mean(), np.abs(r[:, OFF]).mean(), np.abs(r[:, OFF]).max(), np.degrees(np.abs(r[:, HEAD])).mean(),
                  np.sqrt(np.mean((np.diff(r[:, OFF]) * 50) ** 2))])
    v = np.mean(v, axis=0)
    print(f"{g:24s} {v[0]:6.0f} {v[1]:5.1f} {v[2]:5.1f} {v[3]:4.1f} {v[4]:6.1f} {v[5]:6.1f} {v[6]:6.1f} {v[7]:6.1f} {v[8]:10.3f} {v[9]:12.1f} {v[10]:10.2f} {v[11]:6.2f} {v[12]:8.2f} {v[13]:8.2f}")
print()
print('== left / right command share (%) per 40 m of track (frames >= 60) ==')
edges = np.arange(0, 960, 40)
print('bin    curv ' + ' '.join(f'{g[:14]:>15s}' for g in groups))
T = np.load(W / 'track.npz')
for e in edges:
    seg = (T['starts'] >= e) & (T['starts'] < e + 40)
    turn = np.degrees(T['dtheta'][seg].sum())
    line = f'{e:3d}-{e + 40:3d} {turn:+5.0f}'
    for g, rs in groups.items():
        Ls, Rs = [], []
        for r in rs:
            r = r[60:]; m = (r[:, ALONG] >= e) & (r[:, ALONG] < e + 40)
            if m.sum() == 0: continue
            d = DIRS[r[m, CLS].astype(int)]; Ls.append(100 * (d < 0).mean()); Rs.append(100 * (d > 0).mean())
        line += f'   L{np.mean(Ls):4.0f} R{np.mean(Rs):4.0f}  ' if Ls else '      -        '
    print(line)
print()
print('== corner entry: along-track position of first sustained command in the corner direction (>=5 of next 10 frames), searched from corner_start-60 ==')
print('corner                     ' + ' '.join(f'{g[:18]:>22s}' for g in groups))
for (s, e, dch, turn, k) in CORNERS:
    want = -1 if dch == 'L' else 1
    line = f'{dch} {s:5.0f}-{e:5.0f} {turn:+4.0f}deg  '
    for g, rs in groups.items():
        pos = []; steer_at_start = []; off_apex = []
        for r in rs:
            d = DIRS[r[:, CLS].astype(int)]; a = r[:, ALONG]
            idx = np.where((a >= s - 60) & (a <= e) & (np.arange(len(a)) > 60))[0]
            found = None
            for i in idx:
                if d[i] == want and (d[i:i + 10] == want).sum() >= 5: found = a[i]; break
            pos.append(found if found is not None else np.nan)
            m = (a >= s) & (a <= e)
            if m.any(): off_apex.append(r[m, OFF][np.abs(r[m, OFF]).argmax()])
        line += f"  {np.nanmean(pos) - s:+6.1f} [{np.nanmin(pos) - s:+5.0f},{np.nanmax(pos) - s:+5.0f}] off{np.mean(off_apex):+5.1f}"
    print(line)
print('(value = onset position minus corner start, in track units (22.5 units/s => 1 unit = 44 ms, 0.45 units/frame); off = signed peak offset inside the corner, + = left of centre)')
