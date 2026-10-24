import numpy as np
np.set_printoptions(linewidth=250)
d = np.load('/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad/cache/data.npz')
obs, lap, frame, labels = d['obs'], d['lap'], d['frame'], d['labels']
print(obs.shape, np.bincount(lap))
hud = obs[:, 84:96]
for c, n in enumerate('RGB'):
    nz = (hud[..., c] > 0).mean(0)
    print('channel', n, 'fraction nonzero x99 (rows 84..95 x cols 0..63)')
    for r in range(12):
        print(84 + r, ''.join(f'{int(round(v*99)):2d}' if v > 0 else ' .' for v in nz[r, :64]))
for (L, F) in [(0, 800), (0, 0), (0, 10), (0, 30), (0, 60), (0, 100)]:
    i = np.where((lap == L) & (frame == F))[0][0]
    for c, n in enumerate('RGB'):
        print('lap', L, 'frame', F, 'channel', n, 'label', labels[i])
        for r in range(84, 96):
            print(r, ' '.join(f'{v:3d}' for v in obs[i, r, 8:32, c]))
