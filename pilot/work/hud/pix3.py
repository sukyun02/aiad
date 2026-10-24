import numpy as np
np.set_printoptions(linewidth=250, precision=3, suppress=True)
SP='/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad'
d = np.load(SP+'/cache/data.npz')
obs, lap, frame, labels = d['obs'], d['lap'], d['frame'], d['labels']
late = frame >= 150
o = obs[late]
# digit string width: rightmost nonzero col in rows 86:92
w = np.array([np.max(np.where(x.max(0) > 0)[0]) + 11 if x.max() > 0 else 0 for x in o[:, 86:92, 11:30, 0]])
print('rightmost digit column distribution', np.unique(w, return_counts=True))
for r in (92, 93):
    for col, ch in [(12, 0), (18, 2), (20, 2), (22, 2), (24, 2)]:
        v = o[:, r, col, ch]
        print('row', r, 'col', col, 'uniq', dict(zip(*[a.tolist() for a in np.unique(v, return_counts=True)])) if len(np.unique(v)) < 12 else ('n_unique', len(np.unique(v)), 'min', v.min(), 'max', v.max()),
              'mean by rightmost digit col:', {int(k): round(float(v[w == k].mean()), 1) for k in np.unique(w)})
# abs3 clean (rows 92:94, col 24, B) relation to steering label / gyro within fixed digit width
abs3 = o[:, 92:94, 24, 2].astype(float).sum(1) / 255 / 5
gy = o[:, 88, 58:86, 0].astype(float) / 255 / 5; gy[:, :14] *= -1; gy = gy.sum(1)
lab = labels[late]
for k in np.unique(w):
    m = w == k
    print('width', k, 'n', m.sum(), 'abs3 mean %.4f std %.4f' % (abs3[m].mean(), abs3[m].std()), 'corr(abs3,|gyro|) %.3f' % (np.corrcoef(abs3[m], np.abs(gy[m]))[0, 1] if abs3[m].std() > 0 else float('nan')),
          'abs3 mean by label', {int(c): round(float(abs3[m & (lab == c)].mean()), 4) for c in (1, 5, 6) if (m & (lab == c)).any()})
