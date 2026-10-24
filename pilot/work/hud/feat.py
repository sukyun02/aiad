import numpy as np, json
np.set_printoptions(linewidth=250, precision=3, suppress=True)
SP='/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad'
d = np.load(SP+'/cache/data.npz')
obs, lap, frame, labels = d['obs'], d['lap'], d['frame'], d['labels']
o = obs.astype(np.float32)
sp_o = o[:, 84:94, 12, 0].sum(1) / 255 / 5
abs_o = o[:, 84:94, 18:25:2, 2].sum(1) / 255 / 5
sp_c = o[:, 92:94, 12, 0].sum(1) / 255 / 5
abs_c = o[:, 92:94, 18:25:2, 2].sum(1) / 255 / 5
steer = o[:, 88, 38:58, 1] / 255 / 10; steer[:, :10] *= -1; steer = steer.sum(1)
gyro = o[:, 88, 58:86, 0] / 255 / 5; gyro[:, :14] *= -1; gyro = gyro.sum(1)
print('max rows 84:87 cols 0:36:', obs[:, 84:87, 0:36].max(), ' rows 94:96:', obs[:, 94:96].max())
print('rows 92:94 nonzero columns:', np.where(obs[:, 92:94].max(axis=(0, 1, 3)) > 0)[0])
print('rows 86:92 nonzero columns below col 36:', np.where(obs[:, 86:92, :36].max(axis=(0, 1, 3)) > 0)[0])
print('rows 86:92: max |R-G|,|R-B| cols 11:30 (digits white?)', np.abs(o[:, 86:92, 11:30, 0]-o[:, 86:92, 11:30, 1]).max(), np.abs(o[:, 86:92, 11:30, 0]-o[:, 86:92, 11:30, 2]).max())
late = frame >= 150
def st(name, v, m):
    x = v[m]
    print(f'{name:12s} min {x.min():.3f} p1 {np.percentile(x,1):.3f} med {np.median(x):.3f} p99 {np.percentile(x,99):.3f} max {x.max():.3f} std {x.std():.4f} n_unique {len(np.unique(np.round(x,4)))}')
print('\n=== cruise frames (frame>=150), n=%d ===' % late.sum())
for name, v in [('speed orig', sp_o), ('speed clean', sp_c)] + [(f'abs{k} orig', abs_o[:, k]) for k in range(4)] + [(f'abs{k} clean', abs_c[:, k]) for k in range(4)] + [('steer', steer), ('gyro', gyro)]:
    st(name, v, late)
print('\n=== all frames ===')
for name, v in [('speed orig', sp_o), ('speed clean', sp_c)] + [(f'abs{k} orig', abs_o[:, k]) for k in range(4)] + [(f'abs{k} clean', abs_c[:, k]) for k in range(4)] + [('steer', steer), ('gyro', gyro)]:
    st(name, v, np.ones(len(v), bool))
# digit dependence: variance of feature explained by digit pixels in the column it reads
def r2_by_group(v, keys):
    _, inv = np.unique(keys, axis=0, return_inverse=True); inv = inv.ravel()
    mean = np.bincount(inv, weights=v) / np.bincount(inv)
    return round(1 - (v - mean[inv]).var() / max(v.var(), 1e-12), 4), len(mean)
print('\nfraction of variance explained by the digit pixels (rows 86:92) of the column that the feature reads, cruise frames: (R2, n_groups)')
print(' speed orig', r2_by_group(sp_o[late], obs[late][:, 86:92, 12, 0]), ' speed clean', r2_by_group(sp_c[late], obs[late][:, 86:92, 12, 0]))
for k, col in enumerate([18, 20, 22, 24]):
    ds = o[late][:, 86:92, col, 2].sum(1)
    print(f' abs{k}: corr(orig, digit col{col} sum) = {np.corrcoef(abs_o[late, k], ds)[0,1]:.3f} ; corr(clean, digit sum) = {np.corrcoef(abs_c[late, k], ds)[0,1]:.3f}; R2 orig {r2_by_group(abs_o[late,k], obs[late][:, 86:92, col, 2])} clean {r2_by_group(abs_c[late,k], obs[late][:, 86:92, col, 2])}')
ds = o[late][:, 86:92, 12, 0].sum(1)
print(' speed: corr(orig, digit col12 sum) = %.3f ; corr(clean, digit sum) = %.3f' % (np.corrcoef(sp_o[late], ds)[0,1], np.corrcoef(sp_c[late], ds)[0,1]))
# exact identity: orig = clean + digit-rows contribution
print(' identity check max|orig - (clean + digitrows/255/5)| speed:', np.abs(sp_o - sp_c - o[:, 84:92, 12, 0].sum(1)/255/5).max())
# does blanking the digits (kit HUD_BLOB rows 86:92 cols 11:30) change the clean features? (must be exactly 0)
z = o.copy(); z[:, 86:92, 11:30] = 0
print(' after hud_zero: max|clean speed change|', np.abs(z[:, 92:94, 12, 0].sum(1)/255/5 - sp_c).max(), ' max|clean abs change|', np.abs(z[:, 92:94, 18:25:2, 2].sum(1)/255/5 - abs_c).max(),
      ' max|orig speed change|', np.abs(z[:, 84:94, 12, 0].sum(1)/255/5 - sp_o).max(), ' max|orig abs change|', np.abs(z[:, 84:94, 18:25:2, 2].sum(1)/255/5 - abs_o).max(),
      ' steer change', np.abs((z[:, 88, 38:58, 1]-o[:, 88, 38:58, 1])).max(), ' gyro change', np.abs((z[:, 88, 58:86, 0]-o[:, 88, 58:86, 0])).max())
# digits: time or tile dependent? compare digit pattern across laps at same frame index vs same along-track
dig = obs[:, 86:92, 11:30, 0].reshape(len(obs), -1)
print('\nunique digit patterns in data:', len(np.unique(dig, axis=0)), 'of', len(dig), 'frames')
for L in range(7):
    idx = np.where(lap == L)[0]; idx = idx[np.argsort(frame[idx])]
    ch = (np.abs(np.diff(dig[idx].astype(int), axis=0)).sum(1) > 0)
    print(' lap', L, 'frames', len(idx), 'frames where digit pattern changed vs previous frame:', int(ch.sum()), ' first 12 change frames', (np.where(ch)[0][:12]+1).tolist())
# same pattern in different laps?
keys = [set(map(bytes, dig[lap == L])) for L in range(7)]
print(' lap0 patterns:', len(keys[0]), ' shared with lap1:', len(keys[0] & keys[1]), ' shared by all 7:', len(set.intersection(*keys)))
print('\n=== first frames of lap 0 and lap 3: lap frame label | speed_orig speed_clean | abs_orig x4 | abs_clean x4 | steer gyro ===')
for L in (0, 3):
    idx = np.where(lap == L)[0]; idx = idx[np.argsort(frame[idx])]
    for i in list(idx[:6]) + list(idx[6:120:6]) + list(idx[120:400:40]):
        print(L, frame[i], labels[i], f'| {sp_o[i]:.3f} {sp_c[i]:.3f} |', abs_o[i], '|', abs_c[i], f'| {steer[i]:.3f} {gyro[i]:.3f}')
print('\nclass 0 frames per lap (frame indices)')
for L in range(7):
    idx = np.where((lap == L) & (labels == 0))[0]
    print(L, sorted(frame[idx].tolist())[:40], '... n=', len(idx))
print('highest row with blue-only pixel (B-G>40) in cols 16:27:', 84 + np.where(((o[:, 84:94, 16:27, 2] - o[:, 84:94, 16:27, 1]) > 40).any(axis=(0, 2)))[0])
print('clean maxima: abs', abs_c.max(0), 'speed', sp_c.max())
print('speed clean uniques at cruise:', np.unique(np.round(sp_c[late],3), return_counts=True))
for k in range(4): print(f'abs{k} clean uniques at cruise:', np.unique(np.round(abs_c[late,k],3), return_counts=True))
# is clean abs informative of steering label?  mean by label
for c in (0,1,5,6):
    m = late & (labels == c)
    print('label', c, 'n', m.sum(), 'mean clean speed %.4f abs' % sp_c[m].mean(), abs_c[m].mean(0), 'steer %.3f gyro %.3f' % (steer[m].mean(), gyro[m].mean()))
