"""How similar are the trajectories of the final model and the two extra models? (numpy only, read-only)"""
import numpy as np

R = '/home/sukyun02/workspace/aiad/pilot/work/report'
ref = np.load(f'{R}/final_model/trace.npz')['rows']
ref_cls = ref[:, 3].astype(int)
w0 = int(np.argmax(np.isin(ref_cls, (1, 5, 6))))
for name in ('m_20261006_232759', 'm_20261006_233500', 'stage1_model'):
    r = np.load(f'{R}/{name}/trace.npz')['rows']
    c = r[:, 3].astype(int)
    w = int(np.argmax(np.isin(c, (1, 5, 6))))
    a, b = ref[w0:], r[w:]            # align at the first gas frame
    m = min(len(a), len(b))
    ca, cb = a[:m, 3].astype(int), b[:m, 3].astype(int)
    diff = np.flatnonzero(ca != cb)
    d = np.hypot(a[:m, 0] - b[:m, 0], a[:m, 1] - b[:m, 1])
    print('=====', name, 'wait', w, 'vs final', w0, 'aligned frames', m)
    print('  first class mismatch at aligned frame', int(diff[0]) if len(diff) else None, '| mismatches in first 500/1000/1500:',
          int((diff < 500).sum()), int((diff < 1000).sum()), int((diff < 1500).sum()), '| total', len(diff))
    for thr in (0.001, 0.01, 0.1, 1.0, 3.0):
        k = int(np.argmax(d > thr)) if (d > thr).any() else -1
        print(f'  first aligned frame with position difference > {thr}: {k}' + (f' (along {a[k, 5]:.1f})' if k >= 0 else ''))
    print('  max position difference in first 500 / 1000 aligned frames', float(d[:500].max()), float(d[:1000].max()))
    if len(diff):
        k = int(diff[0])
        print('  around first mismatch: final classes', ca[max(0, k - 5):k + 10].tolist(), 'other', cb[max(0, k - 5):k + 10].tolist(), 'along', round(float(a[k, 5]), 1))
