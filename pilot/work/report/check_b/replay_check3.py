"""Where does the final model leave training lap 20261006_225214_080995? (numpy only, read-only)"""
import numpy as np

R = '/home/sukyun02/workspace/aiad/pilot/work/report'
laps = np.load(f'{R}/check_b/lap_classes.npz')
lap = laps['20261006_225214_080995']


def gas0(c):
    return int(np.argmax(np.isin(c, (1, 5, 6))))


for name in ('final_model', 'm_20261006_232759'):
    rows = np.load(f'{R}/{name}/trace.npz')['rows']
    c = rows[:, 3].astype(int)
    g, gl = gas0(c), gas0(lap)
    a, b = c[g:], lap[gl:]
    m = min(len(a), len(b))
    diff = np.flatnonzero(a[:m] != b[:m])
    k = int(diff[0])
    f = g + k
    print(name, 'first gas', g, 'lap first gas', gl, 'identical aligned frames', k, '-> model frame', f, f't={f / 50:.2f}s', 'along', round(float(rows[f, 5]), 1),
          'err', round(float(rows[f, 6]), 2), 'off', bool(rows[f, 2] > 0.5))
    print('  model classes around', a[k - 6:k + 12].tolist())
    print('  lap   classes around', b[k - 6:k + 12].tolist())
    print('  total mismatches after that (same aligned index)', len(diff), 'of', m - k, '; steering frames model', int(np.isin(a, (5, 6)).sum()), 'lap', int(np.isin(b, (5, 6)).sum()))
    print('  share of the lap replayed exactly: %.1f%% of model frames, along %.1f of 959 = %.1f%%' % (100 * f / len(c), rows[f, 5], 100 * rows[f, 5] / 959))
# agreement of identical-prefix lengths: every model vs every training lap already printed in replay_check2
