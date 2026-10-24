"""Q3: which offline indicator predicts closed-loop success across existing checkpoints."""
import sys; sys.path.insert(0, '/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad/work/diag')
from common import *
from scipy.stats import spearmanr
off = {r['name']: r for r in json.load(open(W / 'offline.json'))}
res = results()
rows = []
for k, r in res.items():
    if k not in off: print('missing offline', k); continue
    d = r['drive']; fin = d['status'] == 'finished'
    o = dict(off[k]); o['name'] = k; o['fin'] = int(fin); o['progress'] = d['progress']; o['center'] = d['center']; o['max_offset'] = d['max_offset']
    o['lap'] = d['lap_time'] if fin else np.nan
    o['clean'] = int(fin and d['progress'] >= 99.9)                       # finished without any off-road stretch
    o['score'] = (100 + d['center']) if fin else d['progress']             # ordinal outcome
    hz = r.get('drive_hud_zero')
    o['fin_hz'] = (int(hz['status'] == 'finished') if hz else np.nan)
    o['drive_steer'] = d['steer_frac']
    o['kit_acc'] = r['train_acc']; o['kit_loss'] = r['train_loss_mean']; o['kit_psr'] = r['pred_steer_rate']
    o['fam'] = k.split('_b')[0]; o['bs'] = r['batch_size']; o['epoch'] = r['epoch']
    rows.append(o)
keys = ['acc', 'kit_loss', 'loss', 'bal_acc', 'rec_L', 'rec_R', 'rec_gas', 'prec_L', 'prec_R', 'psr', 'pL', 'pR', 'wrong_dir', 'pred0_after60', 'margin_med', 'low_margin',
        'flip_hudzero', 'flip_noise2', 'burst_det_L', 'burst_det_R', 'burst_lag_L', 'burst_lag_R', 'net25_mae', 'net25_worst', 'binrec_min', 'binrec_p10', 'rec_R_chicane', 'zoom_pred0']
def auc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, int)
    p, n = x[y == 1], x[y == 0]
    if len(p) == 0 or len(n) == 0: return np.nan
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in p for b in n]))
def table(sub, title):
    print(f'\n== {title}: n={len(sub)}, finished {sum(r["fin"] for r in sub)}, clean(progress=100) {sum(r["clean"] for r in sub)} ==')
    print(f"{'indicator':16s} {'rho(score)':>10s} {'AUC(fin)':>9s} {'AUC(clean)':>10s} {'rho(center|fin)':>15s} {'rho(maxoff|fin)':>15s}   fin: med [min,max]      not fin: med [min,max]")
    out = []
    for k in keys:
        x = np.array([r[k] for r in sub], float); ok = np.isfinite(x)
        s = np.array([r['score'] for r in sub]); f = np.array([r['fin'] for r in sub]); c = np.array([r['clean'] for r in sub])
        rho = spearmanr(x[ok], s[ok]).correlation if ok.sum() > 3 else np.nan
        fm = ok & (f == 1)
        rc = spearmanr(x[fm], np.array([r['center'] for r in sub])[fm]).correlation if fm.sum() > 3 else np.nan
        rm = spearmanr(x[fm], np.array([r['max_offset'] for r in sub])[fm]).correlation if fm.sum() > 3 else np.nan
        a = auc(x[ok], f[ok]); a2 = auc(x[ok], c[ok])
        def rng(m):
            return f"{np.median(x[m]):8.3f} [{x[m].min():7.3f},{x[m].max():7.3f}]" if m.any() else '      -'
        print(f"{k:16s} {rho:+10.2f} {a:9.2f} {a2:10.2f} {rc:+15.2f} {rm:+15.2f}   {rng(ok & (f == 1))}   {rng(ok & (f == 0))}")
        out.append((k, rho, a, a2))
    return out
table(rows, 'ALL checkpoints with a closed-loop result')
table([r for r in rows if r['fam'] in ('cnnA', 'cnnA_crop') and r['bs'] == 256], 'cnnA + cnnA_crop, batch 256 (e25/e50/e100)')
table([r for r in rows if r['fam'] in ('cnnA', 'cnnA_crop') and r['bs'] == 256 and r['epoch'] >= 50], 'cnnA + cnnA_crop, batch 256, epochs >= 50')
table([r for r in rows if r['fam'] == 'base'], 'starter arch (b256 e25/50/100 + b4096 e100)')
table([r for r in rows if r['acc'] < 0.85], 'all checkpoints with train acc < 0.85 (undertrained regime)')
print('\n== per-checkpoint table ==')
cols = ['acc', 'loss', 'rec_L', 'rec_R', 'rec_gas', 'psr', 'wrong_dir', 'pred0_after60', 'burst_det_L', 'burst_det_R', 'burst_lag_L', 'burst_lag_R', 'net25_mae', 'net25_worst', 'binrec_min', 'binrec_min_at', 'rec_R_chicane', 'flip_hudzero', 'zoom_pred0']
print(f"{'name':26s} fin hz  prog  ctr maxoff   lap dsteer " + ' '.join(c[:8].rjust(8) for c in cols))
for r in sorted(rows, key=lambda r: (r['fam'], r['bs'], r['epoch'], r['name'])):
    hz = '-' if not np.isfinite(r['fin_hz']) else str(int(r['fin_hz']))
    print(f"{r['name']:26s} {r['fin']:3d} {hz:>2s} {r['progress']:5.1f} {r['center']:4.0f} {r['max_offset']:6.1f} {r['lap']:5.2f} {r['drive_steer']:5.2f}  " + ' '.join(f"{r[c]:8.3f}" for c in cols))
if 'baseline' in off:
    r = off['baseline']; print(f"{'baseline.pth':26s}   1  - 100.0   51    -   45.54   0.49  " + ' '.join(f"{r[c]:8.3f}" for c in cols))
json.dump(rows, open(W / 'offline_joined.json', 'w'), indent=1)
