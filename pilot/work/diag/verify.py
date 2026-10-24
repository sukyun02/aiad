import sys; sys.path.insert(0, '/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad/work/diag')
from common import *
res = results(); ok = bad = 0
for n in names():
    r, meta, _ = load(n)
    hz = n.endswith('_hudzero'); k = n.replace('_hudzero', '')
    if k not in res: continue
    d = res[k].get('drive_hud_zero') if hz else res[k]['drive']
    if d is None: continue
    same = d['status'] == meta['status'] and d['frames'] == meta['frames'] and abs(d['progress'] - meta['progress']) < 0.01 and abs(d['center'] - meta['center']) < 0.01
    ok += same; bad += (not same)
    if not same: print('MISMATCH', n, d['status'], d['frames'], d['progress'], '| trace:', meta['status'], meta['frames'], meta['progress'])
print('verified identical to kit results:', ok, 'mismatch:', bad)
