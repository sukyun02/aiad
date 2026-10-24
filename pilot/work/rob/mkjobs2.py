import sys
S = sys.argv[1]
main, e25, base = {}, {}, {}
for ep, suf, d in ((50, '_e50', main), (100, '', main), (25, '_e25', e25)):
    for arch in ('cnnA', 'cnnA_crop'):
        for s in range(4):
            d[f'{arch}-e{ep}-s{s}'] = f'{S}/cands/{arch}/runs/b256_e100/s{s}{suf}.pth'
base['baseline-e0-s0'] = f'{S}/blobs/baseline.pth'
base['base4096-e100-s0'] = f'{S}/cands/base/runs/b4096_e100/s0.pth'
base['base4096-e100-s1'] = f'{S}/cands/base/runs/b4096_e100/s1.pth'
allc = {**main, **e25, **base}; rest = {**e25, **base}
C = '--device cpu '
P = {'cpu': ('k', '--device cpu'), 'hudzero': ('k', C + '--hud zero'), 'hudshift': ('k', C + '--hud shift'), 'huddim': ('k', C + '--hud dim'),
     'thr4': ('r', '--device cpu --threads 4'), 'ln3': ('r', '--device cpu --logit-noise 0.001 --seed 0')}
for k in (1, 2, 4):
    for s in (0, 1, 2):
        P[f'n{k}s{s}'] = ('k', C + f'--noise {k} --noise-seed {s}')
plan = [(['cpu'], allc), (['n2s0', 'n1s0'], main), (['hudzero', 'n4s0', 'n2s1'], main), (['n2s0', 'n1s0'], rest),
        (['thr4', 'ln3'], main), (['n1s1', 'hudshift', 'huddim'], main), (['n4s1', 'n2s2', 'n1s2'], main),
        (['n4s0', 'hudzero', 'n2s1', 'n1s1'], rest), (['n4s2'], main), (['thr4', 'ln3', 'hudshift', 'huddim', 'n4s1', 'n2s2', 'n1s2', 'n4s2'], rest)]
for probes, cks in plan:
    for pn in probes:
        for cn, path in cks.items():
            print(f'{cn}__{pn}\t{path}\t{P[pn][0]}\t{P[pn][1]}')
