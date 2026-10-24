import sys
S = sys.argv[1]; tier = sys.argv[2]
ck = {}
for ep, suf in ((50, '_e50'), (100, ''), (25, '_e25')):
    for arch in ('cnnA', 'cnnA_crop'):
        for s in range(4):
            ck[f'{arch}-e{ep}-s{s}'] = f'{S}/cands/{arch}/runs/b256_e100/s{s}{suf}.pth'
ck['baseline-e0-s0'] = f'{S}/blobs/baseline.pth'
ck['base4096-e100-s0'] = f'{S}/cands/base/runs/b4096_e100/s0.pth'
ck['base4096-e100-s1'] = f'{S}/cands/base/runs/b4096_e100/s1.pth'
C = '--device cpu '
T = {
 'A': [('cuda', 'k', ''), ('cpu', 'k', '--device cpu'),
       ('n2s0', 'k', C + '--noise 2 --noise-seed 0'), ('hudzero', 'k', C + '--hud zero'),
       ('n1s0', 'k', C + '--noise 1 --noise-seed 0'), ('n4s0', 'k', C + '--noise 4 --noise-seed 0'),
       ('hudshift', 'k', C + '--hud shift'), ('huddim', 'k', C + '--hud dim'),
       ('thr4', 'r', '--device cpu --threads 4'), ('ln3', 'r', '--device cpu --logit-noise 0.001 --seed 0')],
 'B': [('n2s1', 'k', C + '--noise 2 --noise-seed 1'), ('n1s1', 'k', C + '--noise 1 --noise-seed 1'),
       ('n4s1', 'k', C + '--noise 4 --noise-seed 1'), ('n2s2', 'k', C + '--noise 2 --noise-seed 2'),
       ('n1s2', 'k', C + '--noise 1 --noise-seed 2'), ('n4s2', 'k', C + '--noise 4 --noise-seed 2')],
}
for pname, tool, opts in T[tier]:
    for cname, path in ck.items():
        print(f'{cname}__{pname}\t{path}\t{tool}\t{opts}')
