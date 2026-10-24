import json, sys, collections
S='/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad'
names = sys.argv[1:] or ['dyn_cnnA']
rows = collections.defaultdict(list)
for n in names:
    for l in open(f'{S}/results/{n}.jsonl'):
        r = json.loads(l); rows[(r['cand'], r['tag'], r['batch_size'], r['lr'], r['epoch'])].append(r)
def cell(d):
    if d is None: return '   -  '
    return f"{d['lap_time']:6.2f}" if d['status']=='finished' and d.get('lap_time') else f"ab@{d['progress']:3.0f}"
for key in sorted(rows, key=lambda k:(k[0],k[1],k[4])):
    rs = sorted(rows[key], key=lambda r:r['seed'])
    steps = key[4]*-(-15067//key[2])
    out = f"{key[1]:18s} e{key[4]:<3d} steps{steps:6d} acc " + ' '.join(f"{r['train_acc']:.3f}" for r in rs)
    for kind in ['drive','drive_hud_zero','drive_noise2']:
        if kind in rs[0]:
            out += f" | {kind[6:] or 'cuda':8s} " + ' '.join(cell(r.get(kind)) for r in rs)
    out += " | maxoff " + ' '.join(f"{r['drive']['max_offset']:4.1f}" for r in rs)
    print(out)
