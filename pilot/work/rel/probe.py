import sys, json, subprocess, pathlib
from concurrent.futures import ThreadPoolExecutor
S = pathlib.Path('/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad')
K = str(S / 'kit' / 'k')
out = S / 'work' / 'rel' / sys.argv[1]
cand, tag = sys.argv[2], sys.argv[3]
kinds = sys.argv[4].split(',')
FL = {'cuda': [], 'cpu': ['--device', 'cpu'], 'noise2': ['--noise', '2'], 'noise2b': ['--noise', '2', '--noise-seed', '1'],
      'noise4': ['--noise', '4'], 'hud_zero': ['--hud', 'zero']}
jobs = []
for s in range(4):
    for ep, suffix in ((25, '_e25'), (50, '_e50'), (100, '')):
        m = S / 'cands' / cand / 'runs' / tag / f's{s}{suffix}.pth'
        if not m.exists():
            continue
        for k in kinds:
            jobs.append((s, ep, k, m))
def go(j):
    s, ep, k, m = j
    r = subprocess.run([K, 'eval', str(m), '--slot'] + FL[k], capture_output=True, text=True)
    try:
        d = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        d = {'status': 'ERROR', 'err': r.stderr[-500:]}
    rec = dict(cand=cand, tag=tag, seed=s, epoch=ep, kind=k, **d)
    with open(out, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    return rec
with ThreadPoolExecutor(6) as p:
    list(p.map(go, jobs))
print('done', len(jobs))
