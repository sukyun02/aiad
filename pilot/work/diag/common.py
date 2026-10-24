import json, glob, sys
from pathlib import Path
import numpy as np
SP = Path('/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad')
W = SP / 'work/diag'
HALF = 6.6667; L = 959.0
CORNERS = json.load(open(W / 'corners.json'))
DIRS = np.zeros(9); DIRS[[3, 5, 7]] = -1; DIRS[[4, 6, 8]] = 1      # -1 = left command, +1 = right command
# trace columns
ALONG, OFF, CLS, MARGIN, SPEED, STEER, HEAD, DEP, FORCED = range(9)

def load(name):
    z = np.load(W / f'traces/{name}.npz', allow_pickle=True)
    meta = json.loads(str(z['meta'])) if 'meta' in z.files else {}
    return z['rows'], meta, (z['logits'] if 'logits' in z.files else None)

def experts():
    return [np.load(W / f'traces/expert_lap{i}.npz')['rows'] for i in range(7)]

def unwrap(along):
    """along starts near 959 (start line) then wraps to 0; return monotone-ish distance from start."""
    a = along.astype(np.float64).copy()
    d = np.diff(a); d = (d + L / 2) % L - L / 2
    return np.r_[0.0, np.cumsum(d)] + (a[0] - L if a[0] > L / 2 else a[0])

def corner_of(a):
    a = a % L
    for i, (s, e, d, turn, k) in enumerate(CORNERS):
        if s - 25 <= a <= e + 25:
            return f'C{i}{d}({s:.0f}-{e:.0f},{turn:+.0f}deg)'
    return 'straight'

def names(pattern='*'):
    return sorted(Path(p).stem for p in glob.glob(str(W / f'traces/{pattern}.npz')) if not Path(p).stem.startswith('expert'))

def results():
    """closed-loop results from kit results/*.jsonl keyed by trace name."""
    out = {}
    for c in ('cnnA', 'cnnA_crop', 'base'):
        for line in open(SP / f'results/{c}.jsonl'):
            r = json.loads(line)
            bs = r['batch_size']; key = f"{c}_b{bs}_s{r['seed']}_e{r['epoch']}"
            out[key] = r
    return out
